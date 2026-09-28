from __future__ import annotations

from pathlib import Path
from typing import Mapping

from sts1_llm_policy.model_runtime import (
    BaseModelRuntimeConfig,
    configure_local_huggingface_environment,
    load_verified_local_tokenizer,
    verify_model_weights,
)
from sts1_llm_policy.execution_environment import verify_runtime_dependencies, verify_torch_hardware

from .chat_rendering import render_generation_request
from .protocol import GenerationRequest


class TransformersGenerationBackend:
    """Frozen local Transformers backend for one independent decision."""

    def __init__(
        self,
        config: BaseModelRuntimeConfig,
        tokenizer: object,
        model: object,
        torch_module: object,
        generation_config: object,
        adapter_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self._torch = torch_module
        self._generation_config = generation_config
        self.adapter_metadata = (
            dict(adapter_metadata) if adapter_metadata is not None else None
        )

    @classmethod
    def from_config(
        cls,
        config: BaseModelRuntimeConfig,
        *,
        project_root: str | Path,
        lora_checkpoint_dir: str | Path | None = None,
        verified_lora_metadata: Mapping[str, object] | None = None,
    ) -> TransformersGenerationBackend:
        root = Path(project_root).resolve()
        configure_local_huggingface_environment(root)
        verify_runtime_dependencies(config)
        verify_model_weights(config)

        try:
            import torch
            from transformers import AutoModelForCausalLM, GenerationConfig
        except ImportError as error:
            raise RuntimeError(
                "PyTorch and Transformers are required by the Base backend"
            ) from error

        verify_torch_hardware(config, torch)
        tokenizer = load_verified_local_tokenizer(
            config,
            project_root=root,
        )
        dtype = getattr(torch, config.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            config.snapshot_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            attn_implementation=config.attention_implementation,
        )
        adapter_metadata = None
        if lora_checkpoint_dir is not None:
            checkpoint_dir = Path(lora_checkpoint_dir)
            if not checkpoint_dir.is_absolute():
                checkpoint_dir = root / checkpoint_dir
            checkpoint_dir = checkpoint_dir.resolve()
            if not checkpoint_dir.is_relative_to(root):
                raise ValueError("LoRA checkpoint directory escapes the project root")
            if not checkpoint_dir.is_dir():
                raise ValueError(f"LoRA checkpoint directory does not exist: {checkpoint_dir}")
            from sts1_llm_policy.train.lora import load_lora_checkpoint

            _, _, adapter_metadata = load_lora_checkpoint(
                model,
                checkpoint_dir,
                expected_base_model_id=config.model_id,
                expected_base_revision=config.revision,
                verified_metadata=verified_lora_metadata,
            )
        model = model.to(torch.device(config.device))
        model.eval()
        generation_config = GenerationConfig(
            do_sample=config.do_sample,
            num_beams=config.num_beams,
            max_new_tokens=config.max_new_tokens,
            use_cache=True,
            repetition_penalty=config.repetition_penalty,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=list(config.eos_token_ids),
            pad_token_id=config.generation_pad_token_id,
        )
        generation_config.validate()
        return cls(
            config,
            tokenizer,
            model,
            torch,
            generation_config,
            adapter_metadata,
        )

    def _validate_request(self, request: GenerationRequest) -> None:
        expected = {
            "temperature": self.config.temperature,
            "do_sample": self.config.do_sample,
            "max_new_tokens": self.config.max_new_tokens,
        }
        observed = {
            "temperature": request.temperature,
            "do_sample": request.do_sample,
            "max_new_tokens": request.max_new_tokens,
        }
        if observed != expected:
            raise ValueError(
                f"Generation request violates frozen runtime: "
                f"observed={observed!r}, expected={expected!r}"
            )

    def generate(self, request: GenerationRequest) -> str:
        self._validate_request(request)
        model_inputs = render_generation_request(
            self.tokenizer,
            request,
            return_tensors="pt",
            return_dict=True,
        )
        device = self._torch.device(self.config.device)  # type: ignore[attr-defined]
        model_inputs = {
            name: value.to(device)
            for name, value in model_inputs.items()
        }
        input_length = model_inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():  # type: ignore[attr-defined]
            output_ids = self.model.generate(  # type: ignore[attr-defined]
                **model_inputs,
                generation_config=self._generation_config,
            )
        generated_ids = output_ids[0, input_length:]
        return self.tokenizer.decode(  # type: ignore[attr-defined]
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
