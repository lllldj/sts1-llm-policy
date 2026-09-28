from __future__ import annotations

from pathlib import Path
import random
from typing import TYPE_CHECKING

import torch

from sts1_llm_policy.model_runtime import (
    BaseModelRuntimeConfig,
    configure_local_huggingface_environment,
    verify_model_metadata_assets,
    verify_model_weights,
)
from sts1_llm_policy.execution_environment import verify_runtime_dependencies, verify_torch_hardware
from sts1_llm_policy.train.lora import LoraSpec, inject_lora, load_lora_checkpoint
from sts1_llm_policy.artifacts import sha256_file

if TYPE_CHECKING:
    from sts1_llm_policy.train.configured_training import ConfiguredTrainingRun


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def capture_training_rng(device: torch.device) -> dict:
    return {
        "schema_version": "training_rng_v1",
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def read_training_rng(checkpoint: Path, state: dict, device: torch.device) -> dict:
    path = checkpoint / "rng.pt"
    if not path.is_file() or state.get("rng_sha256") != sha256_file(path):
        raise ValueError("Training checkpoint RNG state is missing or its hash differs")
    rng = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(rng, dict) or rng.get("schema_version") != "training_rng_v1":
        raise ValueError("Unsupported training RNG state")
    try:
        random.Random().setstate(rng["python"])
        torch.Generator(device="cpu").set_state(rng["torch_cpu"])
        cuda = rng["torch_cuda"]
        if device.type == "cuda":
            if not isinstance(cuda, torch.Tensor) or cuda.dtype != torch.uint8 or cuda.ndim != 1 or not cuda.numel():
                raise ValueError("Missing CUDA RNG state")
        elif cuda is not None:
            raise ValueError("RNG device type differs from the training device")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Invalid training checkpoint RNG state") from error
    return rng


def restore_training_rng(rng: dict, device: torch.device) -> None:
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng["torch_cuda"], device)


def load_base_model(
    runtime: BaseModelRuntimeConfig,
    *,
    project_root: Path,
    verify_assets: bool,
) -> torch.nn.Module:
    """Load Base; skip verification only after checks in this same execution."""
    configure_local_huggingface_environment(project_root)
    if verify_assets:
        verify_runtime_dependencies(runtime)
        verify_model_metadata_assets(runtime)
        verify_model_weights(runtime)
        verify_torch_hardware(runtime, torch)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        runtime.snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=getattr(torch, runtime.dtype),
        attn_implementation=runtime.attention_implementation,
    )


def load_initial_model(
    run: ConfiguredTrainingRun,
    *,
    verify_assets: bool,
) -> tuple[torch.nn.Module, LoraSpec, tuple[str, ...]]:
    model = load_base_model(
        run.runtime, project_root=run.project_root, verify_assets=verify_assets
    )
    if run.initial_checkpoint is None:
        if run.adapter.requires_initial_checkpoint:
            raise ValueError("This algorithm requires an initial checkpoint")
        raw = run.recipe["lora"]
        spec = LoraSpec(
            target_module_suffixes=tuple(raw["target_module_suffixes"]),
            rank=int(raw["rank"]),
            alpha=float(raw["alpha"]),
            dropout=float(raw["dropout"]),
        )
        replaced = inject_lora(model, spec)
    else:
        spec, replaced, _ = load_lora_checkpoint(
            model,
            run.initial_checkpoint.directory,
            expected_base_model_id=run.runtime.model_id,
            expected_base_revision=run.runtime.revision,
            verified_metadata=run.initial_checkpoint.metadata,
        )
        raw = run.recipe["lora"]
        if (
            spec.target_module_suffixes != tuple(raw["target_module_suffixes"])
            or spec.rank != int(raw["rank"])
            or spec.alpha != float(raw["alpha"])
            or spec.dropout != float(raw["dropout"])
        ):
            raise ValueError("Recipe LoRA shape differs from its initial checkpoint")
    model.to(torch.device(run.runtime.device))
    return model, spec, tuple(replaced)
