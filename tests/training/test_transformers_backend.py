from __future__ import annotations

from pathlib import Path
import unittest
import json
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from .runtime_fixture import runtime_document

from sts1_llm_policy.model_runtime import load_base_model_runtime_config
from sts1_llm_policy.policy.protocol import GenerationRequest
from sts1_llm_policy.policy.transformers_backend import TransformersGenerationBackend




class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.devices: list[object] = []

    def to(self, device: object) -> FakeTensor:
        self.devices.append(device)
        return self


class FakeOutputIds:
    def __init__(self) -> None:
        self.requested_key: object = None

    def __getitem__(self, key: object) -> list[int]:
        self.requested_key = key
        return [101, 102]


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_kwargs: dict[str, object] | None = None
        self.decode_kwargs: dict[str, object] | None = None
        self.input_ids = FakeTensor((1, 12))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> dict[str, FakeTensor]:
        self.template_kwargs = {"messages": messages, **kwargs}
        return {
            "input_ids": self.input_ids,
            "attention_mask": FakeTensor((1, 12)),
        }

    def decode(self, token_ids: object, **kwargs: object) -> str:
        self.decode_kwargs = {"token_ids": token_ids, **kwargs}
        return "ACTION_0"


class FakeModel:
    def __init__(self) -> None:
        self.generate_kwargs: dict[str, object] | None = None
        self.output_ids = FakeOutputIds()

    def generate(self, **kwargs: object) -> FakeOutputIds:
        self.generate_kwargs = kwargs
        return self.output_ids


class FakeInferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class FakeTorch:
    @staticmethod
    def device(name: str) -> str:
        return name

    @staticmethod
    def inference_mode() -> FakeInferenceMode:
        return FakeInferenceMode()


class TransformersGenerationBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = root / "runtime.json"
        path.write_text(json.dumps(runtime_document(legacy=True)), encoding="utf-8")
        self.config = load_base_model_runtime_config(path, project_root=root)
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.backend = TransformersGenerationBackend(
            self.config,
            self.tokenizer,
            self.model,
            FakeTorch(),
            "frozen-generation-config",
        )

    def test_renders_independent_chat_and_decodes_only_new_tokens(self) -> None:
        request = GenerationRequest(
            system_prompt="system",
            user_prompt="state and actions",
        )

        output = self.backend.generate(request)

        self.assertEqual(output, "ACTION_0")
        assert self.tokenizer.template_kwargs is not None
        self.assertEqual(
            self.tokenizer.template_kwargs["messages"],
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "state and actions"},
            ],
        )
        self.assertIsNone(self.tokenizer.template_kwargs["tools"])
        self.assertTrue(self.tokenizer.template_kwargs["add_generation_prompt"])
        self.assertTrue(self.tokenizer.template_kwargs["tokenize"])
        self.assertEqual(self.tokenizer.template_kwargs["return_tensors"], "pt")
        self.assertEqual(self.model.output_ids.requested_key, (0, slice(12, None)))
        assert self.model.generate_kwargs is not None
        self.assertEqual(
            self.model.generate_kwargs["generation_config"],
            "frozen-generation-config",
        )
        assert self.tokenizer.decode_kwargs is not None
        self.assertTrue(self.tokenizer.decode_kwargs["skip_special_tokens"])
        self.assertFalse(
            self.tokenizer.decode_kwargs["clean_up_tokenization_spaces"]
        )

    def test_rejects_request_that_drifts_from_frozen_decoding(self) -> None:
        request = GenerationRequest(
            system_prompt="system",
            user_prompt="state and actions",
            max_new_tokens=9,
        )

        with self.assertRaisesRegex(ValueError, "violates frozen runtime"):
            self.backend.generate(request)

        self.assertIsNone(self.tokenizer.template_kwargs)
        self.assertIsNone(self.model.generate_kwargs)

    def test_records_an_independent_copy_of_adapter_metadata(self) -> None:
        metadata = {"weights_sha256": "abc"}
        backend = TransformersGenerationBackend(
            self.config,
            self.tokenizer,
            self.model,
            FakeTorch(),
            "frozen-generation-config",
            metadata,
        )

        metadata["weights_sha256"] = "changed"

        self.assertEqual(backend.adapter_metadata, {"weights_sha256": "abc"})
        self.assertIsNone(self.backend.adapter_metadata)

    def test_startup_checks_model_metadata_once_through_tokenizer_loader(self):
        from sts1_llm_policy import model_runtime
        from sts1_llm_policy.policy import transformers_backend

        with patch.dict("sys.modules", {"torch": MagicMock(), "transformers": MagicMock()}), patch.object(
            transformers_backend, "verify_runtime_dependencies"
        ), patch.object(transformers_backend, "verify_torch_hardware"), patch.object(
            transformers_backend, "verify_model_weights"
        ) as weights, patch.object(model_runtime, "verify_model_metadata_assets") as metadata, patch.object(
            model_runtime, "verify_tokenizer_assets"
        ) as tokenizer_assets, patch.object(model_runtime, "verify_loaded_tokenizer"), patch.object(
            transformers_backend, "configure_local_huggingface_environment"
        ), patch.object(model_runtime, "configure_local_huggingface_environment"):
            TransformersGenerationBackend.from_config(self.config, project_root=self.config.snapshot_path.parent)
        weights.assert_called_once_with(self.config)
        metadata.assert_called_once_with(self.config)
        tokenizer_assets.assert_called_once_with(self.config)


if __name__ == "__main__":
    unittest.main()
