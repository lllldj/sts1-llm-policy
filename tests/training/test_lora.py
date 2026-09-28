from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from sts1_llm_policy.train.lora import (
    LoraLinear,
    LoraSpec,
    adapter_state_sha256,
    inject_lora,
    load_lora_checkpoint,
    save_lora_checkpoint,
    trainable_parameter_summary,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.q_proj(inputs) + torch.cat(
            (self.v_proj(inputs), self.v_proj(inputs)), dim=-1
        )


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.output = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.attn(inputs))


class LoraTest(unittest.TestCase):
    def test_adapter_layers_inherit_base_device_and_dtype(self) -> None:
        model = _ToyModel().to(dtype=torch.float64)
        inject_lora(model, LoraSpec(("q_proj",), rank=2, alpha=4.0, dropout=0.0))

        wrapped = model.attn.q_proj
        self.assertIsInstance(wrapped, LoraLinear)
        self.assertEqual(wrapped.lora_a.weight.dtype, torch.float64)
        self.assertEqual(wrapped.lora_b.weight.dtype, torch.float64)
        self.assertEqual(wrapped.lora_a.weight.device, wrapped.base_layer.weight.device)

    def setUp(self) -> None:
        self.spec = LoraSpec(("q_proj", "v_proj"), rank=2, alpha=4.0, dropout=0.0)

    def test_injection_is_initially_exact_and_only_adapter_is_trainable(self) -> None:
        torch.manual_seed(7)
        model = _ToyModel()
        inputs = torch.randn(3, 4)
        expected = model(inputs).detach()
        replaced = inject_lora(model, self.spec)

        self.assertEqual(replaced, ("attn.q_proj", "attn.v_proj"))
        self.assertIsInstance(model.attn.q_proj, LoraLinear)
        self.assertTrue(torch.equal(model(inputs), expected))
        summary = trainable_parameter_summary(model)
        self.assertEqual(summary["trainable_parameters"], 28)
        self.assertTrue(
            all(
                parameter.requires_grad
                == (name.endswith("lora_a.weight") or name.endswith("lora_b.weight"))
                for name, parameter in model.named_parameters()
            )
        )

        model(inputs).sum().backward()
        self.assertIsNotNone(model.attn.q_proj.lora_b.weight.grad)
        self.assertIsNone(model.attn.k_proj.weight.grad)

    def test_checkpoint_round_trip_restores_exact_adapter(self) -> None:
        torch.manual_seed(11)
        model = _ToyModel()
        replaced = inject_lora(model, self.spec)
        with torch.no_grad():
            model.attn.q_proj.lora_b.weight.fill_(0.25)
            model.attn.v_proj.lora_b.weight.fill_(-0.5)
        expected_digest = adapter_state_sha256(model)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            save_lora_checkpoint(
                model,
                checkpoint,
                spec=self.spec,
                replaced_modules=replaced,
                base_model_id="toy",
                base_revision="revision",
            )
            restored = _ToyModel()
            loaded_spec, loaded_modules, _ = load_lora_checkpoint(
                restored,
                checkpoint,
                expected_base_model_id="toy",
                expected_base_revision="revision",
            )

        self.assertEqual(loaded_spec, self.spec)
        self.assertEqual(loaded_modules, replaced)
        self.assertEqual(adapter_state_sha256(restored), expected_digest)

    def test_adapter_hash_supports_bfloat16_parameters(self) -> None:
        model = _ToyModel().to(dtype=torch.bfloat16)
        inject_lora(model, self.spec)

        digest = adapter_state_sha256(model)

        self.assertEqual(len(digest), 64)
        self.assertEqual(adapter_state_sha256(model), digest)


if __name__ == "__main__":
    unittest.main()
