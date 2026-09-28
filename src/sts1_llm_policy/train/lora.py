from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class LoraSpec:
    target_module_suffixes: tuple[str, ...]
    rank: int
    alpha: float
    dropout: float

    def __post_init__(self) -> None:
        if not self.target_module_suffixes or any(
            not value for value in self.target_module_suffixes
        ):
            raise ValueError("LoRA target suffixes must be non-empty")
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive and finite")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")


class LoraLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, spec: LoraSpec) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = spec.rank
        self.alpha = spec.alpha
        self.scaling = spec.alpha / spec.rank
        self.dropout = nn.Dropout(spec.dropout)
        self.lora_a = nn.Linear(base_layer.in_features, spec.rank, bias=False)
        self.lora_b = nn.Linear(spec.rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        self.lora_b.to(
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        update = self.lora_b(self.lora_a(self.dropout(inputs)))
        return base + update * self.scaling


def inject_lora(model: nn.Module, spec: LoraSpec) -> tuple[str, ...]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", maxsplit=1)[-1] in set(
            spec.target_module_suffixes
        ):
            replacements.append((name, module))
    if not replacements:
        raise ValueError("No linear modules matched the LoRA targets")

    replaced_names: list[str] = []
    for name, module in replacements:
        parent_path, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_path)
        setattr(parent, child_name, LoraLinear(module, spec))
        replaced_names.append(name)
    return tuple(replaced_names)


def trainable_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
    }


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if name.endswith("lora_a.weight") or name.endswith("lora_b.weight")
    }
    if not state:
        raise ValueError("Model does not contain LoRA adapter parameters")
    return state


def save_lora_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    *,
    spec: LoraSpec,
    replaced_modules: Sequence[str],
    base_model_id: str,
    base_revision: str,
) -> dict[str, object]:
    from safetensors.torch import save_file

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    save_file(adapter_state_dict(model), str(weights_path))
    metadata = {
        "schema_version": 1,
        "adapter_type": "project_lora_v1",
        "base_model_id": base_model_id,
        "base_revision": base_revision,
        "spec": {
            **asdict(spec),
            "target_module_suffixes": list(spec.target_module_suffixes),
        },
        "replaced_modules": list(replaced_modules),
        "weights_file": weights_path.name,
        "weights_sha256": sha256(weights_path.read_bytes()).hexdigest(),
    }
    config_path = checkpoint_dir / "adapter_config.json"
    config_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def inspect_lora_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_base_model_id: str,
    expected_base_revision: str,
) -> dict[str, object]:
    config_path = checkpoint_dir / "adapter_config.json"
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("adapter_type") != "project_lora_v1"
    ):
        raise ValueError(
            f"Unsupported LoRA checkpoint format in {config_path}: expected "
            "project_lora_v1 with schema_version=1. PEFT adapters are not supported."
        )
    if (
        metadata.get("base_model_id") != expected_base_model_id
        or metadata.get("base_revision") != expected_base_revision
    ):
        raise ValueError(
            "LoRA checkpoint base identity mismatch: "
            f"expected {expected_base_model_id}@{expected_base_revision}, "
            f"found {metadata.get('base_model_id')}@{metadata.get('base_revision')}"
        )
    weights_path = (checkpoint_dir / str(metadata.get("weights_file"))).resolve()
    if not weights_path.is_relative_to(checkpoint_dir.resolve()) or not weights_path.is_file():
        raise ValueError("LoRA checkpoint weights path is invalid")
    if sha256(weights_path.read_bytes()).hexdigest() != metadata.get("weights_sha256"):
        raise ValueError("LoRA checkpoint weights hash mismatch")
    spec_raw = metadata["spec"]
    LoraSpec(tuple(spec_raw["target_module_suffixes"]), spec_raw["rank"],
             spec_raw["alpha"], spec_raw["dropout"])
    if not isinstance(metadata.get("replaced_modules"), list) or not metadata["replaced_modules"]:
        raise ValueError("LoRA checkpoint target modules are missing")
    return metadata


def load_lora_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    *,
    expected_base_model_id: str,
    expected_base_revision: str,
    verified_metadata: Mapping[str, object] | None = None,
) -> tuple[LoraSpec, tuple[str, ...], dict[str, object]]:
    from safetensors.torch import load_file

    metadata = dict(verified_metadata) if verified_metadata is not None else inspect_lora_checkpoint(
        checkpoint_dir, expected_base_model_id=expected_base_model_id,
        expected_base_revision=expected_base_revision,
    )
    if (metadata["base_model_id"] != expected_base_model_id
            or metadata["base_revision"] != expected_base_revision):
        raise ValueError("LoRA checkpoint base identity mismatch")
    spec_raw = metadata["spec"]
    spec = LoraSpec(
        target_module_suffixes=tuple(spec_raw["target_module_suffixes"]),
        rank=spec_raw["rank"],
        alpha=spec_raw["alpha"],
        dropout=spec_raw["dropout"],
    )
    replaced = inject_lora(model, spec)
    expected_modules = tuple(metadata["replaced_modules"])
    if replaced != expected_modules:
        raise ValueError("LoRA checkpoint target module list mismatch")
    weights_path = checkpoint_dir / metadata["weights_file"]
    loaded = load_file(str(weights_path), device="cpu")
    current = dict(model.named_parameters())
    if set(loaded) != {
        name
        for name in current
        if name.endswith("lora_a.weight") or name.endswith("lora_b.weight")
    }:
        raise ValueError("LoRA checkpoint tensor names mismatch")
    with torch.no_grad():
        for name, value in loaded.items():
            parameter = current[name]
            if parameter.shape != value.shape:
                raise ValueError(f"LoRA checkpoint tensor shape mismatch: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return spec, replaced, metadata


def adapter_state_sha256(model: nn.Module) -> str:
    digest = sha256()
    for name, value in sorted(adapter_state_dict(model).items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
