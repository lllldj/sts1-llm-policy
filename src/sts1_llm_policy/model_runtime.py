from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Any

from sts1_llm_policy.artifacts import sha256_bytes, sha256_file
from sts1_llm_policy.execution_environment import validate_execution_profile


BASE_MODEL_RUNTIME_SCHEMA_VERSION = "base_model_runtime_v1"


@dataclass(frozen=True)
class BaseModelRuntimeConfig:
    runtime_id: str
    experiment_protocol_id: str
    model_id: str
    revision: str
    snapshot_path: Path
    model_config_sha256: str
    generation_config_sha256: str
    weights_file: str | None
    weights_sha256: str | None
    weight_assets_sha256: tuple[tuple[str, str], ...]
    tokenizer_class: str
    tokenizer_vocab_size: int
    tokenizer_total_size: int
    tokenizer_model_max_length: int
    chat_template_byte_length: int
    chat_template_sha256: str
    bos_token: str | None
    bos_token_id: int | None
    eos_token: str
    eos_token_id: int
    pad_token: str
    pad_token_id: int
    unk_token: str | None
    unk_token_id: int | None
    tokenizer_asset_sha256: tuple[tuple[str, str], ...]
    device: str
    dtype: str
    quantization: str | None
    attention_implementation: str
    temperature: float
    do_sample: bool
    num_beams: int
    repetition_penalty: float
    max_new_tokens: int
    eos_token_ids: tuple[int, ...]
    generation_pad_token_id: int
    dependencies: tuple[tuple[str, str], ...]
    validated_device_name: str
    validated_compute_capability: str
    validated_cuda_runtime: str
    validated_bf16_supported: bool
    tokenizer_local_only_verified: bool
    model_weights_loaded: bool
    generation_smoke_completed: bool
    execution_requirements: dict[str, Any] | None = None


def _require_object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _require_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _require_int(value: object, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{path} must be positive")
    return value


def _require_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    return float(value)


def _require_optional_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, path)


def _require_optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, path)


def _require_sha256(value: object, path: str) -> str:
    digest = _require_string(value, path)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return digest


def _resolve_repository_path(project_root: Path, value: object, path: str) -> Path:
    relative = Path(_require_string(value, path))
    if relative.is_absolute():
        raise ValueError(f"{path} must be repository-relative")
    resolved = (project_root / relative).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"{path} escapes the repository") from error
    return resolved


def load_base_model_runtime_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    execution_profile: dict[str, Any] | None = None,
) -> BaseModelRuntimeConfig:
    config_path = Path(path).resolve()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else config_path.parents[2]
    )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read Base model runtime config: {config_path}") from error
    if not isinstance(raw, dict):
        raise ValueError("Base model runtime config must be an object")
    portable = raw.get("schema_version") == "base_model_runtime_v2"
    if raw.get("schema_version") not in {BASE_MODEL_RUNTIME_SCHEMA_VERSION, "base_model_runtime_v2"}:
        raise ValueError("Unsupported Base model runtime schema")
    if portable:
        if execution_profile is None and "execution_profile" in raw:
            profile_path = _resolve_repository_path(root, raw["execution_profile"], "execution_profile")
            execution_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        validate_execution_profile(execution_profile)
        if any(key in raw for key in ("dependencies", "validated_hardware", "validation_status")):
            raise ValueError("Portable runtime keeps execution requirements and acceptance outside model identity")
    elif execution_profile is not None:
        raise ValueError("Legacy runtime does not accept execution overrides")

    model = _require_object(raw.get("model"), "model")
    tokenizer = _require_object(raw.get("tokenizer"), "tokenizer")
    chat_template = _require_object(tokenizer.get("chat_template"), "tokenizer.chat_template")
    special_tokens = _require_object(tokenizer.get("special_tokens"), "tokenizer.special_tokens")
    prompt_rendering = _require_object(raw.get("prompt_rendering"), "prompt_rendering")
    model_loading = _require_object(raw.get("model_loading"), "model_loading")
    generation = _require_object(raw.get("generation"), "generation")
    dependencies = _require_object(
        execution_profile["dependencies"] if portable else raw.get("dependencies"), "dependencies"
    )
    hardware = {} if portable else _require_object(raw.get("validated_hardware"), "validated_hardware")
    validation_status = (
        {"tokenizer_local_only_verified": False, "model_weights_loaded": False,
         "generation_smoke_completed": False}
        if portable else _require_object(raw.get("validation_status"), "validation_status")
    )

    if model.get("architecture") != "Qwen2ForCausalLM":
        raise ValueError("model.architecture must be Qwen2ForCausalLM")
    revision = _require_string(model.get("revision"), "model.revision")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("model.revision must be a 40-character lowercase revision")
    snapshot_path = _resolve_repository_path(root, model.get("snapshot_path"), "model.snapshot_path")
    if snapshot_path.name != revision:
        raise ValueError("model.snapshot_path must end with model.revision")

    expected_tokenizer_loader = {
        "loader": "AutoTokenizer.from_pretrained",
        "local_files_only": True,
        "trust_remote_code": False,
        "use_fast": True,
    }
    for key, expected in expected_tokenizer_loader.items():
        if tokenizer.get(key) != expected:
            raise ValueError(f"tokenizer.{key} must be {expected!r}")
    if chat_template.get("source") != "tokenizer.chat_template":
        raise ValueError("tokenizer.chat_template.source is unsupported")
    if chat_template.get("encoding") != "utf-8":
        raise ValueError("tokenizer.chat_template.encoding must be utf-8")

    asset_values = _require_object(tokenizer.get("asset_sha256"), "tokenizer.asset_sha256")
    if not asset_values:
        raise ValueError("tokenizer.asset_sha256 must not be empty")
    assets: list[tuple[str, str]] = []
    for filename, digest in asset_values.items():
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError("tokenizer asset names must be plain filenames")
        assets.append((filename, _require_sha256(digest, f"tokenizer.asset_sha256.{filename}")))

    if prompt_rendering != {
        "message_roles": ["system", "user"],
        "tools": None,
        "add_generation_prompt": True,
        "tokenize": True,
        "add_special_tokens": False,
    }:
        raise ValueError("prompt_rendering does not match prompt_v1 chat rendering")

    device = _require_string(model_loading.get("device"), "model_loading.device")
    if re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) is None:
        raise ValueError("model_loading.device must select a CUDA device as cuda:N")
    expected_model_loading = {
        "backend": "transformers",
        "loader": "AutoModelForCausalLM.from_pretrained",
        "local_files_only": True,
        "trust_remote_code": False,
        "device": device,
        "dtype": "bfloat16",
        "device_map": None,
        "quantization": None,
        "attention_implementation": "sdpa",
        "eval_mode": True,
    }
    if model_loading != expected_model_loading:
        raise ValueError("model_loading does not match the frozen local Base runtime")

    expected_generation = {
        "batch_size": 1,
        "strategy": "greedy",
        "temperature": 0.0,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "transformers_sampling_parameters": None,
        "max_new_tokens": 8,
        "use_cache": True,
        "constrained_decoding": False,
        "decode_skip_special_tokens": True,
        "decode_clean_up_tokenization_spaces": False,
        "independent_decisions": True,
        "chat_history": False,
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            raise ValueError(f"generation.{key} violates the frozen policy runtime")
    eos_token_values = generation.get("eos_token_ids")
    if not isinstance(eos_token_values, list) or not eos_token_values:
        raise ValueError("generation.eos_token_ids must be a non-empty array")
    eos_token_ids = tuple(
        _require_int(value, "generation.eos_token_ids[]")
        for value in eos_token_values
    )
    if len(set(eos_token_ids)) != len(eos_token_ids):
        raise ValueError("generation.eos_token_ids must be unique")

    tokenizer_verified = _require_bool(
        validation_status.get("tokenizer_local_only_verified"),
        "validation_status.tokenizer_local_only_verified",
    )
    model_loaded = _require_bool(
        validation_status.get("model_weights_loaded"),
        "validation_status.model_weights_loaded",
    )
    generation_smoke = _require_bool(
        validation_status.get("generation_smoke_completed"),
        "validation_status.generation_smoke_completed",
    )
    if not portable and not tokenizer_verified:
        raise ValueError("The frozen runtime requires verified local tokenizer loading")
    if generation_smoke and not model_loaded:
        raise ValueError("A generation smoke requires model_weights_loaded=true")
    if set(validation_status) != {
        "tokenizer_local_only_verified",
        "model_weights_loaded",
        "generation_smoke_completed",
    }:
        raise ValueError("validation_status contains unsupported fields")

    dependency_values = tuple(
        (name, _require_string(version, f"dependencies.{name}"))
        for name, version in dependencies.items()
    )
    required_dependencies = {
        "python",
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "safetensors",
    }
    if {name for name, _ in dependency_values} != required_dependencies:
        raise ValueError("dependencies must contain the complete frozen runtime set")

    eos_token_id = _require_int(special_tokens.get("eos_token_id"), "tokenizer.special_tokens.eos_token_id")
    pad_token_id = _require_int(special_tokens.get("pad_token_id"), "tokenizer.special_tokens.pad_token_id")
    generation_pad_token_id = _require_int(generation.get("pad_token_id"), "generation.pad_token_id")
    if eos_token_ids[0] != eos_token_id or pad_token_id not in eos_token_ids:
        raise ValueError("generation.eos_token_ids do not match the frozen tokenizer stops")
    if generation_pad_token_id != pad_token_id:
        raise ValueError("generation.pad_token_id must match tokenizer pad_token_id")

    weights_file_value = model.get("weights_file")
    weights_sha256_value = model.get("weights_sha256")
    weight_assets_value = model.get("weight_assets_sha256")
    if weight_assets_value is None:
        weights_file = _require_string(weights_file_value, "model.weights_file")
        if Path(weights_file).name != weights_file:
            raise ValueError("model.weights_file must be a plain filename")
        weights_sha256 = _require_sha256(weights_sha256_value, "model.weights_sha256")
        weight_assets = ((weights_file, weights_sha256),)
    else:
        if weights_file_value is not None or weights_sha256_value is not None:
            raise ValueError(
                "model.weight_assets_sha256 is mutually exclusive with legacy weight fields"
            )
        raw_weight_assets = _require_object(
            weight_assets_value, "model.weight_assets_sha256"
        )
        if not raw_weight_assets:
            raise ValueError("model.weight_assets_sha256 must not be empty")
        parsed_weight_assets: list[tuple[str, str]] = []
        for filename, digest in raw_weight_assets.items():
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
            ):
                raise ValueError("model weight asset names must be plain filenames")
            parsed_weight_assets.append(
                (
                    filename,
                    _require_sha256(
                        digest, f"model.weight_assets_sha256.{filename}"
                    ),
                )
            )
        weights_file = None
        weights_sha256 = None
        weight_assets = tuple(parsed_weight_assets)

    return BaseModelRuntimeConfig(
        runtime_id=_require_string(raw.get("runtime_id"), "runtime_id"),
        experiment_protocol_id=_require_string(raw.get("experiment_protocol_id"), "experiment_protocol_id"),
        model_id=_require_string(model.get("id"), "model.id"),
        revision=revision,
        snapshot_path=snapshot_path,
        model_config_sha256=_require_sha256(
            model.get("config_sha256"),
            "model.config_sha256",
        ),
        generation_config_sha256=_require_sha256(
            model.get("generation_config_sha256"),
            "model.generation_config_sha256",
        ),
        weights_file=weights_file,
        weights_sha256=weights_sha256,
        weight_assets_sha256=weight_assets,
        tokenizer_class=_require_string(tokenizer.get("implementation_class"), "tokenizer.implementation_class"),
        tokenizer_vocab_size=_require_int(tokenizer.get("vocab_size"), "tokenizer.vocab_size", positive=True),
        tokenizer_total_size=_require_int(tokenizer.get("total_size"), "tokenizer.total_size", positive=True),
        tokenizer_model_max_length=_require_int(tokenizer.get("model_max_length"), "tokenizer.model_max_length", positive=True),
        chat_template_byte_length=_require_int(chat_template.get("byte_length"), "tokenizer.chat_template.byte_length", positive=True),
        chat_template_sha256=_require_sha256(chat_template.get("sha256"), "tokenizer.chat_template.sha256"),
        bos_token=_require_optional_string(special_tokens.get("bos_token"), "tokenizer.special_tokens.bos_token"),
        bos_token_id=_require_optional_int(special_tokens.get("bos_token_id"), "tokenizer.special_tokens.bos_token_id"),
        eos_token=_require_string(special_tokens.get("eos_token"), "tokenizer.special_tokens.eos_token"),
        eos_token_id=eos_token_id,
        pad_token=_require_string(special_tokens.get("pad_token"), "tokenizer.special_tokens.pad_token"),
        pad_token_id=pad_token_id,
        unk_token=_require_optional_string(special_tokens.get("unk_token"), "tokenizer.special_tokens.unk_token"),
        unk_token_id=_require_optional_int(special_tokens.get("unk_token_id"), "tokenizer.special_tokens.unk_token_id"),
        tokenizer_asset_sha256=tuple(assets),
        device=_require_string(model_loading.get("device"), "model_loading.device"),
        dtype=_require_string(model_loading.get("dtype"), "model_loading.dtype"),
        quantization=_require_optional_string(model_loading.get("quantization"), "model_loading.quantization"),
        attention_implementation=_require_string(model_loading.get("attention_implementation"), "model_loading.attention_implementation"),
        temperature=_require_float(generation.get("temperature"), "generation.temperature"),
        do_sample=_require_bool(generation.get("do_sample"), "generation.do_sample"),
        num_beams=_require_int(generation.get("num_beams"), "generation.num_beams", positive=True),
        repetition_penalty=_require_float(
            generation.get("repetition_penalty"),
            "generation.repetition_penalty",
        ),
        max_new_tokens=_require_int(generation.get("max_new_tokens"), "generation.max_new_tokens", positive=True),
        eos_token_ids=eos_token_ids,
        generation_pad_token_id=generation_pad_token_id,
        dependencies=dependency_values,
        validated_device_name="" if portable else _require_string(
            hardware.get("device_name"),
            "validated_hardware.device_name",
        ),
        validated_compute_capability="" if portable else _require_string(
            hardware.get("compute_capability"),
            "validated_hardware.compute_capability",
        ),
        validated_cuda_runtime="" if portable else _require_string(
            hardware.get("cuda_runtime"),
            "validated_hardware.cuda_runtime",
        ),
        validated_bf16_supported=False if portable else _require_bool(
            hardware.get("bf16_supported"),
            "validated_hardware.bf16_supported",
        ),
        tokenizer_local_only_verified=tokenizer_verified,
        model_weights_loaded=model_loaded,
        generation_smoke_completed=generation_smoke,
        execution_requirements=execution_profile,
    )


def configure_local_huggingface_environment(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    hf_home = (root / ".model-cache" / "huggingface").resolve()
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return hf_home


def _file_sha256(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError as error:
        raise ValueError(f"Unable to read frozen model asset: {path}") from error


def verify_tokenizer_assets(config: BaseModelRuntimeConfig) -> None:
    for filename, expected_digest in config.tokenizer_asset_sha256:
        asset_path = config.snapshot_path / filename
        actual_digest = _file_sha256(asset_path)
        if actual_digest != expected_digest:
            raise ValueError(f"Tokenizer asset SHA-256 mismatch: {filename}")


def verify_model_metadata_assets(config: BaseModelRuntimeConfig) -> None:
    expected_assets = {
        "config.json": config.model_config_sha256,
        "generation_config.json": config.generation_config_sha256,
    }
    for filename, expected_digest in expected_assets.items():
        asset_path = config.snapshot_path / filename
        actual_digest = _file_sha256(asset_path)
        if actual_digest != expected_digest:
            raise ValueError(f"Model metadata SHA-256 mismatch: {filename}")


def verify_model_weights(config: BaseModelRuntimeConfig) -> None:
    for filename, expected_digest in config.weight_assets_sha256:
        weights_path = config.snapshot_path / filename
        actual_digest = _file_sha256(weights_path)
        if actual_digest != expected_digest:
            raise ValueError(f"Model weights SHA-256 mismatch: {filename}")


def verify_loaded_tokenizer(config: BaseModelRuntimeConfig, tokenizer: object) -> None:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        raise ValueError("Loaded tokenizer does not expose one string chat template")
    encoded_template = chat_template.encode("utf-8")
    observed = {
        "implementation_class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "total_size": len(tokenizer),  # type: ignore[arg-type]
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "chat_template_byte_length": len(encoded_template),
        "chat_template_sha256": sha256_bytes(encoded_template),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token": getattr(tokenizer, "unk_token", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
    }
    expected = {
        "implementation_class": config.tokenizer_class,
        "vocab_size": config.tokenizer_vocab_size,
        "total_size": config.tokenizer_total_size,
        "model_max_length": config.tokenizer_model_max_length,
        "chat_template_byte_length": config.chat_template_byte_length,
        "chat_template_sha256": config.chat_template_sha256,
        "bos_token": config.bos_token,
        "bos_token_id": config.bos_token_id,
        "eos_token": config.eos_token,
        "eos_token_id": config.eos_token_id,
        "pad_token": config.pad_token,
        "pad_token_id": config.pad_token_id,
        "unk_token": config.unk_token,
        "unk_token_id": config.unk_token_id,
    }
    for key, expected_value in expected.items():
        if observed[key] != expected_value:
            raise ValueError(
                f"Loaded tokenizer {key} mismatch: "
                f"{observed[key]!r} != {expected_value!r}"
            )


def load_verified_local_tokenizer(
    config: BaseModelRuntimeConfig,
    *,
    project_root: str | Path,
) -> object:
    configure_local_huggingface_environment(project_root)
    verify_model_metadata_assets(config)
    verify_tokenizer_assets(config)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ValueError("Transformers is required to load the frozen tokenizer") from error
    tokenizer = AutoTokenizer.from_pretrained(
        config.snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    verify_loaded_tokenizer(config, tokenizer)
    return tokenizer
