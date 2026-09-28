from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
from typing import Any, Mapping

from sts1_llm_policy.model_runtime import (
    BaseModelRuntimeConfig,
    load_base_model_runtime_config,
    load_verified_local_tokenizer,
)
from sts1_llm_policy.train.configured_adapters import (
    PreparedTrainingData,
    TrainingAdapter,
    adapter_for_algorithm,
)
from sts1_llm_policy.train.lora import inspect_lora_checkpoint
from sts1_llm_policy.configuration import (
    ConfigDocument,
    load_config_document,
    resolve_bound_document,
)
from sts1_llm_policy.artifacts import (
    iter_jsonl,
    repository_relative,
    resolve_repository_path,
    sha256_file,
)
from sts1_llm_policy.data.manifest import validate_dataset_manifest


TRAINING_RUN_SCHEMA_VERSION = "training_run_v1"
TRAINING_RECIPE_SCHEMA_VERSION = "training_recipe_v1"
TRAINING_MODES = ("preflight", "backward", "smoke", "run")


@dataclass(frozen=True)
class InitialCheckpoint:
    directory: Path
    config_path: Path
    weights_path: Path
    config_sha256: str
    weights_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ConfiguredTrainingRun:
    project_root: Path
    run_document: ConfigDocument
    recipe_document: ConfigDocument
    runtime_document: ConfigDocument
    dataset_document: ConfigDocument
    runtime: BaseModelRuntimeConfig
    adapter: TrainingAdapter
    prepared: PreparedTrainingData
    output_dir: Path
    report_path: Path
    initial_checkpoint: InitialCheckpoint | None
    mode: str
    implementation_hashes: dict[str, str]
    execution_document: ConfigDocument | None = None
    tokenizer: object = None

    @property
    def config(self) -> dict[str, Any]:
        return self.run_document.value

    @property
    def recipe(self) -> dict[str, Any]:
        return self.recipe_document.value

    @property
    def dataset(self) -> dict[str, Any]:
        return self.dataset_document.value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _recipe_fields(value: object, name: str, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing:
        raise ValueError(f"{name}: missing fields {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name}: unknown fields {sorted(unknown)}")


def _recipe_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validate_recipe(recipe: Mapping[str, Any]) -> None:
    algorithm = recipe.get("algorithm")
    if algorithm not in ("sft", "dpo"):
        raise ValueError("recipe.algorithm must be sft or dpo")
    fields = {"schema_version", "recipe_id", "algorithm", "lora", "tokenization", "optimizer", "training", "smoke"}
    if algorithm == "dpo":
        fields.add("dpo")
    _recipe_fields(recipe, "recipe", fields)
    if recipe["schema_version"] != TRAINING_RECIPE_SCHEMA_VERSION:
        raise ValueError("recipe.schema_version must be training_recipe_v1")
    if not isinstance(recipe["recipe_id"], str) or not recipe["recipe_id"].strip():
        raise ValueError("recipe.recipe_id must be a non-empty string")
    for section, required, optional in (
        ("lora", {"target_module_suffixes", "rank", "alpha", "dropout"}, {"source"}),
        ("tokenization", {"max_sequence_tokens", "truncation", "response_only", "supervise_assistant_terminator"}, set()),
        ("optimizer", {"name", "learning_rate", "weight_decay", "max_grad_norm"}, set()),
        ("training", {"epochs", "micro_batch_units", "gradient_accumulation_units", "gradient_checkpointing", "resume_checkpoint_interval_steps"}, set()),
        ("smoke", {"units", "optimizer_steps", "selection"}, set()),
    ):
        _recipe_fields(recipe[section], section, required, optional)
    lora = recipe.get("lora")
    tokenization = recipe.get("tokenization")
    optimizer = recipe.get("optimizer")
    training = recipe.get("training")
    smoke = recipe.get("smoke")
    suffixes = lora.get("target_module_suffixes")
    if lora.get("source") not in (None, "initial_checkpoint"):
        raise ValueError("lora.source must be initial_checkpoint or null")
    if not isinstance(suffixes, list) or not suffixes or any(not isinstance(item, str) or not item for item in suffixes):
        raise ValueError("lora.target_module_suffixes must be a non-empty list of non-empty strings")
    _positive_int(lora.get("rank"), "lora.rank")
    if _recipe_number(lora["alpha"], "lora.alpha") <= 0:
        raise ValueError("lora.alpha must be positive")
    if not 0 <= _recipe_number(lora["dropout"], "lora.dropout") < 1:
        raise ValueError("lora.dropout must be in [0, 1)")
    for key, expected in (("truncation", False), ("response_only", True), ("supervise_assistant_terminator", True)):
        if tokenization[key] is not expected:
            raise ValueError(f"tokenization.{key} must be {expected}")
    _positive_int(tokenization.get("max_sequence_tokens"), "tokenization.max_sequence_tokens")
    if optimizer.get("name") != "adamw":
        raise ValueError("optimizer.name must be adamw")
    for key in ("learning_rate", "max_grad_norm"):
        if _recipe_number(optimizer[key], f"optimizer.{key}") <= 0:
            raise ValueError(f"optimizer.{key} must be positive and finite")
    if _recipe_number(optimizer["weight_decay"], "optimizer.weight_decay") < 0:
        raise ValueError("optimizer.weight_decay must be nonnegative")
    for key in ("epochs", "micro_batch_units", "gradient_accumulation_units", "resume_checkpoint_interval_steps"):
        _positive_int(training.get(key), f"training.{key}")
    if training["micro_batch_units"] != 1:
        raise ValueError("training.micro_batch_units must be 1")
    if training["gradient_checkpointing"] is not True:
        raise ValueError("training.gradient_checkpointing must be true")
    _positive_int(smoke.get("units"), "smoke.units")
    if _positive_int(smoke["optimizer_steps"], "smoke.optimizer_steps") != 1:
        raise ValueError("smoke.optimizer_steps must be 1")
    if smoke["selection"] != "longest_then_hash_spread":
        raise ValueError("smoke.selection must be longest_then_hash_spread")
    if algorithm == "dpo":
        dpo = recipe.get("dpo")
        _recipe_fields(dpo, "dpo", {"beta", "label_smoothing", "reference"})
        if dpo["reference"] != "frozen_initial_checkpoint":
            raise ValueError("dpo.reference must be frozen_initial_checkpoint")
        if _recipe_number(dpo["label_smoothing"], "dpo.label_smoothing") != 0:
            raise ValueError("dpo.label_smoothing must be 0")
        if _recipe_number(dpo["beta"], "dpo.beta") <= 0:
            raise ValueError("dpo.beta must be positive")


def _resolve_initial_checkpoint(
    project_root: Path,
    raw: object,
    *,
    required: bool,
    runtime: BaseModelRuntimeConfig,
    portable: bool,
) -> InitialCheckpoint | None:
    if raw is None:
        if required:
            raise ValueError("This algorithm requires an initial SFT checkpoint")
        return None
    if portable:
        if not isinstance(raw, str) or not raw:
            raise ValueError("initial_checkpoint must be a repository-relative checkpoint directory")
        raw = {"directory": raw}
    if not isinstance(raw, dict):
        raise ValueError("initial_checkpoint must be an object")
    directory = resolve_repository_path(
        project_root, str(raw.get("directory")), expected_kind="directory"
    )
    config_path = directory / "adapter_config.json"
    metadata = inspect_lora_checkpoint(
        directory, expected_base_model_id=runtime.model_id,
        expected_base_revision=runtime.revision,
    )
    weights_path = directory / str(metadata.get("weights_file"))
    config_hash = sha256_file(config_path)
    expected_config = raw.get("expected_config_sha256")
    expected_weights = raw.get("expected_weights_sha256")
    if not portable and (
        config_hash != expected_config
        or metadata.get("weights_sha256") != expected_weights
    ):
        raise ValueError("Initial checkpoint binding mismatch")
    return InitialCheckpoint(
        directory=directory,
        config_path=config_path,
        weights_path=weights_path,
        config_sha256=config_hash,
        weights_sha256=str(metadata["weights_sha256"]),
        metadata=metadata,
    )


def _portable_config(raw: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "run_id", "allowed_modes", "training_recipe", "model_runtime",
                "dataset_manifest", "execution_profile", "seed"}
    optional = {"output_dir", "report", "backward", "smoke", "required_simulator_capabilities",
                "initial_checkpoint"}
    if required - raw.keys() or raw.keys() - required - optional:
        raise ValueError(f"Portable run fields: missing={sorted(required - raw.keys())}, "
                         f"unsupported={sorted(raw.keys() - required - optional)}")
    if not isinstance(raw["run_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw["run_id"]):
        raise ValueError("run_id must be a simple filename identifier")
    if isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int):
        raise ValueError("seed must be an integer")
    modes = raw["allowed_modes"]
    if not isinstance(modes, list) or not modes or any(mode not in TRAINING_MODES for mode in modes):
        raise ValueError("allowed_modes must list supported training modes")
    for key in ("training_recipe", "model_runtime", "dataset_manifest", "execution_profile"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"{key} must be a repository path")
    capabilities = raw.get("required_simulator_capabilities", [])
    if not isinstance(capabilities, list) or any(not isinstance(item, str) or not item for item in capabilities):
        raise ValueError("required_simulator_capabilities must be a list of names")
    config = dict(raw)
    config.setdefault("output_dir", f"outputs/training/{raw['run_id']}")
    config.setdefault("report", f"{config['output_dir']}/report.json")
    for mode in ("backward", "smoke"):
        if not isinstance(raw.get(mode, {}), dict):
            raise ValueError(f"Portable {mode} options must be an object")
        options = dict(raw.get(mode, {}))
        if options.keys() - {"output_dir", "report"}:
            raise ValueError(f"Unsupported portable {mode} fields")
        options.setdefault("output_dir", f"{config['output_dir']}/{mode}")
        options.setdefault("report", f"{options['output_dir']}/report.json")
        config[mode] = options
    for options in (config, config["backward"], config["smoke"]):
        if any(not isinstance(options[key], str) or not options[key] for key in ("output_dir", "report")):
            raise ValueError("output_dir and report must be repository paths")
    # These are execution guarantees, not knobs repeated in every experiment.
    config.update(test_data_read=False, sealed_test_run=False, stable_training_claim=False)
    return config


def load_configured_training_run(
    project_root: str | Path,
    config_path: str | Path,
    *,
    mode: str,
) -> ConfiguredTrainingRun:
    if mode not in TRAINING_MODES:
        raise ValueError(f"Unsupported training mode: {mode}")
    root = Path(project_root).resolve()
    run_document = load_config_document(
        root, config_path, supported_schemas={TRAINING_RUN_SCHEMA_VERSION, "training_run_v2"}
    )
    config = run_document.value
    portable = config["schema_version"] == "training_run_v2"
    if portable:
        config = _portable_config(config)
        run_document = replace(run_document, value=config)

    def reference(key: str, *schemas: str) -> ConfigDocument:
        if portable:
            return load_config_document(root, config[key], supported_schemas=set(schemas))
        return resolve_bound_document(root, config, path_key=key,
                                      sha256_key=f"expected_{key}_sha256", supported_schemas=set(schemas))

    execution_document = reference("execution_profile", "training_execution_v1", "training_execution_v2") if portable else None
    allowed_modes = config.get("allowed_modes")
    if not isinstance(allowed_modes, list) or mode not in allowed_modes:
        raise ValueError(f"Run config does not allow mode={mode}")
    if any(
        config.get(key) is not False
        for key in ("test_data_read", "sealed_test_run", "stable_training_claim")
    ):
        raise ValueError("Training run violates the project data boundary")
    recipe_document = reference("training_recipe", TRAINING_RECIPE_SCHEMA_VERSION)
    _validate_recipe(recipe_document.value)
    runtime_schemas = ("base_model_runtime_v2",) if portable else ("base_model_runtime_v1", "base_model_runtime_v2")
    runtime_document = reference("model_runtime", *runtime_schemas)
    dataset_document = reference("dataset_manifest", "dataset_manifest_v1")
    if portable and set(dataset_document.value.get("splits", {})) != {"train"}:
        raise ValueError("Portable training consumes a train-only manifest; no test/development assets are read")
    dataset = validate_dataset_manifest(
        dataset_document.value, project_root=root, verify_artifacts=True
    )
    algorithm = str(recipe_document.value["algorithm"])
    if dataset["task_type"] != ("sft" if algorithm == "sft" else "preference"):
        raise ValueError("Dataset task type disagrees with training recipe")
    split = dataset["splits"].get("train")
    if not isinstance(split, dict):
        raise ValueError("Training dataset manifest has no train split")
    artifact_path = resolve_repository_path(root, str(split["path"]), expected_kind="file")
    records = list(iter_jsonl(artifact_path))
    if len(records) != int(split["records"]):
        raise ValueError("Training dataset record count differs from its manifest")
    if algorithm == "sft":
        from .sft_groups import validate_weight_semantics
        validate_weight_semantics(records, dataset["semantics"])
    else:
        from .dpo import validate_preference_weight_semantics
        validate_preference_weight_semantics(records, dataset["semantics"])
    runtime = load_base_model_runtime_config(
        runtime_document.path, project_root=root,
        execution_profile=execution_document.value if execution_document else None,
    )
    tokenizer = load_verified_local_tokenizer(runtime, project_root=root)
    adapter = adapter_for_algorithm(algorithm)
    prepared = adapter.prepare(
        tokenizer,
        records,
        dataset_id=str(dataset["dataset_id"]),
        max_sequence_tokens=int(recipe_document.value["tokenization"]["max_sequence_tokens"]),
    )
    if portable:
        implementation_hashes = {}  # Portable compatibility is semantic; Git records source provenance.
    else:
        components = config.get("implementation_components")
        if not isinstance(components, list) or not components or len(set(components)) != len(components):
            raise ValueError("implementation_components must list unique source files")
        implementation_paths = [
            resolve_repository_path(root, str(item), expected_kind="file")
            for item in components
        ]
        implementation_hashes = {
            repository_relative(root, path): sha256_file(path) for path in implementation_paths
        }
    mode_config = config.get(mode) if mode in {"backward", "smoke"} else config
    if not isinstance(mode_config, dict):
        raise ValueError(f"Training run has no configuration for mode={mode}")
    output_dir = resolve_repository_path(
        root, str(mode_config.get("output_dir")), must_exist=False
    )
    report_path = resolve_repository_path(
        root, str(mode_config.get("report")), must_exist=False
    )
    checkpoint_raw = mode_config.get("initial_checkpoint", config.get("initial_checkpoint"))
    initial_checkpoint = _resolve_initial_checkpoint(
        root, checkpoint_raw, required=(adapter.requires_initial_checkpoint
                                       or recipe_document.value["lora"].get("source") == "initial_checkpoint"),
        runtime=runtime, portable=portable,
    )
    return ConfiguredTrainingRun(
        project_root=root,
        run_document=run_document,
        recipe_document=recipe_document,
        runtime_document=runtime_document,
        dataset_document=dataset_document,
        runtime=runtime,
        adapter=adapter,
        prepared=prepared,
        output_dir=output_dir,
        report_path=report_path,
        initial_checkpoint=initial_checkpoint,
        mode=mode,
        implementation_hashes=implementation_hashes,
        execution_document=execution_document,
        tokenizer=tokenizer,
    )
