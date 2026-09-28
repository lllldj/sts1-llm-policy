"""Portable training identity; legacy byte-bound runs keep their original contract."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from typing import TYPE_CHECKING, Sequence

from sts1_llm_policy.artifacts import canonical_json_bytes

if TYPE_CHECKING:
    from .configured_training import ConfiguredTrainingRun


# Bump these when implementation changes invalidate backward or continuation.
# Source formatting, logging and module relocation do not change these contracts.
BACKWARD_CONTRACT = "lora_response_backward_v1"
UPDATE_CONTRACT = "adamw_ordered_accumulation_rng_resume_v2"


def semantic_sha256(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _inputs_sha256(run: ConfiguredTrainingRun, indices: Sequence[int]) -> str:
    digest = sha256()
    for index in indices:
        unit = run.prepared.units[index]
        value = run.adapter.input_identity(unit)
        digest.update(canonical_json_bytes(value) + b"\n")
    return digest.hexdigest()


def portable_binding(
    run: ConfiguredTrainingRun, *, unit_indices: Sequence[int], purpose: str,
) -> dict[str, object]:
    runtime = asdict(run.runtime)
    # These tuples represent JSON maps, so object key order has no meaning.
    for key in ("weight_assets_sha256", "tokenizer_asset_sha256"):
        runtime[key] = dict(runtime[key])
    for key in (
        "runtime_id", "snapshot_path", "device", "dependencies", "execution_requirements",
        "validated_device_name", "validated_compute_capability", "validated_cuda_runtime",
        "validated_bf16_supported", "tokenizer_local_only_verified", "model_weights_loaded",
        "generation_smoke_completed",
    ):
        runtime.pop(key)
    recipe = {key: run.recipe[key] for key in ("schema_version", "algorithm", "lora", "tokenization")}
    recipe.update(run.adapter.loss_configuration(run.recipe))
    recipe["training_execution"] = {key: run.recipe["training"][key]
                                    for key in ("micro_batch_units", "gradient_checkpointing")}
    if purpose not in {"backward", "recovery"}:
        recipe["optimizer"] = run.recipe["optimizer"]
        recipe["training"] = {key: value for key, value in run.recipe["training"].items()
                              if key != "resume_checkpoint_interval_steps" and (purpose != "smoke" or key != "epochs")}
        if purpose == "smoke":
            recipe["smoke"] = run.recipe["smoke"]
    dataset = {key: run.dataset.get(key) for key in (
        "schema_version", "task_type", "observation_version", "identity_fields", "semantics",
    )}
    dataset["train_sha256"] = run.dataset["splits"]["train"]["sha256"]
    return {
        "schema_version": "portable_training_identity_v3",
        "purpose": purpose,
        "seed": run.config["seed"],
        "training_recipe_sha256": semantic_sha256(recipe),
        "model_runtime_sha256": semantic_sha256(runtime),
        "dataset_semantics_sha256": semantic_sha256(dataset),
        "backward_contract": BACKWARD_CONTRACT,
        "loss_contract": run.adapter.loss_contract,
        **({"update_contract": UPDATE_CONTRACT} if purpose not in {"backward", "recovery"} else {}),
        "prepared_inputs_sha256": _inputs_sha256(run, range(len(run.prepared.units))),
        "selected_inputs_sha256": _inputs_sha256(run, unit_indices),
        "initial_checkpoint_sha256": (
            semantic_sha256({key: value for key, value in run.initial_checkpoint.metadata.items()
                             if key != "weights_file"})
            if run.initial_checkpoint is not None else None
        ),
    }


def configuration_snapshot(run: ConfiguredTrainingRun) -> dict:
    """Record resolved documents for provenance, without making locations an identity."""
    return {name: {"path": document.relative_path, "file_sha256": document.sha256,
                   "resolved": document.value}
            for name, document in (
                ("run", run.run_document), ("recipe", run.recipe_document),
                ("model_runtime", run.runtime_document), ("dataset", run.dataset_document),
                ("execution", run.execution_document),
            ) if document is not None}


def training_binding(
    run: ConfiguredTrainingRun,
    *,
    unit_indices: Sequence[int],
) -> dict[str, object]:
    if run.config["schema_version"] == "training_run_v2":
        return portable_binding(run, unit_indices=unit_indices, purpose=run.mode)
    unit_ids = [run.prepared.units[index]["record_id"] for index in unit_indices]
    result: dict[str, object] = {
        "run_config_sha256": run.run_document.sha256,
        "training_recipe_sha256": run.recipe_document.sha256,
        "model_runtime_sha256": run.runtime_document.sha256,
        "dataset_manifest_sha256": run.dataset_document.sha256,
        "dataset_artifact_sha256": run.dataset["splits"]["train"]["sha256"],
        "implementation_components_sha256": run.implementation_hashes,
        "mode": run.mode,
        "selected_units_sha256": sha256("\n".join(unit_ids).encode("utf-8")).hexdigest(),
    }
    if run.initial_checkpoint is not None:
        result["initial_checkpoint"] = {
            "config_sha256": run.initial_checkpoint.config_sha256,
            "weights_sha256": run.initial_checkpoint.weights_sha256,
        }
    return result
