"""Validate recovery evidence before starting a configured training run."""
from __future__ import annotations

from pathlib import Path
import torch

from sts1_llm_policy.execution_environment import observe_environment
from sts1_llm_policy.train.configured_adapters import select_smoke_unit_indices
from sts1_llm_policy.train.configured_training import ConfiguredTrainingRun
from sts1_llm_policy.train.identity import portable_binding, semantic_sha256
from sts1_llm_policy.artifacts import read_json_object, resolve_repository_path, sha256_file


BACKWARD_CHECKS = frozenset({
    "longest_unit_selected", "finite_loss", "gradient_tensors_present", "finite_gradient_norm",
    "nonzero_gradients", "optimizer_not_created", "optimizer_step_count_zero",
    "adapter_unchanged_without_step", "no_test_or_sealed_read",
})


def recovery_binding(run: ConfiguredTrainingRun) -> dict:
    longest = select_smoke_unit_indices(run.prepared.units, count=1, seed=int(run.config["seed"]))
    return portable_binding(run, unit_indices=longest, purpose="recovery")


def reusable_backward(report: dict, binding: dict, environment: dict, execution_sha256: str) -> bool:
    backward = report.get("backward", {})
    return bool(
        environment.get("driver_versions")
        and report.get("schema_version") == "training_recovery_v1"
        and report.get("status") == "ready"
        and report.get("binding") == binding
        and report.get("environment") == environment
        and report.get("execution_profile_sha256") == execution_sha256
        and report.get("model_load") == "executed"
        and report.get("generation", {}).get("status") == "passed"
        and backward.get("status") == "backward_passed"
        and BACKWARD_CHECKS <= backward.get("checks", {}).keys()
        and all(value is True for value in backward["checks"].values())
        and report.get("training_started") is False
    )


def require_recovery(run: ConfiguredTrainingRun, path: Path | None) -> dict:
    if path is None:
        raise ValueError("Portable training requires --recovery-report from scripts/restore_training.py")
    report_path = resolve_repository_path(run.project_root, path, expected_kind="file")
    report = read_json_object(report_path)
    environment = observe_environment(run.runtime, torch)
    if not reusable_backward(report, recovery_binding(run), environment, semantic_sha256(run.execution_document.value)):
        raise ValueError("Recovery evidence does not match this training/environment; run restore_training.py")
    return {"environment": environment, "recovery_report_sha256": sha256_file(report_path),
            "execution_profile_sha256": semantic_sha256(run.execution_document.value)}
