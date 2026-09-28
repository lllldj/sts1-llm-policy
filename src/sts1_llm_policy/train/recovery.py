"""Bounded recovery of a configured training execution on the current host."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import gc
from pathlib import Path
import subprocess
from typing import Any

import torch

from sts1_llm_policy.model_runtime import verify_model_weights
from sts1_llm_policy.train.configured_adapters import select_smoke_unit_indices
from sts1_llm_policy.train.configured_runner import run_training
from sts1_llm_policy.train.configured_training import ConfiguredTrainingRun
from sts1_llm_policy.execution_environment import observe_environment
from sts1_llm_policy.train.identity import configuration_snapshot, training_binding, semantic_sha256
from sts1_llm_policy.train.runtime import load_initial_model, set_training_seed, read_training_rng
from sts1_llm_policy.train.readiness import recovery_binding, reusable_backward
from sts1_llm_policy.artifacts import (
    atomic_write_json,
    read_json_object,
    resolve_repository_path,
    sha256_file,
)


def check_git(root: Path, expected_commit: str | None) -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              check=True, timeout=15).stdout.strip()
    commit = git("rev-parse", "HEAD")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(f"Git HEAD {commit} differs from expected commit {expected_commit}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked worktree changes must be resolved before recovery")
    return commit


def inspect_checkpoint(run: ConfiguredTrainingRun) -> dict:
    """Check the latest saved state without creating an optimizer or changing it."""
    root = resolve_repository_path(run.project_root, run.config["output_dir"], must_exist=False)
    latest_path = root / "latest.json"
    if not latest_path.exists():
        if (root / "resume").exists() or (root / "checkpoint").exists():
            raise ValueError("Checkpoint files exist without latest.json; inspect incomplete migration")
        return {"status": "not_present"}
    latest = read_json_object(latest_path)
    training = replace(run, mode="run", output_dir=root)
    expected = training_binding(training, unit_indices=tuple(range(len(run.prepared.units))))
    if latest.get("binding") != expected:
        raise ValueError("Migrated checkpoint binding differs from this training run")
    relative = latest.get("checkpoint")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Invalid checkpoint pointer")
    checkpoint = resolve_repository_path(root, relative, expected_kind="directory")
    state = read_json_object(checkpoint / "state.json")
    metadata = read_json_object(checkpoint / "adapter/adapter_config.json")
    weights = resolve_repository_path(checkpoint / "adapter", str(metadata.get("weights_file")), expected_kind="file")
    if (
        state.get("binding") != expected or state.get("adapter_metadata") != metadata
        or metadata.get("base_model_id") != run.runtime.model_id
        or metadata.get("base_revision") != run.runtime.revision
        or metadata.get("weights_sha256") != sha256_file(weights)
        or state.get("optimizer_sha256") != sha256_file(checkpoint / "optimizer.pt")
        or any(state.get(key) != latest.get(key) for key in ("optimizer_step_count", "next_order_offset"))
    ):
        raise ValueError("Migrated checkpoint state or asset hash mismatch")
    read_training_rng(checkpoint, state, torch.device(run.runtime.device))
    return {"status": "verified", "optimizer_step_count": state["optimizer_step_count"]}


def generation_probe(run: ConfiguredTrainingRun, model: Any) -> dict:
    index = select_smoke_unit_indices(run.prepared.units, count=1, seed=int(run.config["seed"]))[0]
    prompt = run.adapter.generation_prompt(run.prepared.units[index])
    inputs = torch.tensor([prompt], device=run.runtime.device, dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
            num_beams=run.runtime.num_beams, repetition_penalty=run.runtime.repetition_penalty,
            max_new_tokens=run.runtime.max_new_tokens, use_cache=True,
            eos_token_id=list(run.runtime.eos_token_ids), pad_token_id=run.runtime.generation_pad_token_id,
        )
    tokens = generated[0, len(prompt):].tolist()
    if not 0 < len(tokens) <= run.runtime.max_new_tokens:
        raise ValueError("Generation smoke did not produce a bounded response")
    return {"status": "passed", "execution": "executed", "new_tokens": len(tokens),
            "text": run.tokenizer.decode(tokens, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)}


def recover_training(
    run: ConfiguredTrainingRun, report_path: Path, *, previous: Path | None = None,
    expected_commit: str | None = None,
) -> dict:
    if run.execution_document is None or run.mode != "backward":
        raise ValueError("Recovery requires a portable training configuration in backward mode")
    report_path = resolve_repository_path(run.project_root, report_path, must_exist=False)
    if report_path.exists():
        raise ValueError("Recovery report already exists; choose a new path")
    report: dict = {
        "schema_version": "training_recovery_v1", "status": "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_started": False, "model_load": "not_executed",
        "generation": {"status": "not_executed"}, "backward": {"status": "not_executed"},
    }
    model_bundle = None
    try:
        print("Checking Git and current execution environment...", flush=True)
        report["git_revision"] = check_git(run.project_root, expected_commit)
        report["environment"] = observe_environment(run.runtime, torch)
        report["binding"] = recovery_binding(run)
        report["configuration"] = configuration_snapshot(run)
        report["execution_profile_sha256"] = semantic_sha256(run.execution_document.value)
        report["checkpoint"] = inspect_checkpoint(run)
        capabilities = run.config.get("required_simulator_capabilities", [])
        if capabilities:
            from sts1_llm_policy.env.simulator_execution import resolve_simulator
            execution = resolve_simulator(project_root=run.project_root, required_capabilities=capabilities)
            client = execution.create_client()
            try:
                report["simulator"] = {"status": "passed", "execution": "executed", "hello": client.hello_info}
            finally:
                client.close()
        prior = read_json_object(resolve_repository_path(run.project_root, previous, expected_kind="file")) if previous else {}
        reuse = reusable_backward(prior, report["binding"], report["environment"], semantic_sha256(run.execution_document.value))
        print("Verifying weights, loading model once and running short generation...", flush=True)
        # Config loading checked metadata/tokenizer; observe_environment checked
        # this process's dependencies and device. Only Base weights remain.
        verify_model_weights(run.runtime)
        set_training_seed(int(run.config["seed"]))
        model_bundle = load_initial_model(run, verify_assets=False)
        report["model_load"] = "executed"
        report["assets"] = "verified_from_existing_config_and_manifest"
        report["generation"] = generation_probe(run, model_bundle[0])
        if reuse:
            report["backward"] = {**prior["backward"], "execution": "reused",
                                  "source_report_sha256": sha256_file(run.project_root / previous)}
            print("Reusing backward evidence for matching dependencies and environment.", flush=True)
        else:
            print("Running longest-sample backward (no optimizer)...", flush=True)
            probe = run_training(run, backward_model=model_bundle, persist_report=False)
            report["backward"] = {"status": probe["status"], "execution": "executed",
                                  "checks": probe["checks"], "metrics": probe["backward"]}
            if probe["status"] != "backward_passed":
                raise ValueError("Longest-sample backward did not pass")
        # Do not announce readiness if the checkout changed during the probe.
        check_git(run.project_root, report["git_revision"])
        report["status"] = "ready"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        model_bundle = None
        gc.collect()
        torch.cuda.empty_cache()
    atomic_write_json(report_path, report)
    return report
