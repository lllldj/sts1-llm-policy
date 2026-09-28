from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch

from sts1_llm_policy.model_runtime import verify_model_weights
from sts1_llm_policy.execution_environment import verify_runtime_dependencies, verify_torch_hardware
from sts1_llm_policy.train.configured_adapters import select_smoke_unit_indices
from sts1_llm_policy.train.configured_training import ConfiguredTrainingRun
from sts1_llm_policy.train.readiness import require_recovery
from sts1_llm_policy.train.identity import configuration_snapshot, training_binding
from sts1_llm_policy.train.lora import (
    LoraSpec,
    adapter_state_sha256,
    inspect_lora_checkpoint,
    load_lora_checkpoint,
    save_lora_checkpoint,
    trainable_parameter_summary,
)
from sts1_llm_policy.train.runtime import (
    load_base_model, load_initial_model, set_training_seed,
    capture_training_rng, read_training_rng, restore_training_rng,
)
from sts1_llm_policy.artifacts import (
    atomic_write_json,
    read_json_object,
    repository_relative,
    sha256_file,
)


def _replace_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _enable_training(model: torch.nn.Module) -> None:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()


def _training_order(unit_count: int, *, epochs: int, seed: int) -> tuple[int, ...]:
    order: list[int] = []
    for epoch in range(epochs):
        indices = list(range(unit_count))
        random.Random(f"{seed}|epoch|{epoch}").shuffle(indices)
        order.extend(indices)
    return tuple(order)


def _backward_unit(run, model, unit, reference, *, scale=1.0):
    """Release each independent candidate graph before forwarding the next one."""
    total = 0.0
    terms = run.adapter.loss_terms(model, unit, runtime=run.runtime,
                                   reference=reference, recipe=run.recipe)
    # The generator's forward runs inside autocast; backward runs outside it.
    while True:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = next(terms, None)
        if loss is None:
            break
        if not bool(torch.isfinite(loss).item()):
            raise ValueError("Configured training produced a non-finite loss")
        total += float(loss.detach().cpu())
        (loss * scale).backward()
        del loss
    if not math.isfinite(total):
        raise ValueError("Configured training produced a non-finite unit loss")
    return total


def _save_checkpoint(
    run: ConfiguredTrainingRun,
    *,
    binding: Mapping[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: LoraSpec,
    replaced: Sequence[str],
    state: Mapping[str, object],
) -> Path:
    step = int(state["optimizer_step_count"])
    checkpoint = run.output_dir / "resume" / f"step-{step:06d}"
    if checkpoint.exists():
        saved = read_json_object(checkpoint / "state.json")
        read_training_rng(checkpoint, saved, torch.device(run.runtime.device))
        if (
            saved.get("binding") != dict(binding)
            or saved.get("adapter_state_sha256") != adapter_state_sha256(model)
            or saved.get("optimizer_sha256") != sha256_file(checkpoint / "optimizer.pt")
        ):
            raise ValueError(f"Conflicting training checkpoint: {checkpoint}")
    else:
        temporary = checkpoint.with_name(f".{checkpoint.name}.tmp-{uuid4().hex}")
        temporary.mkdir(parents=True)
        try:
            metadata = save_lora_checkpoint(
                model,
                temporary / "adapter",
                spec=spec,
                replaced_modules=replaced,
                base_model_id=run.runtime.model_id,
                base_revision=run.runtime.revision,
            )
            optimizer_path = temporary / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            rng_path = temporary / "rng.pt"
            torch.save(capture_training_rng(torch.device(run.runtime.device)), rng_path)
            _replace_json(temporary / "state.json", {
                **state,
                "binding": dict(binding),
                **({"configuration": configuration_snapshot(run)} if run.execution_document else {}),
                "adapter_metadata": metadata,
                "adapter_state_sha256": adapter_state_sha256(model),
                "optimizer_sha256": sha256_file(optimizer_path),
                "rng_sha256": sha256_file(rng_path),
            })
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(checkpoint)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    _replace_json(run.output_dir / "latest.json", {
        "schema_version": "configured_training_latest_v1",
        "binding": dict(binding),
        "checkpoint": str(checkpoint.relative_to(run.output_dir)).replace("\\", "/"),
        "optimizer_step_count": state["optimizer_step_count"],
        "next_order_offset": state["next_order_offset"],
    })
    return checkpoint


def _load_or_initialize(
    run: ConfiguredTrainingRun,
    *,
    binding: Mapping[str, object],
) -> tuple[torch.nn.Module, torch.optim.AdamW, LoraSpec, tuple[str, ...], dict[str, Any], bool]:
    # run_training has verified the Base assets and execution environment.
    latest_path = run.output_dir / "latest.json"
    resumed = latest_path.exists()
    if resumed:
        latest = read_json_object(latest_path)
        if latest.get("binding") != dict(binding):
            raise ValueError("Training resume binding changed")
        checkpoint = (run.output_dir / str(latest["checkpoint"])).resolve()
        if not checkpoint.is_relative_to(run.output_dir) or not checkpoint.is_dir():
            raise ValueError("Training resume checkpoint path is invalid")
        model = load_base_model(
            run.runtime, project_root=run.project_root, verify_assets=False
        )
        spec, replaced, metadata = load_lora_checkpoint(
            model,
            checkpoint / "adapter",
            expected_base_model_id=run.runtime.model_id,
            expected_base_revision=run.runtime.revision,
        )
        model.to(torch.device(run.runtime.device))
        state = read_json_object(checkpoint / "state.json")
        if (
            state.get("binding") != dict(binding)
            or state.get("adapter_metadata") != metadata
            or state.get("optimizer_sha256") != sha256_file(checkpoint / "optimizer.pt")
            or state.get("adapter_state_sha256") != adapter_state_sha256(model)
            or state.get("optimizer_step_count") != latest.get("optimizer_step_count")
            or state.get("next_order_offset") != latest.get("next_order_offset")
        ):
            raise ValueError("Training resume checkpoint state is invalid")
    else:
        model, spec, replaced = load_initial_model(run, verify_assets=False)
        state = {
            "schema_version": "configured_training_state_v1",
            "optimizer_step_count": 0,
            "next_order_offset": 0,
            "processed_units": 0,
            "initial_adapter_state_sha256": adapter_state_sha256(model),
            "losses": [],
            "optimizer_steps": [],
            "peak_memory_allocated_bytes": 0,
            "peak_memory_reserved_bytes": 0,
        }
    optimizer_config = run.recipe["optimizer"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    if resumed:
        optimizer.load_state_dict(
            torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
        )
        _optimizer_to_device(optimizer, torch.device(run.runtime.device))
        # Restore after model/adapter construction, which consumes random numbers.
        rng = read_training_rng(checkpoint, state, torch.device(run.runtime.device))
        restore_training_rng(rng, torch.device(run.runtime.device))
    return model, optimizer, spec, tuple(replaced), state, resumed


def _reference_values(
    run: ConfiguredTrainingRun,
    *,
    units: Sequence[Mapping[str, Any]],
    binding: Mapping[str, object],
    model: torch.nn.Module | None = None,
) -> tuple[Mapping[str, Any] | None, dict[str, int]]:
    if not run.adapter.uses_reference:
        return None, {"groups_reused": 0, "groups_computed": 0}
    reference_dir = run.output_dir / "reference"
    values: dict[str, Any] = {}
    pending: list[tuple[int, Mapping[str, Any], Path]] = []
    for index, unit in enumerate(units):
        path = reference_dir / f"group-{index:06d}.json"
        if path.exists():
            artifact = read_json_object(path)
            unit_values = artifact.get("values")
            if (
                artifact.get("schema_version") != "configured_dpo_reference_group_v1"
                or artifact.get("binding") != dict(binding)
                or artifact.get("record_id") != unit["record_id"]
                or not run.adapter.validate_reference_values(unit, unit_values)
            ):
                raise ValueError(f"DPO reference cache binding failed: {path}")
            values[str(unit["record_id"])] = unit_values
        else:
            pending.append((index, unit, path))
    if pending:
        owns_model = model is None
        if owns_model:
            # Reuse the same run's Base validation for reference and training loads.
            model, _, _ = load_initial_model(run, verify_assets=False)
            model.requires_grad_(False)
        for position, (_, unit, path) in enumerate(pending, start=1):
            computed = run.adapter.reference_values(
                model, (unit,), runtime=run.runtime
            )[str(unit["record_id"])]
            if not run.adapter.validate_reference_values(unit, computed):
                raise ValueError("Reference calculation produced invalid values")
            artifact = {
                "schema_version": "configured_dpo_reference_group_v1",
                "binding": dict(binding),
                "record_id": unit["record_id"],
                "values": computed,
            }
            atomic_write_json(path, artifact)
            values[str(unit["record_id"])] = computed
            if position == 1 or position % 25 == 0 or position == len(pending):
                print(json.dumps({
                    "reference_groups": position,
                    "reference_groups_pending": len(pending),
                }), flush=True)
        if owns_model:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return values, {
        "groups_reused": len(units) - len(pending),
        "groups_computed": len(pending),
    }


def run_training(
    run: ConfiguredTrainingRun, *, recovery_report: Path | None = None,
    backward_model: tuple | None = None, persist_report: bool = True,
) -> dict[str, Any]:
    if backward_model is not None and run.mode != "backward":
        raise ValueError("A prepared backward model is only valid in backward mode")
    execution = None
    if run.config["schema_version"] == "training_run_v2" and run.mode != "preflight" and backward_model is None:
        if run.mode == "backward":
            raise ValueError("Use scripts/restore_training.py for portable backward acceptance")
        execution = require_recovery(run, recovery_report)
    recipe = run.recipe
    training = recipe["training"]
    seed = int(run.config["seed"])
    all_units = run.prepared.units
    if run.mode == "backward":
        unit_indices = select_smoke_unit_indices(all_units, count=1, seed=seed)
        epochs = 1
    elif run.mode == "smoke":
        unit_indices = select_smoke_unit_indices(
            all_units, count=int(recipe["smoke"]["units"]), seed=seed
        )
        epochs = 1
    else:
        unit_indices = tuple(range(len(all_units)))
        epochs = int(training["epochs"])
    accumulation = int(training["gradient_accumulation_units"])
    expected_steps = (
        0
        if run.mode == "backward"
        else math.ceil(len(unit_indices) * epochs / accumulation)
    )
    if run.mode == "smoke" and expected_steps != int(recipe["smoke"]["optimizer_steps"]):
        raise ValueError("Smoke unit count does not produce the configured optimizer steps")
    binding = training_binding(run, unit_indices=unit_indices)
    scope = {
        "algorithm": run.adapter.algorithm,
        "dataset_id": run.dataset["dataset_id"],
        "available_units": len(all_units),
        "selected_units": len(unit_indices),
        "epochs": epochs,
        "gradient_accumulation_units": accumulation,
        "derived_optimizer_steps": expected_steps,
    }
    checks = {
        "manifest_and_artifact_bound": True,
        "record_count_derived_from_manifest": len(all_units)
        == int(run.dataset["splits"]["train"]["records"]),
        "optimizer_steps_derived": True,
        "tokenization_without_truncation": run.prepared.tokenization["truncated"] is False,
        "no_test_or_sealed_read": True,
    }
    if run.mode == "preflight":
        return {
            "schema_version": "configured_training_report_v1",
            "run_id": run.config["run_id"],
            "status": "preflight_passed",
            "mode": run.mode,
            "binding": binding,
            **({"configuration": configuration_snapshot(run)} if run.execution_document else {}),
            "scope": scope,
            "tokenization": run.prepared.tokenization,
            "checks": checks,
            "model_weights_loaded": False,
            "training_started": False,
            "test_data_read": False,
        }
    if run.mode == "backward":
        if persist_report and run.report_path.exists():
            prior = read_json_object(run.report_path)
            if (
                prior.get("binding") == binding
                and prior.get("status") == "backward_passed"
            ):
                return prior
            raise ValueError(f"Conflicting completed backward report: {run.report_path}")
        run.output_dir.mkdir(parents=True, exist_ok=True)
        selected_unit = all_units[unit_indices[0]]
        selected_units = (selected_unit,)
        set_training_seed(seed)
        model, _, _ = backward_model if backward_model is not None else load_initial_model(run, verify_assets=True)
        reference, reference_summary = _reference_values(
            run, units=selected_units, binding=binding, model=model
        )
        initial_adapter_hash = adapter_state_sha256(model)
        parameters = trainable_parameter_summary(model)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        _enable_training(model)
        for parameter in trainable:
            parameter.grad = None
        device = torch.device(run.runtime.device)
        torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        loss_value = _backward_unit(run, model, selected_unit, reference)
        finite_loss = math.isfinite(loss_value)
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float("inf"))
        finite_gradient_norm = bool(torch.isfinite(grad_norm).item())
        nonzero_gradients = any(
            bool(torch.count_nonzero(gradient).item()) for gradient in gradients
        )
        final_adapter_hash = adapter_state_sha256(model)
        checks.update({
            "longest_unit_selected": int(selected_unit["sequence_tokens_max"])
            == int(run.prepared.tokenization["sequence_tokens_max"]),
            "finite_loss": finite_loss,
            "gradient_tensors_present": bool(gradients),
            "finite_gradient_norm": finite_gradient_norm,
            "nonzero_gradients": nonzero_gradients,
            "optimizer_not_created": True,
            "optimizer_step_count_zero": True,
            "adapter_unchanged_without_step": final_adapter_hash == initial_adapter_hash,
        })
        status = "backward_passed" if all(checks.values()) else "no_go"
        report = {
            "schema_version": "configured_training_report_v1",
            "run_id": f"{run.config['run_id']}_backward",
            "status": status,
            "mode": run.mode,
            "binding": binding,
            **({"configuration": configuration_snapshot(run)} if run.execution_document else {}),
            "scope": scope,
            "tokenization": run.prepared.tokenization,
            "runtime": {
                "runtime_id": run.runtime.runtime_id,
                "model_id": run.runtime.model_id,
                "revision": run.runtime.revision,
                "device": run.runtime.device,
                "dtype": run.runtime.dtype,
            },
            "lora": {
                "spec": recipe["lora"],
                "parameters": parameters,
                "adapter_state_sha256": initial_adapter_hash,
            },
            "backward": {
                "record_id": selected_unit["record_id"],
                "sequence_tokens": int(selected_unit["sequence_tokens_max"]),
                "loss": loss_value,
                "gradient_tensor_count": len(gradients),
                "gradient_norm": float(grad_norm.detach().cpu()),
                "optimizer_created": False,
                "optimizer_step_count": 0,
                "seconds": round(perf_counter() - started, 3),
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            },
            "reference": reference_summary,
            "checks": checks,
            "model_weights_loaded": True,
            "training_started": False,
            "test_data_read": False,
            "sealed_test_run": False,
            "stable_training_claim": False,
            "git_revision": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=run.project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if persist_report:
            atomic_write_json(run.report_path, report)
        return report
    if run.report_path.exists():
        prior = read_json_object(run.report_path)
        if prior.get("binding") == binding and prior.get("status") in {
            "smoke_passed", "trained_pending_development_evaluation"
        }:
            metadata = inspect_lora_checkpoint(
                run.output_dir / "checkpoint",
                expected_base_model_id=run.runtime.model_id,
                expected_base_revision=run.runtime.revision,
            )
            if (prior.get("checkpoint", {}).get("metadata") != metadata
                    or prior["checkpoint"].get("reload_exact") is not True
                    or prior.get("checks", {}).get("checkpoint_reload_exact") is not True):
                raise ValueError("Completed training report differs from its final checkpoint")
            return prior
        raise ValueError(f"Conflicting completed training report: {run.report_path}")
    # The configured run already validated model metadata and tokenizer assets.
    # Portable recovery validation above also checked this process's environment.
    if execution is None:
        verify_runtime_dependencies(run.runtime)
        verify_torch_hardware(run.runtime, torch)
    verify_model_weights(run.runtime)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    selected_units = tuple(all_units[index] for index in unit_indices)
    reference, reference_summary = _reference_values(
        run, units=selected_units, binding=binding
    )
    set_training_seed(seed)
    model, optimizer, spec, replaced, state, resumed = _load_or_initialize(
        run, binding=binding
    )
    if execution is not None:
        history = list(state.get("execution_history", []))
        if not history or history[-1] != execution:
            history.append(execution)
        state["execution_history"] = history
    initial_adapter_hash = str(state["initial_adapter_state_sha256"])
    parameters = trainable_parameter_summary(model)
    order = _training_order(len(selected_units), epochs=epochs, seed=seed)
    offset = int(state["next_order_offset"])
    losses = [float(item) for item in state["losses"]]
    optimizer_steps = list(state["optimizer_steps"])
    optimizer_step_count = int(state["optimizer_step_count"])
    device = torch.device(run.runtime.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _enable_training(model)
    optimizer.zero_grad(set_to_none=True)
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint_interval = (
        1 if run.mode == "smoke" else int(training["resume_checkpoint_interval_steps"])
    )
    while offset < len(order):
        group = order[offset : offset + accumulation]
        group_losses: list[float] = []
        for position in group:
            unit = selected_units[position]
            group_losses.append(_backward_unit(run, model, unit, reference, scale=1 / len(group)))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(recipe["optimizer"]["max_grad_norm"])
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise ValueError("Configured training produced a non-finite gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        offset += len(group)
        optimizer_step_count += 1
        losses.extend(group_losses)
        optimizer_steps.append({
            "optimizer_step": optimizer_step_count,
            "next_order_offset": offset,
            "group_mean_loss": sum(group_losses) / len(group_losses),
            "grad_norm": float(grad_norm.detach().cpu()),
        })
        checkpoint_state = {
            **state,
            "optimizer_step_count": optimizer_step_count,
            "next_order_offset": offset,
            "processed_units": offset,
            "losses": losses,
            "optimizer_steps": optimizer_steps,
            "peak_memory_allocated_bytes": max(
                int(state.get("peak_memory_allocated_bytes", 0)),
                torch.cuda.max_memory_allocated(device),
            ),
            "peak_memory_reserved_bytes": max(
                int(state.get("peak_memory_reserved_bytes", 0)),
                torch.cuda.max_memory_reserved(device),
            ),
        }
        state = checkpoint_state
        print(json.dumps({
            "algorithm": run.adapter.algorithm,
            "mode": run.mode,
            "optimizer_step": optimizer_step_count,
            "optimizer_steps_total": expected_steps,
            "units": offset,
            "units_total": len(order),
            "group_mean_loss": round(sum(group_losses) / len(group_losses), 6),
        }), flush=True)
        if optimizer_step_count % checkpoint_interval == 0 or offset == len(order):
            _save_checkpoint(
                run,
                binding=binding,
                model=model,
                optimizer=optimizer,
                spec=spec,
                replaced=replaced,
                state=checkpoint_state,
            )
    final_adapter_hash = adapter_state_sha256(model)
    final_checkpoint = run.output_dir / "checkpoint"
    existing_final = final_checkpoint.exists()
    if not existing_final:
        temporary = final_checkpoint.with_name(f".checkpoint.tmp-{uuid4().hex}")
        try:
            metadata = save_lora_checkpoint(
                model,
                temporary,
                spec=spec,
                replaced_modules=replaced,
                base_model_id=run.runtime.model_id,
                base_revision=run.runtime.revision,
            )
            temporary.replace(final_checkpoint)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = load_base_model(
        run.runtime, project_root=run.project_root, verify_assets=False
    )
    _, reload_modules, reload_metadata = load_lora_checkpoint(
        reloaded,
        final_checkpoint,
        expected_base_model_id=run.runtime.model_id,
        expected_base_revision=run.runtime.revision,
    )
    reload_hash = adapter_state_sha256(reloaded)
    if existing_final:
        if reload_modules != replaced or reload_hash != final_adapter_hash:
            raise ValueError("Existing final training checkpoint conflicts")
        metadata = reload_metadata
    reload_exact = (
        reload_modules == replaced
        and reload_metadata == metadata
        and reload_hash == final_adapter_hash
    )
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    checks.update({
        "all_selected_units_processed": offset == len(order),
        "derived_optimizer_steps_completed": optimizer_step_count == expected_steps,
        "finite_losses": all(math.isfinite(item) for item in losses),
        "adapter_changed": final_adapter_hash != initial_adapter_hash,
        "checkpoint_reload_exact": reload_exact,
    })
    status = (
        "smoke_passed"
        if run.mode == "smoke" and all(checks.values())
        else "trained_pending_development_evaluation"
        if run.mode == "run" and all(checks.values())
        else "no_go"
    )
    report = {
        "schema_version": "configured_training_report_v1",
        "run_id": run.config["run_id"] if run.mode == "run" else f"{run.config['run_id']}_smoke",
        "status": status,
        "mode": run.mode,
        "binding": binding,
        **({"configuration": configuration_snapshot(run)} if run.execution_document else {}),
        "scope": scope,
        "tokenization": run.prepared.tokenization,
        "runtime": {
            "runtime_id": run.runtime.runtime_id,
            "model_id": run.runtime.model_id,
            "revision": run.runtime.revision,
            "device": run.runtime.device,
            "dtype": run.runtime.dtype,
        },
        "lora": {
            "spec": recipe["lora"],
            "replaced_module_count": len(replaced),
            "parameters": parameters,
            "initial_adapter_state_sha256": initial_adapter_hash,
            "final_adapter_state_sha256": final_adapter_hash,
        },
        "training": {
            "resumed": resumed,
            "optimizer_step_count": optimizer_step_count,
            "processed_units": offset,
            "minimum_loss": min(losses),
            "maximum_loss": max(losses),
            "optimizer_steps": optimizer_steps,
            "seconds": round(perf_counter() - started, 3),
            "peak_memory_allocated_bytes": state["peak_memory_allocated_bytes"],
            "peak_memory_reserved_bytes": state["peak_memory_reserved_bytes"],
        },
        "reference": reference_summary,
        "checkpoint": {
            "path": repository_relative(run.project_root, final_checkpoint),
            "metadata": metadata,
            "reload_adapter_state_sha256": reload_hash,
            "reload_exact": reload_exact,
        },
        "checks": checks,
        "development_evaluation_required": run.mode == "run",
        "model_weights_loaded": True,
        "training_started": True,
        "test_data_read": False,
        "sealed_test_run": False,
        "stable_training_claim": False,
        "git_revision": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=run.project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
    }
    if execution is not None:
        report["execution_history"] = state["execution_history"]
    atomic_write_json(run.report_path, report)
    return report
