from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sts1_llm_policy.model_runtime import load_base_model_runtime_config
from sts1_llm_policy.execution_environment import verify_torch_hardware
from sts1_llm_policy.train.configured_adapters import SftAdapter
from sts1_llm_policy.train.identity import training_binding
from sts1_llm_policy.train.configured_runner import (
    _save_checkpoint,
    _load_or_initialize,
    run_training,
)
from sts1_llm_policy.train.lora import LoraSpec, inject_lora, adapter_state_sha256
from sts1_llm_policy.train.configured_training import load_configured_training_run, _portable_config
from sts1_llm_policy.train.identity import semantic_sha256
from sts1_llm_policy.train.readiness import (
    BACKWARD_CHECKS,
    recovery_binding,
    require_recovery,
    reusable_backward,
)
from sts1_llm_policy.train.recovery import (
    check_git,
    generation_probe,
    inspect_checkpoint,
    recover_training,
)
from sts1_llm_policy.configuration import load_config_document
from sts1_llm_policy.artifacts import read_json_object, sha256_file

from .runtime_fixture import CONFIG, RUNNER, fixture, EXECUTION
from tests.support import write_json


RECOVERY = "sts1_llm_policy.train.recovery."
ENVIRONMENT = {"device_name": "fixture GPU A", "driver_versions": ["test-driver"]}


def passed_report(run, environment=ENVIRONMENT):
    return {"schema_version": "training_recovery_v1", "status": "ready",
            "binding": recovery_binding(run), "environment": deepcopy(environment),
            "execution_profile_sha256": semantic_sha256(run.execution_document.value),
            "model_load": "executed", "generation": {"status": "passed"},
            "backward": {"status": "backward_passed", "execution": "executed",
                         "checks": dict.fromkeys(BACKWARD_CHECKS, True)},
            "training_started": False}


class TrainingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = fixture(self.root)

    def test_legacy_runtime_rejects_execution_override(self):
        with self.assertRaisesRegex(ValueError, "Legacy runtime"):
            load_base_model_runtime_config(self.root / "legacy.json", project_root=self.root,
                                           execution_profile=self.run.execution_document.value)

    def test_portable_loader_validates_assets_and_allows_execution_profile_changes(self):
        config = read_json_object(self.root / CONFIG)
        artifact = self.root / "train.jsonl"
        artifact.write_text('{"record_id":"long"}\n', encoding="utf-8")
        manifest = {"schema_version": "dataset_manifest_v1", "dataset_id": "test", "task_type": "sft",
                    "observation_version": "observation_v5", "identity_fields": ["record_id"],
                    "lineage": {"source": "test"}, "semantics": {"labels": "test"}, "splits": {"train": {
                        "path": "train.jsonl", "sha256": sha256_file(artifact), "bytes": artifact.stat().st_size,
                        "records": 1, "format": "jsonl", "compression": None}}}
        write_json(self.root / "dataset.json", manifest)
        with patch("sts1_llm_policy.train.configured_training.load_verified_local_tokenizer", return_value=object()), \
             patch.object(SftAdapter, "prepare", return_value=self.run.prepared):
            first = load_configured_training_run(self.root, CONFIG, mode="backward")
            execution = deepcopy(EXECUTION)
            execution["hardware"]["compute_capabilities"] = ["8.9"]
            write_json(self.root / "execution.json", execution)
            loaded = load_configured_training_run(self.root, CONFIG, mode="backward")
            self.assertEqual(first.output_dir, self.root / "outputs/training/fixture/backward")
            self.assertEqual(training_binding(first, unit_indices=(0,)), training_binding(loaded, unit_indices=(0,)))
            self.assertNotEqual(loaded.runtime.execution_requirements, first.runtime.execution_requirements)
            artifact.write_text("damaged", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_configured_training_run(self.root, CONFIG, mode="backward")
            manifest["splits"]["test"] = {**manifest["splits"]["train"], "path": "sealed/must-not-read.jsonl"}
            write_json(self.root / "dataset.json", manifest)
            with self.assertRaisesRegex(ValueError, "train-only"):
                load_configured_training_run(self.root, CONFIG, mode="backward")
        for field in ("expected_dataset_manifest_sha256", "implementation_components", "test_data_read", "typo"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unsupported"):
                _portable_config({**config, field: "invalid"})

    def test_actual_tokens_labels_and_nonlongest_inputs_invalidate_both_bindings(self):
        original = training_binding(self.run, unit_indices=(0,))
        for key in ("input_ids", "attention_mask", "labels", "prompt_token_count"):
            units = deepcopy(self.run.prepared.units)
            value = units[0]["tokenized"][key]
            if isinstance(value, list):
                value[-1] += 1
            else:
                units[0]["tokenized"][key] += 1
            changed = replace(self.run, prepared=replace(self.run.prepared, units=units))
            self.assertNotEqual(original, training_binding(changed, unit_indices=(0,)))
            self.assertNotEqual(recovery_binding(self.run), recovery_binding(changed))
        second = deepcopy(self.run.prepared.units[0])
        second.update(record_id="short", sequence_tokens_max=4)
        run = replace(self.run, prepared=replace(self.run.prepared, units=(*self.run.prepared.units, second)))
        before = recovery_binding(run)
        second["tokenized"]["input_ids"][0] += 1
        self.assertNotEqual(before, recovery_binding(run))

    def test_json_format_paths_and_metadata_do_not_change_identity(self):
        original = training_binding(self.run, unit_indices=(0,))
        path = self.run.recipe_document.path
        path.write_text(json.dumps(dict(reversed(list(self.run.recipe.items()))), indent=4), encoding="utf-8")
        changed = replace(self.run, recipe_document=load_config_document(self.root, path))
        self.assertNotEqual(changed.recipe_document.sha256, self.run.recipe_document.sha256)
        self.assertEqual(original, training_binding(changed, unit_indices=(0,)))
        changed = replace(changed, runtime=replace(changed.runtime, runtime_id="new-name", snapshot_path=self.root / "moved"))
        self.assertEqual(original, training_binding(changed, unit_indices=(0,)))
        self.assertEqual(semantic_sha256(EXECUTION), semantic_sha256(dict(reversed(list(EXECUTION.items())))))

    def test_optimizer_change_invalidates_resume_but_not_backward_evidence(self):
        recipe = deepcopy(self.run.recipe)
        recipe["optimizer"]["learning_rate"] *= 2
        run = replace(self.run, mode="run")
        changed = replace(run, recipe_document=replace(run.recipe_document, value=recipe))
        self.assertNotEqual(training_binding(run, unit_indices=(0,)), training_binding(changed, unit_indices=(0,)))
        self.assertEqual(recovery_binding(run), recovery_binding(changed))
        recipe = deepcopy(self.run.recipe)
        recipe["training"]["resume_checkpoint_interval_steps"] += 1
        changed = replace(run, recipe_document=replace(run.recipe_document, value=recipe))
        self.assertEqual(training_binding(run, unit_indices=(0,)), training_binding(changed, unit_indices=(0,)))

    def test_new_hardware_profile_accepts_another_name_but_rejects_wrong_capability(self):
        cuda = SimpleNamespace(is_available=lambda: True, device=lambda i: nullcontext(),
                               get_device_name=lambda i: "fixture GPU B",
                               get_device_capability=lambda i: (12, 0), is_bf16_supported=lambda **kwargs: True,
                               get_device_properties=lambda i: SimpleNamespace(total_memory=32 * 1024**3))
        fake = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="13.2"))
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="x86_64"):
            verify_torch_hardware(self.run.runtime, fake)
            legacy = load_base_model_runtime_config(self.root / "legacy.json", project_root=self.root)
            with self.assertRaisesRegex(ValueError, "Frozen hardware mismatch"):
                verify_torch_hardware(legacy, fake)
            cuda.get_device_capability = lambda i: (8, 9)
            with self.assertRaisesRegex(ValueError, "capability"):
                verify_torch_hardware(self.run.runtime, fake)

    def test_path_and_execution_changes_preserve_resume_but_seed_and_implementation_do_not(self):
        original = training_binding(self.run, unit_indices=(0,))
        config = deepcopy(self.run.config)
        config.update(output_dir="outputs/new-place", report="report/training/new.json",
                      execution_profile="configs/env/another.json", expected_execution_profile_sha256="c" * 64)
        config["backward"] = {
            "output_dir": "outputs/new-backward",
            "report": "report/training/backward.json",
        }
        moved = replace(self.run, run_document=replace(self.run.run_document, value=config, sha256="d" * 64),
                        project_root=self.root / "relocated")
        self.assertEqual(original, training_binding(moved, unit_indices=(0,)))
        config["seed"] += 1
        self.assertNotEqual(original, training_binding(moved, unit_indices=(0,)))
        changed = replace(self.run, implementation_hashes={"implementation.py": "e" * 64})
        self.assertEqual(original, training_binding(changed, unit_indices=(0,)))
        legacy = replace(self.run, run_document=replace(self.run.run_document,
                         value={**self.run.config, "schema_version": "training_run_v1"}))
        self.assertIn("run_config_sha256", training_binding(legacy, unit_indices=(0,)))

    def test_backward_reuse_requires_matching_environment_dependencies_and_success(self):
        report = passed_report(self.run)
        def matches(value, environment=ENVIRONMENT):
            return reusable_backward(value, recovery_binding(self.run), environment, semantic_sha256(self.run.execution_document.value))
        self.assertTrue(matches(report))
        for key, value in (("status", "failed"), ("binding", {}), ("execution_profile_sha256", "changed")):
            changed = {**report, key: value}
            self.assertFalse(matches(changed))
        self.assertFalse(matches(report, {**ENVIRONMENT, "device_name": "another GPU"}))
        self.assertFalse(matches(report, {**ENVIRONMENT, "driver_versions": None}))
        incomplete = deepcopy(report)
        incomplete["backward"]["checks"].pop("longest_unit_selected")
        self.assertFalse(matches(incomplete))
        report["backward"]["checks"]["finite_loss"] = False
        self.assertFalse(matches(report))

    def test_recovery_always_loads_and_generates_and_only_reuses_matching_backward(self):
        previous = self.root / "previous.json"
        write_json(previous, passed_report(self.run))
        # A historical cached backward report must not be consulted by recovery.
        write_json(self.run.report_path, {"status": "backward_passed"})
        for number, environment in enumerate((ENVIRONMENT, {**ENVIRONMENT, "device_name": "another GPU"})):
            with patch(RECOVERY + "check_git", return_value="commit"), \
                 patch(RECOVERY + "observe_environment", return_value=environment), \
                 patch(RECOVERY + "verify_model_weights"), \
                 patch(RECOVERY + "load_initial_model", return_value=(object(), None, ())) as load, \
                 patch(RECOVERY + "generation_probe", return_value={"status": "passed"}) as generate, \
                 patch(RECOVERY + "run_training", return_value={"status": "backward_passed",
                       "checks": {"finite_loss": True}, "backward": {}}) as backward:
                result = recover_training(self.run, self.root / f"result-{number}.json", previous=previous)
                self.assertEqual(result["status"], "ready")
                load.assert_called_once()
                generate.assert_called_once()
                self.assertEqual(backward.call_count, number)
                if number:
                    self.assertFalse(backward.call_args.kwargs["persist_report"])
                self.assertEqual(result["backward"]["execution"], "reused" if number == 0 else "executed")

    def test_no_gpu_does_not_reuse_old_success_or_load_model(self):
        previous = self.root / "previous.json"
        write_json(previous, passed_report(self.run))
        with patch(RECOVERY + "check_git", return_value="commit"), \
             patch(RECOVERY + "observe_environment", side_effect=ValueError("No CUDA")), \
             patch(RECOVERY + "load_initial_model") as load:
            report = recover_training(self.run, self.root / "failure.json", previous=previous)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["backward"]["status"], "not_executed")
        load.assert_not_called()

    def test_generation_executes_frozen_settings_on_training_prompt_only(self):
        prepared = replace(self.run.prepared, units=({"record_id": "long", "sequence_tokens_max": 5,
                            "tokenized": {"input_ids": [1, 2, 3, 4, 5], "prompt_token_count": 3}},))
        run = replace(self.run, runtime=replace(self.run.runtime, device="cpu"), prepared=prepared,
                      tokenizer=SimpleNamespace(decode=lambda tokens, **kw: "ACTION_0"))
        model = SimpleNamespace(eval=lambda: None)
        def generate(**kwargs):
            self.assertEqual(kwargs["input_ids"].tolist(), [[1, 2, 3]])
            self.assertFalse(kwargs["do_sample"])
            self.assertEqual(kwargs["max_new_tokens"], 8)
            return torch.tensor([[1, 2, 3, 9, 10]])
        model.generate = generate
        self.assertEqual(generation_probe(run, model)["new_tokens"], 2)
        model.generate = lambda **kw: torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "bounded response"):
            generation_probe(run, model)

    def test_generation_failure_is_saved_as_failure_without_backward(self):
        with patch(RECOVERY + "check_git", return_value="commit"), \
             patch(RECOVERY + "observe_environment", return_value=ENVIRONMENT), \
             patch(RECOVERY + "verify_model_weights"), \
             patch(RECOVERY + "load_initial_model", return_value=(object(), None, ())), \
             patch(RECOVERY + "generation_probe", side_effect=ValueError("generation failed")), \
             patch(RECOVERY + "run_training") as backward:
            result = recover_training(self.run, self.root / "failed.json")
        self.assertEqual(result["status"], "failed")
        self.assertIn("generation failed", read_json_object(self.root / "failed.json")["error"])
        backward.assert_not_called()

    def test_training_requires_current_recovery_evidence(self):
        training = replace(self.run, mode="run")
        with self.assertRaisesRegex(ValueError, "requires --recovery-report"):
            run_training(training)
        with self.assertRaisesRegex(ValueError, "restore_training.py"):
            run_training(self.run)
        report = self.root / "recovery.json"
        write_json(report, passed_report(self.run))
        with patch("sts1_llm_policy.train.readiness.observe_environment", return_value=ENVIRONMENT):
            self.assertEqual(require_recovery(training, report)["recovery_report_sha256"], sha256_file(report))
        with patch("sts1_llm_policy.train.readiness.observe_environment", return_value={**ENVIRONMENT, "driver_versions": ["changed"]}):
            with self.assertRaisesRegex(ValueError, "does not match"):
                require_recovery(training, report)

    def test_git_dirty_and_wrong_commit_are_rejected(self):
        with patch("sts1_llm_policy.train.recovery.subprocess.run",
                   side_effect=[SimpleNamespace(stdout="abc"), SimpleNamespace(stdout=" M source.py")]):
            with self.assertRaisesRegex(ValueError, "Tracked worktree changes"):
                check_git(self.root, "abc")
        with patch("sts1_llm_policy.train.recovery.subprocess.run", return_value=SimpleNamespace(stdout="abc")):
            with self.assertRaisesRegex(ValueError, "expected commit"):
                check_git(self.root, "def")

    def test_checkpoint_hash_damage_and_path_escape_are_rejected(self):
        from sts1_llm_policy.train.runtime import capture_training_rng
        self.run = replace(self.run, runtime=replace(self.run.runtime, device="cpu"))
        root = self.root / self.run.config["output_dir"]
        checkpoint = root / "resume/step-000050"
        checkpoint.mkdir(parents=True)
        binding = training_binding(replace(self.run, mode="run"), unit_indices=(0,))
        weights = checkpoint / "adapter/adapter_model.safetensors"
        weights.parent.mkdir()
        weights.write_bytes(b"adapter")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        torch.save(capture_training_rng(torch.device("cpu")), checkpoint / "rng.pt")
        metadata = {"base_model_id": self.run.runtime.model_id, "base_revision": self.run.runtime.revision,
                    "weights_file": weights.name, "weights_sha256": sha256_file(weights)}
        write_json(checkpoint / "adapter/adapter_config.json", metadata)
        state = {"binding": binding, "adapter_metadata": metadata,
                 "optimizer_sha256": sha256_file(checkpoint / "optimizer.pt"),
                 "rng_sha256": sha256_file(checkpoint / "rng.pt"),
                 "optimizer_step_count": 50, "next_order_offset": 1}
        write_json(checkpoint / "state.json", state)
        latest = {"binding": binding, "checkpoint": "resume/step-000050",
                  "optimizer_step_count": 50, "next_order_offset": 1}
        write_json(root / "latest.json", latest)
        self.assertEqual(inspect_checkpoint(self.run)["status"], "verified")
        weights.write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            inspect_checkpoint(self.run)
        latest["checkpoint"] = "../../outside"
        write_json(root / "latest.json", latest)
        with self.assertRaisesRegex(ValueError, "escapes project root"):
            inspect_checkpoint(self.run)

    def test_backward_reuses_loaded_model_and_never_creates_optimizer(self):
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))
        model = Probe()
        run = replace(self.run, runtime=replace(self.run.runtime, device="cpu"))
        with patch.object(SftAdapter, "unit_loss", side_effect=lambda model, *a, **k: model.weight.square()), \
             patch(RUNNER + "load_initial_model", side_effect=AssertionError("duplicate load")), \
             patch(RUNNER + "_enable_training"), patch(RUNNER + "adapter_state_sha256", return_value="same"), \
             patch(RUNNER + "trainable_parameter_summary", return_value={}), \
             patch(RUNNER + "torch.autocast", return_value=nullcontext()), \
             patch(RUNNER + "torch.cuda.reset_peak_memory_stats"), \
             patch(RUNNER + "torch.cuda.max_memory_allocated", return_value=0), \
             patch(RUNNER + "torch.cuda.max_memory_reserved", return_value=0), \
             patch(RUNNER + "torch.optim.AdamW", side_effect=AssertionError("optimizer created")), \
             patch(RUNNER + "subprocess.run", return_value=SimpleNamespace(stdout="commit")):
            result = run_training(run, backward_model=(model, None, ()), persist_report=False)
        self.assertEqual(result["status"], "backward_passed")
        self.assertEqual(model.weight.item(), 2.0)
        self.assertFalse(run.report_path.exists())

    def test_cpu_checkpoint_resumes_after_relocation_and_matches_uninterrupted_step(self):
        def base_model(*args, **kwargs):
            model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)))
            with torch.no_grad():
                model[0][0].weight.fill_(0.5)
            return model

        model = base_model()
        spec = LoraSpec(("0",), rank=1, alpha=2.0, dropout=0.0)
        replaced = inject_lora(model, spec)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        def step(model, optimizer):
            optimizer.zero_grad()
            model(torch.tensor([[1.0, 2.0]])).square().sum().backward()
            optimizer.step()
        step(model, optimizer)
        run = replace(self.run, mode="run", runtime=replace(self.run.runtime, device="cpu"))
        binding = training_binding(run, unit_indices=(0,))
        _save_checkpoint(run, binding=binding, model=model, optimizer=optimizer, spec=spec,
                         replaced=replaced, state={"optimizer_step_count": 1, "next_order_offset": 1})
        import shutil
        destination = self.root / "relocated-output"
        shutil.copytree(run.output_dir, destination)
        moved = replace(run, output_dir=destination)
        with patch(RUNNER + "load_base_model", side_effect=base_model):
            loaded, resumed_optimizer, _, _, state, resumed = _load_or_initialize(moved, binding=binding)
        self.assertTrue(resumed)
        self.assertEqual(state["next_order_offset"], 1)
        self.assertIn("configuration", state)
        self.assertEqual(adapter_state_sha256(model), adapter_state_sha256(loaded))
        step(model, optimizer)
        step(loaded, resumed_optimizer)
        self.assertEqual(adapter_state_sha256(model), adapter_state_sha256(loaded))
        changed = replace(moved, run_document=replace(moved.run_document, value={**moved.config, "seed": 999}))
        with self.assertRaisesRegex(ValueError, "resume binding changed"):
            _load_or_initialize(changed, binding=training_binding(changed, unit_indices=(0,)))


if __name__ == "__main__":
    unittest.main()
