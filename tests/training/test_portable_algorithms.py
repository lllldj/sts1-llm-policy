"""Algorithm substitution uses small inputs and real CPU adapter tensors."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import torch

from sts1_llm_policy.train.configured_adapters import DpoAdapter, PreparedTrainingData
from sts1_llm_policy.train.identity import training_binding
from sts1_llm_policy.train.runtime import load_initial_model
from sts1_llm_policy.train.configured_runner import _reference_values, run_training
from sts1_llm_policy.train.configured_training import (
    _resolve_initial_checkpoint,
    load_configured_training_run,
)
from sts1_llm_policy.train.lora import (
    LoraSpec,
    inject_lora,
    save_lora_checkpoint,
    adapter_state_sha256,
)
from sts1_llm_policy.train.recovery import generation_probe
from sts1_llm_policy.train.readiness import recovery_binding
from sts1_llm_policy.artifacts import sha256_file, read_json_object

from tests.training.model_fixture import TinyLM
from .runtime_fixture import fixture, RUNNER
from tests.support import write_json


class PortableAlgorithmTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        run = fixture(self.root)
        recipe = deepcopy(run.recipe)
        recipe.update(algorithm="dpo", dpo={"beta": 0.1, "label_smoothing": 0.0,
                                           "reference": "frozen_initial_checkpoint"})
        chosen = deepcopy(run.prepared.units[0]["tokenized"])
        chosen["record_id"] = "chosen"
        rejected = deepcopy(chosen)
        rejected["record_id"] = "rejected"
        rejected["input_ids"][-1] = rejected["labels"][-1] = 6
        unit = {"record_id": "preference", "sequence_tokens_max": 5, "edges": [
            {"edge_id": "edge", "weight": 1.0, "chosen": chosen, "rejected": rejected}]}
        self.run = replace(run, adapter=DpoAdapter(), runtime=replace(run.runtime, device="cpu"),
                           recipe_document=replace(run.recipe_document, value=recipe),
                           prepared=PreparedTrainingData("dpo", (unit,), run.prepared.tokenization))
        self.model = TinyLM()
        self.spec = LoraSpec(("q_proj", "v_proj"), rank=8, alpha=16.0, dropout=0.0)
        self.replaced = inject_lora(self.model, self.spec)
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.fill_(0.07 if "lora_a" in name else 0.03)
        save_lora_checkpoint(self.model, self.root / "initial", spec=self.spec,
                             replaced_modules=self.replaced, base_model_id=run.runtime.model_id,
                             base_revision=run.runtime.revision)
        self.run = replace(self.run, initial_checkpoint=self.resolve("initial"))

    def resolve(self, directory):
        return _resolve_initial_checkpoint(self.root, directory, required=True,
                                           runtime=self.run.runtime, portable=True)

    def test_portable_dpo_loader_uses_declared_dataset_model_and_checkpoint(self):
        raw = read_json_object(self.root / "run.json")
        raw["initial_checkpoint"] = "initial"
        write_json(self.root / "run.json", raw)
        write_json(self.root / "recipe.json", self.run.recipe)
        artifact = self.root / "preferences.jsonl"
        artifact.write_text('{"record_id":"preference"}\n', encoding="utf-8")
        manifest = {"schema_version": "dataset_manifest_v1", "dataset_id": "other-input",
                    "task_type": "preference", "observation_version": "observation_v5",
                    "identity_fields": ["record_id"], "lineage": {"source": "fixture"},
                    "semantics": {"edges": "weighted"}, "splits": {"train": {
                        "path": artifact.name, "sha256": sha256_file(artifact),
                        "bytes": artifact.stat().st_size, "records": 1, "format": "jsonl", "compression": None}}}
        write_json(self.root / "dataset.json", manifest)
        with patch("sts1_llm_policy.train.configured_training.load_verified_local_tokenizer", return_value=object()), \
             patch.object(DpoAdapter, "prepare", return_value=self.run.prepared) as prepare:
            loaded = load_configured_training_run(self.root, "run.json", mode="preflight")
            self.assertIsInstance(loaded.adapter, DpoAdapter)
            self.assertEqual(loaded.initial_checkpoint.directory, self.root / "initial")
            self.assertEqual(loaded.runtime.model_id, "fixture/model")
            self.assertEqual(prepare.call_args.kwargs["dataset_id"], "other-input")
            raw.pop("initial_checkpoint")
            write_json(self.root / "run.json", raw)
            with self.assertRaisesRegex(ValueError, "requires an initial"):
                load_configured_training_run(self.root, "run.json", mode="preflight")

    def test_initial_checkpoint_controls_model_and_binds_content_not_location(self):
        shutil.copytree(self.root / "initial", self.root / "moved")
        metadata = read_json_object(self.root / "moved/adapter_config.json")
        write_json(self.root / "moved/adapter_config.json", dict(reversed(list(metadata.items()))))
        relocated = replace(self.run, initial_checkpoint=self.resolve("moved"))
        self.assertEqual(recovery_binding(self.run), recovery_binding(relocated))
        with patch("sts1_llm_policy.train.runtime.load_base_model", side_effect=lambda *a, **k: TinyLM()):
            loaded, _, _ = load_initial_model(relocated, verify_assets=False)
        self.assertEqual(adapter_state_sha256(loaded), adapter_state_sha256(self.model))
        runtime = replace(self.run.runtime, model_id="other/model")
        with self.assertRaisesRegex(ValueError, "base identity"):
            _resolve_initial_checkpoint(self.root, "moved", required=True, runtime=runtime, portable=True)
        (self.root / "moved/adapter_model.safetensors").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.resolve("moved")

    def test_dpo_changes_invalidate_training_and_recovery(self):
        variations = []
        for branch in ("chosen", "rejected"):
            for key in ("input_ids", "attention_mask", "labels", "prompt_token_count"):
                units = deepcopy(self.run.prepared.units)
                tokens = units[0]["edges"][0][branch]
                if isinstance(tokens[key], list):
                    tokens[key][-1] += 1
                else:
                    tokens[key] += 1
                variations.append(replace(self.run, prepared=replace(self.run.prepared, units=units)))
        units = deepcopy(self.run.prepared.units)
        units[0]["edges"][0]["weight"] = 0.5
        variations.append(replace(self.run, prepared=replace(self.run.prepared, units=units)))
        units = deepcopy(self.run.prepared.units)
        units[0]["loss_weight"] = 0.5
        variations.append(replace(self.run, prepared=replace(self.run.prepared, units=units)))
        recipe = deepcopy(self.run.recipe)
        recipe["dpo"]["beta"] = 0.2
        variations.append(replace(self.run, recipe_document=replace(self.run.recipe_document, value=recipe)))
        initial = self.run.initial_checkpoint
        variations.append(replace(self.run, initial_checkpoint=replace(initial,
            metadata={**initial.metadata, "weights_sha256": "f" * 64})))
        for changed in variations:
            self.assertNotEqual(training_binding(self.run, unit_indices=(0,)), training_binding(changed, unit_indices=(0,)))
            self.assertNotEqual(recovery_binding(self.run), recovery_binding(changed))

    def test_dpo_backward_reuses_model_preserves_reference_and_has_real_gradients(self):
        before = adapter_state_sha256(self.model)
        with ExitStack() as stack:
            for name in ("load_initial_model", "torch.optim.AdamW"):
                stack.enter_context(patch(RUNNER + name, side_effect=AssertionError("unexpected model/optimizer")))
            stack.enter_context(patch(RUNNER + "_enable_training", new=lambda model: model.train()))
            stack.enter_context(patch(RUNNER + "torch.cuda.reset_peak_memory_stats"))
            for name in ("torch.cuda.max_memory_allocated", "torch.cuda.max_memory_reserved"):
                stack.enter_context(patch(RUNNER + name, return_value=0))
            stack.enter_context(patch(RUNNER + "torch.autocast", return_value=nullcontext()))
            stack.enter_context(patch(RUNNER + "subprocess.run", return_value=SimpleNamespace(stdout="commit")))
            result = run_training(self.run, backward_model=(self.model, self.spec, self.replaced), persist_report=False)
        self.assertEqual(result["status"], "backward_passed")
        self.assertEqual(before, adapter_state_sha256(self.model))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in self.model.parameters()))
        with patch(RUNNER + "load_initial_model", side_effect=AssertionError("reference must be reused")):
            reference, counts = _reference_values(self.run, units=self.run.prepared.units,
                                                  binding=training_binding(self.run, unit_indices=(0,)))
        self.assertEqual(counts["groups_reused"], 1)
        with torch.no_grad():
            self.model.q_proj.lora_b.weight[4].add_(0.2)
        new_reference = self.run.adapter.reference_values(self.model, self.run.prepared.units, runtime=self.run.runtime)
        self.assertNotEqual(reference, new_reference)
        with self.assertRaisesRegex(ValueError, "cache binding failed"):
            _reference_values(self.run, units=self.run.prepared.units, binding={"changed": True})

    def test_dpo_generation_uses_prompt_only(self):
        model = SimpleNamespace(eval=lambda: None)
        def generate(**kwargs):
            self.assertEqual(kwargs["input_ids"].tolist(), [[1, 2, 3]])
            return torch.tensor([[1, 2, 3, 7]])
        model.generate = generate
        run = replace(self.run, tokenizer=SimpleNamespace(decode=lambda *a, **k: "ACTION_0"))
        self.assertEqual(generation_probe(run, model)["new_tokens"], 1)

    def test_dpo_resumes_after_final_save_and_reloads_without_overlapping_models(self):
        run = replace(self.run, mode="run", output_dir=self.root / "formal",
                      report_path=self.root / "formal/report.json")
        alive = []
        loads = 0
        def base(*args, **kwargs):
            nonlocal loads
            self.assertFalse(any(item() is not None for item in alive), "simultaneous Base models")
            loads += 1
            if loads == 3:
                raise RuntimeError("interrupted after final save")
            model = TinyLM()
            alive.append(weakref.ref(model))
            return model
        with ExitStack() as stack:
            stack.enter_context(patch(RUNNER + "load_base_model", side_effect=base))
            stack.enter_context(patch("sts1_llm_policy.train.runtime.load_base_model", side_effect=base))
            stack.enter_context(patch(RUNNER + "require_recovery", return_value={}))
            stack.enter_context(patch(RUNNER + "verify_model_weights"))
            stack.enter_context(patch(RUNNER + "_enable_training", new=lambda model: model.train()))
            stack.enter_context(patch(RUNNER + "torch.cuda.reset_peak_memory_stats"))
            for name in ("torch.cuda.max_memory_allocated", "torch.cuda.max_memory_reserved"):
                stack.enter_context(patch(RUNNER + name, return_value=0))
            stack.enter_context(patch(RUNNER + "torch.autocast", return_value=nullcontext()))
            stack.enter_context(patch(RUNNER + "subprocess.run", return_value=SimpleNamespace(stdout="commit")))
            with self.assertRaisesRegex(RuntimeError, "interrupted after final save"):
                run_training(run)
            self.assertFalse(run.report_path.exists())
            self.assertTrue((run.output_dir / "checkpoint/adapter_model.safetensors").exists())
            resumed = run_training(run)
            self.assertEqual(resumed["status"], "trained_pending_development_evaluation")
            self.assertTrue(resumed["training"]["resumed"])
            self.assertTrue(all(resumed["checks"].values()))
            self.assertEqual(resumed["training"]["optimizer_step_count"], 1)
            self.assertEqual(resumed["reference"]["groups_reused"], 1)
            self.assertEqual(loads, 5)  # Reference, training, interrupted reload, resumed training, fresh reload.
