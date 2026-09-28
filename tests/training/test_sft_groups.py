"""Weighted candidates change actual gradients without changing state mass."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sts1_llm_policy.data.gold.sft_export import export_records, export_gold_sft
from sts1_llm_policy.train.configured_adapters import SftAdapter
from sts1_llm_policy.train.configured_runner import _backward_unit, run_training
from sts1_llm_policy.train.identity import training_binding
from sts1_llm_policy.train.readiness import recovery_binding
from sts1_llm_policy.artifacts import iter_jsonl
from sts1_llm_policy.data.manifest import validate_dataset_manifest
from sts1_llm_policy.artifacts import canonical_json_bytes

from .model_fixture import _StubTokenizer
from .runtime_fixture import fixture, RUNNER


def source_fixture():
    roots = [
        {"id": "ACTION_0", "card": "Strike_R", "target": 0, "action_type": "play_card",
         "card_semantics": {"card_id": "Strike_R", "card_type": "ATTACK", "upgrades": 1}},
        {"id": "ACTION_1", "card": "Shrug It Off", "target": None, "action_type": "play_card",
         "card_semantics": {"card_id": "Shrug It Off", "card_type": "SKILL", "upgrades": 0}},
        {"id": "ACTION_2", "card": "end_turn", "target": None, "action_type": "end_turn", "card_semantics": None},
    ]
    state = {"sample": {"id": "state", "route": 4, "group": 2, "combat": 1, "step": 3},
             "encounter_family": "normal_weak", "outer_trials_completed": 32,
             "minimum_win_rate": .99, "win_floor_rounding": "floor", "minimum_wins": 31,
             "root_actions": roots, "status": "completed", "executions": 96, "executed_decisions": 100,
             "actions": [{"action": a["id"], "trials": 32, "wins": 32, "expected_carried_hp": hp}
                         for a, hp in zip(roots, (50, 50.5, 40))],
             "estimated_acceptable_actions": ["ACTION_0", "ACTION_1"],
             "observation": "PUBLIC\nLEGAL_ACTIONS:\nACTION_0: PLAY Strike+ -> TARGET_0\nACTION_1: PLAY Shrug It Off\nACTION_2: END_TURN"}
    cfg = {"observation_version": "observation_v7", "expected_hp_tolerance": 1.,
           "minimum_win_rate": {"normal_weak": .99}, "samples": [state["sample"]],
           "source_report": "pool/report.json", "source_partition": {"role": "train_candidate",
                    "excluded_routes": [1], "reserved_routes": [5]}}
    report = {"status": "completed", "configuration": cfg, "states": [state]}
    return report, {"status": "completed", "verified_executions": 96, "verified_decisions": 100,
                    "report_canonical_sha256": sha256(canonical_json_bytes(report)).hexdigest()}


def records_from_fixture():
    r, v = source_fixture()
    return export_records(r, v, dataset_id="test", hp_tolerance=1, nonbasic_weight=1.5)[0]


class TinySftLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.embedding = torch.nn.Embedding(256, 4)
        self.q_proj = torch.nn.Linear(4, 256, bias=False)
        self.v_proj = torch.nn.Linear(4, 256, bias=False)

    def forward(self, input_ids, labels, **kwargs):
        x = self.embedding(input_ids)
        logits = self.q_proj(x) + self.v_proj(x)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 256), labels[:, 1:].reshape(-1))
        return SimpleNamespace(loss=loss)


class SftGroupTests(unittest.TestCase):
    def prepare(self, records=None):
        return SftAdapter().prepare(_StubTokenizer(), records or records_from_fixture(),
                                    dataset_id="test", max_sequence_tokens=4096)

    def test_export_uses_hp_and_nonbasic_parameters_and_rejects_incomplete_evidence(self):
        r, v = source_fixture()
        rows, counts = export_records(r, v, dataset_id="test", hp_tolerance=1, nonbasic_weight=1.5)
        self.assertEqual([c["weight"] for c in rows[0]["candidates"]], [.4, .6])
        other, _ = export_records(r, v, dataset_id="test", hp_tolerance=0, nonbasic_weight=3)
        self.assertEqual(other[0]["candidates"], [{"action_id": "ACTION_1", "weight": 1.}])
        with self.assertRaisesRegex(ValueError, "No proper"):
            export_records(r, v, dataset_id="test", hp_tolerance=20, nonbasic_weight=1.5)
        for change in ("counts", "partial", "partition", "labels"):
            rr, vv = deepcopy(r), deepcopy(v)
            if change == "counts": vv["verified_executions"] -= 1
            if change == "partial": rr["states"][0]["status"] = "partial"
            if change == "partition": rr["configuration"]["source_partition"]["reserved_routes"].append(4)
            if change == "labels": rr["states"][0]["estimated_acceptable_actions"] = ["ACTION_2"]
            # Synthetic receipt: keep exercising the existing semantic checks after binding.
            vv["report_canonical_sha256"] = sha256(canonical_json_bytes(rr)).hexdigest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                export_records(rr, vv, dataset_id="test", hp_tolerance=1, nonbasic_weight=1.5)

    def test_export_is_portable_and_refuses_conflicting_existing_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            r, v = source_fixture()
            cfg = {"schema_version": "gold_sft_export_v1", "dataset_id": "test",
                   "source_report": "report.json", "verification": "verified.json",
                   "output": "outputs/export", "hp_tolerance": 1, "nonbasic_weight": 1.5}
            for name, data in (("report.json", r), ("verified.json", v), ("config.json", cfg)):
                (root / name).write_text(json.dumps(data), encoding="utf-8")
            manifest = export_gold_sft(root, "config.json")
            self.assertEqual(manifest, export_gold_sft(root, "config.json"))
            validate_dataset_manifest(manifest, project_root=root, verify_artifacts=True)
            self.assertEqual(list(iter_jsonl(root / manifest["splits"]["train"]["path"])), records_from_fixture())
            cfg["nonbasic_weight"] = 2
            (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                export_gold_sft(root, "config.json")

    def test_secondary_selection_and_prompt_boundaries(self):
        records = records_from_fixture()
        for action in records[0]["model_actions"]:
            action["action_type"] = "select_card"
        unit = self.prepare(records).units[0]
        self.assertEqual(unit["candidates"][0]["tokenized"]["selected_semantic"], "SELECT_CARD")
        prompt = SftAdapter().generation_prompt(unit)
        for c in unit["candidates"]:
            tokens = c["tokenized"]
            self.assertEqual(tokens["input_ids"][:len(prompt)], prompt)
            self.assertEqual(tokens["labels"][:len(prompt)], [-100] * len(prompt))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            SftAdapter().prepare(_StubTokenizer(), records, dataset_id="test", max_sequence_tokens=4)

    def test_invalid_candidate_groups_fail(self):
        for case in ("illegal", "duplicate", "weights", "nan", "prompt", "hash", "ids"):
            rows = records_from_fixture()
            if case == "illegal": rows[0]["candidates"][0]["action_id"] = "ACTION_8"
            if case == "duplicate": rows[0]["candidates"][1]["action_id"] = "ACTION_0"
            if case == "weights": rows[0]["candidates"][0]["weight"] = .5
            if case == "nan": rows[0]["candidates"][0]["weight"] = float("nan")
            if case == "prompt": rows[0]["model_actions"].pop()
            if case == "hash": rows[0]["observation"] += "changed"
            if case == "ids": rows *= 2
            with self.subTest(case=case), self.assertRaises(ValueError): self.prepare(rows)

    def test_streamed_weighted_gradient_equals_full_loss_and_weights_change_behavior(self):
        prepared = self.prepare()
        run = SimpleNamespace(adapter=SftAdapter(), runtime=SimpleNamespace(device="cpu", generation_pad_token_id=0), recipe={})
        a, b = TinySftLM(), TinySftLM()
        unit = prepared.units[0]
        expected = run.adapter.unit_loss(a, unit, runtime=run.runtime, reference=None, recipe={})
        expected.backward()
        with patch(RUNNER + "torch.autocast", return_value=nullcontext()):
            actual = _backward_unit(run, b, unit, None)
        self.assertAlmostEqual(actual, expected.item(), places=5)
        for p, q in zip(a.parameters(), b.parameters()): torch.testing.assert_close(p.grad, q.grad)
        changed = deepcopy(unit)
        changed["candidates"][0]["weight"], changed["candidates"][1]["weight"] = .8, .2
        c = TinySftLM()
        run.adapter.unit_loss(c, changed, runtime=run.runtime, reference=None, recipe={}).backward()
        self.assertFalse(torch.allclose(c.q_proj.weight.grad, a.q_proj.weight.grad))

    def test_group_weights_and_every_branch_bind_recovery(self):
        with TemporaryDirectory() as tmp:
            run = replace(fixture(Path(tmp)), prepared=self.prepare())
            for field in ("weight", "tokens", "loss_weight"):
                units = deepcopy(run.prepared.units)
                if field == "weight": units[0]["candidates"][1]["weight"] = .3
                elif field == "loss_weight": units[0]["loss_weight"] = 2.
                else: units[0]["candidates"][1]["tokenized"]["labels"][-1] += 1
                changed = replace(run, prepared=replace(run.prepared, units=units))
                self.assertNotEqual(training_binding(run, unit_indices=(0,)), training_binding(changed, unit_indices=(0,)))
                self.assertNotEqual(recovery_binding(run), recovery_binding(changed))

    def test_state_weight_changes_streamed_loss_and_gradients(self):
        unit = self.prepare().units[0]
        weighted = {**unit, "loss_weight": 2.5}
        run = SimpleNamespace(adapter=SftAdapter(), runtime=SimpleNamespace(device="cpu", generation_pad_token_id=0), recipe={})
        a, b = TinySftLM(), TinySftLM()
        with patch(RUNNER + "torch.autocast", return_value=nullcontext()):
            first = _backward_unit(run, a, unit, None)
            second = _backward_unit(run, b, weighted, None)
        self.assertAlmostEqual(second, first * 2.5, places=5)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(q.grad, p.grad * 2.5)

    def test_explicit_state_weights_require_finite_positive_mean_one(self):
        for value in (0, -1, float("nan"), float("inf"), True, 2):
            rows = records_from_fixture()
            rows[0]["loss_weight"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.prepare(rows)
        rows = records_from_fixture()
        rows[0]["loss_weight"] = .5
        rows.append(deepcopy(rows[0]))
        rows[1].update(record_id="other", loss_weight=1.5)
        self.assertEqual([u["loss_weight"] for u in self.prepare(rows).units], [.5, 1.5])
        del rows[1]["loss_weight"]
        with self.assertRaises(ValueError): self.prepare(rows)

    def test_group_training_resume_and_fresh_base_reload(self):
        with TemporaryDirectory() as tmp, ExitStack() as stack:
            run = fixture(Path(tmp))
            records = records_from_fixture()
            records[0]["loss_weight"] = .5
            records.append(deepcopy(records[0]))
            records[1].update(record_id="other", loss_weight=1.5)
            run = replace(run, prepared=self.prepare(records), mode="run", runtime=replace(run.runtime, device="cpu"),
                          output_dir=Path(tmp) / "formal", report_path=Path(tmp) / "formal/report.json")
            manifest = deepcopy(run.dataset_document.value)
            manifest["splits"]["train"]["records"] = 2
            run = replace(run, dataset_document=replace(run.dataset_document, value=manifest))
            loads = 0
            def load(*args, **kwargs):
                nonlocal loads
                loads += 1
                if loads == 2: raise RuntimeError("interrupt reload")
                return TinySftLM()
            stack.enter_context(patch(RUNNER + "load_base_model", side_effect=load))
            stack.enter_context(patch("sts1_llm_policy.train.runtime.load_base_model", side_effect=load))
            stack.enter_context(patch(RUNNER + "require_recovery", return_value={}))
            stack.enter_context(patch(RUNNER + "verify_model_weights"))
            stack.enter_context(patch(RUNNER + "_enable_training", new=lambda m: m.train()))
            stack.enter_context(patch(RUNNER + "torch.cuda.reset_peak_memory_stats"))
            for name in ("torch.cuda.max_memory_allocated", "torch.cuda.max_memory_reserved"):
                stack.enter_context(patch(RUNNER + name, return_value=0))
            stack.enter_context(patch(RUNNER + "torch.autocast", return_value=nullcontext()))
            stack.enter_context(patch(RUNNER + "subprocess.run", return_value=SimpleNamespace(stdout="commit")))
            with self.assertRaisesRegex(RuntimeError, "interrupt reload"): run_training(run)
            result = run_training(run)
            self.assertTrue(result["training"]["resumed"])
            self.assertEqual(result["training"]["optimizer_step_count"], 1)
            self.assertTrue(all(result["checks"].values()))
