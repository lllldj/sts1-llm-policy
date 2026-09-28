"""Preference selection, global HP weighting and portable artifact boundaries."""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sts1_llm_policy.data.gold.dpo_export import export_gold_dpo, preference_records
from sts1_llm_policy.artifacts import iter_jsonl
from sts1_llm_policy.data.manifest import validate_dataset_manifest
from sts1_llm_policy.artifacts import canonical_json_bytes

from tests.training.gold_fixture import PARAMS, sources


class GoldDpoExportTests(unittest.TestCase):
    def test_strict_boundary_common_states_and_global_weighting(self):
        groups, _, normalizers = preference_records(*sources(), **PARAMS)
        self.assertEqual([r["record_id"] for r in groups["a"]], ["state-0", "state-1", "state-3"])
        for arm in "abc":
            self.assertEqual([r["record_id"] for r in groups[arm]], [r["record_id"] for r in groups["a"]])
            self.assertAlmostEqual(math.fsum(r["loss_weight"] for r in groups[arm]), 3.)
        self.assertEqual([e["mean_carried_hp_gap"] for e in groups["a"][0]["edges"]], [3, 6])
        self.assertEqual([e["mean_carried_hp_gap"] for e in groups["b"][0]["edges"]], [6])
        self.assertEqual(groups["b"][0]["edges"][0]["weight"], 1.)
        self.assertEqual([r["loss_weight"] for r in groups["a"]], [1., 1., 1.])
        # A one-edge state still changes its total mass; local normalization must not erase it.
        self.assertNotEqual(groups["c"][-1]["loss_weight"], 1.)
        for a, c in zip(groups["a"], groups["c"]):
            for ae, ce in zip(a["edges"], c["edges"]):
                self.assertEqual(ae["chosen_action_id"], ce["chosen_action_id"])
                self.assertEqual(ae["rejected_action_id"], ce["rejected_action_id"])
                self.assertAlmostEqual(c["loss_weight"] * ce["weight"],
                                       ae["weight"] * min(1, ae["mean_carried_hp_gap"]/5) / normalizers["c"])

    def test_parameters_change_actual_selection_and_weights(self):
        baseline = preference_records(*sources(), **PARAMS)[0]
        coarse = preference_records(*sources(), **{**PARAMS, "coarse_hp_gap": 5.})[0]
        self.assertEqual(len(coarse["a"]), 1)
        capped = preference_records(*sources(), **{**PARAMS, "weight_hp_cap": 3.})[0]
        self.assertNotEqual(capped["c"][0]["edges"], baseline["c"][0]["edges"])
        no_trials = {**PARAMS, "minimum_paired_trials": 128}
        with self.assertRaisesRegex(ValueError, "No common"):
            preference_records(*sources(), **no_trials)
        report, verification = sources()
        for state in report["states"]:
            for pair in state["paired_comparisons"]:
                pair["a"], pair["b"] = pair["b"], pair["a"]
                pair["mean_carried_hp_a_minus_b"] *= -1
        verification["report_canonical_sha256"] = sha256(canonical_json_bytes(report)).hexdigest()
        self.assertEqual(preference_records(report, verification, **PARAMS)[0], baseline)

    def test_noise_win_and_resource_filters(self):
        for reason in ("noise", "wins", "resource"):
            report, verification = sources()
            state = report["states"][0]
            if reason == "noise": state["paired_comparisons"][1]["standard_error"] = 2.5  # exactly 1 adjusted HP
            if reason == "wins": state["actions"][0]["wins"] = 63
            if reason == "resource": state["paired_comparisons"][1]["unpriced_resource_conflict_trials"] = 1
            verification["report_canonical_sha256"] = sha256(canonical_json_bytes(report)).hexdigest()
            groups = preference_records(report, verification, **PARAMS)[0]
            with self.subTest(reason=reason):
                self.assertNotIn("state-0", [r["record_id"] for r in groups["a"]])

    def test_rejects_missing_inconsistent_or_leaking_evidence(self):
        for case in ("missing", "duplicate", "nan", "trials", "mean", "reserved", "verification", "prompt"):
            r, v = sources()
            state = r["states"][0]
            pair = state["paired_comparisons"][0]
            if case == "missing": state["paired_comparisons"].pop()
            if case == "duplicate": state["paired_comparisons"].append(deepcopy(pair))
            if case == "nan": pair["standard_error"] = float("nan")
            if case == "trials": pair["paired_trials"] = 63
            if case == "mean": pair["mean_carried_hp_a_minus_b"] += 1
            if case == "reserved": r["configuration"]["source_partition"]["reserved_routes"].append(4)
            if case == "verification": v["verified_executions"] -= 1
            if case == "prompt": state["observation"] += "\nACTION_8: END_TURN"
            # Synthetic receipts keep the checks below reachable; NaN fails canonical encoding.
            if case != "nan":
                v["report_canonical_sha256"] = sha256(canonical_json_bytes(r)).hexdigest()
            with self.subTest(case=case), self.assertRaises(ValueError):
                preference_records(r, v, **PARAMS)

    def test_export_manifest_roundtrip_and_conflict_protection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            r, v = sources()
            cfg = dict(PARAMS, schema_version="gold_dpo_export_v1", source_report="source.json",
                       verification="verify.json", output="data", report="result.json")
            for name, value in (("source.json", r), ("verify.json", v), ("config.json", cfg)):
                (root/name).write_text(json.dumps(value), encoding="utf-8")
            result = export_gold_dpo(root, "config.json")
            self.assertEqual(result, export_gold_dpo(root, "config.json"))
            for arm in "abc":
                manifest = json.loads((root / result["datasets"][arm]["manifest"]).read_text(encoding="utf-8"))
                validate_dataset_manifest(manifest, project_root=root, verify_artifacts=True)
                rows = list(iter_jsonl(root / manifest["splits"]["train"]["path"]))
                self.assertEqual(len(rows), manifest["splits"]["train"]["records"])
                self.assertEqual(manifest["task_type"], "preference")
            (root/"data/a/train.jsonl.gz").write_bytes(b"other data")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                export_gold_dpo(root, "config.json")


if __name__ == "__main__":
    unittest.main()
