"""Whole-run replay receipts must identify the report consumed by either exporter."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import collect_gold as cli
from sts1_llm_policy.data.gold.dpo_export import export_gold_dpo
from sts1_llm_policy.data.gold.sft_export import export_gold_sft
from sts1_llm_policy.data.gold import verification as probe

from .gold_fixture import PARAMS, sources


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class GoldVerificationBindingTests(unittest.TestCase):
    def replay_fixture(self, root):
        report, _ = sources()
        report["configuration"].update(output="run", outer_trials=64)
        output = root / "run/formal"
        write_json(output / "report.json", report)
        write_json(output / "identity.json", {
            "protocol": probe.PROTOCOL, "native": {"bridge_sha256": "test-native"},
            "configuration": {k: v for k, v in report["configuration"].items() if k != "output"},
        })
        return report, output

    def replay(self, root, report, side_effect=None):
        def state_receipt(_root, _config, _output, state, _env):
            if side_effect:
                side_effect()
            return {"verified_executions": state["executions"],
                    "verified_decisions": state["executed_decisions"],
                    "full_search_trace_executions": 0}, None, None

        with patch.object(probe, "resolve_simulator") as resolve, \
                patch.object(probe, "StsLightspeedEnv"), \
                patch.object(probe, "verify_state", side_effect=state_receipt) as verify:
            resolve.return_value.describe.return_value = {"bridge_sha256": "test-native"}
            result = probe.verify_run(root, report["configuration"])
            self.assertEqual(verify.call_count, len(report["states"]))
            return result

    def export_config(self, kind):
        common = dict(source_report="source.json", verification="receipt.json", output="data")
        if kind == "sft":
            return dict(common, schema_version="gold_sft_export_v1", dataset_id="test-sft",
                        hp_tolerance=1., nonbasic_weight=1.5)
        return dict(common, schema_version="gold_dpo_export_v1", report="export.json", **PARAMS)

    def test_replay_receipt_is_accepted_by_both_exporters_after_reformat_and_move(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, _ = self.replay_fixture(root)
            receipt = self.replay(root, report)
            canonical = json.dumps(report, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.assertEqual(receipt["report_canonical_sha256"], sha256(canonical).hexdigest())
            for kind, export in (("sft", export_gold_sft), ("dpo", export_gold_dpo)):
                with self.subTest(kind=kind):
                    moved = root / kind
                    write_json(moved / "receipt.json", receipt)
                    # JSON key order, indentation and CRLF do not change report identity.
                    text = json.dumps(dict(reversed(list(report.items()))), indent=2, ensure_ascii=False)
                    (moved / "source.json").write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
                    write_json(moved / "config.json", self.export_config(kind))
                    export(moved, "config.json")
                    self.assertTrue((moved / "data").is_dir())

    def test_missing_stale_or_foreign_receipts_fail_before_any_export_write(self):
        for kind, export in (("sft", export_gold_sft), ("dpo", export_gold_dpo)):
            for case in ("missing", "malformed", "foreign", "changed_observation", "changed_score"):
                with self.subTest(kind=kind, case=case), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    report, receipt = sources()
                    if case == "missing":
                        receipt.pop("report_canonical_sha256")
                    if case == "malformed":
                        receipt["report_canonical_sha256"] = {"sha256": "invalid"}
                    if case == "foreign":
                        other = deepcopy(report)
                        other["configuration"]["source_report"] = "another-pool/report.json"
                        payload = json.dumps(other, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, allow_nan=False).encode("utf-8")
                        receipt["report_canonical_sha256"] = sha256(payload).hexdigest()
                    if case == "changed_observation":
                        report["states"][0]["observation"] = report["states"][0]["observation"].replace("PUBLIC", "CHANGED")
                    if case == "changed_score":
                        report["states"][0]["actions"][0]["expected_carried_hp"] += .1
                    self.assertEqual(receipt["verified_executions"], sum(s["executions"] for s in report["states"]))
                    self.assertEqual(receipt["verified_decisions"], sum(s["executed_decisions"] for s in report["states"]))
                    for name, value in (("source.json", report), ("receipt.json", receipt),
                                        ("config.json", self.export_config(kind))):
                        write_json(root / name, value)
                    before = {p.name: p.read_bytes() for p in root.iterdir()}
                    with self.assertRaisesRegex(ValueError, "report_canonical_sha256"):
                        export(root, "config.json")
                    self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)

    def test_report_changed_during_replay_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, output = self.replay_fixture(root)
            changed = deepcopy(report)
            changed["states"][0]["observation"] += " changed"
            with self.assertRaisesRegex(ValueError, "Report changed during verification"):
                self.replay(root, report, lambda: write_json(output / "report.json", changed))

    def test_cli_preserves_old_receipt_and_writes_new_receipt_only_after_replay(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, output = self.replay_fixture(root)
            legacy = b'{"status":"completed","verified_executions":704,"verified_decisions":400}\n'
            (output / "verification.json").write_bytes(legacy)
            args = ["collect_gold.py", "--config", "run.json", "--verify"]
            with patch.object(cli, "ROOT", root), patch.object(cli, "load_config", return_value=report["configuration"]), \
                    patch.object(cli, "verify_run") as verify, patch("sys.argv", args):
                with self.assertRaisesRegex(ValueError, "use --verification-output"):
                    cli.main()
                verify.assert_not_called()
            destination = "run/formal/verification-bound.json"
            with patch.object(cli, "ROOT", root), patch.object(cli, "load_config", return_value=report["configuration"]), \
                    patch.object(cli, "verify_run", side_effect=lambda *a, **kw: self.replay(root, report)), \
                    patch("sys.argv", args + ["--verification-output", destination]), patch("builtins.print"):
                self.assertEqual(cli.main(), 0)
            receipt = json.loads((root / destination).read_text(encoding="utf-8"))
            self.assertIn("report_canonical_sha256", receipt)
            self.assertEqual((output / "verification.json").read_bytes(), legacy)
            self.assertEqual(json.loads((output / "report.json").read_text(encoding="utf-8")), report)

    def test_cli_failed_replay_writes_no_receipt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, output = self.replay_fixture(root)
            with patch.object(cli, "ROOT", root), patch.object(cli, "load_config", return_value=report["configuration"]), \
                    patch.object(cli, "verify_run", side_effect=ValueError("Replay mismatch")), \
                    patch("sys.argv", ["collect_gold.py", "--config", "run.json", "--verify"]):
                with self.assertRaisesRegex(ValueError, "Replay mismatch"):
                    cli.main()
            self.assertFalse((output / "verification.json").exists())


if __name__ == "__main__":
    unittest.main()
