"""A rejected export must not leave other members of that export on disk."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sts1_llm_policy.data.gold.sft_export import export_gold_sft
from sts1_llm_policy.data.gold.dpo_export import export_gold_dpo, preference_records

from tests.support import write_json
from .gold_fixture import PARAMS, sources


class ExportArtifactTests(unittest.TestCase):
    def fixture(self, root, kind):
        report, verification = sources()
        config = dict(source_report="source.json", verification="receipt.json", output="export")
        if kind == "sft":
            config.update(schema_version="gold_sft_export_v1", dataset_id="test",
                          hp_tolerance=1., nonbasic_weight=1.5)
            export = export_gold_sft
        else:
            config.update(schema_version="gold_dpo_export_v1", report="export/result.json", **deepcopy(PARAMS))
            export = export_gold_dpo
        for name, value in (("source.json", report), ("receipt.json", verification), ("config.json", config)):
            write_json(root / name, value)
        return config, export

    def files(self, root):
        return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def test_late_manifest_arm_and_report_conflicts_leave_no_partial_export(self):
        for kind, destination in (("sft", "manifest.json"), ("sft", "manifest.json.tmp"),
                                  ("dpo", "c/train.jsonl.gz"), ("dpo", "c/manifest.json"),
                                  ("dpo", "result.json"), ("dpo", "result.json.tmp")):
            with self.subTest(kind=kind, destination=destination), TemporaryDirectory() as temp:
                root = Path(temp)
                _, export = self.fixture(root, kind)
                target = root / "export" / destination
                target.parent.mkdir(parents=True)
                target.write_bytes(b"retained content")
                before = self.files(root)
                with self.assertRaisesRegex(ValueError, "[Cc]onflict"):
                    export(root, "config.json")
                self.assertEqual(self.files(root), before)

    def test_invalid_manifest_and_output_path_aliases_fail_before_writing(self):
        cases = (("sft", "dataset_id", ""),
                 ("dpo", "report", "export/a/train.jsonl.gz"),
                 ("dpo", "report", "export/a/train.jsonl.gz.tmp"),
                 ("dpo", "report", "export/a"))
        for kind, field, value in cases:
            with self.subTest(field=field, value=value), TemporaryDirectory() as temp:
                root = Path(temp)
                config, export = self.fixture(root, kind)
                config[field] = value
                write_json(root / "config.json", config)
                before = self.files(root)
                with self.assertRaises(ValueError):
                    export(root, "config.json")
                self.assertEqual(self.files(root), before)
                self.assertFalse((root / "export").exists())

    def test_cross_arm_failure_is_checked_before_any_arm_is_published(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, "dpo")
            groups, screening, normalizers = preference_records(*sources(), **PARAMS)
            groups["b"].pop()
            with patch("sts1_llm_policy.data.gold.dpo_export.preference_records",
                       return_value=(groups, screening, normalizers)):
                with self.assertRaisesRegex(ValueError, "Cross-arm"):
                    export_gold_dpo(root, "config.json")
            self.assertFalse((root / "export").exists())
