from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sts1_llm_policy.data.gold.mixed_export import (
    apply_mixture_weights,
    export_mixed_sft,
    training_route_groups,
    validate_source_route,
)
from sts1_llm_policy.train.sft_groups import validate_weight_semantics

from tests.support import write_json


def row(name, source, combat=1, step=0):
    return {"record_id": name, "supervision_source": source,
            "source": {"sample": {"route": 3, "group": 1, "combat": combat, "step": step}}}


class MixedExportTests(unittest.TestCase):
    def test_later_route_defeat_is_not_an_early_demonstration_filter(self):
        pool = {"configuration_identity": {"semantics_sha256": "same"}, "run_id": "pool"}
        route = {"status": "defeated", "death_combat_index": 8,
                 "configuration_identity": deepcopy(pool["configuration_identity"]),
                 "route_lineage": {"route_index": 19, "combat_seed_group_index": 3,
                                   "run_id": "pool", "arm": "teacher"}}
        validate_source_route(route, pool, 19, 3)
        route["status"] = "completed"
        validate_source_route(route, pool, 19, 3)
        route["status"] = "truncated"
        with self.assertRaisesRegex(ValueError, "unfinished"):
            validate_source_route(route, pool, 19, 3)
        route["status"] = "defeated"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_source_route(route, pool, 20, 3)
        route["configuration_identity"]["semantics_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_source_route(route, pool, 19, 3)

    def test_source_mass_and_combat_equal_weight_are_independent_of_length(self):
        rows = [row("g", "gold"), row("g2", "gold", 3), row("end", "forced_end"),
                row("short", "teacher_strategy"), row("long1", "teacher_strategy", 2),
                row("long2", "teacher_strategy", 2, 1), row("long3", "teacher_strategy", 2, 2)]
        result = apply_mixture_weights(rows, {"gold": .7, "teacher_strategy": .25, "forced_end": .05})
        for name, expected in (("gold", .7), ("teacher_strategy", .25), ("forced_end", .05)):
            self.assertAlmostEqual(result[name]["loss_mass"], expected)
        self.assertAlmostEqual(rows[3]["loss_weight"], sum(r["loss_weight"] for r in rows[4:]))
        changed = deepcopy(rows)
        semantics = {"state_weight": "explicit_mean_one_loss_weight", "source_weights": {
            "gold": .7, "teacher_strategy": .25, "forced_end": .05}}
        validate_weight_semantics(rows, semantics)
        with self.assertRaises(ValueError): validate_weight_semantics(rows, {"state_weight": "equal"})
        apply_mixture_weights(changed, {"gold": .5, "teacher_strategy": .4, "forced_end": .1})
        with self.assertRaises(ValueError): validate_weight_semantics(changed, semantics)
        self.assertNotEqual(rows[0]["loss_weight"], changed[0]["loss_weight"])
        for masses in ({"gold": 1}, {"gold": .8, "teacher_strategy": .25, "forced_end": .05},
                       {"gold": .7, "teacher_strategy": float("nan"), "forced_end": .05}):
            with self.assertRaises(ValueError): apply_mixture_weights(rows, masses)
        with self.assertRaises(ValueError):
            apply_mixture_weights(rows[:2], {"gold": .7, "teacher_strategy": .25, "forced_end": .05})

    def test_partition_rejects_reserved_routes_and_multiple_seed_groups(self):
        partition = {"role": "train_candidate", "excluded_routes": [1], "reserved_routes": [2]}
        selection = {"source_partition": partition, "samples": [{"route": 3, "group": 1}]}
        manifest = {"lineage": {"source_partition": deepcopy(partition), "source_routes": [3]}}
        self.assertEqual(training_route_groups(selection, manifest), {3: 1})
        for extra in ({"route": 2, "group": 1}, {"route": 3, "group": 2}, {"route": 4, "group": 1}):
            other = deepcopy(selection)
            other["samples"].append(extra)
            with self.assertRaises(ValueError): training_route_groups(other, manifest)


class MixedExportGuardTests(unittest.TestCase):
    def config(self):
        return dict(schema_version="mixed_sft_export_v1", dataset_id="mixed", gold_manifest="gold.json",
                    source_report="pool.json", source_selection="selection.json", output="output",
                    combat_indices=[1], encounter_families=["normal_weak"], smoke_combats=[],
                    source_weights={"gold": .7, "teacher_strategy": .25, "forced_end": .05})

    def test_invalid_parameters_and_paths_fail_before_reading_sources_or_resolving_native(self):
        for key, value in (("source_weights", {"gold": 1}), ("dataset_id", ""),
                           ("encounter_families", []), ("output", "../outside")):
            with self.subTest(key=key), TemporaryDirectory() as temp:
                root = Path(temp)
                config = self.config()
                config[key] = value
                write_json(root / "config.json", config)
                with patch("sts1_llm_policy.data.gold.mixed_export.resolve_simulator") as native:
                    with self.assertRaises(ValueError):
                        export_mixed_sft(root, "config.json")
                    native.assert_not_called()
                self.assertEqual(list(root.iterdir()), [root / "config.json"])

    def test_train_only_and_source_type_are_checked_before_any_split_asset(self):
        for change in ("test", "development", "preference", "observation_v5"):
            with self.subTest(change=change), TemporaryDirectory() as temp:
                root = Path(temp)
                write_json(root / "config.json", self.config())
                manifest = {"task_type": "sft", "observation_version": "observation_v7",
                            "splits": {"train": {"path": "absent-train.jsonl"}}}
                if change in {"test", "development"}:
                    manifest["splits"][change] = {"path": "must-not-read.jsonl"}
                elif change == "preference":
                    manifest["task_type"] = change
                else:
                    manifest["observation_version"] = change
                write_json(root / "gold.json", manifest)
                with patch("sts1_llm_policy.data.gold.mixed_export.validate_dataset_manifest") as validate:
                    with self.assertRaisesRegex(ValueError, "train-only V7 SFT"):
                        export_mixed_sft(root, "config.json")
                    validate.assert_not_called()
