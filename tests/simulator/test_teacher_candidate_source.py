import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

from sts1_llm_policy.workflows.continuous_plan import (
    load_scope,
    build_route_steps,
    build_route_lineage,
    validate_source_isolation,
)
from sts1_llm_policy.eval.generation_strategies import resolve_generation_strategies

ROOT = Path(__file__).resolve().parents[2]
NEW = "configs/generation/continuous_teacher_pool.json"


def routes(config):
    scope = load_scope(ROOT / config["scope"], project_root=ROOT)
    strategy = resolve_generation_strategies(config["strategies"]).route
    return [{"route_index": i, "combat_seed_group_index": j,
             "steps": build_route_steps(config, scope, i, j, strategy)}
            for i in range(config["route_count"])
            for j in range(config["combat_seed_groups_per_route"])]


class CandidateSourceTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / NEW).read_text(encoding="utf-8"))
        self.config["route_count"] = 2
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.exclusions = json.loads((ROOT / self.config["data_source"]["excluded_sources"][0]).read_text(encoding="utf-8"))
        self.config["data_source"]["excluded_sources"] = ["excluded.json"]
        self.write_exclusions()

    def write_exclusions(self):
        (self.root / "excluded.json").write_text(json.dumps(self.exclusions), encoding="utf-8")

    def test_four_groups_share_lineage_and_new_routes_are_disjoint(self):
        panel = routes(self.config)
        result = validate_source_isolation(self.config, panel, project_root=self.root)
        self.assertEqual(result["source_routes"], 2)
        self.assertEqual(result["route_executions"], 8)
        self.assertEqual(len({build_route_lineage(self.config, r)["source_route_id"] for r in panel[:4]}), 1)
        self.assertNotEqual(build_route_lineage(self.config, panel[0])["source_route_id"],
                            build_route_lineage(self.config, panel[4])["source_route_id"])
        self.assertTrue(all(r["overlap"] == 0 for r in result["excluded_sources"]))

    def test_new_name_does_not_hide_old_source(self):
        panel = routes(self.config)
        self.exclusions["source_route_ids"] = [build_route_lineage(self.config, panel[0])["source_route_id"]]
        self.write_exclusions()
        self.config.update(run_id="renamed", output_dir="outputs/renamed")
        with self.assertRaisesRegex(ValueError, "Source route overlap"):
            validate_source_isolation(self.config, panel, project_root=self.root)

    def test_seed_overlap_is_checked_beyond_base_values(self):
        start, end = next(pair for pair in self.exclusions["seed_intervals"] if pair[1] - pair[0] >= 17)
        self.config["seeds"]["combat"] = start + 17
        with self.assertRaisesRegex(ValueError, "RNG seed overlap"):
            validate_source_isolation(self.config, routes(self.config), project_root=self.root)

    def test_legacy_mode_stays_development_and_invalid_source_is_rejected(self):
        self.assertEqual(validate_source_isolation({}, [], project_root=self.root),
                         {"evidence_class": "development_evaluation"})
        self.config["data_source"]["evidence_class"] = "gold"
        with self.assertRaisesRegex(ValueError, "data_source"):
            validate_source_isolation(self.config, [], project_root=self.root)

    def test_swapping_excluded_source_changes_validation(self):
        self.exclusions["origins"][0]["run_id"] = self.config["run_id"]
        self.write_exclusions()
        with self.assertRaisesRegex(ValueError, "new run ID"):
            validate_source_isolation(self.config, routes(self.config), project_root=self.root)


if __name__ == "__main__":
    unittest.main()
