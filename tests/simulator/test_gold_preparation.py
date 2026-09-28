"""Preparation substitutes real source/config inputs without model or native work."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import textwrap
import unittest
from unittest.mock import patch

from scripts import prepare_gold_collection as cli, collect_gold as run_cli
from sts1_llm_policy.data.gold.preparation import prepare_collection, teacher_floors
from sts1_llm_policy.data.gold.configuration import load_config
from sts1_llm_policy.artifacts import sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def pool(root, name, routes=4):
    base = root / name
    battles = routes * 2
    write(base / "report.json", {
        "status": "completed", "scope": {"routes": routes, "combat_seed_groups_per_route": 2},
        "configuration": {"data_source": {"evidence_class": "teacher_candidate_pool"},
                          "route": {"operations": ["combat:normal_weak", "combat:boss"]}},
        "metrics": {"teacher": {"truncated_routes": 0, "route_execution_count": battles,
                                 "death_combat_index_counts": {}, "combat_victories": battles * 2,
                                 "combat_count": battles * 2}},
    })
    for route in range(routes):
        for group in range(2):
            combats = []
            for combat, family in enumerate(("normal_weak", "boss"), 1):
                path = f"r{route}-g{group}-c{combat}.jsonl"
                rows = [{"schema_version": "trajectory_v1", "record_type": "transition",
                         "step_index": i, "legal_actions": ["A", "B"],
                         "model_legal_actions": ["A", "B"]} for i in range(5)]
                (base / path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                combats.append({"combat_index": combat, "encounter_family": family,
                                "scenario_id": name, "trajectory": path, "summary": {"decisions": 5}})
            write(base / f"teacher/routes/route-{route:03d}-combat-seed-group-{group:02d}.json", {"combats": combats})


def template(source="pool/report.json"):
    return {"schema_version": "teacher_gold_probe_v3", "source_report": source,
            "observation_version": "observation_v7", "output": "old-output",
            "outer_trials": 32, "hidden_samples": 4, "search_budget": 2048,
            "minimum_root_visits": 32, "max_decisions": 1000, "workers": 1, "seed": 1,
            "minimum_win_rate": .9, "confidence": .95, "expected_hp_tolerance": 5.,
            "samples": [{"id": "old", "route": 0, "group": 0, "combat": 1, "step": 0}],
            "smoke_samples": ["old"]}


class GoldPreparationTests(unittest.TestCase):
    def test_state_and_cli_preparation_without_runtime_or_model_imports(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool(root, "pool")
            write(root / "template.json", template())
            script = textwrap.dedent('''
                import sys
                from pathlib import Path

                class BlockExecution:
                    def find_spec(self, fullname, path=None, target=None):
                        if any(fullname == name or fullname.startswith(name + ".") for name in (
                            "sts1_llm_policy.eval",
                            "sts1_llm_policy.env.simulator_installation",
                            "sts1_llm_policy.env.simulator_client",
                            "sts1_llm_policy.env.simulator_adapter",
                            "sts1_llm_policy.env.simulator_env",
                            "sts1_llm_policy.env.simulator_execution",
                            "sts1_llm_policy.env.simulator_search",
                            "torch", "transformers",
                        )):
                            raise AssertionError("Preparation imported execution dependency: " + fullname)

                sys.meta_path.insert(0, BlockExecution())
                from sts1_llm_policy.env.state_schema import CanonicalState
                from scripts import prepare_gold_collection as cli
                root = Path(sys.argv[1])
                cli.__file__ = str(root / "scripts/prepare_gold_collection.py")
                sys.argv = ["prepare_gold_collection.py", "--template-config", "template.json",
                            "--config-output", "new/config.json", "--output", "new/run",
                            "--route-count", "2", "--groups-per-route", "1",
                            "--sampling", "uniform_decisions_v1", "--adaptive", "--compact-storage"]
                cli.main()
            ''')
            result = subprocess.run([sys.executable, "-c", script, str(root)],
                                    cwd=Path(__file__).resolve().parents[2],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = load_config(root / "new/config.json", project_root=root)
            selection = json.loads((root / "new/run/selection.json").read_text(encoding="utf-8"))
            self.assertEqual(config["sampling_ladder"]["mode"], "adaptive")
            self.assertEqual(config["continuation_storage"]["mode"], "compact_v1")
            self.assertEqual(len(config["samples"]), selection["state_count"])
            self.assertGreater(selection["state_count"], 0)

    def prepare(self, root, **options):
        return prepare_collection(root, **{
            "template_config": "template.json", "config_output": "new/config.json", "output": "new/run",
            "route_count": 2, "groups_per_route": 1, "seed": 17,
            "states_per_combat": {"normal_weak": 1, "boss": 2},
            "sampling": "uniform_decisions_v1", **options,
        })

    def test_source_template_and_group_substitution_controls_real_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool(root, "pool")
            pool(root, "other", routes=5)
            write(root / "template.json", template())
            first = self.prepare(root)
            alternate = template("other/report.json")
            alternate.update(search_budget=8192, hidden_samples=2)
            write(root / "other-template.json", alternate)
            second = self.prepare(root, template_config="other-template.json", groups_per_route=2,
                                  config_output="other-run/config.json", output="other-run/results")
            self.assertEqual(first["state_count"], 6)
            self.assertEqual(second["state_count"], 12)
            self.assertTrue(all(s["scenario"] == "other" for s in second["selections"]))
            cfg = load_config(root / "other-run/config.json", project_root=root)
            self.assertEqual((cfg["source_report"], cfg["search_budget"], cfg["hidden_samples"]),
                             ("other/report.json", 8192, 2))
            self.assertEqual(first, self.prepare(root))

    def test_cli_full_trace_clears_inherited_partition_groups_and_storage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool(root, "pool")
            original = template()
            original.update(trace_observation="reconstruct", route_progress=True,
                            source_partition={"role": "train_candidate", "excluded_routes": [1], "reserved_routes": [2, 3]},
                            panel_groups={"old": ["old"]}, resume_from="previous/formal",
                            continuation_storage={"mode": "compact_v1", "full_trace_probability": .01,
                                                  "keep_first_trial": True, "keep_boundary_trials": True})
            write(root / "template.json", original)
            args = ["prepare_gold_collection.py", "--template-config", "template.json",
                    "--config-output", "new/config.json", "--output", "new/run",
                    "--groups-per-route", "1", "--route-count", "2", "--sampling", "uniform_decisions_v1",
                    "--trace-observation", "full"]
            with patch.object(cli, "__file__", str(root / "scripts/prepare.py")), patch("sys.argv", args), patch("builtins.print"):
                cli.main()
            cfg = load_config(root / "new/config.json", project_root=root)
            self.assertEqual(cfg["trace_observation"], "full")
            for key in ("source_partition", "panel_groups", "resume_from", "continuation_storage", "route_progress"):
                self.assertNotIn(key, cfg)
            self.assertEqual(json.loads((root / "template.json").read_text()), original)

    def test_training_partition_and_explicit_storage_are_built_for_new_samples(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool(root, "pool")
            write(root / "template.json", template())
            write(root / "exclude.json", {"source_report": "pool/report.json",
                                          "source_report_sha256": sha256_file(root / "pool/report.json"),
                                          "excluded_routes": [0]})
            selected = self.prepare(root, training_candidates=True, exclusions="exclude.json",
                                    adaptive=True, compact_storage=True, trace_observation="reconstruct")
            cfg = load_config(root / "new/config.json", project_root=root)
            routes = {s["route"] for s in cfg["samples"]}
            self.assertNotIn(0, routes)
            self.assertEqual(cfg["source_partition"]["excluded_routes"], [0])
            self.assertEqual(set(cfg["source_partition"]["reserved_routes"]), {1, 2, 3} - routes)
            self.assertEqual(selected["source_partition"], cfg["source_partition"])
            self.assertEqual((cfg["outer_trials"], cfg["sampling_ladder"]["mode"], cfg["continuation_storage"]["mode"]),
                             (128, "adaptive", "compact_v1"))

    def test_invalid_preparation_never_publishes_either_document(self):
        for options in ({"workers": 0}, {"hp_tolerance": float("nan")}, {"training_candidates": True},
                        {"trace_observation": "invalid"}, {"route_count": 20},
                        {"states_per_combat": {"normal_weak": 0, "boss": 2}}):
            with self.subTest(options=options), TemporaryDirectory() as tmp:
                root = Path(tmp)
                pool(root, "pool")
                write(root / "template.json", template())
                with self.assertRaises(ValueError):
                    self.prepare(root, **options)
                self.assertFalse((root / "new").exists())

    def test_conflict_in_either_destination_prevents_writing_the_other(self):
        for occupied, absent in (("new/config.json", "new/run/selection.json"),
                                 ("new/run/selection.json", "new/config.json")):
            with self.subTest(occupied=occupied), TemporaryDirectory() as tmp:
                root = Path(tmp)
                pool(root, "pool")
                write(root / "template.json", template())
                write(root / occupied, {"existing": True})
                original = (root / occupied).read_bytes()
                with self.assertRaisesRegex(ValueError, "Existing collection inputs differ"):
                    self.prepare(root)
                self.assertEqual((root / occupied).read_bytes(), original)
                self.assertFalse((root / absent).exists())

    def test_cli_requires_explicit_experiment_paths(self):
        for module in (cli, run_cli):
            with self.subTest(module=module.__name__), patch("sys.argv", ["entry.py"]), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 2)

    def test_floor_uses_reached_battles_and_reconciles_source_totals(self):
        report = {"status": "completed", "configuration": {"route": {"operations": ["combat:normal_weak", "heal:24", "combat:elite", "combat:boss"]}},
                  "metrics": {"teacher": {"truncated_routes": 0, "route_execution_count": 100,
                                          "death_combat_index_counts": {"2": 2, "3": 8},
                                          "combat_victories": 288, "combat_count": 298}}}
        floors, rates = teacher_floors(report, .01)
        self.assertAlmostEqual(floors["elite"], .97)
        self.assertEqual(rates["boss"]["battles"], 98)
        self.assertAlmostEqual(floors["boss"], 90 / 98 - .01)
        changed = deepcopy(report)
        changed["metrics"]["teacher"]["combat_count"] = 300
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            teacher_floors(changed, .01)
