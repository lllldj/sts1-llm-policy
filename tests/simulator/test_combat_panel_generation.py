"""Generation boundaries without launching a native simulator."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from sts1_llm_policy.eval import combat_panel_generation as generation


class CombatPanelGenerationTests(unittest.TestCase):
    def test_seed_layout_uses_configured_counts_and_preserves_route_ids(self):
        routes = [{"route_index": 0}, {"route_index": 4}]
        layout = {
            "original_seeds_per_route": 1, "seeds_per_route": 3,
            "additional_episode_index_start": 30,
            "original_combat_seed_base": 100, "original_policy_seed_base": 200,
            "additional_combat_seed_base": 300, "additional_policy_seed_base": 400,
        }
        specs = generation._build_evaluation_specs(routes, layout)
        self.assertEqual([row["episode_index"] for row in specs], [0, 4, 30, 31, 32, 33])
        self.assertEqual([row["combat_seed"] for row in specs], [100, 104, 300, 301, 302, 303])
        self.assertEqual([row["route_index"] for row in specs], [0, 4, 0, 0, 4, 4])
        layout["additional_episode_index_start"] = 0
        with self.assertRaisesRegex(ValueError, "overlap"):
            generation._build_evaluation_specs(routes, layout)

    def test_binding_tracks_rules_strategies_assets_and_routes_without_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = Path(directory) / "bridge"
            bridge.write_bytes(b"native")
            config = {key: {} for key in (
                "source_rules", "source_matrix", "route_panel", "evaluation_panel", "protocol", "strategies",
            )}
            config.update(panel_id="panel", observation_version="observation_v5", simulator_mechanics="corrected_v1")
            validated = {
                "config": config, "routes": [{"route_index": 0}],
                "picker_database_sha256": "database-bytes",
                "source": SimpleNamespace(reward_card_ids=frozenset({"STRIKE_RED", "BASH"})),
                "installation": SimpleNamespace(revision="revision", bridge_executable=bridge),
            }
            original = generation._binding(validated)
            relocated = deepcopy(validated)
            relocated["config"].update(
                output_dir="outputs/elsewhere", scope="elsewhere/scope.json",
                picker_database="elsewhere/picker.db", run_id="another-name",
            )
            self.assertEqual(original, generation._binding(relocated))
            changes = [
                ("protocol", {"search_seed": 9}),
                ("source_rules", {"picker": {"epsilon": 0.1}}),
                ("strategies", {"card_pick": "another_strategy"}),
                ("route_panel", {"seed_bases": {"reward": 123}}),
            ]
            for key, value in changes:
                changed = deepcopy(validated)
                changed["config"][key] = value
                with self.subTest(key=key):
                    self.assertNotEqual(original, generation._binding(changed))
            for key, value in (
                ("source", SimpleNamespace(reward_card_ids=frozenset({"BASH"}))),
                ("routes", [{"route_index": 0, "combat_seed": 42}]),
                ("picker_database_sha256", "different-database"),
            ):
                with self.subTest(key=key):
                    self.assertNotEqual(original, generation._binding({**validated, key: value}))
            bridge.write_bytes(b"different-native")
            self.assertNotEqual(original, generation._binding(validated))

    def test_resume_rejects_corrupt_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.json"
            spec, binding, snapshot = {"route_index": 0}, {"identity": "a"}, {"deck": []}
            row = dict(spec=spec, binding=binding, snapshot=snapshot,
                       snapshot_sha256=generation._canonical_sha(snapshot))
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(generation._load_route(path, binding=binding, spec=spec), row)
            with self.assertRaisesRegex(ValueError, "binding changed"):
                generation._load_route(path, binding={"identity": "changed"}, spec=spec)
            row["snapshot"]["deck"].append("changed")
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding changed"):
                generation._load_route(path, binding=binding, spec=spec)

    def test_preflight_uses_resolver_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            scope_path, database_path = root / "scope.json", root / "picker.db"
            scope_path.write_text('{"reward_cards":{"included_enum_ids":[]}}', encoding="utf-8")
            database_path.write_bytes(b"fixture")
            bridge = root / "bridge"
            bridge.write_bytes(b"native")
            config = {
                "schema_version": "combat_panel_generation_v2", "run_id": "example",
                "simulator_mechanics": "corrected_v1",
                "panel_id": "panel", "observation_version": "observation_v5",
                "simulator_revision": "revision", "test_data_read": False, "sealed_test_run": False,
                "scope": "scope.json",
                "picker_database": "picker.db", "expected_picker_database_sha256": generation._sha(database_path),
                "source_rules": {"deck_stages": [], "picker": {"temperature": 1, "exploration": 0, "epsilon": 0.02}},
                "source_matrix": {},
                "route_panel": {"selected_act": 1, "selected_ascension": 0, "expected_selected_routes": 1},
                "evaluation_panel": {"expected_combats": 1}, "protocol": {},
                "strategies": {
                    "card_pick": "teacher_counterfactual", "card_remove": "starter_alternating",
                    "card_upgrade": "value_priority", "route": "act_topology",
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            execution = MagicMock()
            execution.validate.return_value = SimpleNamespace(revision="revision", bridge_executable=bridge)
            execution.describe.return_value = {"target": "linux"}
            with (
                patch.object(generation, "resolve_simulator", return_value=execution) as resolve,
                patch.object(generation, "_build_source_matrix", return_value=[]),
                patch.object(generation, "_build_route_specs", return_value=[{"act": 1, "ascension": 0}]),
                patch.object(generation, "_build_evaluation_specs", return_value=[{}]),
                patch.object(generation, "SqliteCardRateBackend"),
                patch.object(generation, "_sha", wraps=generation._sha) as file_hash,
            ):
                report = generation.run("config.json", project_root=root, target="linux", preflight_only=True)
                self.assertEqual([call.args[0] for call in file_hash.call_args_list], [database_path, bridge])
                scope_path.write_text('{"description":"changed","reward_cards": {"included_enum_ids": []}}', encoding="utf-8")
                repeated = generation.run("config.json", project_root=root, target="linux", preflight_only=True)
                self.assertEqual(report["binding"], repeated["binding"])
                database_path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "picker database changed"):
                    generation.run("config.json", project_root=root, target="linux", preflight_only=True)
            self.assertEqual(resolve.call_count, 2)
            resolve.assert_called_with(project_root=root.resolve(), target="linux", mechanics="corrected_v1")
            self.assertEqual(execution.validate.call_count, 2)
            execution.create_client.assert_not_called()
            execution.create_environment.assert_not_called()
            self.assertEqual(report["status"], "preflight_passed")
            self.assertFalse((root / "outputs").exists())

    def test_resumed_panel_uses_content_equality_and_checks_each_snapshot_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            route = {"route_index": 0}
            specs = [{"route_index": 0, "combat_seed": 10}]
            snapshot = {"deck": ["BASH"]}
            snapshot_sha = generation._canonical_sha(snapshot)
            panel_sha = generation._canonical_sha({"specs": specs, "decks": {"0": snapshot_sha}})
            binding = {"schema_version": "combat_panel_generation_binding_v3"}
            generation._atomic_json(output / "routes/route-000.json", {
                "spec": route, "binding": binding, "snapshot": snapshot,
                "snapshot_sha256": snapshot_sha, "teacher_search_calls": 0,
            })
            execution = MagicMock()
            execution.describe.return_value = {}
            config = {"run_id": "fixture", "panel_id": "panel", "observation_version": "observation_v5",
                      "simulator_mechanics": "corrected_v1",
                      "protocol": {}, "evaluation_panel": {"expected_panel_sha256": panel_sha}}
            validated = {"config": config, "output": output, "routes": [route], "evaluation_specs": specs,
                         "source": SimpleNamespace(picker_database_path=root / "picker.db"),
                         "strategies": None, "execution": execution,
                         "installation": SimpleNamespace(revision="native")}
            with patch.object(generation, "_validate", return_value=validated), patch.object(
                generation, "_binding", return_value=binding
            ), patch.object(generation, "SqliteCardRateBackend"), patch.object(
                generation, "_sha", side_effect=AssertionError("No output file hash needed")
            ), patch.object(generation, "_canonical_sha", wraps=generation._canonical_sha) as content_hash:
                report = generation.run("config.json", project_root=root)
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["artifact"]["panel_sha256"], panel_sha)
                self.assertEqual(sum(call.args[0] == snapshot for call in content_hash.call_args_list), 1)
                execution.create_client.assert_not_called()
                execution.create_environment.assert_not_called()
                # A changed combat input must still fail the final frozen comparison.
                specs[0]["combat_seed"] = 11
                with self.assertRaisesRegex(ValueError, "differs from frozen input"):
                    generation.run("config.json", project_root=root)


if __name__ == "__main__":
    unittest.main()
