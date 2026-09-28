from __future__ import annotations

import unittest
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from threading import Barrier, Lock
from unittest.mock import patch

from sts1_llm_policy.workflows.continuous_artifacts import (
    IDENTITY_SCHEMA_VERSION,
    summarize_arm,
    summarize_combat,
    load_completed_route,
)
from sts1_llm_policy.workflows.continuous_run import (
    _reward_with_run_state,
    _stateful_snapshot,
    _parallel_route_results,
    run,
)
from sts1_llm_policy.env.combat_snapshot import resign_snapshot


class ContinuousCombatGenerationTests(unittest.TestCase):
    def test_parallel_routes_are_concurrent_and_dispatched_once(self):
        barrier = Barrier(2, timeout=5)
        lock = Lock()
        running = 0
        peak = 0
        calls = []
        routes = [{"route_index": i} for i in range(4)]

        def execute(validated, arm, route):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
                calls.append(route["route_index"])
            barrier.wait()
            with lock:
                running -= 1
            return {"result": route["route_index"]}

        with patch("sts1_llm_policy.workflows.continuous_run._run_teacher_route", side_effect=execute):
            results = list(_parallel_route_results({}, {}, routes, 2))
        self.assertEqual(peak, 2)
        self.assertCountEqual(calls, range(4))
        self.assertEqual(sorted((r["route_index"], v["result"]) for r, v in results),
                         [(i, i) for i in range(4)])

    def test_parallel_failure_does_not_dispatch_the_whole_remaining_pool(self):
        def fail(*args):
            raise RuntimeError("native failure")
        with patch("sts1_llm_policy.workflows.continuous_run._run_teacher_route", side_effect=fail) as execute:
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                list(_parallel_route_results({}, {}, [{"route_index": i} for i in range(20)], 2))
        self.assertLessEqual(execute.call_count, 2)

    def test_workers_reject_invalid_counts_and_llm_arms(self):
        for workers in (True, 0, -1):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                run("unused.json", project_root=".", workers=workers)
        validated = {"routes": [], "arms": [{"config": {"id": "llm", "policy": "llm"}}]}
        with patch("sts1_llm_policy.workflows.continuous_run.prepare_run", return_value=validated):
            with self.assertRaisesRegex(ValueError, "Teacher-only"):
                run("unused.json", project_root=".", workers=2, preflight_only=True)

    def test_batch_continues_after_truncated_route(self):
        from contextlib import ExitStack
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            routes = [{"route_index": i, "combat_seed_group_index": 0} for i in range(2)]
            validated = {
                "config": {"run_id": "test", "route_count": 2, "combat_seed_groups_per_route": 1,
                           "observation_version": "observation_v7", "interaction_contract": "combat_card_selection_v1"},
                "routes": routes, "arms": [{"config": {"id": "test", "policy": "teacher_search"}}],
                "output": root, "paths": {"picker_database": root / "unused"}, "identity": {},
                "execution": SimpleNamespace(describe=lambda: {"status": "resolved"}),
            }
            def finished(validated, arm, route, **kwargs):
                truncated = route["route_index"] == 0
                return {"route": route, "status": "truncated" if truncated else "completed",
                        "death_combat_index": None, "combats_started": 1, "combats": [{
                            "encounter_family": "boss", "summary": {
                                "outcome": "aborted" if truncated else "victory", "decisions": 1,
                                "protocol": {"retry_count": 0, "fallback_count": 0},
                            }}]}
            with ExitStack() as stack:
                prefix = "sts1_llm_policy.workflows.continuous_run."
                stack.enter_context(patch(prefix + "prepare_run", return_value=validated))
                stack.enter_context(patch(prefix + "load_completed_route", return_value=None))
                execute = stack.enter_context(patch(prefix + "_run_route", side_effect=finished))
                for name in ("sts1_llm_policy.data.card_pick_metrics.SqliteCardRateBackend",
                             "sts1_llm_policy.data.card_pick_metrics.MassNormalizedCardRewardPicker",
                             "sts1_llm_policy.eval.counterfactual_reward_picker_v2.StrategicRateTable"):
                    stack.enter_context(patch(name, return_value=object()))
                report = run(root / "unused.json", project_root=root)
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(report["status"], "completed")
            metrics = report["metrics"]["test"]
            self.assertEqual((metrics["route_execution_count"], metrics["boss_victories"], metrics["truncated_routes"]), (2, 1, 1))

    def setUp(self) -> None:
        self.snapshot = resign_snapshot({
            "schema_version": "combat_snapshot_v1", "character": "IRONCLAD",
            "source_reward_seed": 1, "source_reward_id": 0,
            "ascension": 0, "act": 1, "floor": 0,
            "current_hp": 80, "max_hp": 80,
            "deck": [{
                "enum_id": "BASH", "name": "Bash", "string_id": "Bash",
                "upgraded": False,
            }],
            "relics": [
                {"id": "BURNING_BLOOD", "counter": 0},
                {"id": "NUNCHAKU", "counter": 0},
                {"id": "VAJRA", "counter": 0},
            ],
        })
        self.run_state = {
            "current_hp": 37, "max_hp": 84,
            "relics": [
                {"id": "BURNING_BLOOD", "counter": 0},
                {"id": "NUNCHAKU", "counter": 7},
            ],
        }

    def test_stateful_snapshot_carries_hp_max_hp_and_known_relic_counters(self) -> None:
        result = _stateful_snapshot(self.snapshot, self.run_state, floor=11)
        self.assertEqual((result["current_hp"], result["max_hp"], result["floor"]), (37, 84, 11))
        counters = {item["id"]: item["counter"] for item in result["relics"]}
        self.assertEqual(counters, {"BURNING_BLOOD": 0, "NUNCHAKU": 7, "VAJRA": 0})
        self.assertNotEqual(result["fingerprint_fnv1a64"], self.snapshot["fingerprint_fnv1a64"])

    def test_reward_view_uses_carried_state_without_mutating_native_response(self) -> None:
        reward = {
            "reward_id": 1, "choices": [],
            "reward_state": {
                "current_hp": 80, "max_hp": 80, "deck": [],
                "relics": [{"id": "NUNCHAKU", "counter": 0}],
            },
        }
        result = _reward_with_run_state(reward, self.run_state)
        self.assertEqual((result["reward_state"]["current_hp"], result["reward_state"]["max_hp"]), (37, 84))
        self.assertEqual(result["reward_state"]["relics"][0]["counter"], 7)
        self.assertEqual(reward["reward_state"]["current_hp"], 80)

    def test_metrics_record_inclusive_death_combat_index(self) -> None:
        routes = [{
            "status": "defeated", "death_combat_index": 4,
            "route": {"route_index": 0, "combat_seed_group_index": 0},
            "combats": [
                {"summary": {
                    "outcome": "victory" if index < 4 else "defeat",
                    "decisions": 2,
                    "protocol": {"retry_count": 0, "fallback_count": 0},
                }}
                for index in range(1, 5)
            ],
        }]
        metrics = summarize_arm(routes)
        self.assertEqual(metrics["combat_count"], 4)
        self.assertEqual(metrics["combat_victories"], 3)
        self.assertEqual(metrics["route_count"], 1)
        self.assertEqual(metrics["route_execution_count"], 1)
        self.assertEqual(metrics["combat_seed_groups_per_route"], 1)
        self.assertEqual(metrics["death_combat_index_counts"], {4: 1})

    def test_resume_verifies_snapshot_and_trajectory_content(self) -> None:
        identity = {"schema_version": IDENTITY_SCHEMA_VERSION}
        step = {"kind": "combat", "combat_index": 1, "floor": 1, "encounter_family": "boss",
                "scenario_id": "cultist", "combat_seed": 10, "policy_seed": 20}
        route = {"route_index": 0, "steps": [step]}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory = root / "arm/route-work/route-000/combat-01.jsonl"
            trajectory.parent.mkdir(parents=True)
            record = {
                "schema_version": "trajectory_v1", "record_type": "transition", "done": True,
                "episode_id": "fixture", "step_index": 0, "game_seed": 10, "policy_seed": 20,
                "scenario_id": "cultist", "policy_name": "arm", "terminal_outcome": "victory", "turn": 1,
                "policy_result": {}, "next_raw_state": {"source": "simulator", "state": {
                    "terminal": True, "outcome": "PLAYER_VICTORY", "player": {"current_hp": 70, "max_hp": 80},
                    "relics": [], "combat_accounting": {"starting_hp": 80, "total_hp_loss": 10}}},
            }
            trajectory.write_text(json.dumps(record) + "\n", encoding="utf-8")
            snapshot = {"deck": []}
            item = {
                "schema_version": "continuous_combat_route_v1",
                "configuration_identity": identity,
                "arm": "arm",
                "route": route,
                "combats_started": 1, "combats_won": 1, "status": "completed",
                "death_combat_index": None, "test_data_read": False,
                "combats": [{
                    **{key: value for key, value in step.items() if key != "kind"},
                    "summary": summarize_combat([record], "victory"),
                    "input_snapshot": snapshot,
                    "input_snapshot_sha256": sha256(
                        json.dumps(
                            snapshot,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "trajectory": str(trajectory.relative_to(root)).replace("\\", "/"),
                    "trajectory_sha256": sha256(trajectory.read_bytes()).hexdigest(),
                }],
            }
            route_path = root / "arm/routes/route-000.json"
            route_path.parent.mkdir(parents=True)
            route_path.write_text(json.dumps(item), encoding="utf-8")
            self.assertEqual(
                load_completed_route(
                    route_path,
                    identity=identity,
                    arm="arm",
                    route=route,
                    output_root=root,
                ),
                item,
            )
            trajectory.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trajectory changed"):
                load_completed_route(
                    route_path,
                    identity=identity,
                    arm="arm",
                    route=route,
                    output_root=root,
                )

    def test_cached_multi_arm_merge_writes_only_combined_artifacts(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            formal = output / "formal"
            formal.mkdir(parents=True)
            teacher_report = formal / "report-teacher.json"
            teacher_inputs = formal / "inputs-teacher.json"
            original = json.dumps({"configuration_identity": {"schema_version": IDENTITY_SCHEMA_VERSION}})
            teacher_report.write_text(original, encoding="utf-8")
            teacher_inputs.write_text(original, encoding="utf-8")
            route = {
                "route_index": 0,
                "combat_seed_group_index": 0,
                "steps": [],
            }
            identity = {"schema_version": IDENTITY_SCHEMA_VERSION}
            arms = [
                {"config": {"id": "base", "policy": "teacher_search"}},
                {"config": {"id": "teacher", "policy": "teacher_search"}},
            ]
            validated = {
                "config": {
                    "run_id": "test_run",
                    "route_count": 1,
                    "combat_seed_groups_per_route": 1,
                    "observation_version": "observation_v6",
                    "interaction_contract": "combat_card_selection_v1",
                },
                "routes": [route],
                "arms": arms,
                "output": output,
                "paths": {"picker_database": root / "picker.sqlite3"},
                "identity": identity,
                "execution": SimpleNamespace(describe=lambda: {"status": "resolved"}),
            }

            def completed(*args, arm, route, **kwargs):
                del args, kwargs
                return {
                    "schema_version": "continuous_combat_route_v1",
                    "configuration_identity": identity,
                    "arm": arm,
                    "route": route,
                    "status": "completed",
                    "combats_started": 0,
                    "combats_won": 0,
                    "death_combat_index": None,
                    "combats": [],
                    "reward_events": [],
                    "topology_events": [],
                    "test_data_read": False,
                }

            with (
                patch(
                    "sts1_llm_policy.workflows.continuous_run.prepare_run",
                    return_value=validated,
                ),
                patch(
                    "sts1_llm_policy.workflows.continuous_run.load_completed_route",
                    side_effect=completed,
                ) as load_completed,
                patch(
                    "sts1_llm_policy.workflows.continuous_run._run_route"
                ) as run_route,
                patch(
                    "sts1_llm_policy.data.card_pick_metrics.SqliteCardRateBackend",
                    return_value=object(),
                ),
                patch(
                    "sts1_llm_policy.data.card_pick_metrics.MassNormalizedCardRewardPicker",
                    return_value=object(),
                ),
                patch(
                    "sts1_llm_policy.eval.counterfactual_reward_picker_v2.StrategicRateTable",
                    return_value=object(),
                ),
            ):
                report = run(root / "unused.json", project_root=root)

            self.assertEqual(report["scope"]["arms"], 2)
            self.assertEqual(load_completed.call_count, 2)
            run_route.assert_not_called()
            self.assertTrue((formal / "report.json").is_file())
            self.assertTrue((formal / "inputs.json").is_file())
            self.assertEqual(teacher_report.read_text(encoding="utf-8"), original)
            self.assertEqual(teacher_inputs.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
