import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from sts1_llm_policy.data.gold.configuration import validate_ladder
from sts1_llm_policy.data.gold.sampling_ladder import (
    stage_decision,
    ladder_records,
    ladder_statistics,
)
from sts1_llm_policy.data.gold import collection as probe


def config():
    return {"outer_trials": 128, "confidence": .95, "minimum_win_rate": .989,
            "expected_hp_tolerance": 1., "win_floor_rounding": "floor",
            "sampling_ladder": {"mode": "shadow", "stages": [32, 64, 128],
                                "hp_boundary_band": .25, "audit_probability": 0., "audit_seed": 4}}


def summary(count, rows):
    required = {32: 31, 64: 63, 128: 126}[count]
    actions = [{"action": name, "wins": wins, "expected_carried_hp": hp,
                "passes_win_floor": wins >= required} for name, wins, hp in rows]
    best = max((a["expected_carried_hp"] for a in actions if a["passes_win_floor"]), default=None)
    return {"status": "completed", "minimum_wins": required, "actions": actions,
            "estimated_acceptable_actions": [a["action"] for a in actions
                                            if a["passes_win_floor"] and best - a["expected_carried_hp"] <= 1]}


class SamplingLadderTests(unittest.TestCase):
    def test_proper_goes_to_64_but_full_wins_do_not_force_128(self):
        cfg = config()
        a = stage_decision(summary(32, [("A", 32, 40), ("B", 32, 35)]), None, cfg, "x", 32)
        b = stage_decision(summary(64, [("A", 64, 40), ("B", 64, 35)]), a, cfg, "x", 64)
        self.assertTrue(a["would_expand"])
        self.assertFalse(b["would_expand"])
        self.assertEqual(b["win_boundary_actions"], [])

    def test_empty_all_and_near_gate_rules(self):
        cfg = config()
        for rows, expand in (([("A", 29, 40), ("B", 28, 35)], False),
                             ([("A", 30, 40), ("B", 28, 35)], True),
                             ([("A", 32, 40), ("B", 32, 39.5)], False),
                             ([("A", 32, 40), ("B", 32, 39.1)], True),
                             ([("A", 31, 40), ("B", 32, 39.5)], True)):
            self.assertEqual(stage_decision(summary(32, rows), None, cfg, "x", 32)["would_expand"], expand)

    def test_win_boundary_only_matters_for_relevant_roots(self):
        cfg = config()
        old = {"candidates": ["A"]}
        low = stage_decision(summary(64, [("A", 64, 40), ("B", 62, 20)]), old, cfg, "x", 64)
        high = stage_decision(summary(64, [("A", 64, 40), ("B", 62, 50)]), old, cfg, "x", 64)
        self.assertFalse(low["would_expand"])
        self.assertEqual(high["win_boundary_actions"], ["B"])
        self.assertTrue(high["reference_sensitive"])

    def test_set_flip_expands_and_cap_preserves_empirical_candidates(self):
        cfg = config()
        old = {"candidates": ["A", "B"]}
        middle = stage_decision(summary(64, [("A", 64, 40), ("B", 64, 35)]), old, cfg, "x", 64)
        self.assertIn("candidate_set_changed", middle["reasons"])
        cap = stage_decision(summary(128, [("A", 126, 40), ("B", 128, 39.1)]), middle, cfg, "x", 128)
        self.assertFalse(cap["would_expand"])
        self.assertEqual(cap["candidates"], ["A", "B"])
        self.assertEqual(cap["action_stability"], {"A": "borderline", "B": "flip"})

    def test_audit_is_deterministic_and_only_adds_to_stops(self):
        cfg = config()
        cfg["sampling_ladder"]["audit_probability"] = 1.
        data = summary(32, [("A", 32, 40), ("B", 32, 39.5)])
        first = stage_decision(data, None, cfg, "x", 32)
        self.assertEqual(first, stage_decision(data, None, cfg, "x", 32))
        self.assertTrue(first["audit_selected"])
        self.assertTrue(first["would_expand"])
        cfg["sampling_ladder"]["audit_probability"] = 0.
        self.assertFalse(stage_decision(data, None, cfg, "x", 32)["would_expand"])

    def test_run_sample_consumes_ladder_and_replay_recalculation_matches(self):
        cfg = config()
        sample = {"id": "example"}
        execution = MagicMock()
        space = MagicMock(action_ids=["A", "B"])
        rows = []
        def trial(env, source, sample, execution_config, output, trial, action, stop):
            row = {"root_action": action, "trial": trial, "outcome": "victory",
                   "ending_hp": 40 if action == "A" else 35, "max_hp": 80, "relic_counters": [],
                   "search_calls": 4, "decisions": 2}
            rows.append(row)
            return row
        with TemporaryDirectory() as temporary, \
                patch.object(probe, "load_source", return_value=({"scenario_id": "fixture", "encounter_family": "elite"}, [], {})), \
                patch.object(probe, "restore"), patch.object(probe, "StsLightspeedEnv"), \
                patch.object(probe, "build_model_action_space", return_value=space), \
                patch.object(probe, "observation", return_value="fixture"), \
                patch.object(probe, "root_records", return_value=[{"id": "A"}, {"id": "B"}]), \
                patch.object(probe, "execute_trial", side_effect=trial), patch("builtins.print"):
            output = Path(temporary)
            result = probe.run_sample(output, cfg, sample, execution, output, None)
            self.assertEqual(len(rows), 256)  # Full shadow continues beyond hypothetical 64 stop.
            self.assertEqual(result["sampling_ladder"], ladder_records(rows, cfg, "example", probe.summarize))
            for count in (32, 64, 128):
                self.assertEqual(probe.read_json(output / "example" / f"stage-{count:03d}.json")["executions"], count * 2)
            stats = ladder_statistics([result], cfg)["panel"]
            self.assertEqual(stats["proposed_stops"], {"32": 0, "64": 1, "128": 0})
            self.assertEqual(stats["search_saving_fraction"], .5)
            self.assertEqual(stats["early_stop_set_change_rate"], 0.)
            # The same consumer really stops early in adaptive mode, and resumes
            # the completed state without executing its trials again.
            for name, audit, hp, expected in (("early32", 0., 40, 32),
                                              ("early64", 0., 35, 64),
                                              ("audit128", 1., 35, 128)):
                adaptive = {**cfg, "sampling_ladder": {**cfg["sampling_ladder"], "mode": "adaptive", "audit_probability": audit}}
                def adaptive_trial(*args):
                    row = trial(*args)
                    if row["root_action"] == "B":
                        row["ending_hp"] = hp
                    return row
                rows.clear()
                with patch.object(probe, "execute_trial", side_effect=adaptive_trial) as execute:
                    actual = probe.run_sample(output, adaptive, sample, execution, output / name, None)
                    self.assertEqual(actual["outer_trials_completed"], expected)
                    self.assertEqual(len(rows), expected * 2)
                    self.assertEqual(actual["sampling_ladder"], ladder_records(rows, adaptive, "example", probe.summarize))
                    self.assertFalse(ladder_statistics([actual], adaptive)["panel"]["full_128_reference_available"])
                    count = execute.call_count
                    self.assertEqual(probe.run_sample(output, adaptive, sample, execution, output / name, None), actual)
                    self.assertEqual(execute.call_count, count)

    def test_configuration_rejects_unsupported_execution_semantics(self):
        cfg = config()
        validate_ladder(cfg)
        for key, value in (("mode", "unknown"), ("stages", [32, 128, 256]),
                           ("hp_boundary_band", float("nan")), ("audit_probability", 2)):
            with self.assertRaises(ValueError):
                validate_ladder({**cfg, "sampling_ladder": {**cfg["sampling_ladder"], key: value}})

    def test_paired_variance_triggers_only_at_64_without_changing_labels(self):
        cfg = config()
        cfg["sampling_ladder"]["paired_se_multiplier_at_64"] = 2.
        for n in (32, 64, 128):
            data = summary(n, [("A", n, 40), ("B", n, 37.5)])
            data["paired_comparisons"] = [{"a": "A", "b": "B", "standard_error": 1.}]
            stage = stage_decision(data, {"candidates": ["A"]}, cfg, "x", n)
            self.assertEqual(stage["candidates"], ["A"])
            self.assertEqual(stage["variance_boundary_actions"], ["B"] if n == 64 else [])
            if n == 64:
                self.assertTrue(stage["would_expand"])

    def test_route_progress_counts_only_fully_finished_routes(self):
        samples = [{"id": "a", "route": 0}, {"id": "b", "route": 0}, {"id": "c", "route": 1}]
        progress = probe.RouteProgress(samples, "collect")
        with patch("builtins.print") as log:
            progress.update(samples[0], finished=True)
            self.assertIn("routes 0/2 (0.00%)", log.call_args.args[0])
            progress.update(samples[2], finished=True)
            self.assertIn("routes 1/2 (50.00%)", log.call_args.args[0])
            progress.update(samples[1], finished=True)
            self.assertIn("routes 2/2 (100.00%)", log.call_args.args[0])
            progress.update(samples[1], finished=True)
            self.assertIn("states 3/3 (100.00%)", log.call_args.args[0])



if __name__ == "__main__":
    unittest.main()
