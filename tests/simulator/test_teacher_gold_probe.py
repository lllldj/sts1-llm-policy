import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sts1_llm_policy.data.gold.configuration import seed_for, validate_continuation_policy
from sts1_llm_policy.data.gold.continuation import (
    aggregate_searches,
    binomial_lower,
    summarize,
    scoring_config,
    continuation_policy,
    continuation_selection,
    choose,
)
from sts1_llm_policy.data.gold.collection import run
from sts1_llm_policy.data.gold.verification import verify_run


class GoldProbeStatisticsTests(unittest.TestCase):
    def test_attainable_floor_changes_candidates_at_low_and_high_sample_counts(self):
        for count, accepted, rejected in ((32, 31, 30), (128, 126, 125)):
            config = {"outer_trials": count, "confidence": .95, "minimum_win_rate": .989,
                      "expected_hp_tolerance": 5, "win_floor_rounding": "floor"}
            rows = [{"root_action": action, "trial": trial,
                     "outcome": "victory" if trial < wins else "defeat", "ending_hp": hp,
                     "max_hp": 80, "relic_counters": []}
                    for action, wins, hp in (("A", count, 40), ("B", accepted, 42), ("C", rejected, 80))
                    for trial in range(count)]
            result = summarize(rows, config)
            self.assertEqual(result["minimum_wins"], accepted)
            self.assertEqual(result["estimated_acceptable_actions"], ["A", "B"])
            self.assertFalse(result["actions"][1]["passes_win_floor_lcb"])
            self.assertEqual(summarize(rows, {**config, "win_floor_rounding": "exact"})["estimated_acceptable_actions"], ["A"])
            rows.pop()
            self.assertEqual(summarize(rows, config)["estimated_acceptable_actions"], [])

    def test_encounter_floor_changes_actual_candidate_selection(self):
        config = {"outer_trials": 10, "confidence": .95,
                  "minimum_win_rate": {"normal_strong": .99, "boss": .88}, "expected_hp_tolerance": 5}
        rows = [{"root_action": a, "trial": t, "outcome": "victory" if t < wins else "defeat",
                 "ending_hp": hp, "max_hp": 80, "relic_counters": []}
                for a, wins, hp in (("A", 10, 40), ("B", 9, 50)) for t in range(10)]
        self.assertEqual(summarize(rows, scoring_config(config, {"encounter_family": "normal_strong"}))["estimated_acceptable_actions"], ["A"])
        self.assertEqual(summarize(rows, scoring_config(config, {"encounter_family": "boss"}))["estimated_acceptable_actions"], ["A", "B"])
        with self.assertRaisesRegex(ValueError, "No minimum win rate"):
            scoring_config(config, {"encounter_family": "elite"})

    def test_verification_rejects_old_observation_before_replay_without_relabeling(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "formal/report.json"
            report_path.parent.mkdir()
            original = json.dumps({"status": "completed", "configuration": {"observation_version": "observation_v8"}})
            report_path.write_text(original, encoding="utf-8")
            with patch("sts1_llm_policy.data.gold.verification.resolve_simulator") as resolve:
                with self.assertRaisesRegex(ValueError, "Recorded observation version 'observation_v8'.*observation_v7"):
                    verify_run(root, {"output": "."})
                resolve.assert_not_called()
            self.assertEqual(report_path.read_text(encoding="utf-8"), original)

    def test_run_captures_git_provenance_before_executing_samples(self):
        from sts1_llm_policy.data.gold import collection as probe
        events = []
        provenance = {"git_revision": "revision-at-start", "git_dirty": False, "git_provenance_at": "run_start"}
        def capture(root):
            events.append("start")
            return provenance
        def sample(*args):
            events.append("sample")
            return {"sample": {"id": "example"}, "status": "completed"}
        with TemporaryDirectory() as temporary, patch.object(probe, "_git_provenance", side_effect=capture) as capture_mock, \
                patch.object(probe, "resolve_simulator") as resolve, patch.object(probe, "run_sample", side_effect=sample), \
                patch.object(probe, "source_identity", return_value={"fixture": "source"}):
            resolve.return_value.describe.return_value = {"bridge_sha256": "unchanged-runtime"}
            result = run(Path(temporary), {"output": ".", "workers": 1, "samples": [{"id": "example"}]})
            self.assertEqual(events, ["start", "sample"])
            capture_mock.assert_called_once()
            self.assertEqual({key: result[key] for key in provenance}, provenance)

    def test_deaths_count_as_zero_and_floor_precedes_expected_hp(self):
        config = {"outer_trials": 10, "confidence": .95, "minimum_win_rate": .8, "expected_hp_tolerance": 3}
        rows = []
        for action, wins, hp in (("A", 10, 40), ("B", 9, 45), ("C", 7, 70)):
            for trial in range(10):
                rows.append(dict(root_action=action, trial=trial, outcome="victory" if trial < wins else "defeat",
                                 ending_hp=hp if trial < wins else 0, max_hp=80, relic_counters=[]))
        result = summarize(rows, config)
        self.assertEqual(result["estimated_acceptable_actions"], ["A", "B"])
        self.assertEqual([a["expected_carried_hp"] for a in result["actions"]], [40, 40.5, 49])
        self.assertFalse(result["labels_certified"])
        rows[-1]["outcome"] = "aborted"
        self.assertEqual(summarize(rows, config)["estimated_acceptable_actions"], [])

    def test_exact_all_win_bound_and_seed_domains(self):
        self.assertEqual(binomial_lower(0, 32, .05), 0)
        self.assertAlmostEqual(binomial_lower(32, 32, .05), .05 ** (1 / 32))
        self.assertLess(binomial_lower(30, 32, .01), binomial_lower(30, 32, .05))
        self.assertNotEqual(seed_for(1, "outer"), seed_for(1, "inner"))

    def test_worlds_are_equal_weight_even_with_different_visits(self):
        def a(name, win, visits):
            return {"model_action_id": name, "win_rate": win, "visits": visits, "victory_ending_hp_mean": 40}
        evidence = [{"actions": [a("A", 1, 10000), a("B", .6, 1)]},
                    {"actions": [a("A", 0, 1), a("B", .6, 10000)]}]
        self.assertEqual(aggregate_searches(evidence), "B")

    def test_continuation_floor_changes_choice_and_falls_back_below_floor(self):
        evidence = [{"actions": [
            {"model_action_id": "A", "win_rate": .96, "victory_ending_hp_mean": 30},
            {"model_action_id": "B", "win_rate": .92, "victory_ending_hp_mean": 40},
        ]}]
        def policy(floor):
            return {"mode": "win_floor_then_expected_hp", "minimum_search_win_rate": floor}
        self.assertEqual(aggregate_searches(evidence), "A")
        self.assertEqual(aggregate_searches(evidence, policy(.95)), "A")
        decision = continuation_selection(evidence, policy(.90))
        self.assertEqual(decision, {"action": "B", "baseline_action": "A", "hp_mode_applied": True,
                                    "eligible_actions": 2, "changed_from_baseline": True})
        self.assertEqual(aggregate_searches(evidence, policy(.99)), "A")
        self.assertFalse(continuation_selection(evidence, policy(.99))["hp_mode_applied"])

    def test_continuation_uses_expected_hp_not_conditional_hp_or_pooled_visits(self):
        evidence = [{"actions": [
            {"model_action_id": "A", "win_rate": .96, "victory_ending_hp_mean": 40},
            {"model_action_id": "B", "win_rate": .90, "victory_ending_hp_mean": 42},
        ]}]
        policy = {"mode": "win_floor_then_expected_hp", "minimum_search_win_rate": .90}
        self.assertEqual(aggregate_searches(evidence, policy), "A")
        # E[p * HP] differs from E[p] * E[HP]; each world keeps equal weight.
        evidence = [{"actions": [
            {"model_action_id": "A", "win_rate": 1., "victory_ending_hp_mean": 10, "visits": 10000},
            {"model_action_id": "B", "win_rate": .5, "victory_ending_hp_mean": 40, "visits": 1},
        ]}, {"actions": [
            {"model_action_id": "A", "win_rate": .1, "victory_ending_hp_mean": 100, "visits": 1},
            {"model_action_id": "B", "win_rate": .5, "victory_ending_hp_mean": 40, "visits": 10000},
        ]}]
        self.assertEqual(aggregate_searches(evidence), "A")
        self.assertEqual(aggregate_searches(evidence, {**policy, "minimum_search_win_rate": .5}), "B")

    def test_family_specific_continuation_floor_does_not_change_external_gate(self):
        config = {"minimum_win_rate": {"boss": .88, "elite": .99}, "continuation_policy": {
            "mode": "win_floor_then_expected_hp", "minimum_search_win_rate": {"boss": .85, "elite": .95}}}
        evidence = [{"actions": [
            {"model_action_id": "A", "win_rate": .96, "victory_ending_hp_mean": 30},
            {"model_action_id": "B", "win_rate": .86, "victory_ending_hp_mean": 40},
        ]}]
        self.assertEqual(aggregate_searches(evidence, continuation_policy(config, {"encounter_family": "boss"})), "B")
        self.assertEqual(aggregate_searches(evidence, continuation_policy(config, {"encounter_family": "elite"})), "A")
        self.assertEqual(scoring_config(config, {"encounter_family": "boss"})["minimum_win_rate"], .88)
        with self.assertRaisesRegex(ValueError, "No continuation search threshold"):
            continuation_policy(config, {"encounter_family": "normal_weak"})
        for policy in ({"mode": "unknown"}, {"mode": "win_rate_then_hp", "minimum_search_win_rate": .9},
                       {"mode": "win_floor_then_expected_hp", "minimum_search_win_rate": True},
                       {"mode": "win_floor_then_expected_hp", "minimum_search_win_rate": {}},
                       {"mode": "win_floor_then_expected_hp", "minimum_search_win_rate": float("nan")}):
            with self.assertRaises(ValueError):
                validate_continuation_policy(policy)

    def test_choose_runs_configured_policy_on_search_results(self):
        from dataclasses import dataclass
        from types import SimpleNamespace
        from unittest.mock import Mock
        @dataclass
        class Action:
            model_action_id: str
            win_rate: float
            victory_ending_hp_mean: float
        env = Mock()
        env.search_model_actions.return_value = SimpleNamespace(
            model_actions=[Action("A", .96, 30), Action("B", .92, 40)],
            native_result=SimpleNamespace(hidden_order_fingerprint="x", simulations=2048,
                                         root_actions=[0, 1], minimum_root_action_visits=32, elapsed_ms=1))
        config = {"hidden_samples": 4, "seed": 1, "search_budget": 2048, "minimum_root_visits": 32}
        self.assertEqual(choose(env, config, "sample", 0, 1)[0], "A")
        config["continuation_policy"] = {"mode": "win_floor_then_expected_hp", "minimum_search_win_rate": .90}
        chosen, searches = choose(env, config, "sample", 0, 1)
        self.assertEqual(chosen, "B")
        self.assertEqual(len(searches), 4)


if __name__ == "__main__":
    unittest.main()
