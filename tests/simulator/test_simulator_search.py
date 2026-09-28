from __future__ import annotations

from copy import deepcopy
import unittest

from sts1_llm_policy.env.simulator_search import (
    parse_battle_scum_search_result,
    project_battle_scum_search_result,
)
from sts1_llm_policy.env.action_equivalence import (
    ModelActionClass,
    ModelActionSpace,
)
from sts1_llm_policy.env.action_schema import ActionType, CanonicalAction


def _response() -> dict:
    return {
        "source": "BattleScumSearcher2",
        "decision_id": 3,
        "simulations": 10,
        "search_seed": 99,
        "search_quality_normalization": "min_max_guarded_v1",
        "root_action_coverage_policy": "legacy_ucb_no_root_coverage_guarantee",
        "minimum_root_action_visits": 0,
        "hidden_order_seed": None,
        "hidden_order_policy": "exact_privileged_order_v1",
        "hidden_order_fingerprint": "10,7,3,",
        "root_simulation_count": 10,
        "root_actions": [
            {
                "action_id": "PLAY:0:0",
                "kind": "PLAY_CARD",
                "visits": 7,
                "evaluation_sum": 70.0,
                "evaluation_square_sum": 700.0,
                "mean_evaluation": 10.0,
                "terminal_wins": 5,
                "terminal_losses": 2,
                "win_rate": 5 / 7,
                "ending_hp_sum": 50,
                "ending_hp_mean": 50 / 7,
                "victory_ending_hp_sum": 50,
                "victory_ending_hp_mean": 10.0,
            },
            {
                "action_id": "END",
                "kind": "END_TURN",
                "visits": 3,
                "evaluation_sum": 15.0,
                "evaluation_square_sum": 75.0,
                "mean_evaluation": 5.0,
                "terminal_wins": 1,
                "terminal_losses": 2,
                "win_rate": 1 / 3,
                "ending_hp_sum": 5,
                "ending_hp_mean": 5 / 3,
                "victory_ending_hp_sum": 5,
                "victory_ending_hp_mean": 5.0,
            },
        ],
        "suggested_action_id": "PLAY:0:0",
        "selection_rule": "max_win_rate_then_victory_hp_then_visits_then_mean_evaluation_then_action_id",
        "best_sequence_first_action_id": "PLAY:0:0",
        "best_sequence_length": 4,
        "best_action_value": 100.0,
        "min_action_value": 1.0,
        "best_outcome_player_hp": 20,
        "elapsed_ms": 2.5,
        "objective": "upstream_battle_scum_evaluate_end_state_v1",
        "privileged_state": {
            "battle_context_copy": True,
            "ordered_draw_pile": True,
            "future_rng_state": True,
        },
        "transition_semantics": {
            "engine": "shared_corrected_action_execute_v1",
            "bridge_corrections_applied_inside_rollouts": True,
            "lagavulin_rejected": False,
            "upgraded_disarm_decks_rejected": False,
            "corrections_applied": {
                "lagavulin_natural_wake": 0,
                "upgraded_disarm": 0,
                "red_slaver_entangle_once": 0,
                "philosopher_bronze_orb_strength": 0,
                "burning_blood_victory_heal": 0,
            },
        },
    }


class SimulatorSearchResultTest(unittest.TestCase):
    def test_parses_complete_privileged_root_action_evidence(self) -> None:
        result = parse_battle_scum_search_result(
            _response(),
            current_decision_id=3,
            current_native_action_ids=("PLAY:0:0", "PLAY:1:0", "END"),
        )

        self.assertEqual(result.simulations, 10)
        self.assertEqual(result.suggested_action_id, "PLAY:0:0")
        self.assertEqual(sum(item.visits for item in result.root_actions), 10)
        self.assertEqual(result.root_actions[0].mean_evaluation, 10.0)
        self.assertIsNone(result.hidden_order_seed)
        self.assertEqual(result.hidden_order_policy, "exact_privileged_order_v1")

    def test_parses_seeded_hidden_order_replay_evidence(self) -> None:
        response = _response()
        response["hidden_order_seed"] = 701
        response["hidden_order_policy"] = (
            "shuffle_current_draw_pile_and_reseed_shuffle_rng_v1"
        )
        result = parse_battle_scum_search_result(
            response,
            current_decision_id=3,
            current_native_action_ids=("PLAY:0:0", "PLAY:1:0", "END"),
            requested_hidden_order_seed=701,
        )

        self.assertEqual(result.hidden_order_seed, 701)
        self.assertEqual(result.hidden_order_fingerprint, "10,7,3,")

    def test_parses_explicit_root_action_coverage_policy(self) -> None:
        response = _response()
        response["root_action_coverage_policy"] = "visit_every_root_edge_before_ucb_v1"
        response["minimum_root_action_visits"] = 1
        result = parse_battle_scum_search_result(
            response,
            current_decision_id=3,
            current_native_action_ids=("PLAY:0:0", "PLAY:1:0", "END"),
            requested_root_action_coverage=True,
        )
        self.assertEqual(
            result.root_action_coverage_policy,
            "visit_every_root_edge_before_ucb_v1",
        )

    def test_parses_minimum_root_action_visits_policy(self) -> None:
        response = _response()
        response["root_action_coverage_policy"] = (
            "minimum_root_edge_visits_before_ucb_v1"
        )
        response["minimum_root_action_visits"] = 128
        result = parse_battle_scum_search_result(
            response,
            current_decision_id=3,
            current_native_action_ids=("PLAY:0:0", "PLAY:1:0", "END"),
            requested_minimum_root_action_visits=128,
        )
        self.assertEqual(result.minimum_root_action_visits, 128)

    def test_projects_deduplicated_native_edges_to_all_model_classes(self) -> None:
        result = parse_battle_scum_search_result(
            _response(),
            current_decision_id=3,
            current_native_action_ids=("PLAY:0:0", "PLAY:1:0", "END"),
        )
        first_play = CanonicalAction(
            action_id="INTERNAL_0",
            action_type=ActionType.PLAY_CARD,
            hand_index=0,
            card_uuid="card-0",
            card_name="Strike",
            target_index=0,
        )
        duplicate_play = CanonicalAction(
            action_id="INTERNAL_1",
            action_type=ActionType.PLAY_CARD,
            hand_index=1,
            card_uuid="card-1",
            card_name="Strike",
            target_index=0,
        )
        end_turn = CanonicalAction(
            action_id="INTERNAL_2",
            action_type=ActionType.END_TURN,
        )
        action_space = ModelActionSpace(
            classes=(
                ModelActionClass(
                    model_action_id="ACTION_0",
                    action_type=ActionType.PLAY_CARD,
                    card=None,
                    target_index=0,
                    members=(first_play, duplicate_play),
                ),
                ModelActionClass(
                    model_action_id="ACTION_1",
                    action_type=ActionType.END_TURN,
                    card=None,
                    target_index=None,
                    members=(end_turn,),
                ),
            )
        )

        projected = project_battle_scum_search_result(
            result,
            native_action_ids=("PLAY:0:0", "PLAY:1:0", "END"),
            canonical_actions=(first_play, duplicate_play, end_turn),
            action_space=action_space,
        )

        self.assertEqual(
            tuple(item.model_action_id for item in projected.model_actions),
            ("ACTION_0", "ACTION_1"),
        )
        self.assertEqual(projected.suggested_model_action_id, "ACTION_0")
        self.assertEqual(
            projected.model_actions[0].native_action_ids,
            ("PLAY:0:0",),
        )

    def test_rejects_stale_illegal_or_inconsistent_search_evidence(self) -> None:
        cases: list[tuple[str, dict, str]] = []

        stale = deepcopy(_response())
        stale["decision_id"] = 4
        cases.append(("stale", stale, "stale"))

        illegal = deepcopy(_response())
        illegal["root_actions"][0]["action_id"] = "NOT_LEGAL"
        cases.append(("illegal", illegal, "outside"))

        visits = deepcopy(_response())
        visits["root_actions"][0]["visits"] = 8
        visits["root_actions"][0]["mean_evaluation"] = 8.75
        cases.append(("visits", visits, "visits"))

        privilege = deepcopy(_response())
        privilege["privileged_state"]["future_rng_state"] = False
        cases.append(("privilege", privilege, "privileged"))

        for name, response, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    parse_battle_scum_search_result(
                        response,
                        current_decision_id=3,
                        current_native_action_ids=("PLAY:0:0", "END"),
                    )


if __name__ == "__main__":
    unittest.main()
