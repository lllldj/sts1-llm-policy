from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from sts1_llm_policy.data.trajectory import TrajectoryLogger, iter_trajectory_records
from sts1_llm_policy.env.action_schema import ActionType, CanonicalAction
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.env.simulator_adapter import (
    D1_SCENARIOS,
    P0_SCENARIOS,
    SUPPORTED_DECK_PRESETS,
    SUPPORTED_SCENARIOS,
    SimulatorAdapterError,
    from_sts_lightspeed_response,
)
from sts1_llm_policy.env.simulator_env import (
    SimulatorCombatEndedError,
    StsLightspeedEnv,
)
from sts1_llm_policy.data.trajectory import classify_terminal_outcome
from sts1_llm_policy.eval.episode_runner import run_combat_episode
from sts1_llm_policy.policy.random_policy import RandomLegalPolicy

from tests.simulator.state_fixture import _card, _response


class SimulatorAdapterTest(unittest.TestCase):
    def test_maps_upgraded_bash_reachable_from_v3_campfire(self) -> None:
        response = _response()
        bash = _card(
            "BASH",
            "Bash",
            "Bash",
            "ATTACK",
            2,
            has_target=True,
            upgrades=1,
        )
        response["state"]["hand"] = [{**bash, "hand_index": 0}]
        response["state"]["draw_pile"] = []
        response["legal_actions"] = [
            {
                "action_id": "PLAY:0:0",
                "kind": "PLAY_CARD",
                "hand_index": 0,
                "target_index": 0,
                "card_id": "BASH",
                "card_name": "Bash",
                "cost_for_turn": 2,
            },
            {"action_id": "END", "kind": "END_TURN"},
        ]

        decision = from_sts_lightspeed_response(response)

        self.assertEqual(decision.state.combat.hand[0].upgrades, 1)
        self.assertEqual(
            decision.state.combat.hand[0].description,
            "Deal 10 base damage and apply 3 Vulnerable.",
        )

    def test_maps_upgraded_strike_and_defend_reachable_from_v3_campfire(self) -> None:
        response = _response()
        strike = _card(
            "STRIKE_RED", "Strike_R", "Strike", "ATTACK", 1,
            has_target=True, upgrades=1,
        )
        defend = _card(
            "DEFEND_RED", "Defend_R", "Defend", "SKILL", 1,
            has_target=False, upgrades=1,
        )
        response["state"]["hand"] = [
            {**strike, "hand_index": 0},
            {**defend, "hand_index": 1},
        ]
        response["state"]["draw_pile"] = []

        decision = from_sts_lightspeed_response(response)

        self.assertEqual(
            [card.description for card in decision.state.combat.hand],
            ["Deal 9 base damage.", "Gain 8 Block."],
        )


    def test_maps_bridge_state_and_authoritative_actions(self) -> None:
        decision = from_sts_lightspeed_response(_response())

        self.assertEqual(decision.state.seed, 41)
        self.assertEqual(decision.state.character, "IRONCLAD")
        self.assertEqual(decision.state.floor, 1)
        self.assertEqual(decision.deck_preset, "starter")
        self.assertEqual(decision.state.combat.turn, 1)
        monster = decision.state.combat.monsters[0]
        self.assertEqual(monster.monster_id, "Cultist")
        self.assertEqual(monster.intent, "BUFF")
        self.assertEqual(monster.move_id, 3)
        self.assertEqual(monster.move_base_damage, -1)
        self.assertEqual(monster.powers[0].power_id, "Ritual")

        self.assertEqual(
            [action.action_id for action in decision.actions],
            ["ACTION_0", "ACTION_1", "ACTION_2"],
        )
        self.assertEqual(decision.actions[0].action_type, ActionType.PLAY_CARD)
        self.assertEqual(decision.actions[0].target_index, 0)
        self.assertIsNone(decision.actions[1].target_index)
        self.assertEqual(decision.actions[2].action_type, ActionType.END_TURN)
        self.assertEqual(
            decision.native_action_id(decision.actions[0]),
            "PLAY:0:0",
        )

    def test_maps_fixed_deck_cards_powers_and_combat_accounting(self) -> None:
        response = _response()
        response["state"]["deck_preset"] = "boss_ready"
        response["state"]["combat_accounting"] = {
            "starting_hp": 80,
            "total_hp_loss": 15,
            "enemy_damage_taken": 12,
            "self_hp_loss": 3,
        }
        response["state"]["player"].update(
            current_hp=65,
            powers=[
                {"name": "Strength", "amount": 1},
                {"name": "Rage", "amount": 5},
                {"name": "No Draw", "amount": 1},
            ],
        )
        response["state"]["hand"] = [
            {
                **_card(
                    "BLOODLETTING",
                    "Bloodletting",
                    "Bloodletting",
                    "SKILL",
                    0,
                    has_target=False,
                    upgrades=1,
                ),
                "hand_index": 0,
            }
        ]
        response["state"]["draw_pile"] = [
            _card(
                "CARNAGE",
                "Carnage",
                "Carnage",
                "ATTACK",
                2,
                has_target=True,
                upgrades=1,
                ethereal=True,
            )
        ]
        response["legal_actions"] = [
            {
                "action_id": "PLAY:0:-1",
                "kind": "PLAY_CARD",
                "hand_index": 0,
                "target_index": None,
                "card_id": "BLOODLETTING",
                "card_name": "Bloodletting",
                "cost_for_turn": 0,
            },
            {"action_id": "END", "kind": "END_TURN"},
        ]

        decision = from_sts_lightspeed_response(response)
        accounting = decision.state.combat.accounting

        self.assertEqual(SUPPORTED_DECK_PRESETS, (
            "starter",
            "elite_transition",
            "boss_ready",
        ))
        self.assertEqual(decision.deck_preset, "boss_ready")
        self.assertIsNotNone(accounting)
        assert accounting is not None
        self.assertEqual(accounting.total_hp_loss, 15)
        self.assertEqual(accounting.enemy_damage_taken, 12)
        self.assertEqual(accounting.self_hp_loss, 3)
        self.assertEqual(decision.state.combat.hand[0].upgrades, 1)
        self.assertTrue(decision.state.combat.draw_pile[0].ethereal)
        self.assertEqual(
            [power.power_id for power in decision.state.combat.player.powers],
            ["Strength", "Rage", "No Draw"],
        )

    def test_maps_dynamic_loadout_metadata_new_card_and_power(self) -> None:
        response = _response()
        response["state"]["deck_preset"] = "boss_ready"
        response["state"]["loadout"] = {
            "loadout_id": "feature_boss-status_exhaust-000000000000002c",
            "recipe_tier": "feature_boss",
            "base_preset": "boss_ready",
            "feature_family": "status_exhaust",
            "loadout_seed": 44,
            "deck_hash": "a" * 64,
        }
        response["state"]["player"]["powers"] = [
            {"name": "Feel No Pain", "amount": 4}
        ]
        response["state"]["hand"] = [
            {
                **_card(
                    "POWER_THROUGH",
                    "Power Through",
                    "Power Through",
                    "SKILL",
                    1,
                    has_target=False,
                    upgrades=1,
                ),
                "hand_index": 0,
            }
        ]
        response["legal_actions"] = [
            {
                "action_id": "PLAY:0:-1",
                "kind": "PLAY_CARD",
                "hand_index": 0,
                "target_index": None,
                "card_id": "POWER_THROUGH",
                "card_name": "Power Through",
                "cost_for_turn": 1,
            },
            {"action_id": "END", "kind": "END_TURN"},
        ]

        decision = from_sts_lightspeed_response(response)

        self.assertEqual(decision.loadout_id, response["state"]["loadout"]["loadout_id"])
        self.assertEqual(decision.recipe_tier, "feature_boss")
        self.assertEqual(decision.deck_hash, "a" * 64)
        self.assertEqual(decision.state.combat.hand[0].card_id, "Power Through")
        self.assertEqual(
            decision.state.combat.player.powers[0].power_id,
            "Feel No Pain",
        )

    def test_internal_execution_keys_are_deterministic_and_not_serialized(self) -> None:
        first = from_sts_lightspeed_response(_response())
        second = from_sts_lightspeed_response(_response())

        self.assertEqual(first, second)
        observation = serialize_observation(first.state, first.actions)
        self.assertNotIn("sim:cultist", observation)
        self.assertNotIn("PLAY:0:0", observation)
        self.assertIn(
            "ACTION_1: PLAY Strike | COST 1 | UPGRADE 0"
            " -> TARGET_0 | COPIES 1",
            observation,
        )

    def test_bronze_orb_stasis_card_is_public_and_serialized(self) -> None:
        response = _response()
        response["state"]["scenario_id"] = "automaton"
        monster = response["state"]["monsters"][0]
        monster.update(
            {
                "id": "BRONZE_ORB",
                "name": "Bronze Orb",
                "move_name": "BRONZE_ORB_BEAM",
                "previous_move_name": "BRONZE_ORB_STASIS",
                "is_attacking": True,
                "base_damage": 8,
                "adjusted_damage": 8,
                "hits": 1,
                "powers": [
                    {"power_index": 0, "name": "Stasis", "amount": 1}
                ],
                "stasis_card": _card(
                    "CARNAGE", "Carnage", "Carnage", "ATTACK", 2,
                    has_target=True, upgrades=1, ethereal=True,
                ),
            }
        )

        decision = from_sts_lightspeed_response(response)
        held = decision.state.combat.monsters[0].stasis_card
        self.assertIsNotNone(held)
        assert held is not None
        self.assertEqual((held.card_id, held.upgrades), ("Carnage", 1))
        observation = serialize_observation(
            decision.state, decision.actions, version="observation_v4"
        )
        self.assertIn("STASIS Carnage+", observation)
        self.assertIn("Deal 28 base damage. ETHEREAL.", observation)

    def test_maps_jaw_worm_move_ids_to_communicationmod_semantics(self) -> None:
        cases = (
            ("JAW_WORM_BELLOW", False, "DEFEND_BUFF", 2),
            ("JAW_WORM_THRASH", True, "ATTACK_DEFEND", 3),
        )
        for move_name, is_attacking, expected_intent, expected_move_id in cases:
            with self.subTest(move_name=move_name):
                response = _response()
                response["state"]["scenario_id"] = "jaw_worm"
                monster = response["state"]["monsters"][0]
                monster.update(
                    {
                        "id": "JAW_WORM",
                        "name": "JAW_WORM",
                        "move_name": move_name,
                        "is_attacking": is_attacking,
                        "base_damage": 7 if is_attacking else 0,
                        "adjusted_damage": 7 if is_attacking else 0,
                        "hits": 1 if is_attacking else 0,
                        "powers": [],
                    }
                )

                canonical = from_sts_lightspeed_response(response).state.combat
                jaw_worm = canonical.monsters[0]
                self.assertEqual(jaw_worm.intent, expected_intent)
                self.assertEqual(jaw_worm.move_id, expected_move_id)

    def test_maps_bridge_move_history_to_behavior_without_rng_result(self) -> None:
        response = _response()
        response["state"]["scenario_id"] = "jaw_worm"
        monster = response["state"]["monsters"][0]
        monster.update(
            {
                "id": "JAW_WORM",
                "name": "JAW_WORM",
                "move_name": "JAW_WORM_THRASH",
                "previous_move_name": "JAW_WORM_THRASH",
                "is_attacking": True,
                "base_damage": 7,
                "adjusted_damage": 7,
                "hits": 1,
                "powers": [],
            }
        )

        parsed = from_sts_lightspeed_response(response).state.combat.monsters[0]

        self.assertIsNotNone(parsed.behavior)
        assert parsed.behavior is not None
        self.assertEqual(parsed.behavior.previous_move_id, 3)
        self.assertEqual(parsed.behavior.possible_next_move_ids, (1, 2))
        self.assertEqual(parsed.behavior.selection, "stochastic")

    def test_maps_difficulty_expansion_monsters_and_moves(self) -> None:
        self.assertEqual(
            SUPPORTED_SCENARIOS,
            P0_SCENARIOS
            + (
                "gremlin_gang",
                "gremlin_nob",
                "lagavulin",
                "three_sentries",
            ),
        )
        cases = (
            ("FAT_GREMLIN", "FAT_GREMLIN_SMASH", "GremlinFat", "ATTACK_DEBUFF", 1),
            ("GREMLIN_NOB", "GREMLIN_NOB_BELLOW", "GremlinNob", "BUFF", 3),
            ("GREMLIN_NOB", "GREMLIN_NOB_RUSH", "GremlinNob", "ATTACK", 1),
            ("GREMLIN_NOB", "GREMLIN_NOB_SKULL_BASH", "GremlinNob", "ATTACK_DEBUFF", 2),
            ("GREMLIN_WIZARD", "GREMLIN_WIZARD_CHARGING", "GremlinWizard", "UNKNOWN", 2),
            ("GREMLIN_WIZARD", "GREMLIN_WIZARD_ULTIMATE_BLAST", "GremlinWizard", "ATTACK", 1),
            ("LAGAVULIN", "LAGAVULIN_ATTACK", "Lagavulin", "ATTACK", 1),
            ("LAGAVULIN", "LAGAVULIN_SIPHON_SOUL", "Lagavulin", "STRONG_DEBUFF", 2),
            ("LAGAVULIN", "LAGAVULIN_SLEEP", "Lagavulin", "SLEEP", 3),
            ("MAD_GREMLIN", "MAD_GREMLIN_SCRATCH", "GremlinWarrior", "ATTACK", 1),
            ("SENTRY", "SENTRY_BEAM", "Sentry", "ATTACK", 1),
            ("SENTRY", "SENTRY_BOLT", "Sentry", "DEBUFF", 2),
            ("SHIELD_GREMLIN", "SHIELD_GREMLIN_PROTECT", "GremlinTsundere", "DEFEND", 1),
            ("SHIELD_GREMLIN", "SHIELD_GREMLIN_SHIELD_BASH", "GremlinTsundere", "ATTACK", 2),
            ("SNEAKY_GREMLIN", "SNEAKY_GREMLIN_PUNCTURE", "GremlinThief", "ATTACK", 1),
        )
        for native_id, move_name, monster_id, intent, move_id in cases:
            with self.subTest(move_name=move_name):
                response = _response()
                response["state"]["scenario_id"] = "gremlin_gang"
                monster = response["state"]["monsters"][0]
                attacking = "ATTACK" in intent
                monster.update(
                    {
                        "id": native_id,
                        "name": native_id,
                        "move_name": move_name,
                        "is_attacking": attacking,
                        "base_damage": 4 if attacking else 0,
                        "adjusted_damage": 4 if attacking else 0,
                        "hits": 1 if attacking else 0,
                        "powers": [],
                    }
                )

                canonical = from_sts_lightspeed_response(response).state.combat
                parsed = canonical.monsters[0]
                self.assertEqual(parsed.monster_id, monster_id)
                self.assertEqual(parsed.intent, intent)
                self.assertEqual(parsed.move_id, move_id)

    def test_lots_of_slimes_uses_supported_five_monster_multiset(self) -> None:
        self.assertIn("small_slimes", D1_SCENARIOS)
        self.assertIn("lots_of_slimes", D1_SCENARIOS)
        response = _response()
        response["state"]["scenario_id"] = "lots_of_slimes"
        template = response["state"]["monsters"][0]
        monsters = []
        for index, native_id in enumerate(
            ("SPIKE_SLIME_S", "ACID_SLIME_S", "SPIKE_SLIME_S", "ACID_SLIME_S", "SPIKE_SLIME_S")
        ):
            monster = deepcopy(template)
            monster.update(
                index=index,
                id=native_id,
                name=native_id,
                move_name=(
                    "SPIKE_SLIME_S_TACKLE"
                    if native_id == "SPIKE_SLIME_S"
                    else "ACID_SLIME_S_LICK"
                ),
                previous_move_name="INVALID",
                is_attacking=native_id == "SPIKE_SLIME_S",
                base_damage=5 if native_id == "SPIKE_SLIME_S" else 0,
                adjusted_damage=5 if native_id == "SPIKE_SLIME_S" else 0,
                hits=1 if native_id == "SPIKE_SLIME_S" else 0,
                powers=[],
            )
            monsters.append(monster)
        response["state"]["monsters"] = monsters

        combat = from_sts_lightspeed_response(response).state.combat

        self.assertEqual(len(combat.monsters), 5)
        self.assertEqual(
            sorted(monster.monster_id for monster in combat.monsters),
            ["AcidSlime_S", "AcidSlime_S", "SpikeSlime_S", "SpikeSlime_S", "SpikeSlime_S"],
        )
        self.assertTrue(
            all(monster.behavior is not None for monster in combat.monsters)
        )

    def test_maps_ascenders_bane_as_combat_setup_curse(self) -> None:
        response = _response()
        response["state"]["draw_pile"] = [
            {
                **_card(
                    "ASCENDERS_BANE",
                    "AscendersBane",
                    "Ascender's Bane",
                    "CURSE",
                    -3,
                    has_target=False,
                    is_playable=False,
                ),
                "ethereal": True,
            }
        ]

        card = from_sts_lightspeed_response(response).state.combat.draw_pile[0]

        self.assertEqual(card.card_id, "AscendersBane")
        self.assertFalse(card.is_playable)
        self.assertTrue(card.ethereal)

    def test_maps_slaver_looter_and_fungi_moves_and_public_powers(self) -> None:
        cases = (
            ("BLUE_SLAVER", "BLUE_SLAVER_RAKE", "SlaverBlue", "ATTACK_DEBUFF", 4),
            ("RED_SLAVER", "RED_SLAVER_ENTANGLE", "SlaverRed", "DEBUFF", 2),
            ("LOOTER", "LOOTER_SMOKE_BOMB", "Looter", "DEFEND", 2),
            ("LOOTER", "LOOTER_ESCAPE", "Looter", "ESCAPE", 3),
            ("FUNGI_BEAST", "FUNGI_BEAST_GROW", "FungiBeast", "BUFF", 2),
        )
        for native_id, move_name, monster_id, intent, move_id in cases:
            with self.subTest(move_name=move_name):
                response = _response()
                response["state"]["scenario_id"] = "exordium_thugs"
                monster = response["state"]["monsters"][0]
                attacking = "ATTACK" in intent
                monster.update(
                    id=native_id,
                    name=native_id,
                    move_name=move_name,
                    previous_move_name="INVALID",
                    has_used_entangle=native_id == "RED_SLAVER",
                    is_attacking=attacking,
                    base_damage=8 if attacking else 0,
                    adjusted_damage=8 if attacking else 0,
                    hits=1 if attacking else 0,
                    powers=(
                        [{"name": "Spore Cloud", "amount": 2}]
                        if native_id == "FUNGI_BEAST"
                        else [{"name": "Thievery", "amount": 15}]
                        if native_id == "LOOTER"
                        else []
                    ),
                )

                parsed = from_sts_lightspeed_response(response).state.combat.monsters[0]
                self.assertEqual((parsed.monster_id, parsed.intent, parsed.move_id), (monster_id, intent, move_id))
                self.assertIsNotNone(parsed.behavior)

    def test_maps_sentry_artifact_and_generated_dazed(self) -> None:
        response = _response()
        response["state"]["scenario_id"] = "three_sentries"
        monster = response["state"]["monsters"][0]
        monster.update(
            {
                "id": "SENTRY",
                "name": "SENTRY",
                "move_name": "SENTRY_BOLT",
                "powers": [
                    {
                        "name": "Artifact",
                        "amount": 1,
                    }
                ],
            }
        )
        response["state"]["discard_pile"] = [
            {
                **_card(
                    "DAZED",
                    "Dazed",
                    "Dazed",
                    "STATUS",
                    -2,
                    has_target=False,
                    is_playable=False,
                ),
                "ethereal": True,
            }
        ]

        combat = from_sts_lightspeed_response(response).state.combat

        self.assertEqual(combat.monsters[0].powers[0].power_id, "Artifact")
        self.assertEqual(combat.discard_pile[0].card_id, "Dazed")
        self.assertTrue(combat.discard_pile[0].ethereal)

    def test_rejects_forged_equal_action_object(self) -> None:
        decision = from_sts_lightspeed_response(_response())
        forged = CanonicalAction(**decision.actions[2].__dict__)

        with self.assertRaisesRegex(ValueError, "current simulator legal set"):
            decision.native_action_id(forged)

    def test_rejects_native_action_field_disagreement(self) -> None:
        response = _response()
        response["legal_actions"][0]["action_id"] = "PLAY:1:0"

        with self.assertRaisesRegex(
            SimulatorAdapterError,
            "action_id does not match action fields",
        ):
            from_sts_lightspeed_response(response)

    def test_rejects_unsupported_move(self) -> None:
        response = _response()
        response["state"]["monsters"][0]["move_name"] = "UNKNOWN_MOVE"

        with self.assertRaisesRegex(SimulatorAdapterError, "outside support"):
            from_sts_lightspeed_response(response)

    def test_terminal_requires_empty_actions_and_consistent_reward(self) -> None:
        terminal = _response(
            decision_id=4,
            terminal=True,
            outcome="PLAYER_VICTORY",
        )
        decision = from_sts_lightspeed_response(terminal)
        self.assertTrue(decision.terminal)
        self.assertEqual(decision.reward, 1.0)
        self.assertEqual(decision.actions, ())

        terminal["legal_actions"] = [{"action_id": "END", "kind": "END_TURN"}]
        with self.assertRaisesRegex(SimulatorAdapterError, "Terminal state"):
            from_sts_lightspeed_response(terminal)


class _FakeBridgeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.reset_calls: list[dict[str, object]] = []
        self.step_calls: list[tuple[int, str]] = []
        self.search_calls: list[tuple[int, int, int]] = []
        self.closed = False

    def reset(
        self,
        scenario_id: str,
        seed: int,
        *,
        ascension: int = 0,
        player_current_hp: int | None = None,
        deck_preset: str | None = None,
        deck_spec: dict[str, object] | None = None,
        combat_snapshot: dict[str, object] | None = None,
    ) -> dict:
        self.reset_calls.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "ascension": ascension,
                "player_current_hp": player_current_hp,
                "deck_preset": deck_preset,
                "deck_spec": deck_spec,
                "combat_snapshot": combat_snapshot,
            }
        )
        return deepcopy(self.responses[0])

    def step(self, decision_id: int, action_id: str) -> dict:
        self.step_calls.append((decision_id, action_id))
        return deepcopy(self.responses[len(self.step_calls)])

    def search(
        self,
        decision_id: int,
        *,
        simulations: int,
        search_seed: int,
        hidden_order_seed: int | None = None,
        ensure_root_action_coverage: bool = False,
        minimum_root_action_visits: int = 0,
    ) -> dict:
        if hidden_order_seed is not None:
            raise AssertionError("This fixture only covers exact privileged order")
        self.search_calls.append((decision_id, simulations, search_seed))
        return {
            "source": "BattleScumSearcher2",
            "decision_id": decision_id,
            "simulations": simulations,
            "search_seed": search_seed,
            "search_quality_normalization": "min_max_guarded_v1",
            "root_action_coverage_policy": (
                "minimum_root_edge_visits_before_ucb_v1"
                if minimum_root_action_visits > 1
                else (
                    "visit_every_root_edge_before_ucb_v1"
                    if ensure_root_action_coverage or minimum_root_action_visits == 1
                    else "legacy_ucb_no_root_coverage_guarantee"
                )
            ),
            "minimum_root_action_visits": max(
                minimum_root_action_visits,
                1 if ensure_root_action_coverage else 0,
            ),
            "root_simulation_count": simulations,
            "root_actions": [
                {
                    "action_id": "PLAY:0:0",
                    "kind": "PLAY_CARD",
                    "visits": simulations,
                    "evaluation_sum": float(simulations),
                    "evaluation_square_sum": float(simulations),
                    "mean_evaluation": 1.0,
                    "terminal_wins": simulations,
                    "terminal_losses": 0,
                    "win_rate": 1.0,
                    "ending_hp_sum": simulations,
                    "ending_hp_mean": 1.0,
                    "victory_ending_hp_sum": simulations,
                    "victory_ending_hp_mean": 1.0,
                }
            ],
            "suggested_action_id": "PLAY:0:0",
            "selection_rule": (
                "max_win_rate_then_victory_hp_then_visits_then_mean_evaluation_then_action_id"
            ),
            "best_sequence_first_action_id": "PLAY:0:0",
            "best_sequence_length": 1,
            "best_action_value": 1.0,
            "min_action_value": 1.0,
            "best_outcome_player_hp": 1,
            "elapsed_ms": 0.1,
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

    def close(self) -> None:
        self.closed = True


class SimulatorEnvTest(unittest.TestCase):
    def test_reset_forwards_combat_snapshot_without_duplicate_run_state(self) -> None:
        client = _FakeBridgeClient([_response()])
        env = StsLightspeedEnv(client)
        snapshot = {"schema_version": "combat_snapshot_v1"}

        env.reset("cultist", 41, combat_snapshot=snapshot)

        self.assertEqual(client.reset_calls[0]["combat_snapshot"], snapshot)
        self.assertIsNone(client.reset_calls[0]["deck_preset"])
        self.assertIsNone(client.reset_calls[0]["deck_spec"])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            env.reset("cultist", 41, deck_preset="starter", combat_snapshot=snapshot)

    def test_search_uses_current_decision_without_advancing_environment(self) -> None:
        client = _FakeBridgeClient([_response()])
        env = StsLightspeedEnv(client)
        state = env.reset("cultist", 41)

        result = env.search(simulations=8, search_seed=17)

        self.assertEqual(result.suggested_action_id, "PLAY:0:0")
        self.assertEqual(client.search_calls, [(0, 8, 17)])
        self.assertIs(env.get_state(), state)
        self.assertEqual(client.step_calls, [])

    def test_reset_step_and_raw_snapshot(self) -> None:
        next_response = _response(decision_id=1)
        next_response["state"]["player"]["energy"] = 2
        client = _FakeBridgeClient([_response(), next_response])
        env = StsLightspeedEnv(client)

        state = env.reset("cultist", 41)
        action = env.legal_actions()[0]
        raw = env.get_raw_state()
        raw["state"]["seed"] = 999
        next_state = env.step(action)

        self.assertEqual(state.seed, 41)
        self.assertEqual(env.get_raw_state()["state"]["seed"], 41)
        self.assertEqual(client.step_calls, [(0, "PLAY:0:0")])
        self.assertEqual(next_state.combat.player.energy, 2)

    def test_rejects_stale_action_before_calling_bridge(self) -> None:
        next_response = _response(decision_id=1)
        client = _FakeBridgeClient([_response(), next_response])
        env = StsLightspeedEnv(client)
        env.reset("cultist", 41)
        old_end = env.legal_actions()[-1]
        env.step(env.legal_actions()[0])

        with self.assertRaisesRegex(ValueError, "current simulator legal set"):
            env.step(old_end)

        self.assertEqual(len(client.step_calls), 1)

    def test_terminal_exposes_source_outcome_and_reward(self) -> None:
        terminal = _response(
            decision_id=1,
            terminal=True,
            outcome="PLAYER_VICTORY",
        )
        client = _FakeBridgeClient([_response(), terminal])
        env = StsLightspeedEnv(client)
        env.reset("cultist", 41)

        with self.assertRaises(SimulatorCombatEndedError) as caught:
            env.step(env.legal_actions()[0])

        self.assertEqual(caught.exception.reward, 1.0)
        self.assertEqual(caught.exception.raw_state["source"], "simulator")
        self.assertEqual(
            classify_terminal_outcome(caught.exception.raw_state).value,
            "victory",
        )

    def test_close_delegates_to_client(self) -> None:
        client = _FakeBridgeClient([_response()])
        env = StsLightspeedEnv(client)
        env.close()
        self.assertTrue(client.closed)

    def test_existing_episode_runner_records_simulator_terminal_reward(self) -> None:
        terminal = _response(
            decision_id=1,
            terminal=True,
            outcome="PLAYER_VICTORY",
        )
        env = StsLightspeedEnv(_FakeBridgeClient([_response(), terminal]))
        env.reset("cultist", 41)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simulator.jsonl"
            logger = TrajectoryLogger(
                path,
                episode_id="simulator-test",
                game_seed=41,
                policy_seed=7,
                policy_name="random_legal_v1",
                encounter="cultist",
            )
            summary = run_combat_episode(
                env,
                RandomLegalPolicy(policy_seed=7),
                logger,
            )
            record = list(iter_trajectory_records(path))[0]

        self.assertEqual(summary.outcome.value, "victory")
        self.assertEqual(record["reward"], 1.0)
        self.assertEqual(record["next_raw_state"]["source"], "simulator")


if __name__ == "__main__":
    unittest.main()
