import unittest

from sts1_llm_policy.env.live_id_crosswalk import (
    canonicalize_monster_id,
    canonicalize_move_id,
    canonicalize_move_intent,
)
from sts1_llm_policy.env.live_description_fallback import resolve_power_effect
from sts1_llm_policy.env.mechanics_catalog import describe_move
from sts1_llm_policy.env.monster_behavior import UnsupportedMonsterBehaviorError
from sts1_llm_policy.env.real_game_adapter import (
    from_communication_state,
)
from sts1_llm_policy.env.serializer import serialize_state

from tests.live.fixtures import _raw_state


class RealGameAdapterPowerTest(unittest.TestCase):
    def test_reports_apology_slime_as_invalid_basemod_encounter(self) -> None:
        raw = _raw_state()
        raw["game_state"]["combat_state"]["player"]["powers"] = []
        raw["game_state"]["combat_state"]["monsters"][0].update(
            {
                "id": "Apology Slime",
                "name": "Apology Slime",
                "move_id": 1,
                "last_move_id": -1,
                "powers": [],
            }
        )

        with self.assertRaisesRegex(
            UnsupportedMonsterBehaviorError,
            "invalid case-sensitive fight encounter ID",
        ):
            from_communication_state(raw)

    def test_normalizes_audited_native_monster_ids(self) -> None:
        self.assertEqual(canonicalize_monster_id("Champ"), "TheChamp")
        self.assertEqual(canonicalize_monster_id("Healer"), "Mystic")
        self.assertEqual(
            canonicalize_monster_id("Shelled Parasite"),
            "ShelledParasite",
        )
        self.assertEqual(canonicalize_monster_id("SlaverBoss"), "Taskmaster")

    def test_normalizes_other_audited_native_move_ids(self) -> None:
        expected = {
            "AcidSlime_S": {2: 4},
            "BronzeOrb": {2: 3, 3: 2},
            "GremlinFat": {2: 1},
            "Lagavulin": {1: 2, 3: 1, 4: 3, 5: 3, 6: 3},
            "SlimeBoss": {1: 2, 2: 1},
            "Taskmaster": {2: 1},
            "TheCollector": {1: 5},
        }
        for monster_id, mapping in expected.items():
            with self.subTest(monster_id=monster_id):
                self.assertEqual(
                    {
                        native: canonicalize_move_id(monster_id, native)
                        for native in mapping
                    },
                    mapping,
                )

    def test_normalizes_all_guardian_native_move_ids(self) -> None:
        self.assertEqual(
            [canonicalize_move_id("TheGuardian", value) for value in range(1, 8)],
            [5, 2, 6, 7, 4, 1, 3],
        )

    def test_normalizes_guardian_defensive_mode_native_intent(self) -> None:
        self.assertEqual(
            canonicalize_move_intent("TheGuardian", 1, "BUFF"),
            "DEFEND",
        )

        raw_state = _raw_state()
        monster = raw_state["game_state"]["combat_state"]["monsters"][0]
        monster.update(
            {
                "id": "TheGuardian",
                "name": "The Guardian",
                "current_hp": 210,
                "max_hp": 240,
                "block": 20,
                "intent": "BUFF",
                "move_id": 1,
                "last_move_id": 7,
                "second_last_move_id": 2,
                "move_hits": 1,
                "move_base_damage": -1,
                "move_adjusted_damage": -1,
                "powers": [],
            }
        )

        state = from_communication_state(raw_state)
        guardian = state.combat.monsters[0]
        self.assertEqual(guardian.move_id, 5)
        self.assertEqual(guardian.intent, "DEFEND")
        self.assertEqual(guardian.behavior.phase, "defensive_mode")
        self.assertIn("Defensive Mode", describe_move(guardian))

    def test_normalizes_guardian_twin_slam_native_intent(self) -> None:
        self.assertEqual(
            canonicalize_move_intent("TheGuardian", 4, "ATTACK_BUFF"),
            "ATTACK",
        )

        raw_state = _raw_state()
        monster = raw_state["game_state"]["combat_state"]["monsters"][0]
        monster.update(
            {
                "id": "TheGuardian",
                "name": "The Guardian",
                "current_hp": 164,
                "max_hp": 240,
                "intent": "ATTACK_BUFF",
                "move_id": 4,
                "last_move_id": 3,
                "second_last_move_id": 1,
                "move_hits": 2,
                "move_base_damage": 8,
                "move_adjusted_damage": 6,
                "powers": [
                    {"id": "Sharp Hide", "name": "Sharp Hide", "amount": 3}
                ],
            }
        )

        state = from_communication_state(raw_state)
        guardian = state.combat.monsters[0]
        self.assertEqual(guardian.move_id, 7)
        self.assertEqual(guardian.intent, "ATTACK")
        self.assertEqual(guardian.behavior.phase, "defensive_mode")
        self.assertIn("twice", describe_move(guardian))

    def test_normalizes_red_slaver_entangle_native_intent(self) -> None:
        self.assertEqual(
            canonicalize_move_intent("SlaverRed", 2, "STRONG_DEBUFF"),
            "DEBUFF",
        )

        raw_state = _raw_state()
        monster = raw_state["game_state"]["combat_state"]["monsters"][0]
        monster.update(
            {
                "id": "SlaverRed",
                "name": "Slaver",
                "intent": "STRONG_DEBUFF",
                "move_id": 2,
                "last_move_id": 3,
                "second_last_move_id": 1,
                "move_hits": 1,
                "move_base_damage": -1,
                "move_adjusted_damage": -1,
                "powers": [],
            }
        )

        state = from_communication_state(raw_state)
        slaver = state.combat.monsters[0]
        self.assertEqual(slaver.intent, "DEBUFF")
        self.assertIn("Entangled", describe_move(slaver))

    def test_normalizes_sentry_native_move_ids_and_serializes_bolt(self) -> None:
        self.assertEqual(canonicalize_move_id("Sentry", 3), 2)
        self.assertEqual(canonicalize_move_id("Sentry", 4), 1)

        raw_state = _raw_state()
        monster = raw_state["game_state"]["combat_state"]["monsters"][0]
        monster.update(
            {
                "id": "Sentry",
                "name": "Sentry",
                "intent": "DEBUFF",
                "move_id": 3,
                "last_move_id": 4,
                "move_base_damage": -1,
                "move_adjusted_damage": -1,
            }
        )

        state = from_communication_state(raw_state)
        sentry = state.combat.monsters[0]

        self.assertEqual(sentry.move_id, 2)
        self.assertEqual(sentry.behavior.previous_move_id, 1)
        self.assertEqual(sentry.behavior.possible_next_move_ids, (1,))
        self.assertIn("Dazed", describe_move(sentry))

    def test_normalizes_communication_relic_display_id(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["relics"] = [
            {
                "id": "Burning Blood",
                "name": "Burning Blood",
                "counter": -1,
                "description": "At the end of combat, heal 6 HP.",
            }
        ]

        state = from_communication_state(raw_state)

        self.assertEqual(state.relics[0].relic_id, "BURNING_BLOOD")
        self.assertEqual(state.relics[0].name, "Burning Blood")
        self.assertEqual(state.relics[0].counter, -1)
        self.assertEqual(
            state.relics[0].source_description,
            "At the end of combat, heal 6 HP.",
        )

    def test_normalizes_possessive_relic_display_id(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["relics"] = [
            {
                "id": "Philosopher's Stone",
                "name": "Philosopher's Stone",
            }
        ]

        state = from_communication_state(raw_state)

        self.assertEqual(state.relics[0].relic_id, "PHILOSOPHERS_STONE")

    def test_parses_real_cultist_ritual_payload(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["player"][
            "powers"
        ] = []
        raw_state["game_state"]["combat_state"]["monsters"][
            0
        ]["powers"] = [
            {
                "amount": 3,
                "just_applied": False,
                "name": "Ritual",
                "id": "Ritual",
            }
        ]

        state = from_communication_state(raw_state)
        ritual = state.combat.monsters[0].powers[0]

        self.assertEqual(ritual.power_id, "Ritual")
        self.assertEqual(ritual.name, "Ritual")
        self.assertEqual(ritual.amount, 3)
        self.assertFalse(ritual.just_applied)

    def test_builds_behavior_from_real_move_history(self) -> None:
        state = from_communication_state(_raw_state())
        behavior = state.combat.monsters[0].behavior

        self.assertIsNotNone(behavior)
        self.assertEqual(behavior.previous_move_id, 3)
        self.assertEqual(behavior.possible_next_move_ids, (1,))

    def test_missing_real_move_history_is_first_turn_safe(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["turn"] = 1
        del raw_state["game_state"]["combat_state"]["monsters"][0][
            "last_move_id"
        ]

        state = from_communication_state(raw_state)

        self.assertIsNone(
            state.combat.monsters[0].behavior.previous_move_id
        )

    def test_normalizes_guardian_native_move_ids_and_history(self) -> None:
        raw_state = _raw_state()
        monster = raw_state["game_state"]["combat_state"]["monsters"][0]
        monster.update(
            {
                "id": "TheGuardian",
                "name": "The Guardian",
                "intent": "DEFEND",
                "move_id": 6,
                "last_move_id": 5,
                "move_base_damage": -1,
                "move_adjusted_damage": -1,
            }
        )

        state = from_communication_state(raw_state)
        guardian = state.combat.monsters[0]

        self.assertEqual(guardian.move_id, 1)
        self.assertEqual(guardian.behavior.previous_move_id, 4)
        self.assertEqual(guardian.behavior.possible_next_move_ids, (2,))
        self.assertIn("Gain 9 Block", describe_move(guardian))

    def test_normalizes_lagavulin_wake_turn_semantics(self) -> None:
        raw_state = _raw_state()
        monster = raw_state["game_state"]["combat_state"]["monsters"][0]
        monster.update(
            {
                "id": "Lagavulin",
                "name": "Lagavulin",
                "intent": "STUN",
                "move_id": 4,
                "last_move_id": 5,
                "move_base_damage": -1,
                "move_adjusted_damage": -1,
                "powers": [],
            }
        )

        state = from_communication_state(raw_state)
        lagavulin = state.combat.monsters[0]

        self.assertEqual(lagavulin.move_id, 3)
        self.assertEqual(lagavulin.intent, "SLEEP")
        self.assertEqual(lagavulin.behavior.phase, "waking")
        self.assertEqual(lagavulin.behavior.previous_move_id, 3)
        self.assertIn("asleep", describe_move(lagavulin).lower())

    def test_parses_required_and_optional_power_fields(self) -> None:
        state = from_communication_state(_raw_state())

        strength, nightmare = state.combat.player.powers
        ritual = state.combat.monsters[0].powers[0]

        self.assertEqual(strength.power_id, "Strength")
        self.assertEqual(strength.amount, 2)
        self.assertEqual(strength.damage, 0)
        self.assertEqual(strength.misc, 0)
        self.assertFalse(strength.just_applied)
        self.assertIsNone(strength.card)

        self.assertEqual(nightmare.damage, 4)
        self.assertEqual(nightmare.misc, 3)
        self.assertTrue(nightmare.just_applied)
        self.assertIsNotNone(nightmare.card)
        self.assertEqual(nightmare.card.name, "Strike")

        self.assertEqual(ritual.power_id, "Ritual")
        self.assertTrue(ritual.just_applied)

    def test_normalizes_communication_weak_power_id(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["player"]["powers"] = [
            {
                "id": "Weakened",
                "name": "Weakened",
                "amount": 2,
            }
        ]

        state = from_communication_state(raw_state)
        weak = state.combat.player.powers[0]

        self.assertEqual(weak.power_id, "Weak")
        self.assertEqual(weak.name, "Weakened")
        self.assertEqual(weak.amount, 2)

    def test_normalizes_native_gremlin_nob_anger_power_id(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["player"]["powers"] = []
        monster = raw_state["game_state"]["combat_state"]["monsters"][0]
        monster["powers"] = [
            {
                "id": "Anger",
                "name": "Enrage",
                "amount": 2,
            }
        ]

        state = from_communication_state(raw_state)

        self.assertEqual(state.combat.monsters[0].powers[0].power_id, "Enrage")

    def test_normalizes_communication_flex_power_id(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["player"]["powers"] = [
            {
                "id": "Flex",
                "name": "Strength Down",
                "amount": 2,
            }
        ]

        state = from_communication_state(raw_state)
        strength_down = state.combat.player.powers[0]

        self.assertEqual(strength_down.power_id, "Lose Strength")
        self.assertEqual(strength_down.name, "Strength Down")
        self.assertEqual(strength_down.amount, 2)

    def test_normalizes_communication_confusion_power_id(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["player"]["powers"] = [
            {
                "id": "Confusion",
                "name": "Confused",
                "amount": -1,
            }
        ]

        state = from_communication_state(raw_state)

        self.assertEqual(state.combat.player.powers[0].power_id, "Confused")

    def test_normalizes_formatting_for_complete_power_catalog(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["player"]["powers"] = [
            {
                "id": "FeelNoPain",
                "name": "Feel No Pain",
                "amount": 3,
            }
        ]

        state = from_communication_state(raw_state)

        self.assertEqual(
            state.combat.player.powers[0].power_id,
            "Feel No Pain",
        )

    def test_accepts_native_large_slime_split_power(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["player"]["powers"] = []
        monster = raw_state["game_state"]["combat_state"]["monsters"][0]
        monster.update(
            {
                "id": "AcidSlime_L",
                "name": "Acid Slime (L)",
                "intent": "ATTACK",
                "move_id": 2,
                "last_move_id": 1,
                "move_base_damage": 16,
                "move_adjusted_damage": 16,
                "powers": [
                    {
                        "id": "Split",
                        "name": "Split",
                        "amount": -1,
                    }
                ],
            }
        )

        state = from_communication_state(raw_state)
        split = state.combat.monsters[0].powers[0]

        self.assertEqual(split.power_id, "Split")
        self.assertIn(
            "half HP",
            resolve_power_effect(split, allow_source_description=True),
        )
        serialized = serialize_state(
            state,
            version="observation_v5",
            allow_source_descriptions=True,
        )
        self.assertIn("Split(-1):", serialized)
        self.assertIn("schedule Split after falling to half HP", serialized)

    def test_rejects_invalid_power_amount_type(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["player"][
            "powers"
        ][0]["amount"] = True

        with self.assertRaisesRegex(
            ValueError,
            r"combat_state\.player\.powers\[0\]\.amount "
            r"must be an int",
        ):
            from_communication_state(raw_state)

    def test_rejects_non_object_power(self) -> None:
        raw_state = _raw_state()
        raw_state["game_state"]["combat_state"]["monsters"][
            0
        ]["powers"] = ["Ritual"]

        with self.assertRaisesRegex(
            ValueError,
            r"combat_state\.monsters\[\]\.powers\[0\] "
            r"must be a dict",
        ):
            from_communication_state(raw_state)


if __name__ == "__main__":
    unittest.main()
