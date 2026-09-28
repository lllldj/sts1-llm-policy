import unittest
from pathlib import Path

from sts1_llm_policy.eval.real_game_preflight import (
    RealGamePreflightError,
    evaluate_p0_clean_preflight,
    load_p0_clean_capture_config,
    require_p0_clean_preflight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "live" / "real_game_p0_clean_capture_v1.json"


def _card(card_id: str) -> dict:
    return {"id": card_id, "upgrades": 0}


def _clean_two_louse_state(
    monster_ids: tuple[str, str] = (
        "FuzzyLouseDefensive",
        "FuzzyLouseNormal",
    ),
) -> dict:
    deck = (
        [_card("Strike_R") for _ in range(5)]
        + [_card("Defend_R") for _ in range(4)]
        + [_card("Bash")]
    )
    return {
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "seed": 41,
            "class": "IRONCLAD",
            "ascension_level": 0,
            "act": 1,
            "floor": 1,
            "current_hp": 80,
            "max_hp": 80,
            "room_type": "MonsterRoom",
            "room_phase": "COMBAT",
            "action_phase": "WAITING_ON_USER",
            "deck": deck,
            "relics": [{"id": "Burning Blood"}],
            "potions": [{"id": "Potion Slot"} for _ in range(3)],
            "combat_state": {
                "turn": 1,
                "player": {
                    "current_hp": 80,
                    "max_hp": 80,
                    "block": 0,
                    "energy": 3,
                    "powers": [],
                },
                "monsters": [
                    {"id": monster_id}
                    for monster_id in monster_ids
                ],
                "hand": deck[:5],
                "draw_pile": deck[5:],
                "discard_pile": [],
                "exhaust_pile": [],
                "limbo": [],
            },
        },
    }


class RealGamePreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_p0_clean_capture_config(CONFIG_PATH)

    def test_accepts_exact_clean_two_louse_opening(self) -> None:
        report = require_p0_clean_preflight(
            _clean_two_louse_state(),
            self.config,
            "two_louse",
        )

        self.assertTrue(report["accepted"])
        self.assertEqual(report["evidence_class"], "parity_clean")
        self.assertEqual(report["scenario_id"], "two_louse")
        self.assertEqual(report["mismatches"], [])

    def test_accepts_all_seeded_two_louse_compositions(self) -> None:
        valid_compositions = (
            ("FuzzyLouseDefensive", "FuzzyLouseDefensive"),
            ("FuzzyLouseNormal", "FuzzyLouseNormal"),
            ("FuzzyLouseDefensive", "FuzzyLouseNormal"),
            ("FuzzyLouseNormal", "FuzzyLouseDefensive"),
        )

        for monster_ids in valid_compositions:
            with self.subTest(monster_ids=monster_ids):
                report = require_p0_clean_preflight(
                    _clean_two_louse_state(monster_ids),
                    self.config,
                    "two_louse",
                )
                self.assertTrue(report["accepted"])

    def test_rejects_wrong_encounter_before_policy_execution(self) -> None:
        state = _clean_two_louse_state()
        state["game_state"]["combat_state"]["monsters"] = [
            {"id": "JawWorm"}
        ]

        with self.assertRaisesRegex(
            RealGamePreflightError,
            r"combat_state\.monsters\[\]\.id",
        ):
            require_p0_clean_preflight(state, self.config, "two_louse")

    def test_reports_every_loadout_and_initial_state_mismatch(self) -> None:
        state = _clean_two_louse_state()
        state["game_state"]["ascension_level"] = 1
        state["game_state"]["floor"] = 3
        state["game_state"]["deck"].append(_card("Carnage"))
        state["game_state"]["relics"].append({"id": "Vajra"})
        state["game_state"]["combat_state"]["player"]["current_hp"] = 70
        state["game_state"]["potions"][0] = {"id": "Fire Potion"}

        report = evaluate_p0_clean_preflight(
            state,
            self.config,
            "two_louse",
        )
        fields = {item["field"] for item in report["mismatches"]}

        self.assertFalse(report["accepted"])
        self.assertIn("game_state.ascension_level", fields)
        self.assertIn("game_state.floor", fields)
        self.assertIn("game_state.deck", fields)
        self.assertIn("game_state.relics[].id", fields)
        self.assertIn("combat_state.player.current_hp", fields)
        self.assertIn("game_state.actual_potions", fields)

    def test_rejects_neows_lament_even_for_valid_same_color_louse(self) -> None:
        state = _clean_two_louse_state(
            ("FuzzyLouseDefensive", "FuzzyLouseDefensive")
        )
        state["game_state"]["relics"].append({"id": "NeowsBlessing"})

        report = evaluate_p0_clean_preflight(
            state,
            self.config,
            "two_louse",
        )

        self.assertFalse(report["accepted"])
        self.assertEqual(
            [item["field"] for item in report["mismatches"]],
            ["game_state.relics[].id"],
        )

    def test_rejects_unknown_scenario(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown P0 clean scenario"):
            evaluate_p0_clean_preflight(
                _clean_two_louse_state(),
                self.config,
                "gremlin_nob",
            )


if __name__ == "__main__":
    unittest.main()
