from __future__ import annotations

import unittest

from sts1_llm_policy.env.feature_loadout import (
    BASE_PRESET_CARDS,
    FEATURE_FAMILY_ANCHORS,
    FEATURE_RECIPES,
    POOL_BY_ID,
    RANDOM_MIXED_FEATURE_FAMILY,
    build_feature_loadout,
    validate_deck_spec,
)


class FeatureLoadoutTest(unittest.TestCase):


    def test_all_tiers_and_families_are_deterministic_and_valid(self) -> None:
        for tier, recipe in FEATURE_RECIPES.items():
            for offset, family in enumerate(FEATURE_FAMILY_ANCHORS):
                with self.subTest(tier=tier, family=family):
                    first = build_feature_loadout(tier, family, 1000 + offset)
                    second = build_feature_loadout(tier, family, 1000 + offset)
                    self.assertEqual(first, second)
                    self.assertEqual(sum(card.count for card in first.cards), recipe.final_size)
                    self.assertEqual(len(first.additions), recipe.addition_count)
                    self.assertEqual(
                        sum(card.upgrades for card in first.additions),
                        recipe.upgraded_addition_count,
                    )
                    self.assertEqual(len(first.deck_hash), 64)
                    self.assertEqual(validate_deck_spec(first.as_deck_spec()), first)

    def test_elite_and_boss_apply_only_the_new_cumulative_removal(self) -> None:
        elite = build_feature_loadout("feature_elite", "status_exhaust", 7)
        boss = build_feature_loadout("feature_boss", "status_exhaust", 7)
        elite_counts = {card.card_id: card.count for card in elite.cards}
        boss_counts = {card.card_id: card.count for card in boss.cards}
        self.assertEqual(elite_counts["Strike_R"], 4)
        self.assertEqual(elite_counts["Defend_R"], 4)
        self.assertEqual(boss_counts["Strike_R"], 4)
        self.assertEqual(boss_counts["Defend_R"], 3)
        self.assertEqual(boss.relics, ("Burning Blood", "Vajra"))

    def test_additions_do_not_duplicate_cards_already_in_base_preset(self) -> None:
        for tier in FEATURE_RECIPES:
            loadout = build_feature_loadout(tier, "vulnerable_burst", 91)
            inherited_ids = {
                card.card_id for card in BASE_PRESET_CARDS[loadout.base_preset]
            }
            self.assertTrue(
                inherited_ids.isdisjoint(card.card_id for card in loadout.additions)
            )

    def test_random_mixed_loadouts_are_deterministic_and_role_diverse(self) -> None:
        required_roles = {
            "feature_basic": 2,
            "feature_elite": 3,
            "feature_boss": 4,
        }
        for tier, role_count in required_roles.items():
            loadouts = [
                build_feature_loadout(tier, RANDOM_MIXED_FEATURE_FAMILY, seed)
                for seed in range(5000, 5005)
            ]
            with self.subTest(tier=tier):
                self.assertEqual(len({item.deck_hash for item in loadouts}), 5)
                self.assertEqual(
                    loadouts[0],
                    build_feature_loadout(tier, RANDOM_MIXED_FEATURE_FAMILY, 5000),
                )
                for loadout in loadouts:
                    self.assertEqual(
                        len({POOL_BY_ID[card.card_id].role for card in loadout.additions}),
                        role_count,
                    )
                    self.assertEqual(validate_deck_spec(loadout.as_deck_spec()), loadout)


    def test_tampered_spec_is_rejected(self) -> None:
        payload = build_feature_loadout(
            "feature_basic", "block_to_damage", 9
        ).as_deck_spec()
        payload["deck_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_deck_spec(payload)


if __name__ == "__main__":
    unittest.main()
