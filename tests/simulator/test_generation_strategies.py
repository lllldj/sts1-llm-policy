"""Strategy selection and route execution without native simulation."""
from copy import deepcopy
from dataclasses import replace
import random
import unittest
from unittest.mock import Mock

from sts1_llm_policy.eval.reward_generation import generate_frozen_reward_snapshot
from sts1_llm_policy.eval.generation_strategies import (
    CardSelectionContext, RouteContext, act_topology, configured_act1_survival,
    resolve_generation_strategies, starter_alternating, teacher_combat_priority,
    teacher_counterfactual, value_priority,
)
from sts1_llm_policy.env.combat_snapshot import resign_snapshot
from sts1_llm_policy.eval.reward_support import CapabilityScore


NAMES = {
    "card_pick": "teacher_counterfactual", "card_remove": "starter_alternating",
    "card_upgrade": "value_priority", "route": "act_topology",
}


class RewardClient:
    """Only native mutations change this fixture; exports are independent copies."""
    def __init__(self):
        self.deck = []
        self.calls = []
        self.reward_index = 0

    def reward_reset(self, seed, **kwargs):
        self.calls.append(("reset", seed, kwargs))
        self.deck = [{"id": "Strike_R", "enum_id": "STRIKE_RED", "upgraded": False} for _ in range(5)]
        self.deck += [{"id": "Defend_R", "enum_id": "DEFEND_RED", "upgraded": False} for _ in range(4)]
        self.deck += [{"id": "Bash", "enum_id": "BASH", "upgraded": False}]

    def sample_card_reward(self, *, room, floor):
        self.calls.append(("reward", room, floor))
        self.reward_index += 1
        return {"reward_id": self.reward_index, "floor_reward_index": self.reward_index,
                "choices": [], "reward_state": {"deck": deepcopy(self.deck)}}

    def apply_card_reward_choice(self, reward_id, choice):
        self.calls.append(("pick", reward_id, choice))
        return {"reward_state": {"deck": deepcopy(self.deck), "max_hp": 80}}

    def obtain_reward_relic(self, relic_id):
        self.calls.append(("relic", relic_id))

    def export_combat_snapshot(self):
        return {"combat_snapshot": {"deck": deepcopy(self.deck)}}

    def upgrade_reward_card(self, deck_index):
        self.calls.append(("upgrade", deck_index))
        before = deepcopy(self.deck[deck_index])
        self.deck[deck_index]["upgraded"] = True
        return {"upgraded_card_before": before, "upgraded_card_after": deepcopy(self.deck[deck_index])}

    def remove_reward_card(self, card_id):
        self.calls.append(("remove", card_id))
        index = next(i for i, card in enumerate(self.deck) if card["id"] == card_id)
        return {"removed_card": self.deck.pop(index)}

    def advance_reward_act(self, act):
        self.calls.append(("act", act))


class GenerationStrategyTests(unittest.TestCase):
    def test_named_strategies_are_explicit_and_picker_state_is_per_route(self):
        selected = resolve_generation_strategies(NAMES)
        self.assertIs(selected.card_pick, teacher_counterfactual)
        self.assertIs(selected.card_remove, starter_alternating)
        self.assertIs(selected.card_upgrade, value_priority)
        self.assertIs(selected.route, act_topology)
        with self.assertRaisesRegex(ValueError, "Unknown card_remove"):
            resolve_generation_strategies({**NAMES, "card_remove": "missing"})
        with self.assertRaisesRegex(ValueError, "Select"):
            resolve_generation_strategies({"route": "act_topology"})
        kwargs = dict(
            picker=Mock(), evaluator=Mock(), rate_lookup=Mock(),
            spec={"final_boss_scenario_id": "slime_boss", "act_1_boss_scenario_id": "slime_boss"},
            allowed_card_ids=frozenset(), epsilon=0.02, total_rewards=7,
            protocol={"conflict_combat_seed_offsets": [100000, 200000],
                      "opening_damage_multiplier": 1.5, "required_pair_wins_vs_skip": 2,
                      "strategic_weights": {}},
        )
        first, second = selected.card_pick(**kwargs), selected.card_pick(**kwargs)
        first.reward_index += 1
        self.assertEqual(second.reward_index, 0)

    def test_current_route_keeps_checkpoint_order_and_does_not_draw_rng(self):
        rng = random.Random(42)
        before = rng.getstate()
        steps = list(act_topology(RouteContext(1, "pre-boss", 7, 0, ("A", "B", "C"), None, 4, 2), rng))
        self.assertEqual(rng.getstate(), before)
        self.assertEqual([step["kind"] for step in steps], [
            "reset", "reward", "reward", "reward", "ordinary_relic", "upgrade",
            "reward", "reward", "ordinary_relic", "upgrade", "remove",
            "reward", "reward", "ordinary_relic", "upgrade", "upgrade", "remove",
        ])
        rewards = [step for step in steps if step["kind"] == "reward"]
        self.assertEqual([step["floor"] for step in rewards], [1, 3, 5, 7, 9, 11, 13])
        self.assertEqual(rewards[3]["room"], "ELITE")
        a2 = list(act_topology(RouteContext(2, "entry", 7, 0, ("A", "B", "C"), "BOSS", 4, 2), rng))
        self.assertEqual([step["kind"] for step in a2[-3:]], ["reward", "boss_relic", "advance_act"])
        self.assertEqual(a2[-3]["room"], "BOSS")

    def test_executor_uses_selected_policies_in_route_order_with_one_rng(self):
        calls, draws, rngs = [], [], []
        def route(context, rng):
            rngs.append(rng)
            # Deliberately outside the frozen counts: only the chosen policy
            # owns topology. The executor must follow these operations directly.
            yield {"kind": "reset", "act": 1, "relics": []}
            yield {"kind": "reward", "act": 1, "floor": 2, "room": "MONSTER", "source": "fixture"}
            yield {"kind": "remove", "act": 1, "checkpoint": "custom"}
            yield {"kind": "upgrade", "act": 1, "checkpoint": "custom"}
        def pick(reward, act, floor, rng):
            calls.append("pick")
            rngs.append(rng)
            draws.append(rng.random())
            return "SKIP", {"SKIP": 1.0}, {}
        def remove(deck, context, rng):
            calls.append("remove")
            rngs.append(rng)
            draws.append(rng.random())
            self.assertEqual(context, CardSelectionContext(1, "custom", 1))
            return "Defend_R", "selected_removal"
        def upgrade(deck, context, rng):
            calls.append("upgrade")
            rngs.append(rng)
            draws.append(rng.random())
            self.assertEqual(len(deck), 9)
            return len(deck) - 1, "selected_upgrade"
        strategies = replace(resolve_generation_strategies(NAMES), route=route,
                             card_remove=remove, card_upgrade=upgrade)
        client = RewardClient()
        _, rewards, events = generate_frozen_reward_snapshot(
            client, reward_seed=7, ascension=0, target_act=1, stage="custom",
            act_1_regular_rewards=1, act_2_rewards=0, ordinary_relic_ids=(), boss_relic_id=None,
            cumulative_upgrades=1, cumulative_removals=1, picker_seed=42,
            reward_selector=pick, strategies=strategies,
        )
        self.assertEqual(calls, ["pick", "remove", "upgrade"])
        self.assertTrue(all(rng is rngs[0] for rng in rngs))
        expected_rng = random.Random(42)
        self.assertEqual(draws, [expected_rng.random() for _ in range(3)])
        self.assertEqual(rewards[0]["floor"], 2)
        self.assertEqual(events[0]["reason"], "selected_removal")
        self.assertEqual(events[1]["priority"], "selected_upgrade")

    def test_configured_survival_route_resolves_explicit_rewards_and_unique_bags(self):
        operations = (
            "combat:normal_weak", "card_pick", "combat:normal_weak", "card_pick",
            "combat:normal_weak", "card_pick", "upgrade", "remove", "random_relic",
            "combat:normal_strong", "card_pick", "upgrade", "combat:elite",
            "random_relic", "card_pick", "combat:normal_strong", "card_pick", "remove",
            "heal:24", "upgrade", "combat:elite", "random_relic", "card_pick",
            "upgrade", "combat:boss",
        )
        context = RouteContext(
            1, "continuous", 7, 0, (), None, 4, 2,
            operations=operations,
            encounter_pools={
                "normal_weak": ("A", "B", "C", "D"),
                "normal_strong": ("E", "F", "G"),
                "elite": ("H", "I", "J"), "boss": ("K", "L", "M"),
            },
            relic_pool=("R1", "R2", "R3", "R4"),
            combat_floors=(1, 3, 5, 7, 9, 11, 13, 15),
            initial_relics=("START",),
            encounter_seed=700, relic_seed=800,
        )
        first = list(configured_act1_survival(context, random.Random(7)))
        second = list(configured_act1_survival(context, random.Random(7)))
        self.assertEqual(first, second)
        self.assertEqual([item["kind"] for item in first].count("reward"), 7)
        self.assertEqual([item["kind"] for item in first].count("ordinary_relic"), 3)
        self.assertEqual(first[0]["relics"], [{"id": "START"}])
        combats = [item for item in first if item["kind"] == "combat"]
        for family, expected_count in (("normal_weak", 3), ("normal_strong", 2), ("elite", 2)):
            scenarios = [item["scenario_id"] for item in combats if item["encounter_family"] == family]
            self.assertEqual(len(scenarios), expected_count)
            self.assertEqual(len(set(scenarios)), expected_count)
        elite_rewards = [item for item in first if item["kind"] == "reward" and item["room"] == "ELITE"]
        self.assertEqual(len(elite_rewards), 2)
        self.assertTrue(all(item["source"] == "explicit_post_combat_card_reward" for item in elite_rewards))
        self.assertEqual(next(item for item in first if item["kind"] == "heal")["amount"], 24)
        changed_relics = list(configured_act1_survival(
            replace(context, relic_seed=801), random.Random(999),
        ))
        self.assertEqual(
            [item["scenario_id"] for item in first if item["kind"] == "combat"],
            [item["scenario_id"] for item in changed_relics if item["kind"] == "combat"],
        )

    def test_teacher_upgrade_compares_unique_cards_on_a_target_panel(self):
        snapshot = resign_snapshot({
            "schema_version": "combat_snapshot_v1", "character": "IRONCLAD",
            "source_reward_seed": 1, "source_reward_id": 0, "ascension": 0,
            "act": 1, "floor": 1, "current_hp": 50, "max_hp": 80,
            "deck": [
                {"enum_id": "STRIKE_RED", "string_id": "Strike_R", "upgraded": False},
                {"enum_id": "STRIKE_RED", "string_id": "Strike_R", "upgraded": False},
                {"enum_id": "BASH", "string_id": "Bash", "upgraded": False},
            ],
            "relics": [],
        })
        calls = []
        def evaluator(candidate, scenario_id, offset):
            upgraded = next(item["enum_id"] for item in candidate["deck"] if item["upgraded"])
            calls.append((upgraded, scenario_id, offset))
            win = 0.9 if upgraded == "BASH" else 0.5
            return (CapabilityScore(701, win, 40.0, win, "ACTION_0"),)
        decisions = []
        context = CardSelectionContext(
            1, "before_elite", 1, combat_snapshot=snapshot, evaluator=evaluator,
            decision_recorder=decisions.append,
            evaluation_targets=(
                {"encounter_family": "elite", "scenario_id": "gremlin_nob", "combat_seed": 123},
                {"encounter_family": "boss", "scenario_id": "hexaghost", "combat_seed": 456},
            ),
        )
        index, reason = teacher_combat_priority(snapshot["deck"], context, random.Random(9))
        self.assertEqual(index, 2)
        self.assertEqual(reason, "teacher_multi_target_capability")
        self.assertEqual(calls, [
            ("STRIKE_RED", "gremlin_nob", 123),
            ("STRIKE_RED", "hexaghost", 456),
            ("BASH", "gremlin_nob", 123),
            ("BASH", "hexaghost", 456),
        ])
        self.assertEqual(decisions[0]["selected_enum_id"], "BASH")
        self.assertEqual(len(decisions[0]["candidates"]), 2)
        self.assertEqual(
            decisions[0]["candidates"][1]["target_evidence"][0][
                "search_evidence"
            ][0]["hidden_order_seed"],
            701,
        )


    def test_teacher_upgrade_preserves_distinct_searing_counts_in_v2(self):
        snapshot = resign_snapshot({
            "schema_version": "combat_snapshot_v2",
            "deck": [
                {"enum_id": "SEARING_BLOW", "string_id": "Searing Blow", "upgraded": False, "upgrade_count": 0},
                {"enum_id": "SEARING_BLOW", "string_id": "Searing Blow", "upgraded": True, "upgrade_count": 2},
            ],
        })
        original = deepcopy(snapshot)
        calls = []
        def evaluator(candidate, scenario_id, seed):
            counts = tuple(c["upgrade_count"] for c in candidate["deck"])
            calls.append(counts)
            self.assertEqual(candidate, resign_snapshot(candidate))
            self.assertTrue(all(c["upgraded"] == (c["upgrade_count"] > 0) for c in candidate["deck"]))
            return (CapabilityScore(701, 1.0, float(counts[1]), 1.0, "ACTION_0"),)
        context = CardSelectionContext(
            1, "before_elite", 1, combat_snapshot=snapshot, evaluator=evaluator,
            evaluation_targets=({"scenario_id": "jaw_worm", "combat_seed": 712},),
        )
        index, _ = teacher_combat_priority(snapshot["deck"], context, random.Random(9))
        self.assertEqual(calls, [(1, 2), (0, 3)])
        self.assertEqual(index, 1)
        self.assertEqual(snapshot, original)


if __name__ == "__main__":
    unittest.main()
