from dataclasses import replace
import unittest

from sts1_llm_policy.env.action_builder import build_legal_actions
from sts1_llm_policy.env.action_equivalence import (
    build_model_action_space,
)
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.env.state_schema import (
    CanonicalState,
    CardState,
    CombatState,
    MonsterState,
    PlayerState,
)
from sts1_llm_policy.policy.llm_policy import LLMPolicy
from sts1_llm_policy.policy.random_policy import RandomLegalPolicy
from sts1_llm_policy.policy.protocol import (
    GenerationRequest,
    build_retry_request,
)


class StubBackend:
    def __init__(self, output: str) -> None:
        self.output = output
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> str:
        self.requests.append(request)
        return self.output


def _card(
    card_id: str,
    name: str,
    card_type: str,
    *,
    cost: int,
    has_target: bool,
    uuid: str,
    special_data: int = 0,
) -> CardState:
    return CardState(
        card_id=card_id,
        name=name,
        card_type=card_type,
        cost=cost,
        upgrades=0,
        description="source text",
        exhausts=False,
        ethereal=False,
        has_target=has_target,
        is_playable=True,
        uuid=uuid,
        special_data=special_data,
    )


def _state(hand: tuple[CardState, ...]) -> CanonicalState:
    monsters = tuple(
        MonsterState(
            monster_id="FuzzyLouseNormal",
            name=f"Louse {index}",
            current_hp=12,
            max_hp=12,
            block=0,
            intent="ATTACK",
            move_id=3,
            move_hits=1,
            move_base_damage=6,
            move_adjusted_damage=6,
        )
        for index in range(2)
    )
    return CanonicalState(
        seed=9,
        character="IRONCLAD",
        ascension_level=0,
        act=1,
        floor=1,
        combat=CombatState(
            turn=1,
            player=PlayerState(80, 80, 0, 3),
            monsters=monsters,
            hand=hand,
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
        ),
    )


class ActionEquivalenceTest(unittest.TestCase):

    def test_groups_duplicate_instances_but_never_different_targets(self) -> None:
        state = _state(
            (
                _card(
                    "Strike_R",
                    "Strike",
                    "ATTACK",
                    cost=1,
                    has_target=True,
                    uuid="strike-a",
                ),
                _card(
                    "Defend_R",
                    "Defend",
                    "SKILL",
                    cost=1,
                    has_target=False,
                    uuid="defend-a",
                ),
                _card(
                    "Strike_R",
                    "Strike",
                    "ATTACK",
                    cost=1,
                    has_target=True,
                    uuid="strike-b",
                ),
            )
        )
        actions = build_legal_actions(state)
        space = build_model_action_space(state, actions)

        self.assertEqual(space.action_ids, tuple(f"ACTION_{i}" for i in range(4)))
        defend, strike_target_0, strike_target_1, end_turn = space.classes
        self.assertEqual(defend.internal_action_ids, ("ACTION_2",))
        self.assertEqual(
            strike_target_0.internal_action_ids,
            ("ACTION_0", "ACTION_3"),
        )
        self.assertEqual(
            strike_target_1.internal_action_ids,
            ("ACTION_1", "ACTION_4"),
        )
        self.assertEqual(strike_target_0.target_index, 0)
        self.assertEqual(strike_target_1.target_index, 1)
        self.assertEqual(
            space.model_id_for_internal_action("ACTION_0"),
            strike_target_0.model_action_id,
        )
        self.assertEqual(
            space.model_id_for_internal_action("ACTION_3"),
            strike_target_0.model_action_id,
        )
        self.assertIsNone(space.model_id_for_internal_action("ACTION_99"))
        self.assertEqual(end_turn.internal_action_ids, ("ACTION_5",))
        self.assertIs(strike_target_0.representative, actions[0])

        observation = serialize_observation(state, actions)
        self.assertIn("Defend x1", observation)
        self.assertIn("Strike x2", observation)
        self.assertIn(
            "ACTION_1: PLAY Strike | COST 1 | UPGRADE 0"
            " -> TARGET_0 | COPIES 2",
            observation,
        )
        self.assertNotIn("strike-a", observation)
        self.assertNotIn("strike-b", observation)
        self.assertNotIn("[0] Strike", observation)

        retry = build_retry_request(state, actions)
        self.assertIn("ACTION_0\nACTION_1\nACTION_2\nACTION_3", retry.user_prompt)
        self.assertNotIn("ACTION_4\n", retry.user_prompt)

        legacy = serialize_observation(
            state,
            actions,
            version="observation_v1",
        )
        self.assertIn("[0] Strike | COST 1", legacy)
        self.assertIn("[2] Strike | COST 1", legacy)
        self.assertIn("ACTION_0: PLAY Strike -> TARGET_0", legacy)
        self.assertIn("ACTION_3: PLAY Strike -> TARGET_0", legacy)

    def test_current_cost_splits_otherwise_identical_cards(self) -> None:
        state = _state(
            (
                _card(
                    "Strike_R",
                    "Strike",
                    "ATTACK",
                    cost=1,
                    has_target=True,
                    uuid="strike-one",
                ),
                _card(
                    "Strike_R",
                    "Strike",
                    "ATTACK",
                    cost=0,
                    has_target=True,
                    uuid="strike-zero",
                ),
            )
        )
        space = build_model_action_space(state, build_legal_actions(state))

        play_classes = space.classes[:-1]
        self.assertEqual(len(play_classes), 4)
        self.assertEqual({item.card.cost for item in play_classes}, {0, 1})
        self.assertTrue(all(item.copies == 1 for item in play_classes))

    def test_rampage_combat_bonus_splits_action_classes(self) -> None:
        first = _card(
            "Rampage", "Rampage", "ATTACK", cost=1, has_target=True,
            uuid="rampage-zero", special_data=0,
        )
        second = replace(first, uuid="rampage-five", special_data=5)
        state = _state((first, second))
        space = build_model_action_space(state, build_legal_actions(state))
        effects = {item.card.effect for item in space.classes[:-1] if item.card}
        self.assertEqual(len(effects), 2)
        self.assertTrue(any("CURRENT_COMBAT_BONUS=0" in effect for effect in effects))
        self.assertTrue(any("CURRENT_COMBAT_BONUS=5" in effect for effect in effects))

    def test_model_and_random_policy_execute_the_representative(self) -> None:
        strike_a = _card(
            "Strike_R",
            "Strike",
            "ATTACK",
            cost=1,
            has_target=True,
            uuid="strike-a",
        )
        strike_b = replace(strike_a, uuid="strike-b")
        state = _state((strike_a, strike_b))
        actions = build_legal_actions(state)

        backend = StubBackend("ACTION_0")
        result = LLMPolicy(backend, policy_seed=2).select_action(state, actions)
        self.assertEqual(result.selected_model_action_id, "ACTION_0")
        self.assertEqual(
            result.equivalent_internal_action_ids,
            ("ACTION_0", "ACTION_2"),
        )
        self.assertIs(result.action, actions[0])
        self.assertIn("COPIES 2", backend.requests[0].user_prompt)

        random_result = RandomLegalPolicy(policy_seed=1).select_action(
            state,
            actions,
        )
        space = build_model_action_space(state, actions)
        chosen = space.resolve(random_result.selected_model_action_id)
        self.assertIsNotNone(chosen)
        self.assertIs(random_result.action, chosen.representative)

    def test_observation_is_invariant_to_equivalent_hand_order(self) -> None:
        strike_a = _card(
            "Strike_R",
            "Strike",
            "ATTACK",
            cost=1,
            has_target=True,
            uuid="strike-a",
        )
        strike_b = replace(strike_a, uuid="strike-b")
        defend = _card(
            "Defend_R",
            "Defend",
            "SKILL",
            cost=1,
            has_target=False,
            uuid="defend-a",
        )
        first = _state((strike_a, defend, strike_b))
        second = _state((strike_b, strike_a, defend))

        self.assertEqual(
            serialize_observation(first, build_legal_actions(first)),
            serialize_observation(second, build_legal_actions(second)),
        )


if __name__ == "__main__":
    unittest.main()
