from dataclasses import replace
import unittest

from sts1_llm_policy.env.action_builder import build_legal_actions
from sts1_llm_policy.env.action_schema import ActionType, CanonicalAction
from sts1_llm_policy.env.serializer import BEHAVIOR_OBSERVATION_SERIALIZER_VERSION
from sts1_llm_policy.env.state_schema import (
    CanonicalState,
    CardState,
    CombatState,
    MonsterBehaviorState,
    MonsterState,
    PlayerState,
)
from sts1_llm_policy.policy.base import DecisionSource
from sts1_llm_policy.policy.protocol import (
    GenerationRequest,
    build_initial_request_from_observation,
    build_retry_request_from_observation,
    parse_action_id,
)
from sts1_llm_policy.policy.llm_policy import LLMPolicy
from sts1_llm_policy.policy.random_policy import RandomLegalPolicy
from sts1_llm_policy.policy.protocol import index_legal_actions


class StubBackend:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = iter(outputs)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> str:
        self.requests.append(request)
        return next(self._outputs)


def _state() -> CanonicalState:
    defend = CardState(
        card_id="Defend_R",
        name="Defend",
        card_type="SKILL",
        cost=1,
        upgrades=0,
        description="Gain 5 Block.",
        exhausts=False,
        ethereal=False,
        has_target=False,
        is_playable=True,
        uuid="defend-uuid",
    )
    strike = CardState(
        card_id="Strike_R",
        name="Strike",
        card_type="ATTACK",
        cost=1,
        upgrades=0,
        description="Deal 6 damage.",
        exhausts=False,
        ethereal=False,
        has_target=True,
        is_playable=True,
        uuid="strike-uuid",
    )

    return CanonicalState(
        seed=17,
        character="IRONCLAD",
        ascension_level=0,
        act=1,
        floor=1,
        combat=CombatState(
            turn=1,
            player=PlayerState(
                current_hp=80,
                max_hp=80,
                block=0,
                energy=3,
            ),
            monsters=(
                MonsterState(
                    monster_id="Cultist",
                    name="Cultist",
                    current_hp=48,
                    max_hp=48,
                    block=0,
                    intent="BUFF",
                    move_id=3,
                    move_hits=1,
                    move_base_damage=0,
                    move_adjusted_damage=0,
                ),
            ),
            hand=(defend, strike),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
        ),
    )


class ParseActionIdTest(unittest.TestCase):
    def test_accepts_only_one_complete_action_id(self) -> None:
        cases = {
            "ACTION_2": "ACTION_2",
            "  ACTION_2\n": "ACTION_2",
            "ACTION_01": "ACTION_01",
            "ACTION 2": None,
            "I choose ACTION_2": None,
            "ACTION_2 because it is legal": None,
            "ACTION_1\nACTION_2": None,
            "```ACTION_2```": None,
            "": None,
        }

        for raw_output, expected in cases.items():
            with self.subTest(raw_output=raw_output):
                self.assertEqual(parse_action_id(raw_output), expected)

    def test_rejects_non_string_backend_output(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            parse_action_id(None)  # type: ignore[arg-type]


class LegalActionIndexTest(unittest.TestCase):
    def test_rejects_empty_or_duplicate_action_sets(self) -> None:
        duplicate_actions = (
            CanonicalAction("ACTION_0", ActionType.END_TURN),
            CanonicalAction("ACTION_0", ActionType.END_TURN),
        )

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            index_legal_actions(())

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_legal_actions(duplicate_actions)


class SavedObservationRequestTest(unittest.TestCase):
    def test_builds_exact_initial_and_fresh_corrective_requests(self) -> None:
        initial = build_initial_request_from_observation("FROZEN OBSERVATION")
        retry = build_retry_request_from_observation(
            "FROZEN OBSERVATION",
            ("ACTION_0", "ACTION_2"),
        )

        self.assertEqual(initial.user_prompt, "FROZEN OBSERVATION")
        self.assertEqual(initial.temperature, 0.0)
        self.assertIn("FROZEN OBSERVATION", retry.user_prompt)
        self.assertIn("ACTION_0\nACTION_2", retry.user_prompt)
        self.assertNotIn("previous response:", retry.user_prompt.lower())

    def test_rejects_empty_corrective_action_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_retry_request_from_observation("state", ())


class LLMPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _state()
        self.actions = build_legal_actions(self.state)

    def test_returns_first_valid_model_action_without_retry(self) -> None:
        backend = StubBackend(["  ACTION_1\n"])
        policy = LLMPolicy(backend, policy_seed=7)

        result = policy.select_action(self.state, self.actions)

        self.assertEqual(result.action, self.actions[1])
        self.assertEqual(
            result.decision_source,
            DecisionSource.MODEL_FIRST_ATTEMPT,
        )
        self.assertTrue(result.parse_success)
        self.assertTrue(result.legal_on_first_attempt)
        self.assertFalse(result.retry_used)
        self.assertIsNone(result.retry_success)
        self.assertFalse(result.fallback_used)
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(backend.requests[0].temperature, 0.0)
        self.assertFalse(backend.requests[0].do_sample)
        self.assertEqual(backend.requests[0].max_new_tokens, 8)

    def test_live_policy_uses_source_description_for_unknown_card(self) -> None:
        unknown = replace(
            self.state.combat.hand[0],
            card_id="LiveCard",
            name="Live Card",
            description="Gain 7 Block from the live mod.",
        )
        state = replace(
            self.state,
            combat=replace(self.state.combat, hand=(unknown,)),
        )
        actions = build_legal_actions(state)
        backend = StubBackend(["ACTION_0"])
        policy = LLMPolicy(
            backend,
            policy_seed=7,
            allow_source_descriptions=True,
        )

        result = policy.select_action(state, actions)

        self.assertEqual(result.action, actions[0])
        self.assertIn(
            "EFFECT Gain 7 Block from the live mod.",
            backend.requests[0].user_prompt,
        )

    def test_retries_once_after_malformed_output(self) -> None:
        backend = StubBackend(["I choose ACTION_0", "ACTION_2"])
        policy = LLMPolicy(backend, policy_seed=7)

        result = policy.select_action(self.state, self.actions)

        self.assertEqual(result.action, self.actions[2])
        self.assertEqual(result.decision_source, DecisionSource.MODEL_RETRY)
        self.assertFalse(result.parse_success)
        self.assertFalse(result.legal_on_first_attempt)
        self.assertTrue(result.retry_used)
        self.assertEqual(result.retry_parsed_action_id, "ACTION_2")
        self.assertTrue(result.retry_success)
        self.assertFalse(result.fallback_used)
        self.assertEqual(len(backend.requests), 2)

        retry_prompt = backend.requests[1].user_prompt
        self.assertIn("Your previous response was invalid.", retry_prompt)
        self.assertIn("ACTION_0\nACTION_1\nACTION_2", retry_prompt)
        self.assertNotIn("I choose ACTION_0", retry_prompt)

    def test_distinguishes_parse_success_from_legality(self) -> None:
        backend = StubBackend(["ACTION_99", "ACTION_0"])
        policy = LLMPolicy(backend, policy_seed=7)

        result = policy.select_action(self.state, self.actions)

        self.assertTrue(result.parse_success)
        self.assertFalse(result.legal_on_first_attempt)
        self.assertEqual(result.parsed_action_id, "ACTION_99")
        self.assertEqual(result.action, self.actions[0])
        self.assertTrue(result.retry_success)

    def test_uses_seeded_fallback_after_exactly_one_retry(self) -> None:
        backend_a = StubBackend(["bad", "ACTION_99"])
        backend_b = StubBackend(["bad", "ACTION_99"])
        policy_a = LLMPolicy(backend_a, policy_seed=123)
        policy_b = LLMPolicy(backend_b, policy_seed=123)

        result_a = policy_a.select_action(self.state, self.actions)
        result_b = policy_b.select_action(self.state, self.actions)

        self.assertEqual(result_a.action, result_b.action)
        self.assertIn(result_a.action, self.actions)
        self.assertEqual(result_a.decision_source, DecisionSource.FALLBACK)
        self.assertTrue(result_a.retry_used)
        self.assertEqual(result_a.retry_parsed_action_id, "ACTION_99")
        self.assertFalse(result_a.retry_success)
        self.assertTrue(result_a.fallback_used)
        self.assertEqual(len(backend_a.requests), 2)

    def test_resolves_id_against_current_action_objects(self) -> None:
        stale_action = CanonicalAction(
            action_id="ACTION_0",
            action_type=ActionType.PLAY_CARD,
            hand_index=0,
            card_uuid="stale-uuid",
            card_name="Defend",
        )
        backend = StubBackend(["ACTION_0"])
        policy = LLMPolicy(backend, policy_seed=7)

        result = policy.select_action(self.state, self.actions)

        self.assertNotEqual(result.action, stale_action)
        self.assertIs(result.action, self.actions[0])

    def test_can_explicitly_request_behavior_observation(self) -> None:
        monster = replace(
            self.state.combat.monsters[0],
            behavior=MonsterBehaviorState(
                phase="opening_buff",
                previous_move_id=None,
                possible_next_move_ids=(1,),
                selection="deterministic",
                rule="Opens with Incantation, then repeats Dark Strike.",
            ),
        )
        state = replace(
            self.state,
            combat=replace(self.state.combat, monsters=(monster,)),
        )
        backend = StubBackend(["ACTION_0"])
        policy = LLMPolicy(
            backend,
            policy_seed=7,
            observation_version=BEHAVIOR_OBSERVATION_SERIALIZER_VERSION,
        )

        policy.select_action(state, build_legal_actions(state))

        self.assertIn("BEHAVIOR:", backend.requests[0].user_prompt)


class RandomLegalPolicyTest(unittest.TestCase):
    def test_is_reproducible_and_marks_parsing_not_applicable(self) -> None:
        state = _state()
        actions = build_legal_actions(state)
        policy_a = RandomLegalPolicy(policy_seed=11)
        policy_b = RandomLegalPolicy(policy_seed=11)

        sequence_a = [
            policy_a.select_action(state, actions).action.action_id
            for _ in range(8)
        ]
        sequence_b = [
            policy_b.select_action(state, actions).action.action_id
            for _ in range(8)
        ]
        result = policy_a.select_action(state, actions)

        self.assertEqual(sequence_a, sequence_b)
        self.assertIn(result.action, actions)
        self.assertEqual(result.decision_source, DecisionSource.RANDOM)
        self.assertIsNone(result.parse_success)
        self.assertIsNone(result.legal_on_first_attempt)
        self.assertFalse(result.retry_used)
        self.assertFalse(result.fallback_used)

    def test_rejects_empty_action_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            RandomLegalPolicy(policy_seed=0).select_action(_state(), ())


if __name__ == "__main__":
    unittest.main()
