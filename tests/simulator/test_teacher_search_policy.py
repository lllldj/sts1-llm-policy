from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sts1_llm_policy.policy.base import DecisionSource
from sts1_llm_policy.policy.teacher_search_policy import TeacherSearchPolicy


class TeacherSearchPolicyTests(unittest.TestCase):

    def test_selects_projected_search_representative(self) -> None:
        representative = object()
        selected = SimpleNamespace(
            model_action_id="ACTION_3",
            representative=representative,
            internal_action_ids=("PLAY:2:0", "PLAY:5:0"),
        )
        action_space = Mock()
        action_space.resolve.return_value = selected
        env = Mock()
        env.search_model_actions.return_value = SimpleNamespace(
            suggested_model_action_id="ACTION_3",
            native_result=SimpleNamespace(elapsed_ms=12.5),
        )
        policy = TeacherSearchPolicy(env, simulations=8192, search_seed=101)

        with patch(
            "sts1_llm_policy.policy.teacher_search_policy.build_model_action_space",
            return_value=action_space,
        ):
            result = policy.select_action(object(), (object(),))

        env.search_model_actions.assert_called_once_with(
            simulations=8192,
            search_seed=101,
        )
        self.assertIs(result.action, representative)
        self.assertEqual(result.decision_source, DecisionSource.TEACHER_SEARCH)
        self.assertEqual(result.selected_model_action_id, "ACTION_3")
        self.assertEqual(result.equivalent_internal_action_ids, ("PLAY:2:0", "PLAY:5:0"))
        self.assertTrue(result.legal_on_first_attempt)
        self.assertFalse(result.retry_used)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.inference_ms, 12.5)

    def test_rejects_invalid_search_parameters(self) -> None:
        env = Mock()
        for simulations in (True, 0, -1):
            with self.subTest(simulations=simulations):
                with self.assertRaises(ValueError):
                    TeacherSearchPolicy(env, simulations=simulations, search_seed=101)
        for search_seed in (True, -1, 2**32):
            with self.subTest(search_seed=search_seed):
                with self.assertRaises(ValueError):
                    TeacherSearchPolicy(env, simulations=8192, search_seed=search_seed)


if __name__ == "__main__":
    unittest.main()
