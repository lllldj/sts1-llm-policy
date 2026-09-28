from __future__ import annotations

import random
from time import perf_counter
from typing import Sequence

from sts1_llm_policy.env.action_equivalence import build_model_action_space
from sts1_llm_policy.env.action_schema import CanonicalAction
from sts1_llm_policy.env.state_schema import CanonicalState

from .base import DecisionSource, PolicyResult
from .protocol import index_legal_actions


RANDOM_MODEL_ACTION_POLICY_ID = "random_model_action_v1"
RANDOM_EXACT_ACTION_POLICY_ID = "random_legal_v1"


class RandomLegalPolicy:
    """Uniform random baseline over current model action classes.

    ``use_action_equivalence=False`` preserves the exact-instance sampler used
    by the frozen v1 simulator gate and offline dataset builder.
    """

    def __init__(
        self,
        policy_seed: int,
        *,
        use_action_equivalence: bool = True,
    ) -> None:
        self.policy_seed = policy_seed
        self.use_action_equivalence = use_action_equivalence
        self.policy_id = (
            RANDOM_MODEL_ACTION_POLICY_ID
            if use_action_equivalence
            else RANDOM_EXACT_ACTION_POLICY_ID
        )
        self._random = random.Random(policy_seed)

    def select_action(
        self,
        state: CanonicalState,
        legal_actions: Sequence[CanonicalAction],
    ) -> PolicyResult:
        started_at = perf_counter()
        if self.use_action_equivalence:
            action_space = build_model_action_space(state, legal_actions)
            action_class = self._random.choice(action_space.classes)
            action = action_class.representative
            selected_model_action_id = action_class.model_action_id
            equivalent_internal_action_ids = action_class.internal_action_ids
        else:
            exact_actions = tuple(index_legal_actions(legal_actions).values())
            action = self._random.choice(exact_actions)
            selected_model_action_id = action.action_id
            equivalent_internal_action_ids = (action.action_id,)
        inference_ms = (perf_counter() - started_at) * 1000

        return PolicyResult(
            action=action,
            decision_source=DecisionSource.RANDOM,
            selected_model_action_id=selected_model_action_id,
            equivalent_internal_action_ids=equivalent_internal_action_ids,
            raw_output=None,
            parsed_action_id=None,
            parse_success=None,
            legal_on_first_attempt=None,
            retry_used=False,
            retry_raw_output=None,
            retry_parsed_action_id=None,
            retry_success=None,
            fallback_used=False,
            inference_ms=inference_ms,
        )
