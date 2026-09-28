from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from sts1_llm_policy.env.action_equivalence import build_model_action_space
from sts1_llm_policy.env.action_schema import CanonicalAction
from sts1_llm_policy.env.state_schema import CanonicalState

from .base import DecisionSource, PolicyResult

if TYPE_CHECKING:
    from sts1_llm_policy.env.simulator_env import StsLightspeedEnv


class TeacherSearchPolicy:
    """Execute the trajectory action selected by BattleScumSearcher2.

    This is the on-policy behavior used while collecting the promoted teacher
    dataset: one reference-budget search with the contract's first search seed.
    The additional search grid used to accept or reject an SFT label is not part
    of trajectory action selection.
    """

    def __init__(
        self,
        env: StsLightspeedEnv,
        *,
        simulations: int,
        search_seed: int,
    ) -> None:
        if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        if (
            isinstance(search_seed, bool)
            or not isinstance(search_seed, int)
            or search_seed < 0
            or search_seed > 2**32 - 1
        ):
            raise ValueError("search_seed must be an unsigned 32-bit integer")
        self._env = env
        self.simulations = simulations
        self.search_seed = search_seed

    def select_action(
        self,
        state: CanonicalState,
        legal_actions: Sequence[CanonicalAction],
    ) -> PolicyResult:
        action_space = build_model_action_space(state, legal_actions)
        result = self._env.search_model_actions(
            simulations=self.simulations,
            search_seed=self.search_seed,
        )
        selected = action_space.resolve(result.suggested_model_action_id)
        if selected is None:
            raise RuntimeError("Teacher search selected an unknown model action")
        return PolicyResult(
            action=selected.representative,
            decision_source=DecisionSource.TEACHER_SEARCH,
            selected_model_action_id=selected.model_action_id,
            equivalent_internal_action_ids=selected.internal_action_ids,
            raw_output=None,
            parsed_action_id=None,
            parse_success=None,
            legal_on_first_attempt=True,
            retry_used=False,
            retry_raw_output=None,
            retry_parsed_action_id=None,
            retry_success=None,
            fallback_used=False,
            inference_ms=result.native_result.elapsed_ms,
        )
