from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from sts1_llm_policy.env.action_schema import CanonicalAction
from sts1_llm_policy.env.state_schema import CanonicalState


class DecisionSource(str, Enum):
    """How the final action in a policy result was selected."""

    MODEL_FIRST_ATTEMPT = "model_first_attempt"
    MODEL_RETRY = "model_retry"
    FALLBACK = "fallback"
    RANDOM = "random"
    TEACHER_SEARCH = "teacher_search"


@dataclass(frozen=True)
class PolicyResult:
    """Auditable result of one independent policy decision."""

    action: CanonicalAction
    decision_source: DecisionSource

    # The policy selects a decision-local equivalence class. ``action`` is
    # the exact representative executed by the environment.
    selected_model_action_id: str
    equivalent_internal_action_ids: tuple[str, ...]

    # These fields are populated by LLM policies. They are None for policies
    # such as RandomLegalPolicy where model parsing does not apply.
    raw_output: str | None
    parsed_action_id: str | None
    parse_success: bool | None
    legal_on_first_attempt: bool | None

    retry_used: bool
    retry_raw_output: str | None
    retry_parsed_action_id: str | None
    retry_success: bool | None

    fallback_used: bool
    inference_ms: float


class Policy(Protocol):
    def select_action(
        self,
        state: CanonicalState,
        legal_actions: Sequence[CanonicalAction],
    ) -> PolicyResult:
        ...
