from __future__ import annotations

import random
from time import perf_counter
from typing import Sequence

from sts1_llm_policy.env.action_equivalence import (
    ModelActionClass,
    build_model_action_space,
)
from sts1_llm_policy.env.action_schema import CanonicalAction
from sts1_llm_policy.env.serializer import (
    OBSERVATION_SERIALIZER_VERSION,
    SUPPORTED_OBSERVATION_SERIALIZER_VERSIONS,
)
from sts1_llm_policy.env.state_schema import CanonicalState

from .base import DecisionSource, PolicyResult
from .protocol import (
    GenerationBackend,
    build_initial_request,
    build_retry_request,
    parse_action_id,
)


class LLMPolicy:
    """Protocol-v1 model policy with one retry and seeded fallback."""

    def __init__(
        self,
        backend: GenerationBackend,
        policy_seed: int,
        *,
        observation_version: str = OBSERVATION_SERIALIZER_VERSION,
        allow_source_descriptions: bool = False,
    ) -> None:
        if observation_version not in SUPPORTED_OBSERVATION_SERIALIZER_VERSIONS:
            raise ValueError(
                f"Unsupported observation serializer version: {observation_version}"
            )
        self.policy_seed = policy_seed
        self.observation_version = observation_version
        self.allow_source_descriptions = allow_source_descriptions
        self._backend = backend
        self._fallback_random = random.Random(policy_seed)

    def select_action(
        self,
        state: CanonicalState,
        legal_actions: Sequence[CanonicalAction],
    ) -> PolicyResult:
        actions = tuple(legal_actions)
        action_space = build_model_action_space(
            state,
            actions,
            allow_source_descriptions=self.allow_source_descriptions,
        )
        started_at = perf_counter()

        raw_output = self._backend.generate(
            build_initial_request(
                state,
                actions,
                observation_version=self.observation_version,
                allow_source_descriptions=self.allow_source_descriptions,
            )
        )
        parsed_action_id = parse_action_id(raw_output)
        first_class = action_space.resolve(parsed_action_id)
        parse_success = parsed_action_id is not None
        legal_on_first_attempt = first_class is not None

        if first_class is not None:
            return self._build_result(
                action_class=first_class,
                decision_source=DecisionSource.MODEL_FIRST_ATTEMPT,
                raw_output=raw_output,
                parsed_action_id=parsed_action_id,
                parse_success=parse_success,
                legal_on_first_attempt=True,
                retry_used=False,
                retry_raw_output=None,
                retry_parsed_action_id=None,
                retry_success=None,
                fallback_used=False,
                started_at=started_at,
            )

        retry_raw_output = self._backend.generate(
            build_retry_request(
                state,
                actions,
                observation_version=self.observation_version,
                allow_source_descriptions=self.allow_source_descriptions,
            )
        )
        retry_parsed_action_id = parse_action_id(retry_raw_output)
        retry_class = action_space.resolve(retry_parsed_action_id)

        if retry_class is not None:
            return self._build_result(
                action_class=retry_class,
                decision_source=DecisionSource.MODEL_RETRY,
                raw_output=raw_output,
                parsed_action_id=parsed_action_id,
                parse_success=parse_success,
                legal_on_first_attempt=legal_on_first_attempt,
                retry_used=True,
                retry_raw_output=retry_raw_output,
                retry_parsed_action_id=retry_parsed_action_id,
                retry_success=True,
                fallback_used=False,
                started_at=started_at,
            )

        fallback_class = self._fallback_random.choice(action_space.classes)

        return self._build_result(
            action_class=fallback_class,
            decision_source=DecisionSource.FALLBACK,
            raw_output=raw_output,
            parsed_action_id=parsed_action_id,
            parse_success=parse_success,
            legal_on_first_attempt=legal_on_first_attempt,
            retry_used=True,
            retry_raw_output=retry_raw_output,
            retry_parsed_action_id=retry_parsed_action_id,
            retry_success=False,
            fallback_used=True,
            started_at=started_at,
        )

    @staticmethod
    def _build_result(
        *,
        action_class: ModelActionClass,
        decision_source: DecisionSource,
        raw_output: str,
        parsed_action_id: str | None,
        parse_success: bool,
        legal_on_first_attempt: bool,
        retry_used: bool,
        retry_raw_output: str | None,
        retry_parsed_action_id: str | None,
        retry_success: bool | None,
        fallback_used: bool,
        started_at: float,
    ) -> PolicyResult:
        return PolicyResult(
            action=action_class.representative,
            decision_source=decision_source,
            selected_model_action_id=action_class.model_action_id,
            equivalent_internal_action_ids=action_class.internal_action_ids,
            raw_output=raw_output,
            parsed_action_id=parsed_action_id,
            parse_success=parse_success,
            legal_on_first_attempt=legal_on_first_attempt,
            retry_used=retry_used,
            retry_raw_output=retry_raw_output,
            retry_parsed_action_id=retry_parsed_action_id,
            retry_success=retry_success,
            fallback_used=fallback_used,
            inference_ms=(perf_counter() - started_at) * 1000,
        )
