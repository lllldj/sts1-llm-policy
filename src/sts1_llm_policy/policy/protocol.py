from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from sts1_llm_policy.env.action_equivalence import build_model_action_space
from sts1_llm_policy.env.action_schema import CanonicalAction
from sts1_llm_policy.env.serializer import (
    OBSERVATION_SERIALIZER_VERSION,
    serialize_observation,
)
from sts1_llm_policy.env.state_schema import CanonicalState


POLICY_PROTOCOL_VERSION = "policy_protocol_v1"
PROMPT_VERSION = "prompt_v1"

SYSTEM_PROMPT = """You are a Slay the Spire combat policy.
Choose exactly one legal action.
Return only its action ID, with no explanation."""

_ACTION_ID_PATTERN = re.compile(r"^ACTION_[0-9]+$")


@dataclass(frozen=True)
class GenerationRequest:
    """Backend-independent request for one model generation."""

    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 8


class GenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> str:
        ...


def parse_action_id(raw_output: str) -> str | None:
    """Strictly parse one ACTION_n identifier from model output."""

    if not isinstance(raw_output, str):
        raise TypeError("Model output must be a string")

    candidate = raw_output.strip()

    if _ACTION_ID_PATTERN.fullmatch(candidate) is None:
        return None

    return candidate


def index_legal_actions(
    legal_actions: Sequence[CanonicalAction],
) -> dict[str, CanonicalAction]:
    """Build an unambiguous current-decision action lookup."""

    if not legal_actions:
        raise ValueError("legal_actions must not be empty")

    action_by_id: dict[str, CanonicalAction] = {}

    for action in legal_actions:
        if action.action_id in action_by_id:
            raise ValueError(
                f"Duplicate legal action ID: {action.action_id}"
            )

        action_by_id[action.action_id] = action

    return action_by_id


def build_initial_request(
    state: CanonicalState,
    legal_actions: Sequence[CanonicalAction],
    *,
    observation_version: str = OBSERVATION_SERIALIZER_VERSION,
    allow_source_descriptions: bool = False,
) -> GenerationRequest:
    return build_initial_request_from_observation(
        serialize_observation(
            state,
            legal_actions,
            version=observation_version,
            allow_source_descriptions=allow_source_descriptions,
        )
    )


def build_initial_request_from_observation(observation: str) -> GenerationRequest:
    """Build the frozen initial request from an already serialized observation."""

    return GenerationRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=observation,
    )


def build_retry_request(
    state: CanonicalState,
    legal_actions: Sequence[CanonicalAction],
    *,
    observation_version: str = OBSERVATION_SERIALIZER_VERSION,
    allow_source_descriptions: bool = False,
) -> GenerationRequest:
    """Build a fresh corrective request without prior conversation history."""

    action_space = build_model_action_space(
        state,
        legal_actions,
        allow_source_descriptions=allow_source_descriptions,
    )
    return build_retry_request_from_observation(
        serialize_observation(
            state,
            legal_actions,
            version=observation_version,
            allow_source_descriptions=allow_source_descriptions,
        ),
        action_space.action_ids,
    )


def build_retry_request_from_observation(
    observation: str,
    allowed_action_ids: Sequence[str],
) -> GenerationRequest:
    """Build the frozen corrective request from saved offline inputs."""

    if not allowed_action_ids:
        raise ValueError("allowed_action_ids must not be empty")
    allowed_ids = "\n".join(allowed_action_ids)

    correction = (
        "Your previous response was invalid.\n\n"
        "Return exactly one of:\n"
        f"{allowed_ids}\n\n"
        "Return only the action ID."
    )

    return GenerationRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"{observation}\n\n{correction}",
    )
