from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

from sts1_llm_policy.data.trajectory import TerminalOutcome, TrajectoryLogger
from sts1_llm_policy.data.trajectory import classify_terminal_outcome
from sts1_llm_policy.env.action_schema import CanonicalAction
from sts1_llm_policy.env.errors import CombatEndedError
from sts1_llm_policy.env.state_schema import CanonicalState
from sts1_llm_policy.policy.base import Policy, PolicyResult


class CombatEnvironment(Protocol):
    def get_state(self) -> CanonicalState:
        ...

    def get_raw_state(self) -> dict:
        ...

    def legal_actions(self) -> tuple[CanonicalAction, ...]:
        ...

    def step(self, action: CanonicalAction) -> CanonicalState:
        ...


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    steps: int
    outcome: TerminalOutcome
    score: float | None
    termination_reason: str


StepCallback = Callable[[int, PolicyResult], None]


def extract_terminal_score(raw_state: dict) -> float | None:
    """Read the official run score when GAME_OVER exposes one."""

    game_state = raw_state.get("game_state")

    if not isinstance(game_state, dict):
        return None

    if game_state.get("screen_type") != "GAME_OVER":
        return None

    screen_state = game_state.get("screen_state")

    if not isinstance(screen_state, dict):
        return None

    score = screen_state.get("score")

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None

    if not math.isfinite(score):
        return None

    return float(score)


def run_combat_episode(
    env: CombatEnvironment,
    policy: Policy,
    trajectory_logger: TrajectoryLogger,
    *,
    max_steps: int = 1000,
    on_step: StepCallback | None = None,
) -> EpisodeSummary:
    """Run one already-connected combat through a terminal transition."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    for step_index in range(max_steps):
        state = env.get_state()
        raw_state = env.get_raw_state()
        legal_actions = env.legal_actions()
        policy_result = policy.select_action(state, legal_actions)

        try:
            next_state = env.step(policy_result.action)
        except CombatEndedError as terminal:
            outcome = classify_terminal_outcome(terminal.raw_state)
            score = extract_terminal_score(terminal.raw_state)
            reward = getattr(terminal, "reward", None)

            trajectory_logger.log_transition(
                state=state,
                raw_state=raw_state,
                legal_actions=legal_actions,
                policy_result=policy_result,
                next_state=None,
                next_raw_state=terminal.raw_state,
                reward=reward,
                score=score,
                done=True,
                terminal_outcome=outcome,
            )

            if on_step is not None:
                on_step(step_index, policy_result)

            return EpisodeSummary(
                episode_id=trajectory_logger.episode_id,
                steps=step_index + 1,
                outcome=outcome,
                score=score,
                termination_reason="environment_terminal",
            )

        reached_step_limit = step_index + 1 == max_steps

        trajectory_logger.log_transition(
            state=state,
            raw_state=raw_state,
            legal_actions=legal_actions,
            policy_result=policy_result,
            next_state=next_state,
            next_raw_state=env.get_raw_state(),
            done=reached_step_limit,
            terminal_outcome=(
                TerminalOutcome.ABORTED
                if reached_step_limit
                else None
            ),
        )

        if on_step is not None:
            on_step(step_index, policy_result)

        if reached_step_limit:
            return EpisodeSummary(
                episode_id=trajectory_logger.episode_id,
                steps=step_index + 1,
                outcome=TerminalOutcome.ABORTED,
                score=None,
                termination_reason="max_steps",
            )

    raise AssertionError("Unreachable episode loop state")
