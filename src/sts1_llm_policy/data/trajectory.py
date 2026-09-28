from __future__ import annotations

import json
import math
import os
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from sts1_llm_policy.artifacts import sha256_file

from sts1_llm_policy.env.action_equivalence import (
    ACTION_EQUIVALENCE_VERSION,
    build_model_action_space,
    model_action_space_records,
)
from sts1_llm_policy.env.action_schema import CanonicalAction
from sts1_llm_policy.env.live_description_fallback import (
    SOURCE_DESCRIPTION_FALLBACK_ID,
)
from sts1_llm_policy.env.serializer import (
    OBSERVATION_SERIALIZER_VERSION,
    SUPPORTED_OBSERVATION_SERIALIZER_VERSIONS,
    serialize_observation,
)
from sts1_llm_policy.env.state_schema import CanonicalState
from sts1_llm_policy.env.route_context import ROUTE_OBSERVATION_VERSION
from sts1_llm_policy.policy.base import PolicyResult
from sts1_llm_policy.policy.protocol import (
    POLICY_PROTOCOL_VERSION,
    PROMPT_VERSION,
    index_legal_actions,
)


TRAJECTORY_SCHEMA_VERSION = "trajectory_v1"


class TerminalOutcome(str, Enum):
    VICTORY = "victory"
    DEFEAT = "defeat"
    ABORTED = "aborted"


def classify_terminal_outcome(raw_state: dict) -> TerminalOutcome:
    """Classify explicit terminal evidence from a supported environment."""

    if raw_state.get("source") == "simulator":
        simulator_state = raw_state.get("state")
        if not isinstance(simulator_state, dict):
            return TerminalOutcome.ABORTED
        outcome = simulator_state.get("outcome")
        if simulator_state.get("terminal") is not True:
            return TerminalOutcome.ABORTED
        if outcome == "PLAYER_VICTORY":
            return TerminalOutcome.VICTORY
        if outcome == "PLAYER_LOSS":
            return TerminalOutcome.DEFEAT
        return TerminalOutcome.ABORTED

    game_state = raw_state.get("game_state")

    if not isinstance(game_state, dict):
        return TerminalOutcome.ABORTED

    screen_type = game_state.get("screen_type")
    screen_state = game_state.get("screen_state")

    if screen_type == "GAME_OVER" and isinstance(screen_state, dict):
        victory = screen_state.get("victory")

        if victory is True:
            return TerminalOutcome.VICTORY

        if victory is False:
            return TerminalOutcome.DEFEAT

    current_hp = game_state.get("current_hp")

    if (
        isinstance(current_hp, (int, float))
        and not isinstance(current_hp, bool)
        and current_hp <= 0
    ):
        return TerminalOutcome.DEFEAT

    if screen_type in {
        "CARD_REWARD",
        "COMBAT_REWARD",
        "BOSS_REWARD",
        "COMPLETE",
    }:
        return TerminalOutcome.VICTORY

    if game_state.get("room_phase") == "COMPLETE":
        return TerminalOutcome.VICTORY

    return TerminalOutcome.ABORTED


class TrajectoryFormatError(ValueError):
    """Raised when an existing trajectory JSONL record is malformed."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _to_json_value(value: object) -> object:
    """Convert project dataclasses and enums to strict JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_json_value(getattr(value, field.name))
            for field in fields(value)
            if not (field.name == "route_context" and getattr(value, field.name) is None)
            and not (field.name == "hand_upgrade_previews" and not getattr(value, field.name))
            and not (field.name == "source_card" and getattr(value, field.name) is None)
            and not (field.name == "known_draw_top" and getattr(value, field.name) is None)
        }

    if isinstance(value, Enum):
        return _to_json_value(value.value)

    if isinstance(value, Mapping):
        return {
            str(key): _to_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [_to_json_value(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(
        f"Unsupported trajectory value type: {type(value).__name__}"
    )


def _validate_optional_number(
    name: str,
    value: float | None,
) -> None:
    if value is None:
        return

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number or None")

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def load_completed_trajectory(
    path: Path, *, expected_sha256: str, game_seed: int, policy_seed: int,
    scenario_id: str, policy_name: str,
) -> tuple[dict[str, object], ...]:
    """Verify cached combat evidence before reusing its summary."""
    if not path.is_file():
        raise ValueError(f"Completed trajectory is unavailable: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Completed trajectory changed: {path}")
    records = tuple(iter_trajectory_records(path))
    if not records or not isinstance(records[0].get("episode_id"), str) or not records[0]["episode_id"]:
        raise ValueError(f"Completed trajectory has no episode: {path}")
    for index, record in enumerate(records):
        last = index == len(records) - 1
        if (record.get("step_index") != index or record.get("done") is not last
                or record.get("episode_id") != records[0]["episode_id"]
                or record.get("game_seed") != game_seed or record.get("policy_seed") != policy_seed
                or record.get("scenario_id") != scenario_id or record.get("policy_name") != policy_name
                or (not last and record.get("terminal_outcome") is not None)):
            raise ValueError(f"Completed trajectory identity/order is invalid: {path}")
    terminal = records[-1]
    raw = terminal.get("next_raw_state")
    if (not isinstance(raw, dict)
            or terminal.get("terminal_outcome") != classify_terminal_outcome(raw).value):
        raise ValueError(f"Completed trajectory terminal outcome is invalid: {path}")
    return records


def iter_trajectory_records(
    path: str | Path,
) -> Iterator[dict[str, object]]:
    """Read and minimally validate trajectory_v1 JSONL records."""

    trajectory_path = Path(path)

    with trajectory_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise TrajectoryFormatError(
                    f"Blank JSONL record at line {line_number}"
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrajectoryFormatError(
                    f"Invalid JSON at line {line_number}: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise TrajectoryFormatError(
                    f"Record at line {line_number} must be an object"
                )

            if record.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
                raise TrajectoryFormatError(
                    f"Unsupported schema_version at line {line_number}: "
                    f"{record.get('schema_version')!r}"
                )

            if record.get("record_type") != "transition":
                raise TrajectoryFormatError(
                    f"Unsupported record_type at line {line_number}: "
                    f"{record.get('record_type')!r}"
                )

            yield record


class TrajectoryLogger:
    """Append-only JSONL logger for one policy episode."""

    def __init__(
        self,
        path: str | Path,
        *,
        episode_id: str,
        game_seed: int,
        policy_seed: int,
        policy_name: str,
        encounter: str | None = None,
        evidence_class: str | None = None,
        scenario_id: str | None = None,
        preflight: Mapping[str, object] | None = None,
        observation_serializer_version: str = OBSERVATION_SERIALIZER_VERSION,
        allow_source_descriptions: bool = False,
        resume: bool = False,
        fsync: bool = False,
        route_lineage: Mapping[str, object] | None = None,
    ) -> None:
        if not episode_id.strip():
            raise ValueError("episode_id must not be empty")

        if isinstance(game_seed, bool) or not isinstance(game_seed, int):
            raise TypeError("game_seed must be an integer")

        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
            raise TypeError("policy_seed must be an integer")

        if not policy_name.strip():
            raise ValueError("policy_name must not be empty")

        if evidence_class is not None and not evidence_class.strip():
            raise ValueError("evidence_class must be None or a non-empty string")

        if scenario_id is not None and not scenario_id.strip():
            raise ValueError("scenario_id must be None or a non-empty string")

        if (
            observation_serializer_version
            not in SUPPORTED_OBSERVATION_SERIALIZER_VERSIONS
        ):
            raise ValueError(
                "Unsupported observation serializer version: "
                f"{observation_serializer_version}"
            )

        if evidence_class == "parity_clean":
            if scenario_id is None:
                raise ValueError("parity_clean requires scenario_id")
            if preflight is None or preflight.get("accepted") is not True:
                raise ValueError("parity_clean requires an accepted preflight")

        self.path = Path(path)
        self.route_lineage = dict(route_lineage) if route_lineage is not None else None
        self.episode_id = episode_id
        self.game_seed = game_seed
        self.policy_seed = policy_seed
        self.policy_name = policy_name
        self.encounter = encounter
        self.evidence_class = evidence_class
        self.scenario_id = scenario_id
        self.preflight = _to_json_value(preflight)
        self.observation_serializer_version = observation_serializer_version
        self.allow_source_descriptions = allow_source_descriptions
        self._fsync = fsync
        self._next_step_index = 0
        self._finished = False

        if self.path.exists():
            self._inspect_existing_episode(resume=resume)

    @property
    def next_step_index(self) -> int:
        return self._next_step_index

    @property
    def finished(self) -> bool:
        return self._finished

    def _inspect_existing_episode(self, *, resume: bool) -> None:
        episode_records = [
            record
            for record in iter_trajectory_records(self.path)
            if record.get("episode_id") == self.episode_id
        ]

        if not episode_records:
            return

        if not resume:
            raise ValueError(
                f"episode_id already exists in trajectory: {self.episode_id}"
            )

        for expected_index, record in enumerate(episode_records):
            if record.get("route_lineage") != self.route_lineage:
                raise TrajectoryFormatError("Route lineage mismatch for resumed episode")
            if record.get("step_index") != expected_index:
                raise TrajectoryFormatError(
                    f"Non-contiguous step_index for episode "
                    f"{self.episode_id!r}"
                )

            if record.get("game_seed") != self.game_seed:
                raise TrajectoryFormatError(
                    f"game_seed mismatch for episode {self.episode_id!r}"
                )

            if record.get("policy_seed") != self.policy_seed:
                raise TrajectoryFormatError(
                    f"policy_seed mismatch for episode {self.episode_id!r}"
                )

            if record.get("policy_name") != self.policy_name:
                raise TrajectoryFormatError(
                    f"policy_name mismatch for episode {self.episode_id!r}"
                )

            if (
                record.get("observation_serializer_version")
                != self.observation_serializer_version
            ):
                raise TrajectoryFormatError(
                    "observation serializer mismatch for episode "
                    f"{self.episode_id!r}"
                )

            expected_fallback = (
                SOURCE_DESCRIPTION_FALLBACK_ID
                if self.allow_source_descriptions
                else None
            )
            if record.get("source_description_fallback") != expected_fallback:
                raise TrajectoryFormatError(
                    "source description fallback mismatch for episode "
                    f"{self.episode_id!r}"
                )

            selection_record = (
                self.observation_serializer_version in {"observation_v6", ROUTE_OBSERVATION_VERSION}
                and "selection_task" in (record.get("canonical_state") or {})
            )
            expected_action_version = (
                "combat_card_selection_v1" if selection_record else ACTION_EQUIVALENCE_VERSION
            )
            if record.get("action_equivalence_version") != expected_action_version:
                raise TrajectoryFormatError(
                    "action equivalence mismatch for episode "
                    f"{self.episode_id!r}"
                )

            if record.get("evidence_class") != self.evidence_class:
                raise TrajectoryFormatError(
                    f"evidence_class mismatch for episode {self.episode_id!r}"
                )

            if record.get("scenario_id") != self.scenario_id:
                raise TrajectoryFormatError(
                    f"scenario_id mismatch for episode {self.episode_id!r}"
                )

            if record.get("preflight") != self.preflight:
                raise TrajectoryFormatError(
                    f"preflight mismatch for episode {self.episode_id!r}"
                )

        if episode_records[-1].get("done") is True:
            raise ValueError(
                f"Cannot resume finished episode: {self.episode_id}"
            )

        self._next_step_index = len(episode_records)

    def log_transition(
        self,
        *,
        state: CanonicalState,
        legal_actions: Sequence[CanonicalAction],
        policy_result: PolicyResult,
        next_state: CanonicalState | None,
        raw_state: Mapping[str, object] | None = None,
        next_raw_state: Mapping[str, object] | None = None,
        reward: float | None = None,
        score: float | None = None,
        done: bool = False,
        terminal_outcome: TerminalOutcome | None = None,
    ) -> dict[str, object]:
        if self._finished:
            raise RuntimeError("Episode is already finished")

        if state.seed != self.game_seed:
            raise ValueError(
                f"State seed {state.seed} does not match "
                f"game_seed {self.game_seed}"
            )

        if next_state is not None and next_state.seed != self.game_seed:
            raise ValueError(
                f"Next-state seed {next_state.seed} does not match "
                f"game_seed {self.game_seed}"
            )

        actions = tuple(legal_actions)
        action_by_id = index_legal_actions(actions)
        current_action = action_by_id.get(policy_result.action.action_id)

        if current_action is None or current_action != policy_result.action:
            raise ValueError(
                "policy_result.action is not an action from the current "
                "legal action set"
            )

        model_action_space = build_model_action_space(
            state,
            actions,
            allow_source_descriptions=self.allow_source_descriptions,
        )
        selected_class = model_action_space.resolve(
            policy_result.selected_model_action_id
        )
        if selected_class is None:
            raise ValueError(
                "policy_result selected an action outside the current model "
                "action space"
            )
        if (
            policy_result.equivalent_internal_action_ids
            != selected_class.internal_action_ids
        ):
            raise ValueError(
                "policy_result equivalence members disagree with the current "
                "model action space"
            )
        if policy_result.action != selected_class.representative:
            raise ValueError(
                "policy_result.action is not the deterministic representative"
            )

        if done and terminal_outcome is None:
            raise ValueError("A terminal transition requires terminal_outcome")

        if not done and terminal_outcome is not None:
            raise ValueError(
                "A non-terminal transition cannot have terminal_outcome"
            )

        if not done and next_state is None:
            raise ValueError("A non-terminal transition requires next_state")

        _validate_optional_number("reward", reward)
        _validate_optional_number("score", score)

        record = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "record_type": "transition",
            "logged_at_utc": _utc_now(),
            "episode_id": self.episode_id,
            "step_index": self._next_step_index,
            "game_seed": self.game_seed,
            "policy_seed": self.policy_seed,
            "policy_name": self.policy_name,
            "observation_serializer_version": self.observation_serializer_version,
            "source_description_fallback": (
                SOURCE_DESCRIPTION_FALLBACK_ID
                if self.allow_source_descriptions
                else None
            ),
            "action_equivalence_version": model_action_space.version,
            "policy_protocol_version": POLICY_PROTOCOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "encounter": self.encounter,
            "evidence_class": self.evidence_class,
            "scenario_id": self.scenario_id,
            "preflight": self.preflight,
            "turn": state.combat.turn,
            "raw_state": _to_json_value(raw_state),
            "canonical_state": _to_json_value(state),
            "serialized_state": serialize_observation(
                state,
                actions,
                version=self.observation_serializer_version,
                allow_source_descriptions=self.allow_source_descriptions,
            ),
            "legal_actions": _to_json_value(actions),
            "model_legal_actions": _to_json_value(
                model_action_space_records(model_action_space)
            ),
            "action": _to_json_value(policy_result.action),
            "policy_result": _to_json_value(policy_result),
            "reward": reward,
            "score": score,
            "next_raw_state": _to_json_value(next_raw_state),
            "next_canonical_state": _to_json_value(next_state),
            "done": done,
            "terminal_outcome": _to_json_value(terminal_outcome),
        }

        if self.route_lineage is not None:
            record["route_lineage"] = _to_json_value(self.route_lineage)
        if state.route_context is not None:
            route_done = done and (
                terminal_outcome in {TerminalOutcome.DEFEAT, TerminalOutcome.ABORTED}
                or (terminal_outcome == TerminalOutcome.VICTORY
                    and state.route_context.current_encounter_family == "boss")
            )
            record["route_done"] = route_done
            record["route_terminal_outcome"] = (
                _to_json_value(terminal_outcome) if route_done else None
            )

        self._append(record)
        self._next_step_index += 1
        self._finished = done

        return record

    def _append(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()

            if self._fsync:
                os.fsync(stream.fileno())
