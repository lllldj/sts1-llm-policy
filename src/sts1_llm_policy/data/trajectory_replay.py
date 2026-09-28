"""Restore collected simulator decisions from a starting snapshot and actions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any, Mapping, Sequence

from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.env.simulator_env import SimulatorCombatEndedError
from sts1_llm_policy.data.trajectory import classify_terminal_outcome


def comparable_execution_state(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Copy execution evidence without its bridge-session decision ID."""
    result = deepcopy(dict(raw))
    # Decision IDs belong to a bridge session, not to combat mechanics.
    result["state"].pop("decision_id", None)
    return result


def replay_combat(
    env, *, scenario_id: str, combat_seed: int,
    snapshot: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    route_context=None, stop_before: int | None = None,
    public_observation_consumer=None,
) -> dict[str, int]:
    """Verify every transition, or leave env at a verified decision for branching.

    No Teacher search is rerun. The executed canonical representative is replayed,
    including secondary selections. Hidden state remains simulator evidence only.

    An optional consumer receives (index, current public state, legal actions).
    It enables known draw memory without resampling RNG. Historical evidence is
    still checked with its recorded memory exposure; original observations and
    raw evidence are never rewritten.
    """
    if not records:
        raise ValueError("Cannot replay an empty combat")
    if stop_before is not None and (
        isinstance(stop_before, bool) or not isinstance(stop_before, int)
        or not 0 <= stop_before < len(records)
    ):
        raise ValueError("stop_before must identify a recorded decision")
    env.reset(scenario_id, combat_seed, combat_snapshot=snapshot, route_context=route_context)
    def comparable(actual, recorded):
        actual = comparable_execution_state(actual)
        if public_observation_consumer is not None and "public_draw_memory" not in recorded["state"]:
            actual["state"].pop("public_draw_memory", None)
        return actual

    selections = 0
    for index, record in enumerate(records):
        if record["step_index"] != index or record["game_seed"] != combat_seed:
            raise ValueError(f"Replay lineage/index mismatch at decision {index}")
        if comparable(env.get_raw_state(), record["raw_state"]) != comparable_execution_state(record["raw_state"]):
            raise ValueError(f"Replay state mismatch at decision {index}")
        state, actions = env.get_state(), env.legal_actions()
        recorded_state = state
        if public_observation_consumer is not None and "known_draw_top" not in record["canonical_state"]:
            recorded_state = replace(state, known_draw_top=None)
        serialized = serialize_observation(
            recorded_state, actions, version=record["observation_serializer_version"],
        )
        if serialized != record["serialized_state"]:
            raise ValueError(f"Replay observation mismatch at decision {index}")
        if public_observation_consumer is not None:
            public_observation_consumer(index, env.public_state(), env.legal_actions())
            actions = env.legal_actions()
        if stop_before == index:
            return {"verified_transitions": index, "restored_decision": index,
                    "secondary_selection_decisions": selections}
        action = next((a for a in actions if asdict(a) == record["action"]), None)
        if action is None:
            raise ValueError(f"Replay action mismatch at decision {index}")
        selections += "selection_task" in record["canonical_state"]
        terminal_outcome = None
        try:
            env.step(action)
            after = env.get_raw_state()
        except SimulatorCombatEndedError as terminal:
            after = terminal.raw_state
            terminal_outcome = classify_terminal_outcome(after).value
        if comparable(after, record["next_raw_state"]) != comparable_execution_state(record["next_raw_state"]):
            raise ValueError(f"Replay transition mismatch at decision {index}")
        expected = record["terminal_outcome"]
        if terminal_outcome != (None if expected == "aborted" else expected):
            raise ValueError(f"Replay terminal mismatch at decision {index}")
        if record["done"] != (index == len(records) - 1):
            raise ValueError("Replay requires one complete combat trajectory")
    return {"verified_transitions": len(records),
            "secondary_selection_decisions": selections}
