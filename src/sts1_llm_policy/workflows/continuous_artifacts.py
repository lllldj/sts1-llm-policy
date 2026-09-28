"""Write continuous outputs and validate completed routes without executing policies."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from sts1_llm_policy.data.trajectory import load_completed_trajectory
from sts1_llm_policy.artifacts import (
    read_json_object,
    replace_json,
    canonical_json_bytes,
    sha256_bytes,
)


OUTPUT_SCHEMA_VERSION = "continuous_combat_panel_inputs_v1"
REPORT_SCHEMA_VERSION = "continuous_combat_panel_generation_report_v1"
IDENTITY_SCHEMA_VERSION = "continuous_combat_configuration_identity_v2"
CANDIDATE_EVIDENCE_CLASS = "teacher_candidate_pool"


def canonical_sha(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def summarize_combat(records: Sequence[Mapping[str, Any]], outcome: str) -> dict[str, Any]:
    if not records or records[-1].get("done") is not True:
        raise ValueError("Continuous combat trajectory is incomplete")
    envelope = records[-1].get("next_raw_state")
    terminal = envelope.get("state") if isinstance(envelope, Mapping) else None
    if not isinstance(terminal, Mapping):
        raise ValueError("Continuous combat terminal state is missing")
    player = terminal.get("player")
    accounting = terminal.get("combat_accounting")
    if not isinstance(player, Mapping) or not isinstance(accounting, Mapping):
        raise ValueError("Continuous combat terminal accounting is missing")
    policy_rows = [row.get("policy_result") for row in records]
    if any(not isinstance(row, Mapping) for row in policy_rows):
        raise ValueError("Continuous combat policy result is missing")
    return {
        "outcome": outcome,
        "decisions": len(records),
        "turns": max(int(row.get("turn", 0)) for row in records),
        "starting_hp": int(accounting["starting_hp"]),
        "ending_hp": int(player["current_hp"]),
        "max_hp": int(player["max_hp"]),
        "total_hp_loss": int(accounting["total_hp_loss"]),
        "relics": [
            {"id": str(item["id"]), "counter": int(item.get("counter", 0))}
            for item in terminal.get("relics", [])
        ],
        "protocol": {
            "retry_count": sum(row.get("retry_used") is True for row in policy_rows),
            "fallback_count": sum(row.get("fallback_used") is True for row in policy_rows),
            "first_pass_legal_count": sum(
                row.get("legal_on_first_attempt") is True for row in policy_rows
            ),
        },
    }


def _check_resume_identity(actual, expected, path):
    if not isinstance(actual, Mapping) or actual.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported continuous resume identity (legacy or missing): {path}. "
            "Preserve historical artifacts and select a new output directory."
        )
    if actual != expected:
        raise ValueError(f"Continuous resume configuration changed: {path}")


def prepare_output(output: Path, identity: Mapping[str, Any]) -> None:
    # Check every arm before writing, including report-only historical directories.
    marker = output / "configuration_identity.json"
    evidence = [
        *output.glob("report*.json"), *output.glob("inputs*.json"),
        *output.glob("*/routes/*.json"),
    ]
    if marker.exists():
        _check_resume_identity(read_json_object(marker), identity, marker)
    elif output.exists() and any(output.iterdir()) and not evidence:
        raise ValueError(f"Continuous output has no resume identity: {output}; select a new output directory")
    for path in evidence:
        _check_resume_identity(read_json_object(path).get("configuration_identity"), identity, path)
    # Also covers interruption before the first route is complete.
    if not marker.exists():
        replace_json(marker, identity)


def load_completed_route(
    path: Path,
    *,
    identity: Mapping[str, Any],
    arm: str,
    route: Mapping[str, Any],
    output_root: Path,
    evidence_class: str | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    item = read_json_object(path)
    _check_resume_identity(item.get("configuration_identity"), identity, path)
    if (
        item.get("schema_version") != "continuous_combat_route_v1"
        or item.get("arm") != arm
        or item.get("route") != route
    ):
        raise ValueError(f"Completed continuous route configuration changed: {path}")
    combats = item.get("combats")
    planned = [step for step in route["steps"] if step["kind"] == "combat"]
    if (not isinstance(combats, list) or not combats or len(combats) > len(planned)
            or item.get("combats_started") != len(combats)):
        raise ValueError(f"Completed continuous route accounting is invalid: {path}")
    root = output_root.resolve()
    for combat, expected in zip(combats, planned):
        if (not isinstance(combat, Mapping) or any(combat.get(key) != expected[key] for key in (
                "combat_index", "floor", "encounter_family", "scenario_id", "combat_seed", "policy_seed"))):
            raise ValueError(f"Completed continuous combat differs from its planned position: {path}")
        snapshot = combat.get("input_snapshot")
        if combat.get("input_snapshot_sha256") != canonical_sha(snapshot):
            raise ValueError(f"Completed continuous combat snapshot changed: {path}")
        relative = combat.get("trajectory")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Completed continuous combat trajectory is missing: {path}")
        trajectory = (root / relative).resolve()
        if not trajectory.is_relative_to(root) or not trajectory.is_file():
            raise ValueError(f"Completed continuous combat trajectory is unavailable: {path}")
        records = load_completed_trajectory(
            trajectory, expected_sha256=combat.get("trajectory_sha256"),
            game_seed=expected["combat_seed"], policy_seed=expected["policy_seed"],
            scenario_id=expected["scenario_id"], policy_name=arm,
        )
        summary = combat.get("summary")
        actual = summarize_combat(records, records[-1]["terminal_outcome"])
        if not isinstance(summary, dict) or {k: v for k, v in summary.items() if k != "replay"} != actual:
            raise ValueError(f"Completed continuous combat summary differs from its trajectory: {path}")
        if evidence_class == CANDIDATE_EVIDENCE_CLASS:
            lineage = {**item["route_lineage"], "combat_index": combat["combat_index"]}
            if (snapshot.get("schema_version") != "combat_snapshot_v2"
                    or combat["summary"].get("replay", {}).get("verified_transitions") != len(records)
                    or combat["summary"]["decisions"] != len(records)
                    or any(r.get("evidence_class") != evidence_class
                           or r.get("route_lineage") != lineage
                           or r.get("game_seed") != combat["combat_seed"]
                           or r.get("step_index") != i
                           for i, r in enumerate(records))):
                raise ValueError(f"Candidate trajectory source/replay evidence is invalid: {path}")
    outcomes = [combat["summary"]["outcome"] for combat in combats]
    last = outcomes[-1]
    status = {"victory": "completed", "defeat": "defeated", "aborted": "truncated"}[last]
    if (any(outcome != "victory" for outcome in outcomes[:-1]) or item.get("status") != status
            or item.get("combats_won") != outcomes.count("victory")
            or item.get("test_data_read") is not False
            or (status == "completed" and (len(combats) != len(planned)
                                           or planned[-1]["encounter_family"] != "boss"))
            or item.get("death_combat_index") != (combats[-1]["combat_index"] if status == "defeated" else None)):
        raise ValueError(f"Completed continuous route outcome is inconsistent: {path}")
    if status == "truncated":
        if (item.get("truncation_reason") != "decision_limit"
                or item.get("truncation_combat_index") != combats[-1]["combat_index"]):
            raise ValueError(f"Completed continuous route truncation is invalid: {path}")
    elif item.get("truncation_reason") is not None or item.get("truncation_combat_index") is not None:
        raise ValueError(f"Completed continuous route has unexpected truncation: {path}")
    return item


def summarize_arm(routes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    combats = [combat for route in routes for combat in route["combats"]]
    death_indices = [int(route["death_combat_index"]) for route in routes if route["death_combat_index"]]
    decisions = sum(int(item["summary"]["decisions"]) for item in combats)
    return {
        "route_count": len({int(route["route"]["route_index"]) for route in routes}),
        "route_execution_count": len(routes),
        "combat_seed_groups_per_route": len({
            int(route["route"]["combat_seed_group_index"]) for route in routes
        }),
        "boss_victories": sum(route["status"] == "completed" for route in routes),
        "truncated_routes": sum(route["status"] == "truncated" for route in routes),
        "boss_arrivals": sum(
            combat.get("encounter_family") == "boss" for combat in combats
        ),
        "combat_count": len(combats),
        "combat_victories": sum(item["summary"]["outcome"] == "victory" for item in combats),
        "death_combat_index_counts": dict(sorted(Counter(death_indices).items())),
        "death_combat_index_mean": round(mean(death_indices), 3) if death_indices else None,
        "decisions": decisions,
        "retry_count": sum(item["summary"]["protocol"]["retry_count"] for item in combats),
        "fallback_count": sum(item["summary"]["protocol"]["fallback_count"] for item in combats),
    }
