"""Verify a delivered continuous Teacher pool without loading models or simulating."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts1_llm_policy.workflows.continuous_artifacts import (
    CANDIDATE_EVIDENCE_CLASS,
    summarize_arm,
    load_completed_route,
)
from sts1_llm_policy.workflows.continuous_plan import (
    load_scope,
    build_route_lineage,
    build_route_steps,
    validate_source_isolation,
)
from sts1_llm_policy.eval.generation_strategies import resolve_generation_strategies
from sts1_llm_policy.artifacts import read_json_object, resolve_repository_path

ROOT = Path(__file__).resolve().parents[1]


def _verify_arm_metrics(rows, reported):
    # Reports are JSON: integer Counter keys become strings at that boundary.
    computed = json.loads(json.dumps(summarize_arm(rows), allow_nan=False))
    if computed != reported:
        raise ValueError("Aggregate metrics disagree with trajectories")


def verify(report_path: Path, *, project_root: Path = ROOT, exclusions: list[str] | None = None) -> dict:
    report_path = report_path.resolve()
    output = report_path.parent
    report = read_json_object(report_path)
    config = report["configuration"]
    if config.get("data_source", {}).get("evidence_class") != CANDIDATE_EVIDENCE_CLASS:
        raise ValueError("Report does not describe a Teacher candidate pool")
    count = report["scope"]["routes"]
    if report["status"] == "completed":
        if count != config["route_count"]:
            raise ValueError("Formal collection has an incomplete route count")
    elif report["status"] != "smoke_completed" or not 1 <= count <= config["route_count"]:
        raise ValueError("Candidate collection is incomplete")
    scope_path = resolve_repository_path(project_root, config["scope"], expected_kind="file")
    scope = load_scope(scope_path, project_root=project_root)
    strategy = resolve_generation_strategies(config["strategies"]).route
    all_routes = [
        {"route_index": i, "combat_seed_group_index": j,
         "steps": build_route_steps(config, scope, i, j, strategy)}
        for i in range(config["route_count"])
        for j in range(config["combat_seed_groups_per_route"])
    ]
    isolation_config = config
    if exclusions is not None:
        isolation_config = {**config, "data_source": {
            "evidence_class": CANDIDATE_EVIDENCE_CLASS, "excluded_sources": exclusions,
        }}
    elif "excluded_sources" not in config.get("data_source", {}):
        raise ValueError("Report uses retired exclusion configs; supply explicit --exclusions inputs")
    isolation = validate_source_isolation(isolation_config, all_routes, project_root=project_root)
    if isolation != report["source_validation"]:
        raise ValueError("Source isolation evidence disagrees with declared inputs")
    expected_routes = [r for r in all_routes if r["route_index"] < count]
    artifact = (output / report["artifact"]).resolve()
    if not artifact.is_relative_to(output):
        raise ValueError("Inputs must remain inside the delivered output directory")
    inputs = read_json_object(artifact)
    if (inputs["configuration_identity"] != report["configuration_identity"]
            or inputs["run_id"] != config["run_id"]
            or inputs["source_validation"] != isolation
            or inputs["routes"] != expected_routes
            or set(inputs["arms"]) != {a["id"] for a in config["arms"]}
            or report["scope"]["route_executions"] != len(expected_routes)):
        raise ValueError("Candidate inputs/report/config disagree")
    transitions = 0
    for arm, rows in inputs["arms"].items():
        if len(rows) != len(expected_routes):
            raise ValueError("Candidate arm is incomplete")
        for route, item in zip(expected_routes, rows, strict=True):
            lineage = {**build_route_lineage(config, route), "arm": arm}
            if item["route_lineage"] != lineage:
                raise ValueError("Source route lineage mismatch")
            path = output / arm / "routes" / (
                f"route-{route['route_index']:03d}-combat-seed-group-{route['combat_seed_group_index']:02d}.json"
            )
            checked = load_completed_route(
                path, identity=inputs["configuration_identity"], arm=arm, route=route,
                output_root=output, evidence_class=CANDIDATE_EVIDENCE_CLASS,
            )
            if checked != item:
                raise ValueError("Route output differs from inputs manifest")
            combats = item["combats"]
            steps = [s for s in route["steps"] if s["kind"] == "combat"]
            if not combats or len(combats) > len(steps):
                raise ValueError("Invalid combat count")
            for combat, step in zip(combats, steps):
                if any(combat[k] != step[k] for k in (
                    "combat_index", "scenario_id", "combat_seed", "policy_seed",
                )):
                    raise ValueError("Combat schedule mismatch")
                transitions += combat["summary"]["replay"]["verified_transitions"]
            if any(c["summary"]["outcome"] != "victory" for c in combats[:-1]):
                raise ValueError("Route continued after a non-victory")
            outcome = combats[-1]["summary"]["outcome"]
            expected_status = {"victory": "completed", "defeat": "defeated", "aborted": "truncated"}[outcome]
            if (item["status"] != expected_status
                    or (expected_status == "completed" and len(combats) != len(steps))):
                raise ValueError("Route terminal status mismatch")
        _verify_arm_metrics(rows, report["metrics"][arm])
    return {"status": "verified", "collection_status": report["status"],
            "run_id": config["run_id"], "scope": report["scope"],
            "verified_transitions": transitions, "labels_certified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exclusions", action="append", help="Explicit source exclusions for retained reports; repeat for multiple inputs")
    args = parser.parse_args()
    print(json.dumps(verify(args.report, exclusions=args.exclusions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
