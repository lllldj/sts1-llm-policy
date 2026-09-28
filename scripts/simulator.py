"""Resolve, verify, build, or smoke-check the native simulator on this host."""
import argparse
import json
from pathlib import Path
import subprocess

from sts1_llm_policy.env.simulator_execution import CORRECTED_MECHANICS, LEGACY_MECHANICS, resolve_simulator
from sts1_llm_policy.env.simulator_installation import SimulatorInstallationError
from sts1_llm_policy.env.simulator_client import SimulatorRemoteError


def smoke(execution):
    """Check the live protocol in one bounded combat, without a scenario matrix."""
    client = execution.create_client()
    try:
        def current_decision():
            response = client.legal_actions()
            return response["decision_id"], response["legal_actions"]

        first = client.reset("cultist", 12345)
        second = client.reset("cultist", 12345)
        if any(first[key] != second[key] for key in ("state", "legal_actions")):
            raise RuntimeError("Same scenario/seed produced different reset observations")
        decision = second["state"]["decision_id"]
        before = current_decision()
        try:
            client.step(decision, "NOT_A_LEGAL_ACTION")
        except SimulatorRemoteError as error:
            if error.code != "illegal_action":
                raise
        else:
            raise RuntimeError("Bridge accepted an illegal action")
        if current_decision() != before:
            raise RuntimeError("Rejected action changed the bridge state")
        searches = [client.search(decision, simulations=64, search_seed=17,
                                  minimum_root_action_visits=2) for _ in range(2)]
        if searches[0]["root_actions"] != searches[1]["root_actions"]:
            raise RuntimeError("Fixed-seed search is not deterministic")
        roots = searches[0]["root_actions"]
        legal = {action["action_id"] for action in second["legal_actions"]}
        # Native search deduplicates equivalent actions into representative edges.
        if (not roots or not {row["action_id"] for row in roots} <= legal
                or any(row["visits"] < 2 for row in roots)
                or any(row["terminal_wins"] + row["terminal_losses"] != row["visits"] for row in roots)
                or sum(row["visits"] for row in roots) != 64
                or searches[0]["root_simulation_count"] != 64
                or searches[0]["minimum_root_action_visits"] != 2
                or searches[0]["root_action_coverage_policy"] != "minimum_root_edge_visits_before_ucb_v1"
                or searches[0]["suggested_action_id"] not in legal):
            raise RuntimeError("Search violated legal-action coverage or its visit budget")
        if current_decision() != before:
            raise RuntimeError("Search changed the current decision")
        client.step(decision, "END")
        for operation in (lambda: client.step(decision, "END"),
                          lambda: client.search(decision, simulations=1, search_seed=17)):
            try:
                operation()
            except SimulatorRemoteError as error:
                if error.code != "stale_decision":
                    raise
            else:
                raise RuntimeError("Bridge accepted a stale decision")
        return {"status": "passed", "deterministic_reset": True,
                "illegal_and_stale_actions_rejected": True, "stale_search_rejected": True,
                "deterministic_search": True, "search_preserves_state": True,
                "legal_root_coverage_and_budget": True}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("resolve", "verify", "build", "smoke"))
    parser.add_argument("--target", choices=("auto", "windows", "linux"), default="auto")
    parser.add_argument("--mechanics", choices=(CORRECTED_MECHANICS, LEGACY_MECHANICS), default=CORRECTED_MECHANICS)
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--report", type=Path, help="Write a new JSON report; existing files are preserved")
    args = parser.parse_args()
    if args.skip_smoke and args.action != "build":
        parser.error("--skip-smoke only applies to build")
    if args.report and args.report.exists():
        parser.error(f"Report already exists: {args.report}")
    try:
        execution = resolve_simulator(target=args.target, required_capabilities=args.require, mechanics=args.mechanics)
        if args.action == "build":
            execution.build()
        result = execution.describe(verify=args.action != "resolve")
        if args.action == "smoke" or (args.action == "build" and not args.skip_smoke):
            result["smoke"] = smoke(execution)
    except (SimulatorInstallationError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Simulator unavailable: {error}\n")
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
