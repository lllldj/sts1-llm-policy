"""Shared verified GOLD source selection, independent of the training algorithm."""
from collections import Counter
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import math

from sts1_llm_policy.artifacts import canonical_json_bytes


def candidate_ids(state, tolerance):
    count = state["outer_trials_completed"]
    required = int((Decimal(str(state["minimum_win_rate"])) * count).to_integral_value(rounding=ROUND_FLOOR))
    scores = state["actions"]
    if (not isinstance(count, int) or isinstance(count, bool) or count <= 0
            or state.get("win_floor_rounding") != "floor"
            or state.get("minimum_wins") != required):
        raise ValueError("Unsupported or inconsistent executed win gate")
    for a in scores:
        if (a["trials"] != count or not isinstance(a["wins"], int)
                or not 0 <= a["wins"] <= count
                or not math.isfinite(a["expected_carried_hp"]) or a["expected_carried_hp"] < 0):
            raise ValueError("Invalid executed candidate scores")
    eligible = [a for a in scores if a["wins"] >= required]
    best = max((a["expected_carried_hp"] for a in eligible), default=0)
    return {a["action"] for a in eligible if best - a["expected_carried_hp"] <= tolerance}


def replay_selection(states, selection, report_digest):
    """Resolve an explicit, report-bound subset without changing the source panel."""
    if (not isinstance(selection, dict)
            or selection.get("schema_version") != "gold_replay_selection_v1"
            or selection.get("report_canonical_sha256") != report_digest
            or type(selection.get("seed")) is not int):
        raise ValueError("Invalid or stale GOLD replay selection")
    groups = selection.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Replay selection requires named groups")
    ids, names = [], set()
    for group in groups:
        if (not isinstance(group, dict) or not isinstance(group.get("name"), str)
                or not group["name"] or group["name"] in names
                or not isinstance(group.get("state_ids"), list) or not group["state_ids"]
                or any(not isinstance(i, str) for i in group["state_ids"])):
            raise ValueError("Invalid replay selection group")
        names.add(group["name"])
        ids.extend(group["state_ids"])
    by_id = {s["sample"]["id"]: s for s in states}
    if (len(by_id) != len(states) or len(set(ids)) != len(ids)
            or not set(ids) <= set(by_id) or len(ids) >= len(states)):
        raise ValueError("Replay selection must be a unique, nonempty proper subset")
    return [by_id[i] for i in ids]


def select_gold_states(report, verification, hp_tolerance, *, verification_scope="full"):
    """Validate the full collection and select proper GOLD subsets in stable order."""
    cfg, states = report["configuration"], report["states"]
    partition = cfg.get("source_partition", {})
    if (report.get("status") != "completed" or verification.get("status") != "completed"
            or cfg.get("observation_version") != "observation_v7"
            or partition.get("role") != "train_candidate" or not states):
        raise ValueError("Export requires a completed, verified V7 training-candidate collection")
    if verification.get("report_canonical_sha256") != sha256(canonical_json_bytes(report)).hexdigest():
        raise ValueError(
            "GOLD verification is missing a matching report_canonical_sha256; "
            "rerun collect_gold.py --verify on this report and use the new receipt"
        )
    if (isinstance(hp_tolerance, bool) or not isinstance(hp_tolerance, (int, float))
            or not math.isfinite(hp_tolerance) or hp_tolerance < 0):
        raise ValueError("Invalid hp_tolerance")
    ids = [s["sample"]["id"] for s in states]
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(s["id"] for s in cfg["samples"]):
        raise ValueError("State IDs differ from the declared complete panel")
    declared = {s["id"]: s for s in cfg["samples"]}
    excluded = set(partition["excluded_routes"]) | set(partition["reserved_routes"])
    if any(s["sample"] != declared[s["sample"]["id"]] or s["sample"]["route"] in excluded for s in states):
        raise ValueError("State lineage differs from the declared train partition")
    scope = verification.get("scope", "full")
    if verification_scope not in {"full", "sampled"} or scope != verification_scope:
        raise ValueError("Replay scope differs from the export's explicit verification_scope")
    checked_states = states
    if scope == "sampled":
        checked_states = replay_selection(states, verification.get("selection"),
                                          verification["report_canonical_sha256"])
        if (verification.get("verified_states") != len(checked_states)
                or verification.get("total_states") != len(states)):
            raise ValueError("Sampled verification state counts differ from the selection")
    elif verification.get("selection") is not None:
        raise ValueError("Full verification cannot carry a sampled selection")
    for source, verified in (("executions", "verified_executions"), ("executed_decisions", "verified_decisions")):
        if sum(s[source] for s in checked_states) != verification.get(verified):
            raise ValueError("Final verification counts differ from the collection")
    selected, counts = [], Counter()
    for state in sorted(states, key=lambda s: s["sample"]["id"]):
        if state["status"] != "completed":
            raise ValueError("Partial state in completed collection")
        actions = state["root_actions"]
        legal = [a["id"] for a in actions]
        scored = [a["action"] for a in state["actions"]]
        if len(set(legal)) != len(legal) or len(set(scored)) != len(scored) or set(scored) != set(legal):
            raise ValueError("Root metadata and executed action scores differ")
        if state["executions"] != len(actions) * state["outer_trials_completed"]:
            raise ValueError("Executed count differs from complete paired roots")
        if candidate_ids(state, cfg["expected_hp_tolerance"]) != set(state["estimated_acceptable_actions"]):
            raise ValueError("Recorded candidates differ from executed scores")
        accepted = candidate_ids(state, hp_tolerance)
        if not accepted or len(accepted) == len(actions):
            counts["empty_states" if not accepted else "all_pass_states"] += 1
            continue
        selected.append((state, accepted))
    if not selected:
        raise ValueError("No proper candidate subsets remain")
    return selected, dict(counts)


def decision_fields(state, *, dataset_id, observation_version):
    """Build public decision metadata shared by SFT and preference records."""
    observation = state["observation"]
    return {
        "dataset_id": dataset_id, "record_id": state["sample"]["id"], "split": "train",
        "observation_version": observation_version, "observation": observation,
        "observation_sha256": sha256(observation.encode("utf-8")).hexdigest(),
        "source": {"sample": state["sample"], "encounter_family": state["encounter_family"]},
        "model_actions": [{"model_action_id": a["id"], "action_type": a["action_type"],
                           "card_name": a["card"], "target": a["target"]}
                          for a in state["root_actions"]],
    }
