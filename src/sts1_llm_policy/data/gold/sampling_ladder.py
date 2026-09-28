"""Deterministic paired-trial stopping rules and optional shadow calibration.

These are empirical scheduling heuristics, not confidence guarantees or SFT labels.
"""
from __future__ import annotations

import hashlib
import json
import math


def stage_decision(summary, previous, config, sample_id, count):
    rule = config["sampling_ladder"]
    if summary["status"] != "completed":
        raise ValueError("Cannot schedule from incomplete paired trials")
    actions = summary["actions"]
    candidates = sorted(summary["estimated_acceptable_actions"])
    eligible = [a for a in actions if a["passes_win_floor"]]
    best = max((a["expected_carried_hp"] for a in eligible), default=None)
    tolerance, band = config["expected_hp_tolerance"], rule["hp_boundary_band"]
    required = summary["minimum_wins"]
    relevant = [a for a in actions if best is None or a["expected_carried_hp"] >= best - tolerance - band]
    win_boundary = sorted(a["action"] for a in relevant if a["wins"] in {required - 1, required})
    hp_boundary = sorted(a["action"] for a in eligible
                         if abs(best - a["expected_carried_hp"] - tolerance) <= band)
    # Add uncertainty at the 64-trial decision only; old shadow identities retain
    # their original fixed band and exact report shape when the option is absent.
    variance_boundary = []
    multiplier = rule.get("paired_se_multiplier_at_64", 0)
    if count == 64 and multiplier and eligible:
        reference = min(eligible, key=lambda a: (-a["expected_carried_hp"], a["action"]))
        pairs = {frozenset((p["a"], p["b"])): p for p in summary["paired_comparisons"]}
        for action in eligible:
            if action["action"] == reference["action"]:
                continue
            se = pairs[frozenset((reference["action"], action["action"]))]["standard_error"]
            if se is None or not math.isfinite(se):
                raise ValueError("Missing finite paired uncertainty for ladder decision")
            if abs(best - action["expected_carried_hp"] - tolerance) <= max(band, multiplier * se):
                variance_boundary.append(action["action"])
        hp_boundary = sorted(set(hp_boundary) | set(variance_boundary))
    changed = previous is not None and candidates != previous["candidates"]
    kind = "empty" if not candidates else "all" if len(candidates) == len(actions) else "proper"
    reasons = []
    if count == 32 and kind == "proper":
        reasons.append("confirm_proper_set")
    if count >= 64 and changed:
        reasons.append("candidate_set_changed")
    if hp_boundary:
        reasons.append("hp_boundary")
    if win_boundary:
        reasons.append("win_boundary")
    key = json.dumps([rule["audit_seed"], sample_id, count], separators=(",", ":")).encode()
    audit_draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    cap = count == rule["stages"][-1]
    audit = not cap and not reasons and audit_draw < rule["audit_probability"]
    old_candidates = set(previous["candidates"]) if previous else set(candidates)
    statuses = {a["action"]: ("flip" if (a["action"] in old_candidates) != (a["action"] in candidates)
                else "borderline" if a["action"] in win_boundary + hp_boundary
                else "stable_pass" if a["action"] in candidates else "stable_fail") for a in actions}
    # A near-gate, high-HP root may change the reference and hence other labels.
    reference_sensitive = bool(best is not None and any(
        a["action"] in win_boundary and a["expected_carried_hp"] >= best for a in relevant))
    result = {"trials": count, "kind": kind, "candidates": candidates,
            "minimum_wins": required, "actions": actions,
            "hp_boundary_actions": hp_boundary, "win_boundary_actions": win_boundary,
            "candidate_set_changed": changed, "reasons": reasons,
            "audit_selected": audit, "would_expand": not cap and (bool(reasons) or audit),
            "at_cap": cap, "action_stability": statuses,
            "reference_sensitive": reference_sensitive,
            "labels_certified": False}
    if "paired_se_multiplier_at_64" in rule:
        result["variance_boundary_actions"] = sorted(variance_boundary)
    return result


def ladder_records(rows, config, sample_id, summarize):
    records = []
    for count in config["sampling_ladder"]["stages"]:
        subset = [r for r in rows if r["trial"] < count]
        summary = summarize(subset, {**config, "outer_trials": count})
        stage = stage_decision(summary, records[-1] if records else None, config, sample_id, count)
        stage.update(executions=len(subset), search_calls=sum(r["search_calls"] for r in subset),
                     executed_decisions=sum(r["decisions"] for r in subset))
        records.append(stage)
        if config["sampling_ladder"]["mode"] == "adaptive" and not stage["would_expand"]:
            break
    return records


def ladder_statistics(states, config):
    groups = config.get("panel_groups", {"panel": [s["sample"]["id"] for s in states]})
    by_id = {s["sample"]["id"]: s for s in states}
    result = {}
    for name, ids in groups.items():
        rows = [by_id[key] for key in ids]
        stops = [(s, next(r for r in s["sampling_ladder"] if not r["would_expand"])) for s in rows]
        if config["sampling_ladder"]["mode"] == "adaptive":
            result[name] = {"states": len(rows),
                            "actual_stops": {str(n): sum(r["trials"] == n for _, r in stops) for n in (32, 64, 128)},
                            "executions": sum(s["executions"] for s in rows),
                            "search_calls": sum(s["search_calls"] for s in rows),
                            "full_128_reference_available": False}
            continue
        early = [(s, r) for s, r in stops if r["trials"] < 128]
        changed = sum(r["candidates"] != sorted(s["estimated_acceptable_actions"]) for s, r in early)
        total_cost = sum(s["search_calls"] for s in rows)
        proposed_cost = sum(r["search_calls"] for _, r in stops)
        result[name] = {"states": len(rows), "early_stops": len(early),
                        "early_stop_set_changes_at_128": changed,
                        "early_stop_set_change_rate": changed / len(early) if early else None,
                        "proposed_stops": {str(n): sum(r["trials"] == n for _, r in stops) for n in (32, 64, 128)},
                        "proposed_search_calls": proposed_cost, "full_128_search_calls": total_cost,
                        "search_saving_fraction": 1 - proposed_cost / total_cost if total_cost else None}
    return result
