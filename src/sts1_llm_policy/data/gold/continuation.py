"""Shared GOLD continuation rules, source restoration and scoring."""
from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, ROUND_FLOOR
import math
import statistics
import threading

from sts1_llm_policy.artifacts import read_json, canonical_json_bytes, sha256_bytes, sha256_file, resolve_repository_path
from sts1_llm_policy.data.trajectory import iter_trajectory_records
from sts1_llm_policy.data.gold.configuration import (
    OBSERVATION_VERSION,
    seed_for,
    validate_continuation_policy,
)
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.env.route_context import public_route_context
from sts1_llm_policy.data.trajectory_replay import replay_combat


PROTOCOL = "executed_teacher_known_top_fresh_rng_v1"


class RouteProgress:
    """Thread-safe progress over declared route decisions, including resumed states."""
    def __init__(self, samples, label):
        self.label = label
        self.samples = {s["id"]: s for s in samples}
        self.routes = {}
        for s in samples:
            self.routes.setdefault(s["route"], set()).add(s["id"])
        self.done = set()
        self.lock = threading.Lock()

    def update(self, sample=None, *, finished=False, detail=""):
        with self.lock:
            if finished:
                self.done.add(sample["id"])
            routes_done = sum(ids <= self.done for ids in self.routes.values())
            state_count, route_count = len(self.samples), len(self.routes)
            message = (f"[{self.label}] routes {routes_done}/{route_count} ({100 * routes_done / route_count:.2f}%)"
                       f" | states {len(self.done)}/{state_count} ({100 * len(self.done) / state_count:.2f}%)")
            if sample:
                message += f" | {sample['id']}"
            print(message + (f" | {detail}" if detail else ""), flush=True)


def actual_trials(config, state):
    if config.get("sampling_ladder", {}).get("mode") != "adaptive":
        return config["outer_trials"]
    count = state.get("outer_trials_completed")
    if isinstance(count, bool) or count not in config["sampling_ladder"]["stages"]:
        raise ValueError("Missing valid adaptive trial count")
    stages = state.get("sampling_ladder", [])
    if not stages or stages[-1]["trials"] != count or stages[-1]["would_expand"]:
        raise ValueError("Adaptive run did not stop at its declared decision")
    return count


def _source_report(root, config):
    report_path = resolve_repository_path(root, config["source_report"], expected_kind="file")
    report = read_json(report_path)
    if report["status"] != "completed" or report["configuration"]["data_source"]["evidence_class"] != "teacher_candidate_pool":
        raise ValueError("A completed fresh Teacher candidate pool is required")
    return report_path, report


def _source_route_path(base, sample):
    return base / f"teacher/routes/route-{sample['route']:03d}-combat-seed-group-{sample['group']:02d}.json"


def source_identity(root, config):
    """Verify and bind the source files consumed by this panel, once per run."""
    report_path, report = _source_report(root, config)
    base = report_path.parent
    files = {report_path.name: sha256_bytes(canonical_json_bytes(report))}
    routes = {}
    for sample in config["samples"]:
        route_path = _source_route_path(base, sample)
        if route_path not in routes:
            routes[route_path] = read_json(route_path)
            files[route_path.relative_to(base).as_posix()] = sha256_bytes(canonical_json_bytes(routes[route_path]))
        combat = next((c for c in routes[route_path]["combats"] if c["combat_index"] == sample["combat"]), None)
        if combat is None:
            raise ValueError(f"Missing GOLD source combat for sample {sample['id']}")
        trajectory = resolve_repository_path(base, combat["trajectory"], expected_kind="file")
        relative = trajectory.relative_to(base).as_posix()
        if relative not in files:
            files[relative] = sha256_file(trajectory)
        if "trajectory_sha256" in combat and combat["trajectory_sha256"] != files[relative]:
            raise ValueError(f"GOLD source trajectory hash mismatch: {relative}")
    return {"schema_version": "teacher_gold_source_identity_v1", "files": files}


def load_source(root, config, sample):
    report_path, _ = _source_report(root, config)
    base = report_path.parent
    route_path = _source_route_path(base, sample)
    route = read_json(route_path)
    combat = next(c for c in route["combats"] if c["combat_index"] == sample["combat"])
    steps = route["route"]["steps"]
    position = next(i for i, s in enumerate(steps) if s["kind"] == "combat" and s["combat_index"] == sample["combat"])
    records = list(iter_trajectory_records(base / combat["trajectory"]))
    return combat, records, public_route_context(steps, position)


def restore(env, source, sample):
    combat, records, context = source
    replay_combat(env, scenario_id=combat["scenario_id"], combat_seed=combat["combat_seed"],
                  snapshot=combat["input_snapshot"], records=records, route_context=context, stop_before=sample["step"])


def observation(env):
    return serialize_observation(env.get_state(), env.legal_actions(), version=OBSERVATION_VERSION)


def scoring_config(config, combat):
    floor = config["minimum_win_rate"]
    if isinstance(floor, dict):
        family = combat["encounter_family"]
        if family not in floor:
            raise ValueError(f"No minimum win rate declared for encounter family {family!r}")
        return {**config, "minimum_win_rate": floor[family]}
    return config


def continuation_policy(config, combat):
    """Resolve search thresholds independently of the external certification gate."""
    policy = config.get("continuation_policy", {"mode": "win_rate_then_hp"})
    if policy is None:
        policy = {"mode": "win_rate_then_hp"}
    validate_continuation_policy(policy)
    floor = policy.get("minimum_search_win_rate")
    if isinstance(floor, dict):
        family = combat["encounter_family"]
        if family not in floor:
            raise ValueError(f"No continuation search threshold declared for encounter family {family!r}")
        return {**policy, "minimum_search_win_rate": floor[family]}
    return policy


def continuation_selection(searches, policy=None):
    """Equal weight per sampled hidden world; visits do not weight worlds."""
    validate_continuation_policy(policy)
    ids = {a["model_action_id"] for a in searches[0]["actions"]}
    values = {key: [] for key in ids}
    for search in searches:
        if {a["model_action_id"] for a in search["actions"]} != ids:
            raise ValueError("Public sampling changed legal action classes")
        for action in search["actions"]:
            if action["win_rate"] is None:
                raise ValueError("An action received no search support")
            values[action["model_action_id"]].append(action)
    scores = {}
    for key, rows in values.items():
        win = statistics.mean(r["win_rate"] for r in rows)
        weighted_hp = sum(r["win_rate"] * (r["victory_ending_hp_mean"] or 0) for r in rows)
        total_win = sum(r["win_rate"] for r in rows)
        scores[key] = (win, weighted_hp / total_win if total_win else -1, weighted_hp / len(rows))
    baseline = min(ids, key=lambda key: (-scores[key][0], -scores[key][1], key))
    eligible = []
    if policy and policy["mode"] == "win_floor_then_expected_hp":
        floor = policy["minimum_search_win_rate"]
        if isinstance(floor, dict):
            raise ValueError("Resolve encounter-family continuation threshold before selecting an action")
        eligible = [key for key in ids if scores[key][0] >= floor]
    selected = min(eligible, key=lambda key: (-scores[key][2], -scores[key][0], key)) if eligible else baseline
    return {"action": selected, "baseline_action": baseline, "hp_mode_applied": bool(eligible),
            "eligible_actions": len(eligible), "changed_from_baseline": selected != baseline}


def aggregate_searches(searches, policy=None):
    return continuation_selection(searches, policy)["action"]


def choose(env, config, sample_id, trial, decision):
    searches = []
    for hidden in range(config["hidden_samples"]):
        public_seed = seed_for(config["seed"], sample_id, "inner-world", trial, decision, hidden)
        search_seed = seed_for(config["seed"], sample_id, "search", trial, decision, hidden)
        result = env.search_model_actions(
            simulations=config["search_budget"], search_seed=search_seed,
            public_state_seed=public_seed, minimum_root_action_visits=config["minimum_root_visits"],
        )
        searches.append({"public_state_seed": public_seed, "search_seed": search_seed,
                         "draw_fingerprint": result.native_result.hidden_order_fingerprint,
                         "simulations": result.native_result.simulations,
                         "native_root_count": len(result.native_result.root_actions),
                         "minimum_root_action_visits": result.native_result.minimum_root_action_visits,
                         "elapsed_ms": result.native_result.elapsed_ms,
                         "actions": [asdict(a) for a in result.model_actions]})
    return aggregate_searches(searches, config.get("continuation_policy")), searches


def binomial_lower(wins, trials, alpha):
    """Exact one-sided Clopper-Pearson lower bound, without a scipy dependency."""
    if wins == 0:
        return 0.0
    def tail(p):
        if p == 0:
            return 0.0
        if p == 1:
            return 1.0
        return sum(math.exp(math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1)
                            + k * math.log(p) + (trials - k) * math.log1p(-p))
                   for k in range(wins, trials + 1))
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if tail(middle) < alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def summarize(rows, config):
    groups = {}
    for row in rows:
        groups.setdefault(row["root_action"], {})[row["trial"]] = row
    complete = all(len(g) == config["outer_trials"] and all(r["outcome"] in {"victory", "defeat"} for r in g.values()) for g in groups.values())
    action_stats = []
    alpha = (1 - config["confidence"]) / max(1, len(groups))
    for action, group in sorted(groups.items()):
        outcomes = list(group.values())
        finished = [r for r in outcomes if r["outcome"] in {"victory", "defeat"}]
        wins = sum(r["outcome"] == "victory" for r in finished)
        carried = [r["ending_hp"] if r["outcome"] == "victory" else 0 for r in finished]
        count = len(finished)
        lcb = binomial_lower(wins, count, alpha) if count else 0
        action_stats.append({"action": action, "trials": count, "wins": wins,
                             "win_rate": wins / count if count else None,
                             "win_rate_lcb_simultaneous": lcb,
                             "passes_win_floor_lcb": lcb >= config["minimum_win_rate"],
                             "expected_carried_hp": statistics.mean(carried) if carried else None,
                             "victory_hp_mean": statistics.mean([r["ending_hp"] for r in finished if r["outcome"] == "victory"]) if wins else None})
    rounded = config.get("win_floor_rounding", "exact") == "floor"
    threshold = {}
    if rounded:
        required = int((Decimal(str(config["minimum_win_rate"])) * config["outer_trials"]).to_integral_value(rounding=ROUND_FLOOR))
        threshold = {"win_floor_rounding": "floor", "minimum_wins": required,
                     "effective_minimum_win_rate": required / config["outer_trials"]}
        for action in action_stats:
            action["passes_win_floor"] = action["trials"] == config["outer_trials"] and action["wins"] >= required
        eligible = [a for a in action_stats if a["passes_win_floor"]]
    else:
        eligible = [a for a in action_stats if a["win_rate"] is not None and a["win_rate"] >= config["minimum_win_rate"]]
    best = max((a["expected_carried_hp"] for a in eligible), default=None)
    candidates = [a["action"] for a in eligible if best - a["expected_carried_hp"] <= config["expected_hp_tolerance"]] if complete else []
    pairs = []
    keys = sorted(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            differences, resource_conflicts = [], 0
            for trial in sorted(groups[a].keys() & groups[b].keys()):
                x, y = groups[a][trial], groups[b][trial]
                if x["outcome"] not in {"victory", "defeat"} or y["outcome"] not in {"victory", "defeat"}:
                    continue
                differences.append((x["ending_hp"] if x["outcome"] == "victory" else 0) - (y["ending_hp"] if y["outcome"] == "victory" else 0))
                if x["outcome"] == y["outcome"] == "victory":
                    resource_conflicts += (x["max_hp"], x["relic_counters"]) != (y["max_hp"], y["relic_counters"])
            pairs.append({"a": a, "b": b, "paired_trials": len(differences),
                          "mean_carried_hp_a_minus_b": statistics.mean(differences) if differences else None,
                          "standard_error": statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else None,
                          "unpriced_resource_conflict_trials": resource_conflicts})
    return {"status": "completed" if complete else "partial", **threshold, "actions": action_stats, "paired_comparisons": pairs,
            "estimated_acceptable_actions": candidates, "labels_certified": False,
            "interpretation": "diagnostic estimates only; HP uncertainty and persistent-resource tradeoffs are not production-certified"}
