"""Export matched fine, coarse and HP-weighted preferences from executed GOLD."""
from collections import Counter
from copy import deepcopy
import math
import re

from sts1_llm_policy.data.gold.export_source import decision_fields, select_gold_states
from sts1_llm_policy.data.export_artifacts import (
    json_artifact_bytes,
    validate_export_destinations,
    write_export_artifacts,
)
from sts1_llm_policy.data.manifest import build_dataset_manifest
from sts1_llm_policy.artifacts import (
    gzip_jsonl_bytes,
    read_json_object,
    repository_relative,
    resolve_repository_path,
    sha256_bytes,
    sha256_file,
)
from sts1_llm_policy.configuration import load_config_document


def preference_records(report, verification, *, dataset_ids, hp_tolerance,
                       minimum_paired_trials, se_multiplier, minimum_adjusted_hp_gap,
                       coarse_hp_gap, weight_hp_cap, verification_scope="full"):
    """Keep common states; normalize HP factors globally, not away within states."""
    if (set(dataset_ids) != {"a", "b", "c"} or len(set(dataset_ids.values())) != 3
            or any(not isinstance(v, str) or not v for v in dataset_ids.values())):
        raise ValueError("Three distinct dataset IDs are required")
    if (isinstance(minimum_paired_trials, bool) or not isinstance(minimum_paired_trials, int)
            or minimum_paired_trials < 2):
        raise ValueError("minimum_paired_trials must be an integer >= 2")
    for value in (hp_tolerance, se_multiplier, minimum_adjusted_hp_gap, coarse_hp_gap, weight_hp_cap):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("HP filtering parameters must be finite and nonnegative")
    if weight_hp_cap <= 0 or minimum_adjusted_hp_gap <= 0:
        raise ValueError("HP cap and adjusted gap must be positive")
    selected_states, _ = select_gold_states(report, verification, hp_tolerance, verification_scope=verification_scope)
    groups = {arm: [] for arm in dataset_ids}
    screening = Counter()
    for state, gold in selected_states:
        template = decision_fields(state, dataset_id=dataset_ids["a"],
                                   observation_version=report["configuration"]["observation_version"])
        legal = {a["model_action_id"] for a in template["model_actions"]}
        serialized = re.findall(r"^(ACTION_\d+):", template["observation"], re.MULTILINE)
        if set(serialized) != legal or len(serialized) != len(legal):
            raise ValueError("Root action metadata differs from observation")
        scores = {a["action"]: a for a in state["actions"]}
        seen, edges = set(), []
        for pair in state["paired_comparisons"]:
            a, b = pair["a"], pair["b"]
            key = frozenset((a, b))
            if a == b or a not in legal or b not in legal or key in seen:
                raise ValueError("Duplicate or invalid paired comparison")
            seen.add(key)
            n, se, difference = pair["paired_trials"], pair["standard_error"], pair["mean_carried_hp_a_minus_b"]
            conflicts = pair["unpriced_resource_conflict_trials"]
            if (isinstance(n, bool) or not isinstance(n, int) or n != state["outer_trials_completed"]
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                           for v in (se, difference)) or se < 0
                    or isinstance(conflicts, bool) or not isinstance(conflicts, int) or not 0 <= conflicts <= n
                    or not math.isclose(difference, scores[a]["expected_carried_hp"] - scores[b]["expected_carried_hp"], abs_tol=1e-9)):
                raise ValueError("Invalid or inconsistent paired statistics")
            if (a in gold) == (b in gold):
                continue
            screening["gold_to_outside_pairs"] += 1
            chosen, rejected = (a, b) if a in gold else (b, a)
            delta = difference if chosen == a else -difference
            if n < minimum_paired_trials:
                screening["insufficient_trials"] += 1
                continue
            if conflicts:
                screening["resource_conflict"] += 1
                continue
            if scores[chosen]["wins"] < scores[rejected]["wins"]:
                screening["lower_chosen_win_count"] += 1
                continue
            if delta - se_multiplier * se <= minimum_adjusted_hp_gap:
                screening["insufficient_adjusted_gap"] += 1
                continue
            edges.append({"chosen_action_id": chosen, "rejected_action_id": rejected,
                          "mean_carried_hp_gap": delta, "paired_standard_error": se,
                          "paired_trials": n})
        if len(seen) != len(legal) * (len(legal) - 1) // 2:
            raise ValueError("Incomplete paired comparisons")
        if not edges:
            continue
        screening["fine_states_before_common_selection"] += 1
        screening["fine_pairs_before_common_selection"] += len(edges)
        coarse = [e for e in edges if e["mean_carried_hp_gap"] > coarse_hp_gap]
        if not coarse:
            continue
        base = {"schema_version": "decision_preference_group_v1", **template}
        for arm in groups:
            selected = coarse if arm == "b" else edges
            factors = [min(1.0, e["mean_carried_hp_gap"] / weight_hp_cap) if arm == "c" else 1.0
                       for e in selected]
            total = math.fsum(factors)
            groups[arm].append({**base, "dataset_id": dataset_ids[arm],
                                "loss_weight": total / len(selected),
                                "edges": [{**e, "weight": w / total} for e, w in zip(selected, factors)]})
    if not groups["a"]:
        raise ValueError("No common preference states remain")
    normalizers = {}
    for arm, records in groups.items():
        normalizers[arm] = math.fsum(r["loss_weight"] for r in records) / len(records)
        for record in records:
            record["loss_weight"] /= normalizers[arm]
    return groups, dict(screening), normalizers


def _statistics(records, coarse_hp_gap):
    families = {}
    end_mass, small_mass = [], []
    for record in records:
        family = record["source"]["encounter_family"]
        counts = families.setdefault(family, {"states": 0, "pairs": 0})
        counts["states"] += 1
        counts["pairs"] += len(record["edges"])
        end = {a["model_action_id"] for a in record["model_actions"] if a["action_type"] == "end_turn"}
        for edge in record["edges"]:
            mass = record["loss_weight"] * edge["weight"]
            if edge["rejected_action_id"] in end:
                end_mass.append(mass)
            if edge["mean_carried_hp_gap"] <= coarse_hp_gap:
                small_mass.append(mass)
    weights = [r["loss_weight"] for r in records]
    return {"states": len(records), "pairs": sum(len(r["edges"]) for r in records),
            "routes": len({r["source"]["sample"]["route"] for r in records}),
            "max_pairs_per_state": max(len(r["edges"]) for r in records),
            "loss_weight_sum": math.fsum(weights), "loss_weight_min": min(weights), "loss_weight_max": max(weights),
            "rejected_end_loss_mass": math.fsum(end_mass) / math.fsum(weights),
            "gap_at_most_coarse_threshold_loss_mass": math.fsum(small_mass) / math.fsum(weights),
            "families": families}


def export_gold_dpo(root, config_path):
    doc = load_config_document(root, config_path, supported_schemas={"gold_dpo_export_v1"})
    cfg = doc.value
    parameters = {"hp_tolerance", "minimum_paired_trials", "se_multiplier", "minimum_adjusted_hp_gap",
                  "coarse_hp_gap", "weight_hp_cap", "dataset_ids"}
    if set(cfg) - {"verification_scope"} != parameters | {"schema_version", "source_report", "verification", "output", "report"}:
        raise ValueError("Unexpected or missing GOLD DPO export config fields")
    output = resolve_repository_path(root, cfg["output"], must_exist=False, expected_kind="directory")
    report_path = resolve_repository_path(root, cfg["report"], must_exist=False, expected_kind="file")
    validate_export_destinations([output / arm / name for arm in "abc"
                                  for name in ("train.jsonl.gz", "manifest.json")] + [report_path])
    source = resolve_repository_path(root, cfg["source_report"], expected_kind="file")
    verified = resolve_repository_path(root, cfg["verification"], expected_kind="file")
    report, verification = read_json_object(source), read_json_object(verified)
    groups, screening, normalizers = preference_records(report, verification, **{k: cfg[k] for k in parameters},
                                                        verification_scope=cfg.get("verification_scope", "full"))
    lineage = {"source_report": cfg["source_report"], "source_report_sha256": sha256_file(source),
               "verification": cfg["verification"], "verification_sha256": sha256_file(verified),
               "verification_result": verification, "source_run_revision": report.get("git_revision"),
               "source_pool": report["configuration"]["source_report"],
               "source_partition": report["configuration"]["source_partition"],
               "source_routes": sorted({r["source"]["sample"]["route"] for r in groups["a"]})}
    exported, artifacts = {}, []
    for arm, records in groups.items():
        artifact = output / arm / "train.jsonl.gz"
        payload = gzip_jsonl_bytes(records)
        digest = sha256_bytes(payload)
        manifest = build_dataset_manifest(
            dataset_id=cfg["dataset_ids"][arm], task_type="preference", observation_version="observation_v7",
            identity_fields=["record_id"], splits={"train": {
                "path": repository_relative(root, artifact), "format": "jsonl", "compression": "gzip",
                "records": len(records), "bytes": len(payload), "sha256": digest}}, lineage=deepcopy(lineage),
            semantics={"record_schema": "decision_preference_group_v1", "arm": arm,
                       "state_weight": "explicit_mean_one_loss_weight", "edge_normalization": "within_state",
                       "base_weight": "equal_states_then_equal_edges", "targets": "preserved",
                       "common_state_selection": "at_least_one_fine_pair_with_gap_strictly_above_coarse_hp_gap",
                       "chosen": "gold", "rejected": "outside_gold", "chosen_wins_not_lower": True,
                       "unpriced_resource_conflict_trials": 0, "adjusted_gap_comparison": "strict_gt",
                       **{k: cfg[k] for k in parameters - {"dataset_ids"}},
                       "pair_selection": "strict_gt_coarse_hp_gap" if arm == "b" else "all_fine_pairs",
                       "hp_factor": "min(1,mean_carried_hp_gap/weight_hp_cap)" if arm == "c" else "one",
                       "global_hp_factor_mean": normalizers[arm],
                       "labels_statistically_certified": False, "source_trial_counts": "actual_adaptive_stops",
                       "resource_utility": "carried_hp_only_defeat_zero"})
        manifest["export_counts"] = _statistics(records, cfg["coarse_hp_gap"])
        manifest_path = output / arm / "manifest.json"
        artifacts.extend([(artifact, payload), (manifest_path, json_artifact_bytes(manifest))])
        exported[arm] = {"dataset_id": manifest["dataset_id"], "manifest": repository_relative(root, manifest_path),
                         "train": manifest["splits"]["train"], "counts": manifest["export_counts"]}
    result = {"schema_version": "gold_dpo_export_report_v1", "status": "completed",
              "configuration": cfg, "screening": screening, "datasets": exported,
              "checks": {"common_ordered_states": all([r["record_id"] for r in groups[a]] ==
                                                        [r["record_id"] for r in groups["a"]] for a in groups),
                         "a_c_same_pairs": all([(e["chosen_action_id"], e["rejected_action_id"]) for e in a["edges"]] ==
                                               [(e["chosen_action_id"], e["rejected_action_id"]) for e in c["edges"]]
                                               for a, c in zip(groups["a"], groups["c"])),
                         "state_weights_mean_one": all(math.isclose(math.fsum(r["loss_weight"] for r in rs), len(rs))
                                                       for rs in groups.values())},
              "training_started": False, "test_data_read": False}
    if not all(result["checks"].values()):
        raise ValueError("Cross-arm export checks failed")
    artifacts.append((report_path, json_artifact_bytes(result)))
    write_export_artifacts(artifacts)
    return result
