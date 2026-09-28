"""Export verified executed-GOLD estimates into state-weighted SFT groups."""
from collections import Counter
import math

from sts1_llm_policy.data.gold.candidate_statistics import categories
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


def export_records(report, verification, *, dataset_id, hp_tolerance, nonbasic_weight, verification_scope="full"):
    if (isinstance(nonbasic_weight, bool) or not isinstance(nonbasic_weight, (int, float))
            or not math.isfinite(nonbasic_weight) or nonbasic_weight <= 0):
        raise ValueError("nonbasic_weight must be positive and finite")
    selected, dropped = select_gold_states(report, verification, hp_tolerance, verification_scope=verification_scope)
    records, counts = [], Counter(dropped)
    for state, accepted in selected:
        chosen = [a for a in state["root_actions"] if a["id"] in accepted]
        weights = [float(nonbasic_weight) if "nonbasic" in categories(a) else 1.0 for a in chosen]
        total = sum(weights)
        records.append({
            "schema_version": "decision_sft_group_v1",
            **decision_fields(state, dataset_id=dataset_id,
                              observation_version=report["configuration"]["observation_version"]),
            "candidates": [{"action_id": a["id"], "weight": w / total} for a, w in zip(chosen, weights)],
        })
        counts["states"] += 1
        counts["candidate_answers"] += len(chosen)
        counts["single_candidate_states" if len(chosen) == 1 else "multiple_candidate_states"] += 1
    return records, dict(counts)


def export_gold_sft(root, config_path):
    doc = load_config_document(root, config_path, supported_schemas={"gold_sft_export_v1"})
    cfg = doc.value
    required = {"schema_version", "dataset_id", "source_report", "verification", "output",
                "hp_tolerance", "nonbasic_weight"}
    if set(cfg) - {"verification_scope"} != required:
        raise ValueError("Unexpected or missing GOLD SFT export config fields")
    output = resolve_repository_path(root, cfg["output"], must_exist=False, expected_kind="directory")
    artifact, manifest_path = output / "train.jsonl.gz", output / "manifest.json"
    validate_export_destinations([artifact, manifest_path])
    source = resolve_repository_path(root, cfg["source_report"], expected_kind="file")
    verification_path = resolve_repository_path(root, cfg["verification"], expected_kind="file")
    report, verification = read_json_object(source), read_json_object(verification_path)
    records, counts = export_records(report, verification, dataset_id=cfg["dataset_id"],
                                    hp_tolerance=cfg["hp_tolerance"], nonbasic_weight=cfg["nonbasic_weight"],
                                    verification_scope=cfg.get("verification_scope", "full"))
    payload = gzip_jsonl_bytes(records)
    digest = sha256_bytes(payload)
    manifest = build_dataset_manifest(
        dataset_id=cfg["dataset_id"], task_type="sft", observation_version="observation_v7",
        identity_fields=["record_id"], splits={"train": {
            "path": repository_relative(root, artifact), "format": "jsonl", "compression": "gzip",
            "records": len(records), "bytes": len(payload), "sha256": digest}},
        lineage={"source_report": cfg["source_report"], "source_report_sha256": sha256_file(source),
                 "verification": cfg["verification"], "verification_result": verification,
                 "source_run_revision": report.get("git_revision"),
                 "source_pool": report["configuration"]["source_report"],
                 "source_partition": report["configuration"]["source_partition"],
                 "source_routes": sorted({s["sample"]["route"] for s in report["states"]})},
        semantics={"record_schema": "decision_sft_group_v1", "state_weight": "equal",
                   "candidate_loss": "weighted_mean_response_token_cross_entropy",
                   "nonbasic_weight": cfg["nonbasic_weight"], "other_action_weight": 1.0,
                   "candidate_normalization": "within_state", "targets": "preserved",
                   "basic_cards": ["Strike_R", "Defend_R", "Bash"],
                   "hp_tolerance": cfg["hp_tolerance"], "win_floor_rounding": "floor",
                   "minimum_win_rate": report["configuration"]["minimum_win_rate"],
                   "drop_empty_and_all_pass": True, "labels_statistically_certified": False,
                   "source_trial_counts": "actual_adaptive_stops", "resource_utility": "carried_hp_only"},
    )
    manifest["export_counts"] = counts
    write_export_artifacts([(artifact, payload), (manifest_path, json_artifact_bytes(manifest))])
    return manifest
