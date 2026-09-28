"""Rebuild single-combat datasets from Teacher V2 exports, then migrate V4 text.

Record construction follows prepare_dataset.py and migrate_observation_v5.py
before their retirement at 2411f01; execution consumes only declared inputs.
"""
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

from sts1_llm_policy.env.observation_v5_catalog import upgrade_v4_observation
from sts1_llm_policy.data.manifest import build_dataset_manifest, validate_dataset_manifest
from sts1_llm_policy.artifacts import (
    gzip_jsonl_bytes,
    iter_jsonl,
    read_json_object,
    repository_relative,
    resolve_repository_path,
    sha256_file,
)
from sts1_llm_policy.configuration import load_config_document
from .export_artifacts import json_artifact_bytes, write_export_artifacts


def _identity(record: Mapping[str, object], *, source: bool) -> tuple[int, int]:
    prefix = "" if source else "source_"
    episode = record.get(f"{prefix}episode_index")
    decision = record.get(f"{prefix}decision_index")
    if (
        isinstance(episode, bool)
        or not isinstance(episode, int)
        or isinstance(decision, bool)
        or not isinstance(decision, int)
        or episode < 0
        or decision < 0
    ):
        raise ValueError("Dataset record has an invalid source identity")
    return episode, decision


def _record_id(identity: tuple[int, int]) -> str:
    return f"episode-{identity[0]:06d}-decision-{identity[1]:04d}"


def _action_ids(record: Mapping[str, object]) -> tuple[str, ...]:
    actions = record.get("model_actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Dataset record must contain model_actions")
    values = tuple(
        item.get("model_action_id") if isinstance(item, dict) else None
        for item in actions
    )
    if any(not isinstance(item, str) or not item.startswith("ACTION_")
           or not item[7:].isdigit() for item in values):
        raise ValueError("Dataset record has an invalid model action")
    if len(set(values)) != len(values):
        raise ValueError("Dataset record has duplicate model action IDs")
    return values


def derive_datasets(
    historical: Mapping[tuple[int, int], Mapping[str, Any]],
    gold_exports: Sequence[Mapping[str, Any]],
    silver_exports: Sequence[Mapping[str, Any]],
    *,
    gold_dataset_id: str,
    silver_dataset_id: str,
    observation_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    gold_records: list[dict[str, Any]] = []
    silver_records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    changed_gold = 0
    for export in gold_exports:
        identity = _identity(export, source=False)
        source = historical.get(identity)
        if source is None or identity in seen:
            raise ValueError("Gold export identity is missing or duplicated")
        seen.add(identity)
        if (
            export.get("schema_version") != "public_information_teacher_gold_sft_v1"
            or export.get("evidence_tier") != "gold"
            or export.get("observation_version") != observation_version
            or export.get("public_teacher_contract")
            != "public_information_teacher_contract_v2"
            or export.get("source_observation_sha256") != source.get("observation_sha256")
            or export.get("observation") != source.get("observation")
            or export.get("model_actions") != source.get("model_actions")
        ):
            raise ValueError("Gold export diverges from its historical public state")
        target = export.get("target_action_id")
        if target not in _action_ids(export):
            raise ValueError("Gold target is not a legal model action")
        original_target = source.get("teacher_action_id")
        changed = target != original_target
        changed_gold += int(changed)
        gold_records.append({
            "schema_version": "decision_sft_record_v1",
            "dataset_id": gold_dataset_id,
            "split": "train",
            "record_id": _record_id(identity),
            "source_episode_index": identity[0],
            "source_decision_index": identity[1],
            "observation_version": observation_version,
            "observation_sha256": export["source_observation_sha256"],
            "observation": export["observation"],
            "model_actions": export["model_actions"],
            "teacher_action_id": target,
            "historical_teacher_action_id": original_target,
            "teacher_action_changed": changed,
            "evidence_tier": "gold",
            "teacher_contract": export["public_teacher_contract"],
        })
    for export in silver_exports:
        identity = _identity(export, source=False)
        source = historical.get(identity)
        if source is None or identity in seen:
            raise ValueError("Silver export identity is missing, duplicated, or overlaps Gold")
        seen.add(identity)
        if (
            export.get("schema_version")
            != "public_information_teacher_silver_preference_v1"
            or export.get("evidence_tier") != "silver"
            or export.get("observation_version") != observation_version
            or export.get("public_teacher_contract")
            != "public_information_teacher_contract_v2"
            or export.get("source_observation_sha256") != source.get("observation_sha256")
            or export.get("observation") != source.get("observation")
            or export.get("model_actions") != source.get("model_actions")
        ):
            raise ValueError("Silver export diverges from its historical public state")
        action_ids = set(_action_ids(export))
        preferred = export.get("preferred_action_id")
        rejected = export.get("rejected_action_ids")
        if (
            preferred not in action_ids
            or not isinstance(rejected, list)
            or not rejected
            or len(set(rejected)) != len(rejected)
            or preferred in rejected
            or any(item not in action_ids for item in rejected)
        ):
            raise ValueError("Silver preference actions are invalid")
        weight = 1.0 / len(rejected)
        edges = [
            {
                "chosen_action_id": preferred,
                "rejected_action_id": rejected_action,
                "weight": weight,
            }
            for rejected_action in rejected
        ]
        if not math.isclose(math.fsum(item["weight"] for item in edges), 1.0):
            raise AssertionError("Silver preference edge weights are not normalized")
        silver_records.append({
            "schema_version": "decision_preference_group_v1",
            "dataset_id": silver_dataset_id,
            "split": "train",
            "record_id": _record_id(identity),
            "source_episode_index": identity[0],
            "source_decision_index": identity[1],
            "observation_version": observation_version,
            "observation_sha256": export["source_observation_sha256"],
            "observation": export["observation"],
            "model_actions": export["model_actions"],
            "preferred_action_ids": [preferred],
            "rejected_action_ids": list(rejected),
            "edges": edges,
            "evidence_tier": "silver",
            "teacher_contract": export["public_teacher_contract"],
        })
    gold_records.sort(key=lambda item: (item["source_episode_index"], item["source_decision_index"]))
    silver_records.sort(key=lambda item: (item["source_episode_index"], item["source_decision_index"]))
    return gold_records, silver_records, {
        "gold_actions_changed": changed_gold,
        "gold_actions_unchanged": len(gold_records) - changed_gold,
        "accepted_source_identities": len(seen),
    }


def _source(root, path, expected_sha256, expected_records=None):
    """Read the declared historical train/dev view, never discover local runs."""
    path = resolve_repository_path(root, path, expected_kind="file")
    if sha256_file(path) != expected_sha256:
        raise ValueError("Historical source hash mismatch")
    selected = {"train": {}, "dev": {}}
    episodes = {"train": set(), "dev": set()}
    count = 0
    for row in iter_jsonl(path):
        count += 1
        split = row.get("split")
        if split not in selected:
            raise ValueError("Historical source must contain only train/dev records")
        identity = _identity(row, source=True)
        episodes[split].add(identity[0])
        if row.get("classification") != "accepted" or row.get("record_type") != "strategic_teacher":
            continue
        if identity in selected[split]:
            raise ValueError("Duplicate historical source identity")
        _public_state(row)
        if row.get("observation_version") != "observation_v4":
            raise ValueError("Historical source must use observation_v4")
        if row.get("teacher_action_id") not in _action_ids(row):
            raise ValueError("Historical teacher action is not legal")
        selected[split][identity] = row
    if episodes["train"] & episodes["dev"]:
        raise ValueError("Train/development source episodes overlap")
    if expected_records is not None and count != expected_records:
        raise ValueError("Historical source record count mismatch")
    return selected


def _public_state(record):
    observation = record.get("observation")
    if (not isinstance(observation, str) or not observation
            or sha256(observation.encode("utf-8")).hexdigest() != record.get("observation_sha256")):
        raise ValueError("Source observation hash mismatch")
    return set(_action_ids(record))


def _training_record(record, task, split):
    _identity(record, source=False)
    legal = _public_state(record)
    if record.get("split") != split:
        raise ValueError("Record split disagrees with its manifest")
    if task == "sft":
        if record.get("teacher_action_id") not in legal:
            raise ValueError("SFT target is not a legal action")
    else:
        edges = record.get("edges")
        if not isinstance(edges, list) or not edges:
            raise ValueError("Preference edges are required")
        pairs = set()
        for edge in edges:
            chosen, rejected, weight = (edge.get(k) for k in
                                        ("chosen_action_id", "rejected_action_id", "weight"))
            if (chosen not in legal or rejected not in legal or chosen == rejected
                    or (chosen, rejected) in pairs or isinstance(weight, bool)
                    or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0):
                raise ValueError("Invalid preference edge")
            pairs.add((chosen, rejected))
        if not math.isclose(math.fsum(e["weight"] for e in edges), 1.):
            raise ValueError("Preference weights must sum to one")


def _publish(root, output, products, version, lineage, semantics, counts):
    output = resolve_repository_path(root, output, must_exist=False, expected_kind="directory")
    artifacts, manifests = [], {}
    for name, (dataset_id, task, split, records) in products.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Product names must be identifiers, not paths")
        payload = gzip_jsonl_bytes(records)
        artifact = {"path": repository_relative(root, output / f"{name}.jsonl.gz"),
                    "format": "jsonl", "compression": "gzip", "records": len(records),
                    "bytes": len(payload), "sha256": sha256(payload).hexdigest()}
        manifest = build_dataset_manifest(
            dataset_id=dataset_id, task_type=task, observation_version=version,
            identity_fields=["source_episode_index", "source_decision_index"],
            splits={split: artifact}, lineage=lineage, semantics=semantics[name])
        manifests[name] = manifest
        artifacts.extend([(output / f"{name}.jsonl.gz", payload),
                          (output / f"{name}.manifest.json", json_artifact_bytes(manifest))])
    report = {"schema_version": "single_dataset_export_report_v1", "status": "completed",
              "observation_version": version, "lineage": lineage, "counts": counts,
              "artifacts": {name: {"manifest": repository_relative(root, output / f"{name}.manifest.json"),
                                   "splits": m["splits"]} for name, m in manifests.items()},
              "teacher_search_run": False, "training_started": False, "test_data_read": False}
    artifacts.append((output / "report.json", json_artifact_bytes(report)))
    write_export_artifacts(artifacts)
    return report


def export_teacher_v2(root, config_path):
    doc = load_config_document(root, config_path, supported_schemas={"teacher_v2_dataset_export_v1"})
    cfg = doc.value
    if set(cfg) != {"schema_version", "teacher_report", "historical_source", "dataset_ids", "output"}:
        raise ValueError("Unexpected or missing Teacher export fields")
    if set(cfg["dataset_ids"]) != {"gold_sft", "silver_preferences"}:
        raise ValueError("Teacher V2 export requires Gold and Silver dataset IDs")
    report_path = resolve_repository_path(root, cfg["teacher_report"], expected_kind="file")
    teacher = read_json_object(report_path)
    checks = teacher.get("checks")
    if (teacher.get("schema_version") != "public_information_teacher_train_collection_report_v1"
            or teacher.get("status") != "completed" or not isinstance(checks, dict) or not checks
            or any(v is not True for v in checks.values())
            or any(teacher.get(k) is not False for k in ("test_data_read", "sealed_test_run", "training_started"))):
        raise ValueError("A completed train-only Teacher V2 report is required")
    source = teacher["source"]
    historical = _source(root, cfg["historical_source"], source["source_sha256"], source["source_record_count"])
    if len(historical["train"]) != source["target_state_count"]:
        raise ValueError("Teacher target count disagrees with historical train source")
    exports = {}
    for name in cfg["dataset_ids"]:
        entry = teacher["artifacts"][name]
        path = resolve_repository_path(root, entry["path"], expected_kind="file")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Teacher artifact hash mismatch: {name}")
        exports[name] = list(iter_jsonl(path))
        tier = "gold" if name == "gold_sft" else "silver"
        if len(exports[name]) != entry["records"] or len(exports[name]) != teacher["tiers"][tier]:
            raise ValueError("Teacher export count mismatch")
    gold, silver, counts = derive_datasets(
        historical["train"], exports["gold_sft"], exports["silver_preferences"],
        gold_dataset_id=cfg["dataset_ids"]["gold_sft"],
        silver_dataset_id=cfg["dataset_ids"]["silver_preferences"], observation_version="observation_v4")
    lineage = {"teacher_report": cfg["teacher_report"], "teacher_report_sha256": sha256_file(report_path),
               "historical_source": cfg["historical_source"], "historical_source_sha256": source["source_sha256"],
               "teacher_profile": "public_information_teacher_contract_v2"}
    products = {"gold_sft": (cfg["dataset_ids"]["gold_sft"], "sft", "train", gold),
                "silver_preferences": (cfg["dataset_ids"]["silver_preferences"], "preference", "train", silver)}
    return _publish(root, cfg["output"], products, "observation_v4", lineage,
                    {"gold_sft": {"target_field": "teacher_action_id", "teacher_evidence_tier": "gold",
                                  "historical_label_policy": "replace_same_source_identity"},
                     "silver_preferences": {"target_field": "edges", "teacher_evidence_tier": "silver",
                                            "edge_weighting": "uniform_within_group"}}, counts)


def upgrade_record(record, dataset_id):
    if record.get("observation_version") != "observation_v4":
        raise ValueError("Source record must use observation_v4")
    _public_state(record)
    upgraded = upgrade_v4_observation(record["observation"])
    return {**record, "dataset_id": dataset_id, "observation_version": "observation_v5",
            "source_observation_v4_sha256": record["observation_sha256"],
            "observation_sha256": sha256(upgraded.encode("utf-8")).hexdigest(), "observation": upgraded}


def migrate_v5(root, config_path):
    doc = load_config_document(root, config_path, supported_schemas={"single_observation_v5_migration_v1"})
    cfg = doc.value
    if (not {"schema_version", "sources", "output"} <= set(cfg)
            or set(cfg) - {"schema_version", "sources", "development", "output"}
            or not isinstance(cfg["sources"], dict) or not cfg["sources"]):
        raise ValueError("Unexpected or missing V5 migration fields")
    products, lineage, train_episodes, dev_episodes = {}, {"source_manifests": {}}, set(), set()
    all_identities = set()
    for name, source in cfg["sources"].items():
        if set(source) != {"manifest", "dataset_id"}:
            raise ValueError("Each migration source needs manifest and dataset_id")
        path = resolve_repository_path(root, source["manifest"], expected_kind="file")
        manifest = validate_dataset_manifest(read_json_object(path))
        if manifest["observation_version"] != "observation_v4" or len(manifest["splits"]) != 1:
            raise ValueError("Migration requires a single train/development V4 split per manifest")
        split, artifact = next(iter(manifest["splits"].items()))
        if split not in {"train", "development"}:
            raise ValueError("Migration does not consume test/sealed data")
        validate_dataset_manifest(manifest, project_root=root, verify_artifacts=True)
        seen, records = set(), []
        for row in iter_jsonl(resolve_repository_path(root, artifact["path"], expected_kind="file")):
            _training_record(row, manifest["task_type"], split)
            identity = _identity(row, source=False)
            if identity in seen or identity in all_identities:
                raise ValueError("Duplicate migration source identity")
            seen.add(identity)
            all_identities.add(identity)
            (train_episodes if split == "train" else dev_episodes).add(identity[0])
            records.append(upgrade_record(row, source["dataset_id"]))
        if len(records) != artifact["records"]:
            raise ValueError("Migration source count mismatch")
        products[name] = (source["dataset_id"], manifest["task_type"], split, records)
        lineage["source_manifests"][name] = {"path": source["manifest"], "sha256": sha256_file(path)}
    if "development" in cfg:
        source = cfg["development"]
        if set(source) != {"path", "sha256", "dataset_id"} or "development_sft" in products:
            raise ValueError("Invalid or duplicate historical development source")
        selected = _source(root, source["path"], source["sha256"])
        records = []
        train_episodes.update(i[0] for i in selected["train"])
        for identity, row in sorted(selected["dev"].items()):
            dev_episodes.add(identity[0])
            compact = {"schema_version": "decision_sft_record_v1", "split": "development",
                       "record_id": _record_id(identity), "source_episode_index": identity[0],
                       "source_decision_index": identity[1], "evidence_tier": "historical_accepted",
                       "teacher_contract": row["contract"]["contract_id"],
                       **{k: row[k] for k in ("observation_version", "observation_sha256", "observation",
                                             "model_actions", "teacher_action_id")}}
            records.append(upgrade_record(compact, source["dataset_id"]))
        products["development_sft"] = (source["dataset_id"], "sft", "development", records)
        lineage["development_source"] = source
    if train_episodes & dev_episodes:
        raise ValueError("Train/development source episodes overlap")
    semantics = {name: {"migration": "observation_v4_to_v5_semantic_closure_v1",
                        "teacher_labels_changed": False, "model_action_space_changed": False,
                        "target_field": "teacher_action_id" if p[1] == "sft" else "edges"}
                 for name, p in products.items()}
    return _publish(root, cfg["output"], products, "observation_v5", lineage, semantics,
                    {name: len(p[3]) for name, p in products.items()})
