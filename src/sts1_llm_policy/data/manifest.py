from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sts1_llm_policy.artifacts import resolve_repository_path, sha256_file


DATASET_MANIFEST_SCHEMA_VERSION = "dataset_manifest_v1"
SUPPORTED_TASK_TYPES = frozenset({"sft", "preference"})
SUPPORTED_SPLITS = frozenset({"train", "development", "test"})


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    digest = _nonempty_string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def build_dataset_manifest(
    *,
    dataset_id: str,
    task_type: str,
    observation_version: str,
    identity_fields: Sequence[str],
    splits: Mapping[str, Mapping[str, object]],
    lineage: Mapping[str, object],
    semantics: Mapping[str, object],
) -> dict[str, Any]:
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "task_type": task_type,
        "observation_version": observation_version,
        "identity_fields": list(identity_fields),
        "splits": {name: dict(artifact) for name, artifact in sorted(splits.items())},
        "lineage": dict(lineage),
        "semantics": dict(semantics),
    }
    validate_dataset_manifest(manifest)
    return manifest


def validate_dataset_manifest(
    manifest: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    if manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported dataset manifest schema")
    _nonempty_string(manifest.get("dataset_id"), "dataset_id")
    task_type = _nonempty_string(manifest.get("task_type"), "task_type")
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported dataset task_type: {task_type}")
    _nonempty_string(manifest.get("observation_version"), "observation_version")
    identity_fields = manifest.get("identity_fields")
    if (
        not isinstance(identity_fields, list)
        or not identity_fields
        or any(not isinstance(item, str) or not item for item in identity_fields)
        or len(set(identity_fields)) != len(identity_fields)
    ):
        raise ValueError("identity_fields must contain unique non-empty strings")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("splits must be a non-empty object")
    normalized_splits: dict[str, dict[str, object]] = {}
    for split, raw in splits.items():
        if split not in SUPPORTED_SPLITS or not isinstance(raw, dict):
            raise ValueError(f"Invalid dataset split: {split}")
        path = _nonempty_string(raw.get("path"), f"splits.{split}.path")
        records = raw.get("records")
        size = raw.get("bytes")
        if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
            raise ValueError(f"splits.{split}.records must be positive")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"splits.{split}.bytes must be positive")
        digest = _sha256(raw.get("sha256"), f"splits.{split}.sha256")
        if raw.get("format") != "jsonl" or raw.get("compression") not in {None, "gzip"}:
            raise ValueError(f"Unsupported artifact encoding for split {split}")
        if verify_artifacts:
            if project_root is None:
                raise ValueError("project_root is required when verifying artifacts")
            artifact = resolve_repository_path(
                project_root, path, must_exist=True, expected_kind="file"
            )
            if artifact.stat().st_size != size or sha256_file(artifact) != digest:
                raise ValueError(f"Dataset artifact binding mismatch: {path}")
        normalized_splits[split] = dict(raw)
    if not isinstance(manifest.get("lineage"), dict) or not manifest["lineage"]:
        raise ValueError("lineage must be a non-empty object")
    if not isinstance(manifest.get("semantics"), dict) or not manifest["semantics"]:
        raise ValueError("semantics must be a non-empty object")
    return {**dict(manifest), "splits": normalized_splits}
