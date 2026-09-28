from __future__ import annotations

import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sts1_llm_policy.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    gzip_jsonl_bytes,
    iter_jsonl,
    sha256_file,
)
from sts1_llm_policy.data.manifest import build_dataset_manifest, validate_dataset_manifest
from sts1_llm_policy.configuration import load_config_document, resolve_bound_document


class ConfigurationArtifactTests(unittest.TestCase):
    def test_bound_document_and_repository_boundary(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "configs" / "profile.json"
            atomic_write_json(profile, {"schema_version": "profile_v1", "name": "x"})
            owner = {
                "profile": "configs/profile.json",
                "expected_profile_sha256": sha256_file(profile),
            }
            document = resolve_bound_document(
                root,
                owner,
                path_key="profile",
                sha256_key="expected_profile_sha256",
                supported_schemas={"profile_v1"},
            )
            self.assertEqual(document.value["name"], "x")
            self.assertEqual(document.relative_path, "configs/profile.json")
            owner["expected_profile_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resolve_bound_document(
                    root,
                    owner,
                    path_key="profile",
                    sha256_key="expected_profile_sha256",
                )
            outside = root.parent / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            try:
                with self.assertRaisesRegex(ValueError, "escapes project root"):
                    load_config_document(root, outside)
            finally:
                outside.unlink()

    def test_atomic_artifact_is_idempotent_but_rejects_conflict(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            first = atomic_write_bytes(path, b"first")
            self.assertEqual(atomic_write_bytes(path, b"first"), first)
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                atomic_write_bytes(path, b"second")

    def test_gzip_jsonl_and_manifest_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "data" / "train.jsonl.gz"
            payload = gzip_jsonl_bytes([{"record_id": "a"}, {"record_id": "b"}])
            digest = atomic_write_bytes(artifact, payload)
            self.assertEqual([item["record_id"] for item in iter_jsonl(artifact)], ["a", "b"])
            with gzip.open(artifact, "rt", encoding="utf-8") as handle:
                self.assertEqual(len(handle.readlines()), 2)
            manifest = build_dataset_manifest(
                dataset_id="dataset-v1",
                task_type="sft",
                observation_version="observation_v4",
                identity_fields=("record_id",),
                splits={"train": {
                    "path": "data/train.jsonl.gz",
                    "format": "jsonl",
                    "compression": "gzip",
                    "records": 2,
                    "bytes": len(payload),
                    "sha256": digest,
                }},
                lineage={"source": "fixture"},
                semantics={"target_field": "teacher_action_id"},
            )
            validated = validate_dataset_manifest(
                manifest, project_root=root, verify_artifacts=True
            )
            self.assertEqual(validated["splits"]["train"]["records"], 2)


if __name__ == "__main__":
    unittest.main()
