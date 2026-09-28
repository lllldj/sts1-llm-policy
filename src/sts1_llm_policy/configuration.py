from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping

from sts1_llm_policy.artifacts import (
    read_json_object,
    repository_relative,
    resolve_repository_path,
    sha256_file,
)


@dataclass(frozen=True)
class ConfigDocument:
    path: Path
    relative_path: str
    sha256: str
    value: dict[str, Any]


def load_config_document(
    project_root: str | Path,
    path: str | Path,
    *,
    supported_schemas: Collection[object] | None = None,
) -> ConfigDocument:
    resolved = resolve_repository_path(
        project_root, path, must_exist=True, expected_kind="file"
    )
    value = read_json_object(resolved)
    if supported_schemas is not None and value.get("schema_version") not in supported_schemas:
        raise ValueError(
            f"Unsupported schema_version in {resolved}: {value.get('schema_version')!r}"
        )
    return ConfigDocument(
        path=resolved,
        relative_path=repository_relative(project_root, resolved),
        sha256=sha256_file(resolved),
        value=value,
    )


def resolve_bound_document(
    project_root: str | Path,
    owner: Mapping[str, object],
    *,
    path_key: str,
    sha256_key: str,
    supported_schemas: Collection[object] | None = None,
) -> ConfigDocument:
    path_value = owner.get(path_key)
    expected_sha256 = owner.get(sha256_key)
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{path_key} must be a non-empty repository path")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(f"{sha256_key} must be a lowercase SHA-256 digest")
    document = load_config_document(
        project_root,
        path_value,
        supported_schemas=supported_schemas,
    )
    if document.sha256 != expected_sha256:
        raise ValueError(f"Bound document hash mismatch: {path_value}")
    return document
