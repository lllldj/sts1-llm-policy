"""Check every export destination before publishing with the existing atomic writer."""
import json
from pathlib import Path

from sts1_llm_policy.artifacts import atomic_write_bytes, sha256_bytes, sha256_file


def json_artifact_bytes(value):
    """Preserve the existing manifest/report JSON encoding."""
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def validate_export_destinations(paths):
    """Reject path/type/temp conflicts without creating directories or files."""
    paths = [Path(path).resolve() for path in paths]
    temporaries = [path.with_suffix(path.suffix + ".tmp") for path in paths]
    all_paths = paths + temporaries
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Export destinations and temporary paths must be distinct")
    for path in all_paths:
        if any(parent in all_paths or (parent.exists() and not parent.is_dir())
               for parent in path.parents):
            raise ValueError(f"Export destination has a conflicting parent: {path}")
    for path, temporary in zip(paths, temporaries):
        if path.exists() and not path.is_file():
            raise ValueError(f"Export destination must be a file: {path}")
        if temporary.exists():
            raise ValueError(f"Conflicting temporary artifact exists: {temporary}")


def write_export_artifacts(artifacts):
    """Precheck the entire export; individual writes remain atomic, not a transaction."""
    artifacts = [(Path(path), payload) for path, payload in artifacts]
    validate_export_destinations([path for path, _ in artifacts])
    pending = []
    for path, payload in artifacts:
        if path.exists():
            if sha256_file(path) != sha256_bytes(payload):
                raise ValueError(f"Refusing to overwrite conflicting artifact: {path}")
        else:
            pending.append((path, payload))
    for path, payload in pending:
        atomic_write_bytes(path, payload)
