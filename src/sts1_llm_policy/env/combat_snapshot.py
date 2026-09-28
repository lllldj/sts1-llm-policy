"""Copy and sign native combat snapshots without changing their schema semantics."""
from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Any


def resign_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the native FNV-1a fingerprint on an independent snapshot copy."""
    result = deepcopy(dict(snapshot))
    result.pop("fingerprint_fnv1a64", None)
    payload = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = 14695981039346656037
    for byte in payload:
        fingerprint ^= byte
        fingerprint = (fingerprint * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    result["fingerprint_fnv1a64"] = f"{fingerprint:016x}"
    return result


def with_snapshot_relics(
    combat_snapshot: Mapping[str, Any],
    relics: list[dict[str, object]],
) -> dict[str, Any]:
    """Replace relics and re-sign, leaving both caller-owned inputs untouched."""
    return resign_snapshot({**combat_snapshot, "relics": relics})
