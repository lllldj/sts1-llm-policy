"""Per-state continuation archives: executed actions, terminal rows, sparse traces."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

from sts1_llm_policy.artifacts import read_json as _read, replace_json, sha256_file as _sha


VERSION = "teacher_gold_compact_state_v1"


def retained_trials(sample, rows, stages, config):
    policy = config["continuation_storage"]
    keep = set()
    for row in rows:
        key = (row["trial"], row["root_action"])
        encoded = json.dumps([config["seed"], sample["id"], *key, "full-trace-retention"], separators=(",", ":")).encode()
        draw = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") / 2**64
        if draw < policy["full_trace_probability"] or (policy["keep_first_trial"] and row["trial"] == 0):
            keep.add(key)
    if policy["keep_boundary_trials"]:
        previous = None
        for stage in stages:
            actions = set(stage["hp_boundary_actions"]) | set(stage["win_boundary_actions"])
            if previous is not None:
                actions |= set(previous["candidates"]) ^ set(stage["candidates"])
            keep.update((stage["trials"] - 1, action) for action in actions)
            previous = stage
    return keep


def read_archive(path, sample):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        archive = json.load(stream)
    if archive["schema_version"] != VERSION or archive["sample"] != sample:
        raise ValueError("Compact archive identity mismatch")
    rows = archive["executions"]
    keys = [(r["trial"], r["root_action"]) for r in rows]
    if len(set(keys)) != len(keys) or not rows:
        raise ValueError("Duplicate or empty compact executions")
    for row in rows:
        if row["sample_id"] != sample["id"] or row["outcome"] not in {"victory", "defeat"}:
            raise ValueError("Incomplete or foreign compact execution")
        if not row["model_actions"] or row["model_actions"][0] != row["root_action"] or len(row["model_actions"]) != row["decisions"]:
            raise ValueError("Compact action sequence differs from trial metadata")
    return archive


def compact_verified_state(output, sample, rows, stages, config, tapes, verification):
    """Atomically publish an archive; original traces remain until explicit cleanup."""
    if verification["status"] != "completed" or verification["verified_executions"] != len(rows):
        raise ValueError("Complete state verification is required before compaction")
    if verification["verified_decisions"] != sum(r["decisions"] for r in rows):
        raise ValueError("Verification decision count mismatch")
    if verification["full_search_trace_executions"] != len(rows):
        raise ValueError("Every original search trace must be verified before compaction")
    keep = retained_trials(sample, rows, stages, config)
    packed = []
    for row in rows:
        key = (row["trial"], row["root_action"])
        packed.append({**row, "model_actions": tapes[key], "full_trace_retained": key in keep})
    archive = {"schema_version": VERSION, "sample": sample, "executions": packed,
               "verification_before_compaction": verification}
    destination = Path(output) / sample["id"] / "continuations.json.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(archive, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if read_archive(temporary, sample) != archive:
        raise ValueError("Compact archive roundtrip mismatch")
    temporary.replace(destination)
    return archive


def cleanup_compacted_state(output, sample, archive):
    """Remove only this verified state's per-execution files after archive commit.

    Repeated cleanup is safe after an interruption. No recursive filesystem delete.
    """
    if read_archive(Path(output) / sample["id"] / "continuations.json.gz", sample) != archive:
        raise ValueError("Cannot clean up before the compact archive is committed")
    state_dir = (Path(output) / sample["id"]).resolve()
    for row in archive["executions"]:
        stem = f"trial-{row['trial']:03d}-{row['root_action']}"
        for suffix in ((".json",) if row["full_trace_retained"] else (".json", ".jsonl.gz")):
            path = state_dir / (stem + suffix)
            if path.resolve().parent != state_dir:
                raise ValueError("Compact cleanup path escapes its state directory")
            path.unlink(missing_ok=True)


def write_json(path, value):
    # Preserve the platform newlines used by historical GOLD artifacts.
    replace_json(path, value, newline=None)


def prepare_resume(root, config, output, identity):
    """Bind a stopped, full-trace run as an explicit read-only import source."""
    source = (root / config["resume_from"]).resolve()
    destination = output.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Resume source and destination must be separate directories")
    path = output / "resume-source.json"
    receipt = _read(path) if path.exists() else None
    original = receipt["identity"] if receipt else _read(source / "identity.json")
    if identity.get("source") is None or original.get("source") != identity["source"]:
        raise ValueError("Resume source content binding is missing or differs from the current source")
    ignored = {"workers", "output", "continuation_storage", "resume_from"}
    semantic = lambda c: {k: v for k, v in c.items() if k not in ignored}
    if (original["protocol"], semantic(original["configuration"]), original["native"]["bridge_sha256"]) != (identity["protocol"], semantic(identity["configuration"]), identity["native"]["bridge_sha256"]):
        raise ValueError("Resume source changes execution inputs, samples or native runtime")
    if original["configuration"].get("continuation_storage") or original["configuration"].get("resume_from"):
        raise ValueError("Only a full-trace run can be imported; use the same config to resume a compact run")
    if receipt is None:
        original_report = _read(source / "report.json")
        available = [s["id"] for s in config["samples"] if (source / s["id"] / "report.json").is_file()
                     or next((source / s["id"]).glob("trial-*.json"), None) is not None]
        receipt = {"source": config["resume_from"], "identity": original,
                   "available_samples": available, "report_sha256": _sha(source / "report.json"),
                   "provenance": {k: original_report.get(k) for k in ("git_revision", "git_dirty", "git_provenance_at")}}
        write_json(path, receipt)
    if receipt["source"] != config["resume_from"]:
        raise ValueError("Resume source differs from its import manifest")
    pending = [sid for sid in receipt["available_samples"] if not (output / sid / "import.json").is_file()]
    if pending and (_read(source / "identity.json") != original or _sha(source / "report.json") != receipt["report_sha256"]):
        raise ValueError("Resume source changed after import began; keep it stopped and unchanged")
    return receipt


def import_sample(root, config, sample, output, stop, manifest):
    """Copy completed trial pairs atomically; an interrupted trial is searched again."""
    source = root / config["resume_from"] / sample["id"]
    destination = output / sample["id"]
    receipt_path = destination / "import.json"
    if sample["id"] not in manifest["available_samples"]:
        return
    if receipt_path.exists():
        if _read(receipt_path)["source"] != config["resume_from"]:
            raise ValueError("Sample import source mismatch")
        return
    if not source.is_dir():
        raise ValueError("Declared import sample is missing from the stopped source")
    files = []
    if source.exists():
        for path in sorted(source.iterdir()):
            if re.fullmatch(r"trial-\d{3}-ACTION_\d+\.json", path.name):
                trace = path.with_suffix(".jsonl.gz")
                if not trace.is_file():
                    raise ValueError("Imported completed trial is missing its trace")
                files.extend((path, trace))
            elif path.name == "report.json" or re.fullmatch(r"stage-\d{3}\.json", path.name):
                files.append(path)
    copied = []
    for path in files:
        if stop.is_set():
            raise InterruptedError("Import interrupted; source files remain unchanged")
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / path.name
        expected = _sha(path)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(path, temporary)
            if _sha(temporary) != expected:
                raise ValueError("Imported file checksum mismatch")
            temporary.replace(target)
        elif _sha(target) != expected:
            raise ValueError("Destination differs from interrupted import")
        copied.append({"file": path.name, "sha256": expected, "bytes": path.stat().st_size})
    write_json(receipt_path, {"source": config["resume_from"], "files": copied})
