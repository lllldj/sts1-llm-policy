"""Rebuild the historical expanded-SFT JSONL views from episodes and audit.json.

Uses the frozen export rule from commit 7b7a76e, without models or simulation.
The original audit, observation strings and classifications are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/expanded_sft/expanded-sft-data-v3-8192-full-4680"
ARTIFACTS = {"source_records": "source_records.jsonl",
             "eligible_source_records": "eligible_source_records.jsonl"}


def rebuild(source: Path, output: Path | None = None) -> dict:
    """Verify reconstructed hashes in memory, or publish both missing JSONL views."""
    audit = json.loads((source / "audit.json").read_text(encoding="utf-8"))
    if audit["schema_version"] != "expanded_sft_medium_audit_v1" or audit["status"] not in {"complete", "complete_with_quarantine"}:
        raise ValueError("A completed historical expanded-SFT audit is required")
    expected = audit["schedule"]["expected_episodes"]
    paths = sorted((source / "episodes").glob("*.json"), key=lambda p: int(p.stem))
    if [int(p.stem) for p in paths] != list(range(expected)):
        raise ValueError("Missing, duplicate or unexpected episode files")
    quarantined = set(audit["schedule"]["quarantined_dev_episode_indices"])
    hidden = set(audit["records"]["hidden_order_unvalidated_observation_sha256"])
    hashes = {key: hashlib.sha256() for key in ARTIFACTS}
    counts, sizes = Counter(), Counter()
    train_decks, dev_decks = set(), {}
    with ExitStack() as stack:
        streams, temporary = {}, None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            if any((output / name).exists() for name in ARTIFACTS.values()):
                raise FileExistsError("Output views already exist; choose another output directory or --verify-only")
            temporary = Path(stack.enter_context(TemporaryDirectory(prefix=".rebuild-sft-", dir=output)))
            streams = {key: stack.enter_context((temporary / name).open("wb")) for key, name in ARTIFACTS.items()}
        for index, path in enumerate(paths):
            episode = json.loads(path.read_text(encoding="utf-8"))
            if episode["episode_index"] != index:
                raise ValueError("Episode filename and identity disagree")
            split = episode["episode_spec"]["split"]
            if split == "train":
                train_decks.add(episode["deck_hash"])
            elif split == "dev":
                dev_decks[index] = episode["deck_hash"]
            else:
                raise ValueError("Only historical train/dev episodes are supported")
            for row in episode["source_records"]:
                if row["episode_index"] != index:
                    raise ValueError("Record belongs to another episode")
                encoded = (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
                keys = ["source_records"]
                if index not in quarantined and row["observation_sha256"] not in hidden:
                    keys.append("eligible_source_records")
                for key in keys:
                    hashes[key].update(encoded)
                    counts[key] += 1
                    sizes[key] += len(encoded)
                    if key in streams:
                        streams[key].write(encoded)
        if {index for index, deck in dev_decks.items() if deck in train_decks} != quarantined:
            raise ValueError("Audit quarantine differs from episode train/dev deck overlap")
        for key, count_key in (("source_records", "source_raw"), ("eligible_source_records", "source_eligible")):
            if counts[key] != audit["records"][count_key] or hashes[key].hexdigest() != audit["artifacts"][key]["sha256"]:
                raise ValueError(f"Rebuilt content differs from the original audit: {key}")
        if output is not None:
            for stream in streams.values():
                stream.close()
            for name in ARTIFACTS.values():
                if (output / name).exists():
                    raise FileExistsError("Destination appeared during rebuild")
            for name in ARTIFACTS.values():
                (temporary / name).rename(output / name)
    return {"status": "verified" if output is None else "rebuilt", "episodes": len(paths),
            "records": dict(counts), "bytes": dict(sizes),
            "sha256": {key: value.hexdigest() for key, value in hashes.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true", help="Compute rebuilt hashes without writing files")
    modes.add_argument("--output-dir", type=Path, help="Default: restore the two missing files in the source directory")
    args = parser.parse_args()
    result = rebuild(args.source_dir, None if args.verify_only else args.output_dir or args.source_dir)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
