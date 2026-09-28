"""Check the current instance after migrating its system and data disks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from sts1_llm_policy.train.configured_training import load_configured_training_run
from sts1_llm_policy.train.recovery import recover_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--previous", type=Path, help="Previous recovery result; only matching backward evidence is reused")
    parser.add_argument("--report", type=Path, help="New output path; defaults to a unique file under outputs/recovery")
    parser.add_argument("--expected-commit", help="Exact published commit expected on this instance")
    args = parser.parse_args()
    report_path = args.report or Path("outputs/recovery") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8] + ".json"
    )
    try:
        print("Checking configuration, dataset hashes and tokenizer (no training)...", flush=True)
        run = load_configured_training_run(PROJECT_ROOT, args.config, mode="backward")
        report = recover_training(run, report_path, previous=args.previous, expected_commit=args.expected_commit)
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "backward": report["backward"].get("execution"), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
