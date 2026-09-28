from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts1_llm_policy.train.configured_runner import run_training
from sts1_llm_policy.train.configured_training import (
    TRAINING_MODES,
    load_configured_training_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manifest-bound SFT or preference training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=TRAINING_MODES, required=True)
    parser.add_argument("--recovery-report", type=Path, help="Passed restore_training report for portable training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configured = load_configured_training_run(
        PROJECT_ROOT, args.config, mode=args.mode
    )
    report = run_training(configured, recovery_report=args.recovery_report)
    print(json.dumps({
        "status": report["status"],
        "mode": report["mode"],
        "scope": report["scope"],
        "tokenization": report["tokenization"],
        "training": report.get("training"),
        "checkpoint": report.get("checkpoint"),
        "checks": report["checks"],
        "development_evaluation_required": report.get("development_evaluation_required"),
        "test_data_read": report["test_data_read"],
    }, indent=2, ensure_ascii=False))
    return 0 if report["status"] != "no_go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
