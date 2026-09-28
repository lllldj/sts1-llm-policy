from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts1_llm_policy.eval.frozen_policy_panel import run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Base or adapter policy on a frozen panel.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "run"), default="run")
    parser.add_argument("--smoke-combats", type=int, help="Positive combats count; required with --mode smoke")
    args = parser.parse_args()
    if args.mode == "smoke":
        if args.smoke_combats is None or args.smoke_combats <= 0:
            parser.error("--mode smoke requires --smoke-combats > 0")
    elif args.smoke_combats is not None:
        parser.error("--smoke-combats requires --mode smoke")
    result = run(args.config, project_root=PROJECT_ROOT,
                 preflight_only=args.mode == "preflight", smoke_combats=args.smoke_combats)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
