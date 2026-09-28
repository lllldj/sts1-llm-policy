"""Run configured continuous routes for Teacher collection or policy evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts1_llm_policy.artifacts import resolve_repository_path
from sts1_llm_policy.workflows.continuous_run import run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", choices=("auto", "windows", "linux"), default="auto")
    parser.add_argument("--arm", help="Run one configured continuous-route policy arm")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent continuous Teacher routes (default: 1)")
    parser.add_argument("--mode", choices=("preflight", "smoke", "run"), default="run")
    parser.add_argument("--smoke-routes", type=int, help="Positive routes count; required with --mode smoke")
    args = parser.parse_args()
    if args.mode == "smoke":
        if args.smoke_routes is None or args.smoke_routes <= 0:
            parser.error("--mode smoke requires --smoke-routes > 0")
    elif args.smoke_routes is not None:
        parser.error("--smoke-routes requires --mode smoke")
    config_path = resolve_repository_path(PROJECT_ROOT, args.config, expected_kind="file")
    result = run(
        config_path, project_root=PROJECT_ROOT, target=args.target,
        preflight_only=args.mode == "preflight", smoke_routes=args.smoke_routes,
        arm_id=args.arm,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
