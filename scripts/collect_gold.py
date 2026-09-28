"""Execute or replay a declared GOLD collection or continuation panel."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from sts1_llm_policy.data.gold.configuration import load_config
from sts1_llm_policy.data.gold.collection import run
from sts1_llm_policy.data.gold.verification import verify_run
from sts1_llm_policy.artifacts import atomic_write_json, resolve_repository_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "run"), default="run")
    parser.add_argument("--verify", action="store_true", help="Replay every execution from full traces or compact action sequences; combine with --mode smoke for its outputs")
    parser.add_argument("--verification-output", help="New receipt path relative to the project root; defaults to verification.json alongside the report")
    parser.add_argument("--verification-selection", help="Report-bound JSON selection of complete states; requires --verify")
    args = parser.parse_args()
    if args.verify and args.mode == "preflight":
        parser.error("--verify reads final execution traces; it cannot be combined with --mode preflight")
    if args.verification_output and not args.verify:
        parser.error("--verification-output requires --verify")
    if args.verification_selection and not args.verify:
        parser.error("--verification-selection requires --verify")
    if args.verify:
        config = load_config(ROOT / args.config, project_root=ROOT)
        destination = resolve_repository_path(
            ROOT, args.verification_output or Path(config["output"]) / ("smoke" if args.mode == "smoke" else "formal") / "verification.json",
            must_exist=False, expected_kind="file",
        )
        if destination.exists():
            raise ValueError(f"Verification receipt already exists: {destination}; use --verification-output with a new path")
        selection = None
        if args.verification_selection:
            path = resolve_repository_path(ROOT, args.verification_selection, expected_kind="file")
            selection = json.loads(path.read_text(encoding="utf-8"))
        verification = verify_run(ROOT, config, smoke=args.mode == "smoke", selection=selection)
        atomic_write_json(destination, verification, allow_identical=False)
        print(json.dumps(verification))
        return 0
    report = run(
        ROOT, load_config(ROOT / args.config, project_root=ROOT),
        smoke=args.mode == "smoke", preflight=args.mode == "preflight",
    )
    print(json.dumps({"status": report["status"], "states": len(report["states"]),
                      "elapsed_seconds": report["elapsed_seconds"], "labels_certified": False}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
