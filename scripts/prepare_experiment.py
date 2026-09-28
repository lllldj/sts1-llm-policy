"""Prepare independent training/evaluation configs from explicit member lists."""
import argparse
import json
from pathlib import Path

from sts1_llm_policy.workflows.experiment_preparation import prepare_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True,
                        help="Experiment member list; repeat to prepare multiple experiments")
    parser.add_argument("--output", type=Path, required=True, help="Independent output root")
    args = parser.parse_args()
    result = prepare_experiments(Path(__file__).resolve().parents[1], args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
