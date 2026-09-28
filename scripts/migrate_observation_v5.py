"""Migrate manifest-declared V4 train/development data without changing labels."""
import argparse
import json
from pathlib import Path

from sts1_llm_policy.data.single_dataset_export import migrate_v5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = migrate_v5(Path(__file__).resolve().parents[1], args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
