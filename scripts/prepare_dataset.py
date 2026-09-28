"""Build single-combat training manifests from declared Teacher V2 exports."""
import argparse
import json
from pathlib import Path

from sts1_llm_policy.data.single_dataset_export import export_teacher_v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = export_teacher_v2(Path(__file__).resolve().parents[1], args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
