"""Export the matched A/B/C DPO datasets from a declared executed-GOLD collection."""
import argparse
import json
from pathlib import Path

from sts1_llm_policy.data.gold.dpo_export import export_gold_dpo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = export_gold_dpo(Path(__file__).resolve().parents[1], args.config)
    print(json.dumps({"status": result["status"], "datasets": result["datasets"],
                      "checks": result["checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
