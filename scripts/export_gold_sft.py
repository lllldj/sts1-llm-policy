"""Export GOLD or mixed GOLD/Teacher SFT groups from explicit source configs."""
import argparse
import json
from pathlib import Path

from sts1_llm_policy.data.gold.sft_export import export_gold_sft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("smoke", "run"), default="run",
                        help="smoke only replays the configured mixed-data smoke combats")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from sts1_llm_policy.artifacts import read_json_object, resolve_repository_path
    config = read_json_object(resolve_repository_path(root, args.config, expected_kind="file"))
    if config.get("schema_version") == "mixed_sft_export_v1":
        from sts1_llm_policy.data.gold.mixed_export import export_mixed_sft
        result = export_mixed_sft(root, args.config, smoke=args.mode == "smoke")
    else:
        if args.mode == "smoke":
            parser.error("--mode smoke requires a mixed SFT export config")
        result = export_gold_sft(root, args.config)
    print(json.dumps({"dataset_id": result["dataset_id"], "counts": result["export_counts"],
                      "train": result["splits"]["train"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
