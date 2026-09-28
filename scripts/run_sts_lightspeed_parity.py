from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from sts1_llm_policy.artifacts import atomic_write_json

from sts1_llm_policy.env.simulator_installation import discover_sts_lightspeed
from sts1_llm_policy.eval.simulator_parity import (
    evaluate_simulator_parity,
    load_parity_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "runs" / "evaluation" / "sts_lightspeed_parity_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare curated real STS1 transitions with sts_lightspeed."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="New report path; defaults to outputs/parity/<unique-id>.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = (args.output or PROJECT_ROOT / "outputs/parity" /
              (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8] + ".json")).resolve()
    if output.exists():
        raise FileExistsError(f"Parity report already exists: {output}")
    config = load_parity_config(args.config, project_root=PROJECT_ROOT)
    installation = discover_sts_lightspeed(project_root=PROJECT_ROOT)
    if installation.revision != config.backend_revision:
        raise RuntimeError(
            "Parity/backend revision mismatch: "
            f"{config.backend_revision} != {installation.revision}"
        )
    report = evaluate_simulator_parity(config)
    manifest = json.loads(installation.manifest_path.read_text(encoding="utf-8"))
    report["environment"] = {
        "bridge_sha256": manifest.get("artifacts", {})
        .get("bridge", {})
        .get("sha256"),
        "transport": "native_process_jsonl_stdio",
    }
    atomic_write_json(output, report, allow_identical=False)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "gates": report["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
