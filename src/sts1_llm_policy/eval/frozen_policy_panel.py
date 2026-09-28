from __future__ import annotations

from sts1_llm_policy.artifacts import (
    read_json_object as _read_object,
    replace_json as _atomic_json,
    sha256_file,
)

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Mapping, Sequence

from sts1_llm_policy.data.trajectory import (
    TrajectoryLogger, iter_trajectory_records, load_completed_trajectory,
)
from sts1_llm_policy.env.simulator_execution import CORRECTED_MECHANICS, LEGACY_MECHANICS, resolve_simulator
from sts1_llm_policy.eval.episode_runner import run_combat_episode
from sts1_llm_policy.model_runtime import BaseModelRuntimeConfig, load_base_model_runtime_config
from sts1_llm_policy.policy.llm_policy import LLMPolicy
from sts1_llm_policy.policy.transformers_backend import TransformersGenerationBackend
from sts1_llm_policy.train.lora import inspect_lora_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(value: str, *, root: Path, kind: str = "file", must_exist: bool = True) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes project root: {value}")
    if must_exist:
        valid = path.is_file() if kind == "file" else path.is_dir()
        if not valid:
            raise ValueError(f"Required {kind} is missing: {value}")
    return path


def _canonical_sha(value: object) -> str:
    return sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    victories = [row for row in rows if row["outcome"] == "victory"]
    losses = [int(row["total_hp_loss"]) for row in rows]
    decisions = sum(int(row["decisions"]) for row in rows)
    inference_ms = sum(float(row["protocol"]["inference_ms_total"]) for row in rows)
    return {
        "combat_count": len(rows),
        "outcome_counts": dict(sorted(outcomes.items())),
        "win_rate": round(len(victories) / len(rows), 6) if rows else None,
        "total_hp_loss": {
            "mean": round(mean(losses), 3) if losses else None,
            "median": round(median(losses), 3) if losses else None,
            "min": min(losses) if losses else None,
            "max": max(losses) if losses else None,
        },
        "victory_hp_loss_mean": (
            round(mean(int(row["total_hp_loss"]) for row in victories), 3)
            if victories else None
        ),
        "enemy_damage_taken_mean": (
            round(mean(int(row["enemy_damage_taken"]) for row in rows), 3)
            if rows else None
        ),
        "self_hp_loss_mean": (
            round(mean(int(row["self_hp_loss"]) for row in rows), 3)
            if rows else None
        ),
        "decisions": decisions,
        "decisions_mean": (
            round(mean(int(row["decisions"]) for row in rows), 3)
            if rows else None
        ),
        "accounting_consistent_count": sum(
            row["accounting_consistent"] is True for row in rows
        ),
        "protocol": {
            "retry_count": sum(int(row["protocol"]["retry_count"]) for row in rows),
            "fallback_count": sum(int(row["protocol"]["fallback_count"]) for row in rows),
            "first_pass_legal_rate": (
                round(
                    sum(int(row["protocol"]["first_pass_legal_count"]) for row in rows)
                    / decisions,
                    6,
                )
                if decisions else None
            ),
            "inference_ms_mean": round(inference_ms / decisions, 3) if decisions else None,
        },
    }


def _arm_report(episodes: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    rows = [dict(item["summary"]) for item in episodes]
    scenarios = sorted({str(row["scenario_id"]) for row in rows})
    return {
        "combat": _aggregate(rows),
        "by_scenario": {
            scenario: _aggregate(
                [row for row in rows if str(row["scenario_id"]) == scenario]
            )
            for scenario in scenarios
        },
    }


def _episode_summary(
    spec: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, outcome: str
) -> dict[str, object]:
    if not records or records[-1].get("done") is not True:
        raise ValueError("Episode trajectory is incomplete")
    envelope = records[-1].get("next_raw_state")
    terminal = envelope.get("state") if isinstance(envelope, Mapping) else None
    accounting = terminal.get("combat_accounting") if isinstance(terminal, Mapping) else None
    player = terminal.get("player") if isinstance(terminal, Mapping) else None
    if not isinstance(accounting, Mapping) or not isinstance(player, Mapping):
        raise ValueError("Terminal combat accounting is missing")
    policy_results = [record.get("policy_result") for record in records]
    if any(not isinstance(item, Mapping) for item in policy_results):
        raise ValueError("Trajectory policy result is missing")
    typed = [item for item in policy_results if isinstance(item, Mapping)]
    total_hp_loss = int(accounting["total_hp_loss"])
    enemy_damage = int(accounting["enemy_damage_taken"])
    self_hp_loss = int(accounting["self_hp_loss"])
    return {
        "episode_index": int(spec["episode_index"]),
        "scenario_id": str(spec["scenario_id"]),
        "ascension": int(spec["ascension"]),
        "combat_seed": int(spec["combat_seed"]),
        "outcome": outcome,
        "decisions": len(records),
        "turns": max(int(record.get("turn", 0)) for record in records),
        "starting_hp": int(accounting["starting_hp"]),
        "ending_hp": int(player["current_hp"]),
        "total_hp_loss": total_hp_loss,
        "enemy_damage_taken": enemy_damage,
        "self_hp_loss": self_hp_loss,
        "accounting_consistent": total_hp_loss == enemy_damage + self_hp_loss,
        "protocol": {
            "retry_count": sum(item.get("retry_used") is True for item in typed),
            "fallback_count": sum(item.get("fallback_used") is True for item in typed),
            "first_pass_legal_count": sum(
                item.get("legal_on_first_attempt") is True for item in typed
            ),
            "inference_ms_total": round(
                sum(float(item.get("inference_ms", 0.0)) for item in typed), 3
            ),
        },
    }


def _binding(
    config: Mapping[str, Any], runtime: BaseModelRuntimeConfig, *,
    panel_hash: str, simulator_revision: str, simulator_bridge_sha256: str,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Locations and acceptance bookkeeping are not inference behavior.
    model = asdict(runtime)
    for name in (
        "runtime_id", "experiment_protocol_id", "snapshot_path", "device",
        "validated_device_name", "validated_compute_capability", "validated_cuda_runtime",
        "validated_bf16_supported", "tokenizer_local_only_verified", "model_weights_loaded",
        "generation_smoke_completed", "execution_requirements",
    ):
        model.pop(name)
    for name in ("weight_assets_sha256", "tokenizer_asset_sha256", "dependencies"):
        model[name] = dict(model[name])
    return {
        "schema_version": "frozen_policy_panel_binding_v4",
        "model_runtime_sha256": _canonical_sha(model),
        "panel_sha256": panel_hash,
        "simulator_revision": simulator_revision,
        "simulator_mechanics": config["simulator_mechanics"],
        "simulator_bridge_sha256": simulator_bridge_sha256,
        "model_revision": runtime.revision,
        "observation_version": config["observation_version"],
        "max_decisions": int(config["max_decisions"]),
        "action_protocol": "single_action_v1_no_secondary_selection",
        **({"adapter_sha256": _canonical_sha({
            key: value for key, value in checkpoint_metadata.items() if key != "weights_file"
        })} if checkpoint_metadata is not None else {}),
    }


def _validate(config_path: Path, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = (root / config_path).resolve()
    if not config_path.is_relative_to(root):
        raise ValueError("Evaluation config must stay inside the repository")
    config = _read_object(config_path)
    allowed = {
        "schema_version", "run_id", "purpose", "arm", "model_runtime", "execution_profile",
        "observation_version", "simulator_mechanics", "panel_manifest", "expected_routes",
        "expected_combats", "expected_panel_sha256", "max_decisions", "output_dir", "checkpoint",
    }
    if config.keys() - allowed:
        raise ValueError(f"Unsupported evaluation fields: {sorted(config.keys() - allowed)}")
    if (
        config.get("schema_version") != "frozen_policy_panel_evaluation_v1"
        or config.get("simulator_mechanics") not in (CORRECTED_MECHANICS, LEGACY_MECHANICS)
        or any(not isinstance(config.get(name), str) or not config[name]
               for name in ("purpose", "arm", "run_id"))
        or config.get("observation_version") != "observation_v5"
        or int(config.get("expected_routes", 0)) <= 0
        or int(config.get("expected_combats", 0)) <= 0
        or int(config.get("max_decisions", 0)) <= 0
    ):
        raise ValueError("Evaluation config violates its frozen development scope")
    paths = {
        "model_runtime": _resolve(str(config["model_runtime"]), root=root),
        "execution_profile": _resolve(str(config["execution_profile"]), root=root),
        "panel_manifest": _resolve(str(config["panel_manifest"]), root=root),
    }
    panel = _read_object(paths["panel_manifest"])
    specs = panel.get("specs")
    raw_decks = panel.get("decks")
    if (
        panel.get("schema_version") != "frozen_combat_panel_inputs_v1"
        or panel.get("observation_version") != "observation_v5"
        or panel.get("test_data_read") is not False
        or panel.get("sealed_test_run") is not False
        or not isinstance(specs, list)
        or not isinstance(raw_decks, dict)
    ):
        raise ValueError("Frozen panel manifest is invalid")
    expected_count = int(config["expected_combats"])
    identities = [json.dumps(spec, sort_keys=True) for spec in specs]
    if (
        len(specs) != expected_count
        or len(set(identities)) != expected_count
        or any(
            not isinstance(spec, dict)
            or not 1 <= int(spec.get("act", -1)) <= 3
            or not 0 <= int(spec.get("ascension", -1)) <= 20
            for spec in specs
        )
    ):
        raise ValueError("Frozen panel must contain the declared number of unique valid specs")
    episode_indices = [int(spec["episode_index"]) for spec in specs]
    if len(set(episode_indices)) != len(specs) or any(index < 0 for index in episode_indices):
        raise ValueError("Frozen panel episode indices must be unique and nonnegative")
    route_indices = sorted({int(spec["route_index"]) for spec in specs})
    if len(route_indices) != int(config["expected_routes"]):
        raise ValueError("Focused panel route count changed")
    decks: dict[int, dict[str, Any]] = {}
    for route_index in route_indices:
        deck = raw_decks.get(str(route_index))
        if (
            not isinstance(deck, dict)
            or not isinstance(deck.get("snapshot"), dict)
            or not isinstance(deck.get("snapshot_sha256"), str)
            or _canonical_sha(deck["snapshot"]) != deck["snapshot_sha256"]
        ):
            raise ValueError(f"Frozen deck is invalid: route {route_index}")
        decks[route_index] = deck
    if set(raw_decks) != {str(index) for index in route_indices}:
        raise ValueError("Frozen panel contains unexpected deck routes")
    execution = resolve_simulator(project_root=root, mechanics=config["simulator_mechanics"])
    installation = execution.validate()
    if (
        panel.get("simulator_revision") != installation.revision
    ):
        raise ValueError("Simulator revision differs from the frozen panel")
    execution_profile = _read_object(paths["execution_profile"])
    runtime = load_base_model_runtime_config(
        paths["model_runtime"],
        project_root=root,
        execution_profile=execution_profile,
    )
    panel_hash = _canonical_sha(
        {
            "specs": specs,
            "decks": {str(index): decks[index]["snapshot_sha256"] for index in route_indices},
        }
    )
    if (
        panel_hash != str(config["expected_panel_sha256"])
        or panel.get("panel_sha256") != panel_hash
    ):
        raise ValueError("Frozen combat specs or deck snapshots changed")
    checkpoint = None
    checkpoint_metadata = None
    if config.get("checkpoint") is not None:
        if not isinstance(config["checkpoint"], str) or not config["checkpoint"]:
            raise ValueError("checkpoint must be a repository-relative directory")
        checkpoint = _resolve(config["checkpoint"], root=root, kind="directory")
        checkpoint_metadata = inspect_lora_checkpoint(
            checkpoint, expected_base_model_id=runtime.model_id,
            expected_base_revision=runtime.revision,
        )
    binding = _binding(config, runtime, panel_hash=panel_hash, simulator_revision=installation.revision,
                       simulator_bridge_sha256=sha256_file(installation.bridge_executable),
                       checkpoint_metadata=checkpoint_metadata)
    return {
        "config": config,
        "runtime": runtime,
        "specs": specs,
        "decks": decks,
        "installation": installation,
        "execution": execution,
        "binding": binding,
        "execution_profile": execution_profile,
        "checkpoint": checkpoint,
        "checkpoint_metadata": checkpoint_metadata,
    }


def _load_completed(
    path: Path,
    *,
    binding: Mapping[str, Any],
    spec: Mapping[str, Any],
    arm: str,
    deck_snapshot_sha256: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    item = _read_object(path)
    if (
        item.get("schema_version") != "frozen_policy_panel_episode_v1"
        or item.get("binding") != binding
        or item.get("arm") != arm
        or item.get("spec") != spec
        or item.get("test_data_read") is not False
    ):
        raise ValueError(f"Completed episode binding changed: {path}")
    if item.get("deck_snapshot_sha256") != deck_snapshot_sha256:
        raise ValueError(f"Completed episode deck snapshot changed: {path}")
    trajectory = path.parent.parent / "trajectories" / f"episode-{int(spec['episode_index']):03d}.jsonl"
    records = load_completed_trajectory(
        trajectory, expected_sha256=item.get("trajectory_sha256"),
        game_seed=spec["combat_seed"], policy_seed=spec["policy_seed"],
        scenario_id=spec["scenario_id"], policy_name=arm,
    )
    if item.get("summary") != _episode_summary(spec, records, outcome=records[-1]["terminal_outcome"]):
        raise ValueError(f"Completed episode summary differs from its trajectory: {path}")
    return item


def run(config_path: Path, *, project_root: Path = PROJECT_ROOT,
        preflight_only: bool = False, smoke_combats: int | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    validated = _validate(config_path, project_root=root)
    config = validated["config"]
    specs = list(validated["specs"])
    if preflight_only:
        return {
            "status": "preflight_passed",
            "run_id": config["run_id"],
            "scope": {"routes": len(validated["decks"]), "combat_identities": len(specs)},
            "binding": validated["binding"],
            "model_weights_loaded": False,
            "simulator_started": False,
            "test_data_read": False,
        }
    if smoke_combats is not None:
        if not 1 <= smoke_combats <= len(specs):
            raise ValueError(f"--smoke-combats must be between 1 and {len(specs)}")
        specs = specs[:smoke_combats]
        output = _resolve(str(config["output_dir"]), root=root, kind="directory", must_exist=False) / "smoke"
    else:
        output = _resolve(str(config["output_dir"]), root=root, kind="directory", must_exist=False) / "formal"
    episode_dir = output / "episodes"
    trajectory_dir = output / "trajectories"
    report_path = output / "report.json"
    arm = str(config["arm"])
    episodes: list[dict[str, Any]] = []
    pending: list[Mapping[str, Any]] = []
    for spec in specs:
        path = episode_dir / f"episode-{int(spec['episode_index']):03d}.json"
        item = _load_completed(
            path, binding=validated["binding"], spec=spec, arm=arm,
            deck_snapshot_sha256=validated["decks"][int(spec["route_index"])]["snapshot_sha256"],
        )
        (pending if item is None else episodes).append(spec if item is None else item)
    print(f"{arm}: validated {len(episodes)} completed; {len(pending)} pending", flush=True)
    started = perf_counter()
    loaded = started
    if pending:
        backend = TransformersGenerationBackend.from_config(
            validated["runtime"], project_root=root,
            lora_checkpoint_dir=validated["checkpoint"],
            verified_lora_metadata=validated["checkpoint_metadata"],
        )
        loaded = perf_counter()
        env = validated["execution"].create_environment()
        try:
            for position, spec in enumerate(pending, 1):
                index = int(spec["episode_index"])
                partial = trajectory_dir / f"episode-{index:03d}.partial.jsonl"
                final = trajectory_dir / f"episode-{index:03d}.jsonl"
                if partial.exists():
                    partial.unlink()
                logger = TrajectoryLogger(
                    partial,
                    episode_id=f"{config['run_id']}-{index:03d}",
                    game_seed=int(spec["combat_seed"]),
                    policy_seed=int(spec["policy_seed"]),
                    policy_name=arm,
                    encounter=str(spec["scenario_id"]),
                    scenario_id=str(spec["scenario_id"]),
                    observation_serializer_version="observation_v5",
                )
                deck = validated["decks"][int(spec["route_index"])]
                env.reset(str(spec["scenario_id"]), int(spec["combat_seed"]), combat_snapshot=deck["snapshot"])
                result = run_combat_episode(
                    env,
                    LLMPolicy(backend, int(spec["policy_seed"]), observation_version="observation_v5"),
                    logger,
                    max_steps=int(config["max_decisions"]),
                )
                summary = _episode_summary(
                    spec, tuple(iter_trajectory_records(partial)), outcome=result.outcome.value
                )
                partial.replace(final)
                item = {
                    "schema_version": "frozen_policy_panel_episode_v1",
                    "binding": validated["binding"],
                    "arm": arm,
                    "spec": spec,
                    "deck_snapshot_sha256": deck["snapshot_sha256"],
                    "trajectory_sha256": sha256_file(final),
                    "summary": summary,
                    "test_data_read": False,
                }
                _atomic_json(episode_dir / f"episode-{index:03d}.json", item)
                episodes.append(item)
                elapsed = perf_counter() - loaded
                eta = elapsed / position * (len(pending) - position) / 60
                print(
                    f"[{position:03d}/{len(pending):03d}] {spec['scenario_id']} "
                    f"{result.outcome.value} hp_loss={summary['total_hp_loss']} eta~{eta:.1f}m",
                    flush=True,
                )
        finally:
            env.close()
    ended = perf_counter()
    episodes_by_index = {int(item["spec"]["episode_index"]): item for item in episodes}
    episodes = [episodes_by_index[int(spec["episode_index"])] for spec in specs]
    checks = {
        "all_selected_combats_completed": len(episodes) == len(specs),
        "all_combat_accounting_consistent": all(item["summary"]["accounting_consistent"] for item in episodes),
        "candidate_matches_frozen_panel": [item["spec"] for item in episodes] == specs,
        "old_action_protocol": validated["binding"]["action_protocol"] == "single_action_v1_no_secondary_selection",
        "no_test_data_read": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Evaluation completion checks failed: {checks}")
    summaries_path = output / "episodes.jsonl"
    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summaries_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(
        json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in episodes
    ), encoding="utf-8", newline="\n")
    temporary.replace(summaries_path)
    report = {
        "schema_version": "frozen_policy_panel_evaluation_report_v1",
        "run_id": config["run_id"],
        "status": "smoke_completed" if smoke_combats is not None else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding": validated["binding"],
        "configuration": config,
        "execution_profile": validated["execution_profile"],
        "scope": {
            "routes": len({int(spec["route_index"]) for spec in specs}),
            "combat_identities": len(specs),
            "smoke_only": smoke_combats is not None,
        },
        "candidate": _arm_report(episodes),
        "timing": {
            "model_load_seconds": round(loaded - started, 3) if pending else 0.0,
            "run_seconds": round(ended - loaded, 3) if pending else 0.0,
            "new_combats": len(pending),
        },
        "checks": checks,
        "local_output": str(output.relative_to(root)).replace("\\", "/"),
        "episode_summaries": "episodes.jsonl",
        "test_data_read": False,
        "sealed_test_run": False,
        "training_started": False,
        "stable_claim": False,
    }
    _atomic_json(report_path, report)
    return report
