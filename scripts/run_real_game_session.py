from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from sts1_llm_policy.data.trajectory import TrajectoryLogger
from sts1_llm_policy.env.communication_client import CommunicationClient
from sts1_llm_policy.env.live_description_fallback import (
    SOURCE_DESCRIPTION_FALLBACK_ID,
)
from sts1_llm_policy.env.mechanics_catalog import UnsupportedObservationMechanicError
from sts1_llm_policy.env.monster_behavior import UnsupportedMonsterBehaviorError
from sts1_llm_policy.env.real_game_env import RealGameEnv
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.env.state_schema import CanonicalState
from sts1_llm_policy.model_runtime import load_base_model_runtime_config
from sts1_llm_policy.eval.real_game_session import load_real_game_session_config
from sts1_llm_policy.eval.episode_runner import run_combat_episode
from sts1_llm_policy.policy.llm_policy import LLMPolicy
from sts1_llm_policy.policy.base import PolicyResult
from sts1_llm_policy.policy.transformers_backend import TransformersGenerationBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "live" / "real_game_gold_sft_v5_session.json"
_ACTIVE_LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "real_game_session.log"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log(message: str) -> None:
    _ACTIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _ACTIVE_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"[{_utc_now()}] {message}\n")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"real-game-{timestamp}-{uuid4().hex[:8]}"


def _encounter_name(state: CanonicalState) -> str:
    monsters = state.combat.monsters
    names = [monster.name for monster in monsters]
    return " + ".join(names) if names else "UNKNOWN"


def _entry_record(state: CanonicalState, trajectory: Path, index: int) -> dict[str, Any]:
    combat = state.combat
    cards = combat.hand + combat.draw_pile + combat.discard_pile + combat.exhaust_pile
    deck_identity = sorted((card.card_id, card.upgrades) for card in cards)
    deck_digest = sha256(
        json.dumps(deck_identity, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "combat_index": index,
        "status": "running",
        "episode_id": trajectory.stem,
        "trajectory": str(trajectory.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "game_seed": state.seed,
        "ascension_level": state.ascension_level,
        "act": state.act,
        "floor": state.floor,
        "encounter": _encounter_name(state),
        "entry_hp": combat.player.current_hp,
        "max_hp": combat.player.max_hp,
        "entry_deck_multiset_sha256": deck_digest,
        "entry_relic_ids": [relic.relic_id for relic in state.relics],
    }


def _terminal_hp(raw_state: dict) -> int | None:
    game_state = raw_state.get("game_state")
    if not isinstance(game_state, dict):
        return None
    value = game_state.get("current_hp")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _log_step(combat_index: int):
    def callback(step_index: int, result: PolicyResult) -> None:
        log(
            f"combat={combat_index} step={step_index} "
            f"action={result.action.action_id} source={result.decision_source.value}"
        )

    return callback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one operator-controlled real STS run with the Gold SFT V5 combat policy."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--mode", choices=("preflight", "run"), default="run",
        help="preflight validates all file and checkpoint bindings without loading the model or connecting",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path | None:
    global _ACTIVE_LOG_PATH
    config = load_real_game_session_config(args.config, project_root=PROJECT_ROOT)
    allow_source_descriptions = (
        config.source_description_fallback
        == SOURCE_DESCRIPTION_FALLBACK_ID
    )
    _ACTIVE_LOG_PATH = config.log_path
    log(f"profile={config.session_profile_id} bindings_validated=true")
    if args.mode == "preflight":
        log("preflight_only=true completed=true")
        return None

    client = CommunicationClient(logger=log)
    env_class = RealGameEnv
    if config.observation_version == "observation_v6":
        from sts1_llm_policy.env.real_game_card_selection import CardSelectionRealGameEnv
        env_class = CardSelectionRealGameEnv
    env = env_class(
        client,
        state_poll_interval_seconds=config.state_poll_interval_seconds,
    )
    # CommunicationMod has a short child-process readiness timeout. Complete
    # the lightweight protocol handshake before loading model weights.
    env.handshake()

    runtime = load_base_model_runtime_config(config.model_runtime_path, project_root=PROJECT_ROOT)
    backend = TransformersGenerationBackend.from_config(
        runtime,
        project_root=PROJECT_ROOT,
        lora_checkpoint_dir=config.checkpoint_dir,
    )
    if (
        backend.adapter_metadata is None
        or backend.adapter_metadata.get("weights_sha256")
        != config.expected_checkpoint_weights_sha256
    ):
        raise ValueError("Loaded adapter identity disagrees with the live session profile")
    policy = LLMPolicy(
        backend,
        config.policy_seed,
        observation_version=config.observation_version,
        allow_source_descriptions=allow_source_descriptions,
    )

    session_id = args.session_id or _session_id()
    if not session_id.strip() or Path(session_id).name != session_id:
        raise ValueError("session-id must be one non-empty path-safe name")
    session_dir = (config.output_root / session_id).resolve()
    if not session_dir.is_relative_to(config.output_root.resolve()):
        raise ValueError("session directory escapes configured output root")
    if session_dir.exists():
        raise ValueError(f"session-id already exists: {session_id}")
    report_path = session_dir / "session_report.json"
    report: dict[str, Any] = {
        "schema_version": "real_game_session_report_v1",
        "session_id": session_id,
        "session_profile_id": config.session_profile_id,
        "policy_id": config.policy_id,
        "observation_version": config.observation_version,
        "evidence_class": "coverage_collection",
        "automatic_training_ingest": False,
        "bindings": {
            "session_config": str(args.config.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "session_config_sha256": sha256(args.config.resolve().read_bytes()).hexdigest(),
            "base_model_id": runtime.model_id,
            "base_model_revision": runtime.revision,
            "checkpoint_weights_sha256": config.expected_checkpoint_weights_sha256,
            "source_description_fallback": config.source_description_fallback,
            "native_id_crosswalk": config.native_id_crosswalk,
        },
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "status": "loading",
        "combats": [],
    }
    _atomic_json(report_path, report)

    active: dict[str, Any] | None = None
    try:
        report["status"] = "waiting_for_combat"
        report["updated_at_utc"] = _utc_now()
        _atomic_json(report_path, report)
        state = env.connect_session()
        while state is not None and len(report["combats"]) < config.max_combats:
            combat_index = len(report["combats"]) + 1
            episode_id = f"{session_id}-combat-{combat_index:03d}"
            trajectory_path = session_dir / f"combat-{combat_index:03d}.jsonl"
            active = _entry_record(state, trajectory_path, combat_index)
            report["combats"].append(active)
            report["status"] = "in_combat"
            report["updated_at_utc"] = _utc_now()
            _atomic_json(report_path, report)

            legal_actions = env.legal_actions()
            serialize_observation(
                state,
                legal_actions,
                version=config.observation_version,
                allow_source_descriptions=allow_source_descriptions,
            )
            trajectory_logger = TrajectoryLogger(
                trajectory_path,
                episode_id=episode_id,
                game_seed=state.seed,
                policy_seed=config.policy_seed,
                policy_name=config.policy_id,
                encounter=active["encounter"],
                evidence_class="coverage_collection",
                observation_serializer_version=config.observation_version,
                allow_source_descriptions=allow_source_descriptions,
                fsync=True,
            )
            log(
                f"combat={combat_index} start act={state.act} floor={state.floor} "
                f"hp={state.combat.player.current_hp}/{state.combat.player.max_hp} "
                f"encounter={active['encounter']!r}"
            )
            summary = run_combat_episode(
                env,
                policy,
                trajectory_logger,
                max_steps=config.max_steps_per_combat,
                on_step=_log_step(combat_index),
            )
            active.update(
                status="completed",
                steps=summary.steps,
                outcome=summary.outcome.value,
                termination_reason=summary.termination_reason,
                terminal_hp=_terminal_hp(env.get_raw_state()),
            )
            active = None
            report["status"] = "waiting_for_operator"
            report["updated_at_utc"] = _utc_now()
            _atomic_json(report_path, report)
            log(f"combat={combat_index} complete outcome={summary.outcome.value}")
            state = env.wait_for_next_combat()

        report["status"] = "completed" if state is None else "stopped_limit"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        if active is not None:
            active["status"] = "interrupted"
        log("session interrupted by operator")
    except (UnsupportedObservationMechanicError, UnsupportedMonsterBehaviorError) as error:
        report["status"] = "stopped_unsupported_mechanic"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        if active is not None:
            active["status"] = "stopped_unsupported_mechanic"
        log(f"unsupported mechanic safe stop: {type(error).__name__}: {error}")
        raise
    except Exception as error:
        report["status"] = "stopped_safely"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        if active is not None:
            active["status"] = "stopped_safely"
        log(f"session safe stop: {type(error).__name__}: {error}")
        raise
    finally:
        report["updated_at_utc"] = _utc_now()
        report["finished_at_utc"] = _utc_now()
        _atomic_json(report_path, report)

    log(f"session={session_id} status={report['status']} report={report_path}")
    return report_path


def main() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(parse_args())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"fatal error: {type(error).__name__}: {error}")
        raise
