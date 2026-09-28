import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sts1_llm_policy.data.trajectory import TrajectoryLogger
from sts1_llm_policy.env.communication_client import CommunicationClient
from sts1_llm_policy.env.real_game_env import RealGameEnv
from sts1_llm_policy.eval.episode_runner import EpisodeSummary, run_combat_episode
from sts1_llm_policy.eval.real_game_preflight import (
    RealGamePreflightError,
    load_p0_clean_capture_config,
    require_p0_clean_preflight,
)
from sts1_llm_policy.policy.random_policy import RANDOM_MODEL_ACTION_POLICY_ID, RandomLegalPolicy
from sts1_llm_policy.policy.base import PolicyResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "trajectories"
    / "random_model_action_v1.jsonl"
)
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "random_combat.log"
DEFAULT_PREFLIGHT_CONFIG = (
    PROJECT_ROOT / "configs" / "live" / "real_game_p0_clean_capture_v1.json"
)


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"[{timestamp}] {message}\n")


def _episode_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"random-{timestamp}-{uuid4().hex[:8]}"


def _project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Random Legal Policy combat through CommunicationMod."
        )
    )
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--state-poll-interval",
        type=float,
        default=0.25,
        help=(
            "seconds between explicit state requests while waiting "
            "outside automatic game updates (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY_PATH,
    )
    parser.add_argument(
        "--fsync",
        action="store_true",
        help="fsync every JSONL transition for stronger durability",
    )
    parser.add_argument(
        "--p0-clean-scenario",
        choices=("cultist", "jaw_worm", "two_louse"),
        default=None,
        help=(
            "require the exact P0 parity-clean opening for this scenario "
            "before the policy may act"
        ),
    )
    parser.add_argument(
        "--preflight-config",
        type=Path,
        default=DEFAULT_PREFLIGHT_CONFIG,
        help="versioned P0 clean-capture preflight configuration",
    )
    return parser.parse_args()


def _encounter_name(env: RealGameEnv) -> str:
    names = [monster.name for monster in env.get_state().combat.monsters]
    return " + ".join(names) if names else "UNKNOWN"


def _log_step(step_index: int, result: PolicyResult) -> None:
    log(
        f"step={step_index} action={result.action.action_id} "
        f"type={result.action.action_type.value} "
        f"source={result.decision_source.value}"
    )


def run(args: argparse.Namespace) -> EpisodeSummary:
    episode_id = args.episode_id or _episode_id()
    trajectory_path = _project_path(args.trajectory)
    policy = RandomLegalPolicy(policy_seed=args.policy_seed)
    client = CommunicationClient(logger=log)
    env = RealGameEnv(
        client=client,
        state_poll_interval_seconds=args.state_poll_interval,
    )

    log(
        f"episode={episode_id} starting policy_seed={args.policy_seed} "
        f"max_steps={args.max_steps} "
        f"state_poll_interval={args.state_poll_interval}"
    )
    state = env.connect()
    encounter = _encounter_name(env)
    preflight = None

    if args.p0_clean_scenario is not None:
        preflight_config = load_p0_clean_capture_config(
            _project_path(args.preflight_config),
            project_root=PROJECT_ROOT,
        )
        preflight = require_p0_clean_preflight(
            env.get_raw_state(),
            preflight_config,
            args.p0_clean_scenario,
        )
        log(
            f"episode={episode_id} preflight="
            + json.dumps(
                preflight,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        log(
            f"episode={episode_id} unclassified capture: "
            "no --p0-clean-scenario guard"
        )

    logger = TrajectoryLogger(
        trajectory_path,
        episode_id=episode_id,
        game_seed=state.seed,
        policy_seed=args.policy_seed,
        policy_name=RANDOM_MODEL_ACTION_POLICY_ID,
        encounter=encounter,
        evidence_class=(
            preflight["evidence_class"]
            if preflight is not None
            else None
        ),
        scenario_id=args.p0_clean_scenario,
        preflight=preflight,
        fsync=args.fsync or preflight is not None,
    )

    log(
        f"episode={episode_id} connected game_seed={state.seed} "
        f"encounter={encounter!r} trajectory={trajectory_path}"
    )

    summary = run_combat_episode(
        env,
        policy,
        logger,
        max_steps=args.max_steps,
        on_step=_log_step,
    )

    log(
        f"episode={summary.episode_id} finished steps={summary.steps} "
        f"outcome={summary.outcome.value} score={summary.score} "
        f"reason={summary.termination_reason}"
    )
    return summary


def main() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except RealGamePreflightError as error:
        log(
            "preflight rejection="
            + json.dumps(
                error.report,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        log(f"fatal error: {type(error).__name__}: {error}")
        raise
    except Exception as error:
        log(f"fatal error: {type(error).__name__}: {error}")
        raise
