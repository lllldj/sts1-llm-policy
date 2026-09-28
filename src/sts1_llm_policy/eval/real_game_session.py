from __future__ import annotations

from sts1_llm_policy.artifacts import sha256_file

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from sts1_llm_policy.env.live_description_fallback import (
    SOURCE_DESCRIPTION_FALLBACK_ID,
)
from sts1_llm_policy.env.live_id_crosswalk import LIVE_ID_CROSSWALK_ID
from sts1_llm_policy.env.serializer import (
    SEMANTIC_OBSERVATION_SERIALIZER_VERSION,
)


REAL_GAME_SESSION_SCHEMA_VERSION = "real_game_session_v2"
COMMUNICATION_MOD_DESCRIPTION_FALLBACK = {
    "mode": SOURCE_DESCRIPTION_FALLBACK_ID,
    "native_id_crosswalk": LIVE_ID_CROSSWALK_ID,
    "unknown_cards": "use_nonempty_description",
    "unknown_relics": "use_nonempty_description",
    "unknown_powers": "fail_closed",
    "unknown_monster_moves": "fail_closed",
    "unknown_monster_behaviors": "fail_closed",
}


@dataclass(frozen=True)
class RealGameSessionConfig:
    session_profile_id: str
    policy_id: str
    policy_seed: int
    observation_version: str
    model_runtime_path: Path
    checkpoint_dir: Path
    output_root: Path
    log_path: Path
    max_combats: int
    max_steps_per_combat: int
    state_poll_interval_seconds: float
    expected_checkpoint_weights_sha256: str
    source_description_fallback: str
    native_id_crosswalk: str


def _repository_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a repository-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be repository-relative")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} escapes the repository")
    return resolved


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _read_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read {field}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain one JSON object")
    return value


def load_real_game_session_config(
    path: str | Path,
    *,
    project_root: str | Path,
) -> RealGameSessionConfig:
    """Validate the exact live policy identity without loading GPU weights."""

    root = Path(project_root).resolve()
    config_path = Path(path).resolve()
    if not config_path.is_relative_to(root):
        raise ValueError("Real-game session config escapes the repository")
    raw = _read_object(config_path, "real-game session config")
    if raw.get("schema_version") != REAL_GAME_SESSION_SCHEMA_VERSION:
        raise ValueError("Unsupported real-game session schema")

    policy = raw.get("policy")
    safety = raw.get("safety")
    evidence = raw.get("evidence")
    mechanics_fallback = raw.get("mechanics_fallback")
    if not isinstance(policy, dict) or not isinstance(safety, dict):
        raise ValueError("policy and safety must be objects")
    if evidence != {
        "classification": "coverage_collection",
        "test_data_read": False,
        "sealed_test_run": False,
        "stable_policy_claim": False,
        "automatic_training_ingest": False,
    }:
        raise ValueError("evidence must preserve the exploratory safety boundary")
    if mechanics_fallback != COMMUNICATION_MOD_DESCRIPTION_FALLBACK:
        raise ValueError(
            "mechanics_fallback must preserve the configured live-only boundary"
        )

    observation_version = policy.get("observation_version")
    if observation_version == "observation_v6":
        if raw.get("interaction_contract") != "combat_card_selection_v1":
            raise ValueError("V6 requires an explicit combat card-selection contract")
    elif observation_version != SEMANTIC_OBSERVATION_SERIALIZER_VERSION:
        raise ValueError("The live Gold SFT profile requires observation_v5")
    elif "interaction_contract" in raw:
        raise ValueError("V5 cannot silently enable card selection")

    runtime_path = _repository_path(root, policy.get("model_runtime"), "policy.model_runtime")
    training_config_path = _repository_path(
        root, policy.get("training_config"), "policy.training_config"
    )
    training_report_path = _repository_path(
        root, policy.get("training_report"), "policy.training_report"
    )
    checkpoint_dir = _repository_path(root, policy.get("checkpoint_dir"), "policy.checkpoint_dir")
    for required in (runtime_path, training_config_path, training_report_path):
        if not required.is_file():
            raise ValueError(f"Required bound file is missing: {required}")
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory is missing: {checkpoint_dir}")

    bindings = {
        runtime_path: _require_sha256(
            policy.get("expected_model_runtime_sha256"),
            "policy.expected_model_runtime_sha256",
        ),
        training_config_path: _require_sha256(
            policy.get("expected_training_config_sha256"),
            "policy.expected_training_config_sha256",
        ),
        training_report_path: _require_sha256(
            policy.get("expected_training_report_sha256"),
            "policy.expected_training_report_sha256",
        ),
    }
    for bound_path, expected in bindings.items():
        if sha256_file(bound_path) != expected:
            raise ValueError(f"Bound file SHA-256 mismatch: {bound_path}")

    training_config = _read_object(training_config_path, "training config")
    training_report = _read_object(training_report_path, "training report")
    expected_run_id = policy.get("training_run_id")
    if (
        not isinstance(expected_run_id, str)
        or training_config.get("run_id") != expected_run_id
        or training_report.get("run_id") != expected_run_id
        or training_report.get("mode") != "run"
        or training_report.get("training_started") is not True
    ):
        raise ValueError("Gold SFT training identity or completion status mismatch")
    if any(
        training_config.get(flag) is not False
        for flag in ("test_data_read", "sealed_test_run", "stable_training_claim")
    ):
        raise ValueError("Training config violates the development-only boundary")

    checkpoint_config_path = checkpoint_dir / "adapter_config.json"
    checkpoint_metadata = _read_object(checkpoint_config_path, "checkpoint config")
    weights_name = checkpoint_metadata.get("weights_file")
    if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
        raise ValueError("Checkpoint weights_file must be one plain filename")
    weights_path = checkpoint_dir / weights_name
    expected_weights = _require_sha256(
        policy.get("expected_checkpoint_weights_sha256"),
        "policy.expected_checkpoint_weights_sha256",
    )
    if (
        checkpoint_metadata.get("weights_sha256") != expected_weights
        or not weights_path.is_file()
        or sha256_file(weights_path) != expected_weights
    ):
        raise ValueError("Gold SFT checkpoint weights SHA-256 mismatch")

    policy_seed = policy.get("policy_seed")
    max_combats = safety.get("max_combats")
    max_steps = safety.get("max_steps_per_combat")
    poll_interval = safety.get("state_poll_interval_seconds")
    if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
        raise ValueError("policy.policy_seed must be an integer")
    for value, field in ((max_combats, "max_combats"), (max_steps, "max_steps_per_combat")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"safety.{field} must be a positive integer")
    if (
        isinstance(poll_interval, bool)
        or not isinstance(poll_interval, (int, float))
        or not 0 <= float(poll_interval) <= 10
    ):
        raise ValueError("safety.state_poll_interval_seconds must be in [0, 10]")

    for field in ("session_profile_id",):
        if not isinstance(raw.get(field), str) or not raw[field]:
            raise ValueError(f"{field} must be a non-empty string")
    policy_id = policy.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy.policy_id must be a non-empty string")

    return RealGameSessionConfig(
        session_profile_id=raw["session_profile_id"],
        policy_id=policy_id,
        policy_seed=policy_seed,
        observation_version=observation_version,
        model_runtime_path=runtime_path,
        checkpoint_dir=checkpoint_dir,
        output_root=_repository_path(root, raw.get("output_root"), "output_root"),
        log_path=_repository_path(root, raw.get("log_path"), "log_path"),
        max_combats=max_combats,
        max_steps_per_combat=max_steps,
        state_poll_interval_seconds=float(poll_interval),
        expected_checkpoint_weights_sha256=expected_weights,
        source_description_fallback=mechanics_fallback["mode"],
        native_id_crosswalk=mechanics_fallback["native_id_crosswalk"],
    )
