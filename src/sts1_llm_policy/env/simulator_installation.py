from __future__ import annotations

from dataclasses import dataclass
import json
import platform
from pathlib import Path
from typing import Any

from sts1_llm_policy.artifacts import sha256_file

LEGACY_MECHANICS = "legacy_v1"
CORRECTED_MECHANICS = "corrected_v1"


class SimulatorInstallationError(RuntimeError):
    """Raised when the isolated simulator build is absent or inconsistent."""


@dataclass(frozen=True)
class StsLightspeedInstallation:
    project_root: Path
    isolation_root: Path
    source_dir: Path
    build_dir: Path
    manifest_path: Path
    bridge_executable: Path
    revision: str
    platform: str | None = None
    features: tuple[str, ...] = ()
    mechanics: str | None = None


def default_sts_lightspeed_config_name() -> str:
    """Select the native build config without changing the Windows identity."""

    system = platform.system()
    names = {"Linux": "sts_lightspeed_build_linux.json", "Windows": "sts_lightspeed_build.json"}
    if system not in names:
        raise SimulatorInstallationError(f"Unsupported simulator platform: {system}")
    return names[system]


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SimulatorInstallationError(f"Missing {label}: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SimulatorInstallationError(
            f"Unable to read {label} as JSON: {path}"
        ) from error

    if not isinstance(value, dict):
        raise SimulatorInstallationError(f"{label} must contain a JSON object")

    return value


def _resolve_isolated_path(
    project_root: Path,
    isolation_root: Path,
    relative_path: object,
    *,
    label: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise SimulatorInstallationError(f"{label} must be a non-empty path")

    candidate = (project_root / relative_path).resolve()
    try:
        candidate.relative_to(isolation_root)
    except ValueError as error:
        raise SimulatorInstallationError(
            f"{label} escapes simulator isolation root: {candidate}"
        ) from error

    return candidate


def _sha256(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError as error:
        raise SimulatorInstallationError(
            f"Unable to hash simulator artifact: {path}"
        ) from error


def discover_sts_lightspeed(
    *,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
    verify_hashes: bool = True,
) -> StsLightspeedInstallation:
    """Validate and return the isolated native sts_lightspeed installation."""

    if project_root is None:
        root = Path(__file__).resolve().parents[3]
    else:
        root = Path(project_root).resolve()

    if config_path is None:
        resolved_config = root / "configs" / "env" / default_sts_lightspeed_config_name()
    else:
        supplied_config = Path(config_path)
        resolved_config = (
            supplied_config.resolve()
            if supplied_config.is_absolute()
            else (root / supplied_config).resolve()
        )

    config = _load_json_object(resolved_config, label="simulator build config")
    if config.get("schema_version") != 1:
        raise SimulatorInstallationError("Unsupported simulator config schema")
    if config.get("backend") != "sts_lightspeed":
        raise SimulatorInstallationError("Simulator config backend mismatch")
    if config.get("transport") != "native_process":
        raise SimulatorInstallationError("Simulator transport must be native_process")
    mechanics = config.get("mechanics")
    if mechanics not in (LEGACY_MECHANICS, CORRECTED_MECHANICS):
        raise SimulatorInstallationError("Simulator build config must declare supported mechanics")
    configured_platform = config.get("platform")
    if configured_platform is not None:
        if configured_platform not in {"linux", "windows"}:
            raise SimulatorInstallationError("Unsupported simulator platform")
        if configured_platform != platform.system().lower():
            raise SimulatorInstallationError(
                f"Simulator config platform mismatch: expected {platform.system().lower()}"
            )
    configured_features = config.get("features", [])
    if (
        not isinstance(configured_features, list)
        or any(not isinstance(value, str) or not value for value in configured_features)
        or len(set(configured_features)) != len(configured_features)
    ):
        raise SimulatorInstallationError("Simulator features must be unique strings")

    isolation_value = config.get("isolation_root")
    if not isinstance(isolation_value, str) or not isolation_value.strip():
        raise SimulatorInstallationError("isolation_root must be a non-empty path")
    isolation_root = (root / isolation_value).resolve()
    try:
        isolation_root.relative_to(root)
    except ValueError as error:
        raise SimulatorInstallationError(
            "Simulator isolation root must remain inside the project"
        ) from error

    source_dir = _resolve_isolated_path(
        root,
        isolation_root,
        config.get("source_dir"),
        label="source_dir",
    )
    build_dir = _resolve_isolated_path(
        root,
        isolation_root,
        config.get("build_dir"),
        label="build_dir",
    )
    manifest_path = _resolve_isolated_path(
        root,
        isolation_root,
        config.get("manifest_path"),
        label="manifest_path",
    )

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SimulatorInstallationError("artifacts must be a JSON object")
    bridge_executable = _resolve_isolated_path(
        root,
        isolation_root,
        artifacts.get("bridge"),
        label="bridge artifact",
    )

    manifest = _load_json_object(manifest_path, label="simulator build manifest")
    if manifest.get("schema_version") != 1:
        raise SimulatorInstallationError("Unsupported simulator manifest schema")
    if manifest.get("backend") != config["backend"]:
        raise SimulatorInstallationError("Simulator manifest backend mismatch")
    if manifest.get("transport") != config["transport"]:
        raise SimulatorInstallationError("Simulator manifest transport mismatch")
    if manifest.get("mechanics") != mechanics:
        raise SimulatorInstallationError("Simulator manifest mechanics mismatch; rebuild the simulator")
    if manifest.get("platform") != configured_platform:
        raise SimulatorInstallationError("Simulator manifest platform mismatch")
    if manifest.get("features", []) != configured_features:
        raise SimulatorInstallationError("Simulator manifest features mismatch")
    if manifest.get("revision") != config.get("revision"):
        raise SimulatorInstallationError(
            "Simulator revision mismatch; rebuild the isolated backend"
        )

    manifest_submodules = manifest.get("submodules")
    if manifest_submodules != config.get("submodules"):
        raise SimulatorInstallationError(
            "Simulator submodule revisions do not match the build config"
        )

    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, dict):
        raise SimulatorInstallationError("Manifest artifacts must be an object")

    for name, executable in (
        ("bridge", bridge_executable),
    ):
        if not executable.is_file():
            raise SimulatorInstallationError(
                f"Missing simulator {name} artifact: {executable}"
            )
        artifact_record = manifest_artifacts.get(name)
        if not isinstance(artifact_record, dict):
            raise SimulatorInstallationError(
                f"Missing manifest record for {name} artifact"
            )
        if artifact_record.get("path") != artifacts[name]:
            raise SimulatorInstallationError(
                f"Manifest path mismatch for {name} artifact"
            )
        expected_hash = artifact_record.get("sha256")
        if (
            verify_hashes
            and (
                not isinstance(expected_hash, str)
                or _sha256(executable) != expected_hash.lower()
            )
        ):
            raise SimulatorInstallationError(
                f"SHA-256 mismatch for simulator {name} artifact"
            )

    revision = config.get("revision")
    if not isinstance(revision, str):
        raise SimulatorInstallationError("Simulator revision must be a string")

    return StsLightspeedInstallation(
        project_root=root,
        isolation_root=isolation_root,
        source_dir=source_dir,
        build_dir=build_dir,
        manifest_path=manifest_path,
        bridge_executable=bridge_executable,
        revision=revision,
        platform=configured_platform,
        features=tuple(configured_features),
        mechanics=mechanics,
    )
