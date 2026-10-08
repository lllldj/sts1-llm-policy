"""Resolve native simulator capabilities on the machine executing this process.

Existing builders and byte-bound manifests remain the compatibility boundary.
This layer never selects models, changes experiments, or dispatches SSH jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .simulator_installation import (
    CORRECTED_MECHANICS,
    LEGACY_MECHANICS,
    SimulatorInstallationError,
    StsLightspeedInstallation,
    discover_sts_lightspeed,
)

SELECTION = "combat_card_selection_v1"


@dataclass(frozen=True)
class NativeProfile:
    config_name: str
    builder: str
    extension_manifest: str | None = None
    extension_executable: str | None = None
    extension_builder: str | None = None


PROFILES = {
    "windows": NativeProfile(
        "sts_lightspeed_build.json", "scripts/build_sts_lightspeed.ps1",
        "outputs/card-selection/build_manifest.json",
        "outputs/card-selection/sts_lightspeed_bridge.exe",
        "scripts/build_card_selection_bridge.ps1",
    ),
    "linux": NativeProfile(
        "sts_lightspeed_build_linux.json", "scripts/build_sts_lightspeed_linux.py",
    ),
}


def _hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise SimulatorInstallationError(f"Cannot read bound artifact: {path}") from error


def _object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SimulatorInstallationError(f"Cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise SimulatorInstallationError(f"Expected JSON object: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise SimulatorInstallationError(f"Path escapes project root: {path}")
    return path


@dataclass(frozen=True)
class SimulatorExecution:
    project_root: Path
    target: str
    machine: str
    required_capabilities: frozenset[str]
    mechanics: str
    config_path: Path
    config_sha256: str
    profile: NativeProfile

    def _config(self) -> dict:
        if _hash(self.config_path) != self.config_sha256:
            raise SimulatorInstallationError("Simulator config changed after resolution")
        return _object(self.config_path)

    def _base(self) -> StsLightspeedInstallation:
        self._config()
        return discover_sts_lightspeed(
            project_root=self.project_root, config_path=self.config_path,
        )

    def validate(self) -> StsLightspeedInstallation:
        """Validate once per execution object; resolve again after external changes."""
        return self._installation

    @cached_property
    def _installation(self) -> StsLightspeedInstallation:
        base = self._base()
        if self.mechanics == LEGACY_MECHANICS:
            if base.mechanics != LEGACY_MECHANICS:
                raise SimulatorInstallationError("Native build does not provide legacy mechanics")
            return base
        source = self.project_root / "tools/sts_lightspeed/decision_bridge.cpp"
        if base.mechanics == CORRECTED_MECHANICS:
            if SELECTION in self.required_capabilities and SELECTION not in base.features:
                raise SimulatorInstallationError(f"Native build does not declare {SELECTION}")
            manifest = _object(base.manifest_path)
            overlays = manifest.get("source_overlays")
            binding = overlays.get("decision_bridge") if isinstance(overlays, dict) else None
            if (not isinstance(binding, dict)
                    or binding.get("path") != "tools/sts_lightspeed/decision_bridge.cpp"
                    or binding.get("sha256") != _hash(source)):
                raise SimulatorInstallationError("Corrected bridge source binding changed")
            action_binding = overlays.get("actions_upgrade_hand")
            action_path = "tools/sts_lightspeed/actions_upgrade_hand.patch"
            if (not isinstance(action_binding, dict)
                    or action_binding.get("path") != action_path
                    or action_binding.get("sha256") != _hash(self.project_root / action_path)):
                raise SimulatorInstallationError("Hand upgrade action binding changed; rebuild the simulator")
            rage_binding = overlays.get("cards_rage_cost")
            rage_path = "tools/sts_lightspeed/cards_rage_cost.patch"
            if (not isinstance(rage_binding, dict)
                    or rage_binding.get("path") != rage_path
                    or rage_binding.get("sha256") != _hash(self.project_root / rage_path)):
                raise SimulatorInstallationError("Rage cost binding changed; rebuild the simulator")
            mechanics_binding = overlays.get("card_mechanics")
            mechanics_path = "tools/sts_lightspeed/card_mechanics.patch"
            if (not isinstance(mechanics_binding, dict)
                    or mechanics_binding.get("path") != mechanics_path
                    or mechanics_binding.get("sha256") != _hash(self.project_root / mechanics_path)):
                raise SimulatorInstallationError("Card mechanics binding changed; rebuild the simulator")
            for name, filename in (("public_draw_memory_patch", "public_draw_memory.patch"),
                                   ("public_draw_memory_header", "public_draw_memory.h")):
                binding = overlays.get(name)
                path = "tools/sts_lightspeed/" + filename
                if not isinstance(binding, dict) or binding.get("path") != path or binding.get("sha256") != _hash(self.project_root / path):
                    raise SimulatorInstallationError("Public draw memory binding changed; rebuild the simulator")
            return base
        if self.profile.extension_manifest is None:
            raise SimulatorInstallationError("Native build does not provide corrected mechanics")
        manifest_path = _inside(self.project_root, self.profile.extension_manifest)
        manifest = _object(manifest_path)
        if (manifest.get("schema_version") != "card_selection_bridge_build_v1"
                or manifest.get("revision") != base.revision
                or manifest.get("mechanics") != CORRECTED_MECHANICS):
            raise SimulatorInstallationError("Unsupported card-selection bridge identity")
        executable = _inside(self.project_root, self.profile.extension_executable)
        for key, path in {
            "base_manifest_sha256": base.manifest_path,
            "bridge_source_sha256": source,
            "cards_seeing_red_sha256": self.project_root / "tools/sts_lightspeed/cards_seeing_red.patch",
            "actions_upgrade_hand_sha256": self.project_root / "tools/sts_lightspeed/actions_upgrade_hand.patch",
            "cards_rage_cost_sha256": self.project_root / "tools/sts_lightspeed/cards_rage_cost.patch",
            "card_mechanics_sha256": self.project_root / "tools/sts_lightspeed/card_mechanics.patch",
            "public_draw_memory_patch_sha256": self.project_root / "tools/sts_lightspeed/public_draw_memory.patch",
            "public_draw_memory_header_sha256": self.project_root / "tools/sts_lightspeed/public_draw_memory.h",
            "bridge_sha256": executable,
        }.items():
            if manifest.get(key) != _hash(path):
                raise SimulatorInstallationError(f"Card-selection build binding changed: {key}")
        return replace(base, bridge_executable=executable, features=(*base.features, SELECTION),
                       mechanics=CORRECTED_MECHANICS)

    def create_client(self):
        from .simulator_client import StsLightspeedClient
        return StsLightspeedClient(self.validate())

    def create_environment(self):
        from .simulator_env import StsLightspeedEnv
        return StsLightspeedEnv(
            self.create_client(), allow_card_selection=SELECTION in self.required_capabilities,
        )

    def build_commands(self, *, portable: bool = False) -> list[list[str]]:
        """Describe native commands; selection never executes these implicitly."""
        self._config()
        builder = self.profile.builder if portable else str(self.project_root / self.profile.builder)
        config = self.config_path.relative_to(self.project_root).as_posix() if portable else str(self.config_path)
        if self.target == "linux":
            python = ["uv", "run", "--locked", "python"] if portable else [sys.executable]
            command = [*python, builder, "--config", config]
            return [command]
        shell = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        if portable:
            shell = Path(shell).name
        prefix = [shell, "-NoProfile", "-NonInteractive", "-File"]
        command = [*prefix, builder, "-ConfigPath", config]
        commands = [command]
        if self.mechanics == CORRECTED_MECHANICS:
            extension = self.profile.extension_builder if portable else str(self.project_root / self.profile.extension_builder)
            commands.append([*prefix, extension])
        return commands

    def build(self) -> StsLightspeedInstallation:
        """Build only absent installations; never overwrite an existing binding."""
        # Builds are an explicit mutation boundary, unlike repeated client creation.
        self.__dict__.pop("_installation", None)
        config = self._config()
        manifest = _inside(self.project_root, config["manifest_path"])
        commands = self.build_commands()
        if len(commands) > 1 and _inside(self.project_root, self.profile.extension_manifest).exists():
            # A retained extension binds its base too; validate before any mutation.
            return self.validate()
        if manifest.exists():
            self._base()
        else:
            artifacts = [_inside(self.project_root, path) for path in config["artifacts"].values()]
            if any(path.exists() for path in artifacts):
                raise SimulatorInstallationError("Unbound base artifacts exist; inspect before building")
            subprocess.run(commands[0], cwd=self.project_root, check=True)
            self._base()
        if len(commands) > 1:
            extension_manifest = _inside(self.project_root, self.profile.extension_manifest)
            executable = _inside(self.project_root, self.profile.extension_executable)
            if not extension_manifest.exists():
                if executable.exists():
                    raise SimulatorInstallationError("Unbound selection artifact exists; inspect before building")
                subprocess.run(commands[1], cwd=self.project_root, check=True)
        return self.validate()

    def describe(self, *, verify: bool = False) -> dict:
        config = self._config()
        result = {
            "schema_version": "simulator_execution_v1",
            "target": self.target,
            "machine": self.machine,
            "project_root": ".",
            "required_capabilities": sorted(self.required_capabilities),
            "mechanics": self.mechanics,
            "config": self.config_path.relative_to(self.project_root).as_posix(),
            "config_sha256": self.config_sha256,
            "revision": config["revision"],
            "build_commands": self.build_commands(portable=True),
            "status": "resolved",
        }
        if verify:
            installation = self.validate()
            manifests = [installation.manifest_path]
            if (self.mechanics == CORRECTED_MECHANICS
                    and config.get("mechanics") != CORRECTED_MECHANICS
                    and self.profile.extension_manifest):
                manifests.append(_inside(self.project_root, self.profile.extension_manifest))
            result.update(
                status="ready", features=list(installation.features),
                bridge_executable=installation.bridge_executable.relative_to(self.project_root).as_posix(),
                bridge_sha256=_hash(installation.bridge_executable),
                manifests=[{"path": path.relative_to(self.project_root).as_posix(), "sha256": _hash(path)} for path in manifests],
            )
        return result


def resolve_simulator(
    *, project_root: str | Path | None = None, target: str = "auto",
    required_capabilities=(), mechanics: str = CORRECTED_MECHANICS,
) -> SimulatorExecution:
    """Resolve on the execution host; explicit targets must match that host."""
    system = platform.system().lower()
    if system not in PROFILES:
        raise SimulatorInstallationError(f"Unsupported simulator platform: {system}")
    if target == "auto":
        target = system
    if target not in PROFILES or target != system:
        raise SimulatorInstallationError(f"Simulator target {target!r} does not match host {system!r}")
    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64"}:
        raise SimulatorInstallationError(f"Unsupported simulator architecture: {machine}")
    if isinstance(required_capabilities, str):
        raise SimulatorInstallationError("required_capabilities must be a collection, not a string")
    required = frozenset(required_capabilities) | {"simulator"}
    if not required <= {"simulator", SELECTION}:
        raise SimulatorInstallationError(f"Unsupported simulator capabilities: {required}")
    if mechanics not in (LEGACY_MECHANICS, CORRECTED_MECHANICS):
        raise SimulatorInstallationError(f"Unsupported simulator mechanics: {mechanics!r}")
    if mechanics == LEGACY_MECHANICS and (target != "windows" or SELECTION in required):
        raise SimulatorInstallationError("legacy_v1 supports only Windows without card selection")
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[3]
    profile = PROFILES[target]
    config_path = root / "configs/env" / profile.config_name
    config = _object(config_path)
    if (config.get("schema_version") != 1 or config.get("backend") != "sts_lightspeed"
            or config.get("transport") != "native_process"
            or config.get("platform", "windows") != target
            or config.get("mechanics") != (LEGACY_MECHANICS if target == "windows" else CORRECTED_MECHANICS)):
        raise SimulatorInstallationError("Simulator profile config identity mismatch")
    if SELECTION in required and not profile.extension_manifest and SELECTION not in config.get("features", []):
        raise SimulatorInstallationError(f"Profile does not provide {SELECTION}")
    return SimulatorExecution(root, target, machine, required, mechanics, config_path, _hash(config_path), profile)
