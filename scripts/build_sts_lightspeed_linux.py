"""Build the pinned sts_lightspeed bridge as separate Linux-native artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/env/sts_lightspeed_build_linux.json"


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path = PROJECT_ROOT,
    capture: bool = False,
) -> str:
    rendered = [str(value) for value in command]
    result = subprocess.run(
        rendered,
        cwd=cwd,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout if capture else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(relative: object, parent: Path, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = (PROJECT_ROOT / relative).resolve()
    try:
        candidate.relative_to(parent.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must remain inside {parent}: {candidate}") from error
    return candidate


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Build config must be a JSON object")
    if value.get("schema_version") != 1 or value.get("backend") != "sts_lightspeed":
        raise ValueError("Unsupported sts_lightspeed build configuration")
    if value.get("platform") != "linux" or not sys.platform.startswith("linux"):
        raise ValueError("The Linux simulator builder must run on Linux with a Linux config")
    if value.get("transport") != "native_process":
        raise ValueError("Simulator transport must be native_process")
    if value.get("mechanics") != "corrected_v1":
        raise ValueError("The Linux builder provides corrected_v1 mechanics only")
    return value


def _object_name(source: Path, source_dir: Path, overlay_source: Path) -> str:
    if source == overlay_source:
        relative = Path("source_overlay/src/sim/search/BattleScumSearcher2.cpp")
    else:
        relative = source.relative_to(source_dir)
    return re.sub(r"[\\/:]", "_", str(relative)).removesuffix(".cpp") + ".o"


def build(config_path: Path) -> Path:
    config = _load_config(config_path)
    isolation_root = _resolve_inside(
        config.get("isolation_root"), PROJECT_ROOT, "isolation_root"
    )
    source_dir = _resolve_inside(config.get("source_dir"), isolation_root, "source_dir")
    build_dir = _resolve_inside(config.get("build_dir"), isolation_root, "build_dir")
    manifest_path = _resolve_inside(
        config.get("manifest_path"), isolation_root, "manifest_path"
    )
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")
    artifact_paths = {
        name: _resolve_inside(artifacts.get(name), isolation_root, f"{name} artifact")
        for name in ("bridge",)
    }

    tracked = PROJECT_ROOT / "tools/sts_lightspeed"
    compat_header = tracked / "compat.hpp"
    bridge_source = tracked / "decision_bridge.cpp"
    search_patch = tracked / "battle_scum_searcher2_bridge.patch"
    cards_patch = tracked / "cards_seeing_red.patch"
    actions_patch = tracked / "actions_upgrade_hand.patch"
    rage_patch = tracked / "cards_rage_cost.patch"
    mechanics_patch = tracked / "card_mechanics.patch"
    memory_patch = tracked / "public_draw_memory.patch"
    memory_header = tracked / "public_draw_memory.h"
    for path in (compat_header, bridge_source, search_patch, cards_patch, actions_patch, rage_patch, mechanics_patch, memory_patch, memory_header):
        if not path.is_file():
            raise FileNotFoundError(f"Missing tracked simulator build input: {path}")

    git = shutil.which("git")
    compiler = shutil.which("g++")
    if git is None or compiler is None:
        raise RuntimeError("Linux simulator build requires git and g++")

    isolation_root.mkdir(parents=True, exist_ok=True)
    created_checkout = False
    if not source_dir.exists():
        print("Cloning pinned sts_lightspeed source into the Linux isolation root...", flush=True)
        _run((git, "clone", str(config["upstream_url"]), source_dir), cwd=isolation_root)
        created_checkout = True
    if not (source_dir / ".git").exists():
        raise RuntimeError(f"Existing source_dir is not a Git checkout: {source_dir}")

    git_prefix = (git, "-c", f"safe.directory={source_dir}", "-C", source_dir)
    if not created_checkout:
        dirty = _run((*git_prefix, "status", "--porcelain", "--untracked-files=no"), capture=True)
        if dirty.strip():
            raise RuntimeError("Isolated upstream checkout has tracked modifications")
    try:
        _run((*git_prefix, "cat-file", "-e", f"{config['revision']}^{{commit}}"))
    except subprocess.CalledProcessError:
        _run((*git_prefix, "fetch", "origin", str(config["revision"])))
    _run((*git_prefix, "checkout", "--detach", str(config["revision"])))
    _run((*git_prefix, "submodule", "sync", "--recursive"))
    _run((*git_prefix, "submodule", "update", "--init", "--recursive"))
    actual_revision = _run((*git_prefix, "rev-parse", "HEAD"), capture=True).strip()
    if actual_revision != config.get("revision"):
        raise RuntimeError(f"Revision mismatch: expected {config.get('revision')}, got {actual_revision}")

    actual_submodules: dict[str, str] = {}
    configured_submodules = config.get("submodules")
    if not isinstance(configured_submodules, dict):
        raise ValueError("submodules must be an object")
    for submodule_path, expected_revision in configured_submodules.items():
        resolved = source_dir / submodule_path
        actual = _run(
            (git, "-c", f"safe.directory={resolved}", "-C", resolved, "rev-parse", "HEAD"),
            capture=True,
        ).strip()
        if actual != expected_revision:
            raise RuntimeError(
                f"Submodule {submodule_path} mismatch: expected {expected_revision}, got {actual}"
            )
        actual_submodules[submodule_path] = actual

    object_dir = build_dir / "obj"
    overlay_root = build_dir / "source_overlay"
    overlay_include = overlay_root / "include"
    overlay_header = overlay_include / "sim/search/BattleScumSearcher2.h"
    overlay_source = overlay_root / "src/sim/search/BattleScumSearcher2.cpp"
    overlay_cards = overlay_include / "constants/Cards.h"
    overlay_actions = overlay_root / "src/combat/Actions.cpp"
    overlay_actions.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "src/combat/Actions.cpp", overlay_actions)
    overlay_manager = overlay_root / "src/combat/CardManager.cpp"
    shutil.copy2(source_dir / "src/combat/CardManager.cpp", overlay_manager)
    overlay_card = overlay_root / "src/combat/CardInstance.cpp"
    overlay_battle = overlay_root / "src/combat/BattleContext.cpp"
    shutil.copy2(source_dir / "src/combat/CardInstance.cpp", overlay_card)
    shutil.copy2(source_dir / "src/combat/BattleContext.cpp", overlay_battle)
    object_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths["bridge"].parent.mkdir(parents=True, exist_ok=True)
    overlay_header.parent.mkdir(parents=True, exist_ok=True)
    overlay_source.parent.mkdir(parents=True, exist_ok=True)
    overlay_cards.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "include/sim/search/BattleScumSearcher2.h", overlay_header)
    shutil.copy2(source_dir / "src/sim/search/BattleScumSearcher2.cpp", overlay_source)
    shutil.copy2(source_dir / "include/constants/Cards.h", overlay_cards)
    overlay_relative = overlay_root.relative_to(PROJECT_ROOT).as_posix()
    _run((git, "apply", f"--directory={overlay_relative}", search_patch))
    _run((git, "apply", f"--directory={overlay_relative}", cards_patch))
    _run((git, "apply", f"--directory={overlay_relative}", rage_patch))
    _run((git, "apply", f"--directory={overlay_relative}", mechanics_patch))
    _run((git, "apply", f"--directory={overlay_relative}", actions_patch))

    _run((git, "apply", f"--directory={overlay_relative}", memory_patch))

    compile_flags = [
        "-DSTS_PUBLIC_DRAW_MEMORY", "-I", str(tracked),
        "-DSTS_CORRECTED_CARD_MECHANICS",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-Wno-shift-count-overflow",
        "-pthread",
        "-include",
        str(compat_header),
        "-I",
        str(overlay_include),
        "-I",
        str(source_dir / "include"),
        "-I",
        str(source_dir / "json/single_include"),
    ]
    upstream_searcher = source_dir / "src/sim/search/BattleScumSearcher2.cpp"
    engine_sources = sorted(
        (path for path in (source_dir / "src").rglob("*.cpp") if path != upstream_searcher),
        key=str,
    ) + [overlay_source]
    engine_objects: list[Path] = []
    print(f"Compiling {len(engine_sources)} engine translation units...", flush=True)
    for source in engine_sources:
        target = object_dir / _object_name(source, source_dir, overlay_source)
        compiled_source = {
            source_dir / "src/combat/Actions.cpp": overlay_actions,
            source_dir / "src/combat/CardManager.cpp": overlay_manager,
            source_dir / "src/combat/CardInstance.cpp": overlay_card,
            source_dir / "src/combat/BattleContext.cpp": overlay_battle,
        }.get(source, source)
        _run((compiler, *compile_flags, "-c", compiled_source, "-o", target))
        engine_objects.append(target)

    bridge_object = object_dir / "app_decision_bridge.o"
    _run((compiler, *compile_flags, "-c", bridge_source, "-o", bridge_object))

    link_flags = ["-pthread", "-static-libgcc", "-static-libstdc++"]
    _run((compiler, bridge_object, *engine_objects, *link_flags, "-o", artifact_paths["bridge"]))

    compiler_version = _run((compiler, "--version"), capture=True).splitlines()[0]
    manifest = {
        "schema_version": 1,
        "mechanics": "corrected_v1",
        "backend": "sts_lightspeed",
        "platform": "linux",
        "features": config.get("features", []),
        "transport": "native_process",
        "upstream_url": config["upstream_url"],
        "revision": actual_revision,
        "submodules": actual_submodules,
        "compiler": {
            "path": compiler,
            "version": compiler_version,
            "compile_flags": compile_flags,
            "link_flags": link_flags,
        },
        "source_overlays": {
            "public_draw_memory_patch": {"path": "tools/sts_lightspeed/public_draw_memory.patch", "sha256": _sha256(memory_patch)},
            "public_draw_memory_header": {"path": "tools/sts_lightspeed/public_draw_memory.h", "sha256": _sha256(memory_header)},
            "card_mechanics": {
                "path": "tools/sts_lightspeed/card_mechanics.patch",
                "sha256": _sha256(mechanics_patch),
            },
            "cards_rage_cost": {
                "path": "tools/sts_lightspeed/cards_rage_cost.patch",
                "sha256": _sha256(rage_patch),
            },
            "actions_upgrade_hand": {
                "path": "tools/sts_lightspeed/actions_upgrade_hand.patch",
                "sha256": _sha256(actions_patch),
            },
            "battle_scum_searcher2_bridge_patch": {
                "path": "tools/sts_lightspeed/battle_scum_searcher2_bridge.patch",
                "sha256": _sha256(search_patch),
            },
            "cards_seeing_red_patch": {
                "path": "tools/sts_lightspeed/cards_seeing_red.patch",
                "sha256": _sha256(cards_patch),
            },
            "decision_bridge": {
                "path": "tools/sts_lightspeed/decision_bridge.cpp",
                "sha256": _sha256(bridge_source),
            },
        },
        "artifacts": {
            name: {"path": artifacts[name], "sha256": _sha256(path)}
            for name, path in artifact_paths.items()
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print("sts_lightspeed Linux build complete.", flush=True)
    print(f"  source:   {source_dir}", flush=True)
    print(f"  bridge:   {artifact_paths['bridge']}", flush=True)
    print(f"  manifest: {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    started = time.monotonic()
    build(args.config.resolve())
    print(f"Elapsed seconds: {time.monotonic() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
