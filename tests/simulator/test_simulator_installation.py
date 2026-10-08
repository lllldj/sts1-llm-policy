import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sts1_llm_policy.env.simulator_installation import (
    SimulatorInstallationError,
    default_sts_lightspeed_config_name,
    discover_sts_lightspeed,
)


REVISION = "a" * 40
JSON_REVISION = "b" * 40
PYBIND_REVISION = "c" * 40


class SimulatorInstallationTest(unittest.TestCase):
    def _create_installation(self, root: Path) -> Path:
        isolation = root / ".simulator" / "sts_lightspeed"
        source = isolation / "source"
        build = isolation / "build"
        bin_dir = isolation / "bin"
        source.mkdir(parents=True)
        build.mkdir()
        bin_dir.mkdir()

        bridge = bin_dir / "sts_lightspeed_bridge.exe"
        bridge.write_bytes(b"bridge")

        config = {
            "schema_version": 1,
            "mechanics": "legacy_v1",
            "backend": "sts_lightspeed",
            "revision": REVISION,
            "submodules": {
                "json": JSON_REVISION,
                "pybind11": PYBIND_REVISION,
            },
            "isolation_root": ".simulator/sts_lightspeed",
            "source_dir": ".simulator/sts_lightspeed/source",
            "build_dir": ".simulator/sts_lightspeed/build",
            "manifest_path": ".simulator/sts_lightspeed/build_manifest.json",
            "artifacts": {
                "bridge": ".simulator/sts_lightspeed/bin/sts_lightspeed_bridge.exe",
            },
            "transport": "native_process",
        }
        config_path = root / "simulator.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "mechanics": "legacy_v1",
            "backend": "sts_lightspeed",
            "transport": "native_process",
            "revision": REVISION,
            "submodules": config["submodules"],
            "artifacts": {
                "bridge": {
                    "path": config["artifacts"]["bridge"],
                    "sha256": hashlib.sha256(b"bridge").hexdigest(),
                },
            },
        }
        manifest["source_overlays"] = {}
        for name in ("battle_scum_searcher2_bridge", "cards_seeing_red"):
            relative = f"tools/sts_lightspeed/{name}.patch"
            patch_path = root / relative
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_bytes(name.encode())
            manifest["source_overlays"][f"{name}_patch"] = {
                "path": relative, "sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
        (isolation / "build_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return config_path

    def test_discovers_valid_isolated_native_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._create_installation(root)

            installation = discover_sts_lightspeed(
                project_root=root,
                config_path=config_path,
            )

            self.assertEqual(installation.revision, REVISION)
            self.assertTrue(installation.bridge_executable.is_file())

    def test_rejects_artifact_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._create_installation(root)
            bridge = (
                root
                / ".simulator"
                / "sts_lightspeed"
                / "bin"
                / "sts_lightspeed_bridge.exe"
            )
            bridge.write_bytes(b"changed")

            with self.assertRaisesRegex(
                SimulatorInstallationError,
                "SHA-256 mismatch",
            ):
                discover_sts_lightspeed(
                    project_root=root,
                    config_path=config_path,
                )

    def test_rejects_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._create_installation(root)
            manifest_path = (
                root
                / ".simulator"
                / "sts_lightspeed"
                / "build_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["revision"] = "d" * 40
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                SimulatorInstallationError,
                "revision mismatch",
            ):
                discover_sts_lightspeed(
                    project_root=root,
                    config_path=config_path,
                )

    def test_rejects_artifact_outside_isolation_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._create_installation(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["artifacts"]["bridge"] = "outside.exe"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(
                SimulatorInstallationError,
                "escapes simulator isolation root",
            ):
                discover_sts_lightspeed(
                    project_root=root,
                    config_path=config_path,
                )

    def test_default_config_is_platform_specific(self) -> None:
        with patch("sts1_llm_policy.env.simulator_installation.platform.system", return_value="Linux"):
            self.assertEqual(
                default_sts_lightspeed_config_name(),
                "sts_lightspeed_build_linux.json",
            )
        with patch("sts1_llm_policy.env.simulator_installation.platform.system", return_value="Windows"):
            self.assertEqual(
                default_sts_lightspeed_config_name(),
                "sts_lightspeed_build.json",
            )

    def test_rejects_declared_platform_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._create_installation(root)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["platform"] = "linux"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with patch("sts1_llm_policy.env.simulator_installation.platform.system", return_value="Windows"):
                with self.assertRaisesRegex(
                    SimulatorInstallationError,
                    "platform mismatch",
                ):
                    discover_sts_lightspeed(
                        project_root=root,
                        config_path=config_path,
                    )
