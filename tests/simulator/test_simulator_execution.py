"""Capability resolution, native dispatch, and retained-artifact safety."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sts1_llm_policy.env.simulator_execution import PROFILES, SELECTION, resolve_simulator, discover_sts_lightspeed
from sts1_llm_policy.env.simulator_installation import SimulatorInstallationError, default_sts_lightspeed_config_name

from tests.simulator import native_support


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SimulatorExecutionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def configure(self, system, *, installed=True, extension=True):
        repo = Path(__file__).resolve().parents[2]
        profile = PROFILES[system.lower()]
        config_path = self.root / "configs/env" / profile.config_name
        config = json.loads((repo / "configs/env" / profile.config_name).read_text(encoding="utf-8"))
        write_json(config_path, config)
        source = self.root / "tools/sts_lightspeed/decision_bridge.cpp"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"bridge source\r\n")
        actions_patch = source.with_name("actions_upgrade_hand.patch")
        actions_patch.write_bytes(b"hand upgrade patch\n")
        rage_patch = source.with_name("cards_rage_cost.patch")
        rage_patch.write_bytes(b"rage cost patch\n")
        mechanics_patch = source.with_name("card_mechanics.patch")
        mechanics_patch.write_bytes(b"card mechanics patch\n")
        memory_patch = source.with_name("public_draw_memory.patch")
        memory_patch.write_bytes(b"public draw memory patch\n")
        memory_header = source.with_name("public_draw_memory.h")
        memory_header.write_bytes(b"public draw memory header\n")
        search_patch = source.with_name("battle_scum_searcher2_bridge.patch")
        search_patch.write_bytes(b"search patch\n")
        cards_patch = source.with_name("cards_seeing_red.patch")
        cards_patch.write_bytes(b"card exhaust patch\n")
        if installed:
            records = {}
            for name, relative in config["artifacts"].items():
                artifact = self.root / relative
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(name.encode())
                records[name] = {"path": relative, "sha256": digest(artifact)}
            manifest = {key: config[key] for key in (
                "schema_version", "backend", "transport", "revision", "submodules", "mechanics",
            )}
            if system == "Linux":
                manifest.update(platform="linux", features=[SELECTION])
            manifest.update(artifacts=records, source_overlays={"decision_bridge": {
                "path": "tools/sts_lightspeed/decision_bridge.cpp", "sha256": digest(source),
            }})
            manifest_path = self.root / config["manifest_path"]
            manifest["source_overlays"]["actions_upgrade_hand"] = {
                "path": "tools/sts_lightspeed/actions_upgrade_hand.patch", "sha256": digest(actions_patch),
            }
            manifest["source_overlays"]["cards_rage_cost"] = {
                "path": "tools/sts_lightspeed/cards_rage_cost.patch", "sha256": digest(rage_patch),
            }
            manifest["source_overlays"]["card_mechanics"] = {
                "path": "tools/sts_lightspeed/card_mechanics.patch", "sha256": digest(mechanics_patch),
            }
            manifest["source_overlays"].update({
                "battle_scum_searcher2_bridge_patch": {"path": "tools/sts_lightspeed/battle_scum_searcher2_bridge.patch", "sha256": digest(search_patch)},
                "cards_seeing_red_patch": {"path": "tools/sts_lightspeed/cards_seeing_red.patch", "sha256": digest(cards_patch)},
                "public_draw_memory_patch": {"path": "tools/sts_lightspeed/public_draw_memory.patch", "sha256": digest(memory_patch)},
                "public_draw_memory_header": {"path": "tools/sts_lightspeed/public_draw_memory.h", "sha256": digest(memory_header)},
            })
            write_json(manifest_path, manifest)
            if system == "Windows" and extension:
                executable = self.root / profile.extension_executable
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.write_bytes(b"selection")
                write_json(self.root / profile.extension_manifest, {
                    "schema_version": "card_selection_bridge_build_v1",
                    "mechanics": "corrected_v1",
                    "revision": config["revision"],
                    "base_manifest_sha256": digest(manifest_path),
                    "bridge_source_sha256": digest(source),
                    "cards_seeing_red_sha256": digest(cards_patch),
                    "actions_upgrade_hand_sha256": digest(actions_patch),
                    "cards_rage_cost_sha256": digest(rage_patch),
                    "card_mechanics_sha256": digest(mechanics_patch),
                    "public_draw_memory_patch_sha256": digest(memory_patch),
                    "public_draw_memory_header_sha256": digest(memory_header),
                    "bridge_sha256": digest(executable),
                })
        return config

    def resolve(self, system, *, selection=True, target="auto"):
        with patch("platform.system", return_value=system), patch("platform.machine", return_value="AMD64"):
            return resolve_simulator(project_root=self.root, target=target,
                                     required_capabilities={SELECTION} if selection else ())

    def test_native_classes_skip_absent_builds_without_hiding_python_tests(self):
        class NativeProbe(native_support.NativeSelectionTestCase):
            def test_requires_native(self):
                self.fail("Native test must not run without an installation")

        class OtherNativeProbe(NativeProbe):
            pass

        class PythonProbe(unittest.TestCase):
            def test_python_only(self):
                pass

        for system in ("Windows", "Linux"):
            with self.subTest(system=system):
                self.configure(system, installed=False)
                execution = self.resolve(system)
                suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                                           for cls in (NativeProbe, OtherNativeProbe, PythonProbe))
                with patch.object(native_support, "resolve_simulator", return_value=execution):
                    result = unittest.TestResult()
                    suite.run(result)
                self.assertEqual(result.errors, [])
                self.assertEqual(result.failures, [])
                self.assertEqual(len(result.skipped), 2)  # One skip per native class.
                self.assertEqual(result.testsRun, 1)  # The pure Python test still runs.
                self.assertTrue(all("not installed" in reason for _, reason in result.skipped))

    def test_native_prerequisites_reject_partial_and_corrupt_builds(self):
        for system in ("Windows", "Linux"):
            for damage in (None, "binary", "missing_binary", "missing_manifest", "invalid_manifest", "identity"):
                with self.subTest(system=system, damage=damage):
                    config = self.configure(system)
                    execution = self.resolve(system)
                    manifest = self.root / config["manifest_path"]
                    binary = self.root / config["artifacts"]["bridge"]
                    if damage == "binary": binary.write_bytes(b"corrupt")
                    if damage == "missing_binary": binary.unlink()
                    if damage == "missing_manifest": manifest.unlink()
                    if damage == "invalid_manifest": manifest.write_text("invalid", encoding="utf-8")
                    if damage == "identity":
                        value = json.loads(manifest.read_text(encoding="utf-8"))
                        value["revision"] = "wrong-revision"
                        write_json(manifest, value)
                    with patch("platform.system", return_value=system), \
                            patch.object(native_support, "resolve_simulator", return_value=execution):
                        if damage is None:
                            self.assertIs(native_support.require_native_selection(), execution)
                        else:
                            with self.assertRaises(SimulatorInstallationError):
                                native_support.require_native_selection()

    def test_absent_windows_extension_skips_only_with_valid_base(self):
        config = self.configure("Windows", extension=False)
        with patch("platform.system", return_value="Windows"), \
                patch.object(native_support, "resolve_simulator", side_effect=lambda **kw: self.resolve("Windows")):
            with self.assertRaisesRegex(unittest.SkipTest, "extension is not installed"):
                native_support.require_native_selection()
            executable = self.root / PROFILES["windows"].extension_executable
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(b"unbound extension")
            with self.assertRaises(SimulatorInstallationError):
                native_support.require_native_selection()
            executable.unlink()
            (self.root / config["artifacts"]["bridge"]).write_bytes(b"corrupt base")
            with self.assertRaises(SimulatorInstallationError):
                native_support.require_native_selection()

    def test_native_handshake_failure_is_an_error_not_a_skip(self):
        from sts1_llm_policy.env.simulator_client import StsLightspeedClient
        from tests.simulator.client_fixture import _FakeProcess

        class HandshakeTest(native_support.NativeSelectionTestCase):
            def test_handshake(self):
                client = StsLightspeedClient(self.execution.validate(),
                                            popen_factory=lambda *a, **kw: _FakeProcess(mismatch_hello_id=True))
                self.addCleanup(client.close)
                client.start()

        self.configure("Linux")
        execution = self.resolve("Linux")
        with patch("platform.system", return_value="Linux"), \
                patch.object(native_support, "resolve_simulator", return_value=execution):
            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromTestCase(HandshakeTest).run(result)
        self.assertEqual(result.skipped, [])
        self.assertEqual(len(result.errors), 1)
        self.assertIn("SimulatorProtocolError", result.errors[0][1])

    def test_same_api_selects_native_capability_and_records_bindings(self):
        for system in ("Windows", "Linux"):
            with self.subTest(system=system):
                self.configure(system)
                execution = self.resolve(system)
                with patch("platform.system", return_value=system):
                    installation = execution.validate()
                    report = execution.describe(verify=True)
                expected = ("outputs/card-selection/sts_lightspeed_bridge.exe" if system == "Windows"
                            else ".simulator/sts_lightspeed-linux/bin/sts_lightspeed_bridge")
                self.assertEqual(installation.bridge_executable, self.root / expected)
                self.assertIn(SELECTION, installation.features)
                self.assertEqual(report["bridge_sha256"], digest(self.root / expected))
                self.assertEqual(report["status"], "ready")
                self.assertEqual(len(report["manifests"]), 2 if system == "Windows" else 1)

    def test_default_selects_corrected_windows_bridge(self):
        self.configure("Windows")
        execution = self.resolve("Windows", selection=False)
        with patch("platform.system", return_value="Windows"):
            self.assertEqual(execution.validate().bridge_executable, self.root / PROFILES["windows"].extension_executable)
        self.assertEqual(len(execution.build_commands()), 2)

    def test_changed_mechanics_patch_requires_rebuild_on_both_platforms(self):
        for system in ("Windows", "Linux"):
            for name in ("actions_upgrade_hand.patch", "cards_rage_cost.patch", "card_mechanics.patch",
                         "public_draw_memory.patch", "public_draw_memory.h"):
                with self.subTest(system=system, patch=name):
                    self.configure(system)
                    (self.root / "tools/sts_lightspeed" / name).write_bytes(b"changed")
                    with patch("platform.system", return_value=system):
                        with self.assertRaises(SimulatorInstallationError):
                            self.resolve(system).validate()

    def test_startup_reuses_validation_and_build_rechecks_changed_artifacts(self):
        self.configure("Windows")
        execution = self.resolve("Windows")
        with patch("platform.system", return_value="Windows"), patch(
            "sts1_llm_policy.env.simulator_execution.discover_sts_lightspeed",
            wraps=discover_sts_lightspeed,
        ) as discover:
            installation = execution.validate()
            execution.create_client()
            execution.create_environment()
            self.assertIs(execution.validate(), installation)
            discover.assert_called_once()
            (self.root / execution.profile.extension_executable).write_bytes(b"corrupt")
            with self.assertRaisesRegex(SimulatorInstallationError, "bridge_sha256"):
                execution.build()
            with self.assertRaisesRegex(SimulatorInstallationError, "bridge_sha256"):
                self.resolve("Windows").validate()

    def test_explicit_target_cannot_select_foreign_host(self):
        with self.assertRaisesRegex(SimulatorInstallationError, "does not match host"):
            self.resolve("Windows", target="linux")

    def test_unsupported_host_and_architecture_fail_before_config_read(self):
        with patch("platform.system", return_value="Darwin"):
            with self.assertRaisesRegex(SimulatorInstallationError, "Unsupported simulator platform"):
                resolve_simulator(project_root=self.root)
            with self.assertRaises(SimulatorInstallationError):
                default_sts_lightspeed_config_name()
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="aarch64"):
            with self.assertRaisesRegex(SimulatorInstallationError, "architecture"):
                resolve_simulator(project_root=self.root)

    def test_unknown_capability_fails_without_fallback(self):
        with self.assertRaisesRegex(SimulatorInstallationError, "capabilities"):
            resolve_simulator(project_root=self.root, required_capabilities={"live_game"})

    def test_linux_profile_must_declare_selection(self):
        config = self.configure("Linux")
        config["features"] = []
        write_json(self.root / "configs/env/sts_lightspeed_build_linux.json", config)
        with self.assertRaisesRegex(SimulatorInstallationError, "does not provide"):
            self.resolve("Linux")

    def test_changed_config_requires_new_resolution(self):
        self.configure("Windows")
        execution = self.resolve("Windows")
        execution.config_path.write_bytes(b"{}")
        with self.assertRaisesRegex(SimulatorInstallationError, "changed after resolution"):
            execution.validate()

    def test_raw_source_hash_still_rejects_newline_changes_on_both_platforms(self):
        for system in ("Windows", "Linux"):
            self.configure(system)
            execution = self.resolve(system)
            (self.root / "tools/sts_lightspeed/decision_bridge.cpp").write_bytes(b"bridge source\n")
            with patch("platform.system", return_value=system), patch("subprocess.run") as run:
                with self.assertRaisesRegex(SimulatorInstallationError, "binding changed"):
                    execution.build()
                run.assert_not_called()

    def test_existing_valid_build_is_a_noop(self):
        for system in ("Windows", "Linux"):
            self.configure(system)
            execution = self.resolve(system)
            before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
            with patch("platform.system", return_value=system), patch("subprocess.run") as run:
                execution.build()
                run.assert_not_called()
            self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_missing_windows_extension_runs_only_extension_builder(self):
        self.configure("Windows", extension=False)
        execution = self.resolve("Windows")
        def build_extension(*args, **kwargs):
            self.configure("Windows")
        with patch("platform.system", return_value="Windows"), patch("subprocess.run", side_effect=build_extension) as run:
            execution.build()
        self.assertEqual(run.call_count, 1)
        self.assertTrue(run.call_args.args[0][-1].endswith("build_card_selection_bridge.ps1"))
        self.assertEqual(run.call_args.kwargs, {"cwd": self.root, "check": True})

    def test_fresh_linux_build_dispatches_python_and_validates_result(self):
        self.configure("Linux", installed=False)
        execution = self.resolve("Linux")
        with patch("platform.system", return_value="Linux"), patch("subprocess.run", side_effect=lambda *a, **k: self.configure("Linux")) as run:
            execution.build()
        self.assertEqual(run.call_count, 1)
        self.assertTrue(run.call_args.args[0][1].endswith("build_sts_lightspeed_linux.py"))

    def test_failed_builder_does_not_return_ready(self):
        self.configure("Linux", installed=False)
        execution = self.resolve("Linux")
        with patch("platform.system", return_value="Linux"), patch("subprocess.run") as run:
            with self.assertRaises(SimulatorInstallationError):
                execution.build()
        run.assert_called_once()

    def test_unbound_artifacts_are_preserved(self):
        config = self.configure("Windows", installed=False)
        artifact = self.root / config["artifacts"]["bridge"]
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"retained")
        execution = self.resolve("Windows")
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(SimulatorInstallationError, "Unbound base artifacts"):
                execution.build()
            run.assert_not_called()
        self.assertEqual(artifact.read_bytes(), b"retained")

    def test_windows_build_plan_covers_base_then_extension(self):
        self.configure("Windows", installed=False)
        execution = self.resolve("Windows")
        commands = execution.build_commands()
        self.assertEqual(len(commands), 2)
        self.assertIn("-File", commands[1])

    def test_fresh_windows_build_runs_base_before_extension(self):
        self.configure("Windows", installed=False)
        execution = self.resolve("Windows")
        calls = []
        def builder(command, **kwargs):
            calls.append(command)
            self.configure("Windows", extension=len(calls) == 2)
        with patch("platform.system", return_value="Windows"), patch("subprocess.run", side_effect=builder):
            execution.build()
        self.assertEqual(len(calls), 2)
        self.assertTrue(any(arg.endswith("build_sts_lightspeed.ps1") for arg in calls[0]))
        self.assertTrue(calls[1][-1].endswith("build_card_selection_bridge.ps1"))

    def test_retained_extension_with_missing_base_does_not_trigger_build(self):
        config = self.configure("Windows")
        (self.root / config["manifest_path"]).unlink()
        execution = self.resolve("Windows")
        with patch("platform.system", return_value="Windows"), patch("subprocess.run") as run:
            with self.assertRaises(SimulatorInstallationError):
                execution.build()
            run.assert_not_called()

    def test_corrupt_extension_is_rejected_without_base_fallback(self):
        self.configure("Windows")
        execution = self.resolve("Windows")
        (self.root / execution.profile.extension_executable).write_bytes(b"corrupt")
        with patch("platform.system", return_value="Windows"), patch("subprocess.run") as run:
            with self.assertRaisesRegex(SimulatorInstallationError, "bridge_sha256"):
                execution.build()
            run.assert_not_called()

    def test_environment_enables_selection_only_when_requested(self):
        self.configure("Windows")
        for selection in (False, True):
            execution = self.resolve("Windows", selection=selection)
            with patch("platform.system", return_value="Windows"):
                environment = execution.create_environment()
            self.assertEqual(environment._allow_card_selection, selection)
