"""Native test prerequisites: absent builds skip; invalid builds remain errors."""
from dataclasses import replace
import json
import unittest

from sts1_llm_policy.env.simulator_execution import LEGACY_MECHANICS, SELECTION, resolve_simulator


def require_native_selection():
    execution = resolve_simulator(required_capabilities={SELECTION})
    config = json.loads(execution.config_path.read_text(encoding="utf-8"))
    base_paths = [config["manifest_path"], *config["artifacts"].values()]
    extension_paths = [p for p in (execution.profile.extension_manifest,
                                    execution.profile.extension_executable) if p]

    def present(paths):
        return any((execution.project_root / p).exists() or
                   (execution.project_root / p).is_symlink() for p in paths)

    if not present(base_paths + extension_paths):
        raise unittest.SkipTest("Native simulator is not installed; build it to run native tests")
    if extension_paths and not present(extension_paths):
        # A valid Windows base without its optional selection extension can skip.
        # A partial/corrupt base must still fail its ordinary installation checks.
        base = replace(execution, required_capabilities=frozenset({"simulator"}), mechanics=LEGACY_MECHANICS).validate()
        if SELECTION not in base.features:
            raise unittest.SkipTest("Native card-selection extension is not installed; build it to run native tests")
    execution.validate()
    return execution


class NativeSelectionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.execution = require_native_selection()
