import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sts1_llm_policy.eval.real_game_session import load_real_game_session_config


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RealGameSessionConfigTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        runtime = root / "configs/runtime.json"
        training = root / "configs/training.json"
        report = root / "report/training/training.json"
        checkpoint = root / "outputs/checkpoint"
        weights = checkpoint / "adapter_model.safetensors"
        runtime_hash = _write_json(runtime, {"schema_version": "placeholder"})
        training_hash = _write_json(
            training,
            {
                "run_id": "gold-v5",
                "test_data_read": False,
                "sealed_test_run": False,
                "stable_training_claim": False,
            },
        )
        report_hash = _write_json(
            report,
            {"run_id": "gold-v5", "mode": "run", "training_started": True},
        )
        checkpoint.mkdir(parents=True)
        weights.write_bytes(b"gold-sft")
        weights_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        _write_json(
            checkpoint / "adapter_config.json",
            {"weights_file": weights.name, "weights_sha256": weights_hash},
        )
        config = root / "configs/session.json"
        _write_json(
            config,
            {
                "schema_version": "real_game_session_v2",
                "session_profile_id": "gold-live-v1",
                "policy": {
                    "policy_id": "gold-v5",
                    "policy_seed": 7,
                    "observation_version": "observation_v5",
                    "model_runtime": "configs/runtime.json",
                    "expected_model_runtime_sha256": runtime_hash,
                    "training_run_id": "gold-v5",
                    "training_config": "configs/training.json",
                    "expected_training_config_sha256": training_hash,
                    "training_report": "report/training/training.json",
                    "expected_training_report_sha256": report_hash,
                    "checkpoint_dir": "outputs/checkpoint",
                    "expected_checkpoint_weights_sha256": weights_hash,
                },
                "safety": {
                    "max_combats": 3,
                    "max_steps_per_combat": 50,
                    "state_poll_interval_seconds": 0.25,
                },
                "mechanics_fallback": {
                    "mode": "communication_mod_descriptions_with_audited_powers_v2",
                    "native_id_crosswalk": "communication_mod_to_sts_lightspeed_v7",
                    "unknown_cards": "use_nonempty_description",
                    "unknown_relics": "use_nonempty_description",
                    "unknown_powers": "fail_closed",
                    "unknown_monster_moves": "fail_closed",
                    "unknown_monster_behaviors": "fail_closed",
                },
                "output_root": "outputs/sessions",
                "log_path": "outputs/logs/live.log",
                "evidence": {
                    "classification": "coverage_collection",
                    "test_data_read": False,
                    "sealed_test_run": False,
                    "stable_policy_claim": False,
                    "automatic_training_ingest": False,
                },
            },
        )
        return config

    def test_loads_fully_bound_development_session(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            loaded = load_real_game_session_config(
                self._fixture(root), project_root=root
            )

            self.assertEqual(loaded.policy_id, "gold-v5")
            self.assertEqual(loaded.observation_version, "observation_v5")
            self.assertEqual(loaded.max_combats, 3)
            self.assertEqual(
                loaded.source_description_fallback,
                "communication_mod_descriptions_with_audited_powers_v2",
            )
            self.assertEqual(
                loaded.native_id_crosswalk,
                "communication_mod_to_sts_lightspeed_v7",
            )

    def test_rejects_broader_or_missing_mechanics_fallback(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config_path = self._fixture(root)
            value = json.loads(config_path.read_text(encoding="utf-8"))
            value["mechanics_fallback"]["unknown_powers"] = (
                "use_nonempty_description"
            )
            config_path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "live-only boundary"):
                load_real_game_session_config(config_path, project_root=root)

    def test_rejects_training_ingest_or_stable_claim(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config_path = self._fixture(root)
            value = json.loads(config_path.read_text(encoding="utf-8"))
            value["evidence"]["automatic_training_ingest"] = True
            config_path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exploratory safety boundary"):
                load_real_game_session_config(config_path, project_root=root)

    def test_rejects_checkpoint_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config_path = self._fixture(root)
            (root / "outputs/checkpoint/adapter_model.safetensors").write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "checkpoint weights"):
                load_real_game_session_config(config_path, project_root=root)


if __name__ == "__main__":
    unittest.main()
