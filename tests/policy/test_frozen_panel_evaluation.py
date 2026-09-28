from __future__ import annotations

import unittest
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sts1_llm_policy.eval.frozen_policy_panel import _aggregate, _binding, _load_completed
from sts1_llm_policy.eval import frozen_policy_panel as panel_runner
from sts1_llm_policy.model_runtime import load_base_model_runtime_config
from sts1_llm_policy.train.lora import LoraSpec, inject_lora, save_lora_checkpoint

from tests.training.runtime_fixture import runtime_document, EXECUTION
from tests.support import write_json
from sts1_llm_policy.artifacts import sha256_file
from tests.training.model_fixture import TinyLM


def _summary(outcome: str, hp_loss: int) -> dict:
    return {
        "outcome": outcome,
        "total_hp_loss": hp_loss,
        "enemy_damage_taken": hp_loss,
        "self_hp_loss": 0,
        "decisions": 2,
        "accounting_consistent": True,
        "protocol": {
            "retry_count": 1,
            "fallback_count": 0,
            "first_pass_legal_count": 1,
            "inference_ms_total": 4.0,
        },
    }


class FrozenPanelEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = self.root / "runtime.json"
        path.write_text(json.dumps(runtime_document(legacy=True)), encoding="utf-8")
        self.runtime = load_base_model_runtime_config(path, project_root=self.root)
        self.config = {"observation_version": "observation_v5", "max_decisions": 1000,
                       "simulator_mechanics": "corrected_v1"}

    def binding(self, *, runtime=None, config=None):
        return _binding(config or self.config, runtime or self.runtime,
                        panel_hash="panel", simulator_revision="native", simulator_bridge_sha256="bridge")

    def write_episode(self, path, spec, binding, arm, snapshot_hash):
        trajectory = path.parent.parent / "trajectories" / f"episode-{spec['episode_index']:03d}.jsonl"
        record = {
            "schema_version": "trajectory_v1", "record_type": "transition", "done": True,
            "episode_id": "fixture", "step_index": 0, "game_seed": spec["combat_seed"],
            "policy_seed": spec["policy_seed"], "scenario_id": spec["scenario_id"], "policy_name": arm,
            "terminal_outcome": "victory", "turn": 1, "policy_result": {},
            "next_raw_state": {"source": "simulator", "state": {"terminal": True, "outcome": "PLAYER_VICTORY",
                "player": {"current_hp": 76, "max_hp": 80},
                "combat_accounting": {"starting_hp": 80, "total_hp_loss": 4, "enemy_damage_taken": 4, "self_hp_loss": 0}}},
        }
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        trajectory.write_text(json.dumps(record) + "\n", encoding="utf-8")
        write_json(path, {"schema_version": "frozen_policy_panel_episode_v1", "binding": binding,
                          "spec": spec, "arm": arm, "test_data_read": False,
                          "deck_snapshot_sha256": snapshot_hash, "trajectory_sha256": sha256_file(trajectory),
                          "summary": panel_runner._episode_summary(spec, [record], outcome="victory")})

    def test_resume_ignores_locations_and_metadata_but_rejects_inference_changes(self):
        original = self.binding()
        relocated = replace(self.runtime, snapshot_path=self.root / "elsewhere",
                            runtime_id="renamed", device="cuda:1",
                            dependencies=tuple(reversed(self.runtime.dependencies)))
        self.assertEqual(original, self.binding(runtime=relocated, config={
            **self.config, "run_id": "renamed", "output_dir": "elsewhere",
        }))
        path = self.root / "episodes/episode-001.json"
        spec = {"episode_index": 1, "scenario_id": "cultist", "ascension": 0, "combat_seed": 10, "policy_seed": 20}
        self.write_episode(path, spec, original, "base", "snapshot")
        self.assertIsNotNone(_load_completed(path, binding=self.binding(runtime=relocated), spec=spec,
                                            arm="base", deck_snapshot_sha256="snapshot"))
        changed = [
            self.binding(config={**self.config, "max_decisions": 2000}),
            self.binding(runtime=replace(self.runtime, dtype="float32")),
            self.binding(runtime=replace(self.runtime, repetition_penalty=1.1)),
            self.binding(runtime=replace(self.runtime, weight_assets_sha256=(("model.safetensors", "changed"),))),
            self.binding(runtime=replace(self.runtime, dependencies=(("torch", "changed"),))),
            {**original, "panel_sha256": "different panel"},
            {**original, "simulator_revision": "different simulator"},
            {"config_sha256": "legacy byte binding"},
        ]
        for binding in changed:
            with self.subTest(binding=binding), self.assertRaisesRegex(ValueError, "binding changed"):
                _load_completed(path, binding=binding, spec=spec, arm="base", deck_snapshot_sha256="snapshot")

    def test_aggregate_keeps_protocol_and_outcome_accounting(self) -> None:
        result = _aggregate([_summary("victory", 4), _summary("defeat", 8)])
        self.assertEqual(result["combat_count"], 2)
        self.assertEqual(result["win_rate"], 0.5)
        self.assertEqual(result["total_hp_loss"]["mean"], 6)
        self.assertEqual(result["protocol"]["first_pass_legal_rate"], 0.5)

    def test_alternate_panel_and_candidate_drive_preflight_dispatch_and_resume(self):
        snapshot = {"fixture_deck": ["Strike", "Defend"]}
        snapshot_hash = panel_runner._canonical_sha(snapshot)
        specs = [{"episode_index": index, "route_index": 9, "act": 2, "ascension": 7,
                  "scenario_id": "fixture-boss", "combat_seed": index + 10, "policy_seed": 5}
                 for index in (8, 3)]
        fingerprint = panel_runner._canonical_sha({"specs": specs, "decks": {"9": snapshot_hash}})
        panel = {"schema_version": "frozen_combat_panel_inputs_v1", "observation_version": "observation_v5",
                 "specs": specs, "decks": {"9": {"snapshot": snapshot, "snapshot_sha256": snapshot_hash}},
                 "simulator_revision": "fixture-native", "panel_sha256": fingerprint,
                 "test_data_read": False, "sealed_test_run": False}
        config = {"schema_version": "frozen_policy_panel_evaluation_v1", "purpose": "alternate-experiment",
                  "simulator_mechanics": "corrected_v1",
                  "run_id": "alternate", "arm": "candidate-alpha", "model_runtime": "runtime.json",
                  "execution_profile": "execution.json",
                  "panel_manifest": "panel.json", "expected_routes": 1, "expected_combats": 2,
                  "expected_panel_sha256": fingerprint, "observation_version": "observation_v5",
                  "max_decisions": 50, "output_dir": "results"}
        for name, value in (("runtime", runtime_document()), ("execution", EXECUTION), ("panel", panel)):
            write_json(self.root / (name + ".json"), value)
        path = self.root / "eval.json"
        write_json(path, config)
        model = TinyLM()
        spec = LoraSpec(("q_proj",), rank=2, alpha=4.0, dropout=0.0)
        replaced = inject_lora(model, spec)
        metadata = save_lora_checkpoint(model, self.root / "adapter", spec=spec, replaced_modules=replaced,
                                       base_model_id=self.runtime.model_id, base_revision=self.runtime.revision)
        bridge = self.root / "bridge"
        bridge.write_bytes(b"native fixture")
        installation = SimpleNamespace(revision="fixture-native", bridge_executable=bridge)
        execution = MagicMock()
        execution.validate.return_value = installation
        with patch.object(panel_runner, "resolve_simulator", return_value=execution), \
             patch.object(panel_runner.TransformersGenerationBackend, "from_config") as backend:
            base = panel_runner.run(path, project_root=self.root, preflight_only=True)
            self.assertEqual(base["scope"], {"routes": 1, "combat_identities": 2})
            self.assertNotIn("adapter_sha256", base["binding"])
            # Relocation/formatting preserves identity; changed panel contents do not.
            for key in ("model_runtime", "execution_profile", "panel_manifest"):
                document = json.loads((self.root / config[key]).read_text())
                config[key] = "moved-" + config[key]
                (self.root / config[key]).write_text(json.dumps(document, indent=4), encoding="utf-8")
            write_json(path, config)
            self.assertEqual(base["binding"], panel_runner._validate(path, project_root=self.root)["binding"])
            specs[0]["combat_seed"] += 1
            write_json(self.root / config["panel_manifest"], panel)
            with self.assertRaisesRegex(ValueError, "specs or deck snapshots changed"):
                panel_runner._validate(path, project_root=self.root)
            specs[0]["combat_seed"] -= 1
            write_json(self.root / config["panel_manifest"], panel)
            write_json(path, {**config, "checkpont": "adapter"})
            with self.assertRaisesRegex(ValueError, "Unsupported evaluation fields"):
                panel_runner.run(path, project_root=self.root, preflight_only=True)
            config["checkpoint"] = "adapter"
            write_json(path, config)
            adapter = panel_runner.run(path, project_root=self.root, preflight_only=True)
            self.assertNotEqual(base["binding"], adapter["binding"])
            backend.assert_not_called()
            backend.side_effect = RuntimeError("dispatch reached")
            with self.assertRaisesRegex(RuntimeError, "dispatch reached"):
                panel_runner.run(path, project_root=self.root)
            self.assertEqual(backend.call_args.kwargs["lora_checkpoint_dir"], self.root / "adapter")
            self.assertEqual(backend.call_args.kwargs["verified_lora_metadata"], metadata)
            backend.reset_mock()
            for item in specs:
                self.write_episode(
                    self.root / f"results/formal/episodes/episode-{item['episode_index']:03d}.json",
                    item, adapter["binding"], config["arm"], snapshot_hash,
                )
            result = panel_runner.run(path, project_root=self.root)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["timing"]["new_combats"], 0)
            backend.assert_not_called()
            exported = [json.loads(line) for line in (self.root / "results/formal/episodes.jsonl").read_text().splitlines()]
            self.assertEqual([item["spec"] for item in exported], specs)
            self.assertEqual(result["candidate"]["combat"], _aggregate([item["summary"] for item in exported]))
            # Same arm/output with a different candidate must not reuse completed combats.
            config.pop("checkpoint")
            write_json(path, config)
            with self.assertRaisesRegex(ValueError, "binding changed"):
                panel_runner.run(path, project_root=self.root)

if __name__ == "__main__":
    unittest.main()
