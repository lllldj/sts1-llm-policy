"""Exercise continuous validation and cache consumption without large model assets."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sts1_llm_policy.workflows import continuous_run as continuous
from sts1_llm_policy.workflows import continuous_plan as plan
from sts1_llm_policy.workflows import continuous_artifacts as artifacts
from sts1_llm_policy.artifacts import read_json_object, sha256_file
from sts1_llm_policy.workflows.continuous_config import load_continuous_config

ROOT = Path(__file__).resolve().parents[2]


class ContinuousResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = load_continuous_config(ROOT, "configs/runs/evaluation/continuous_qwen2_5_7b_base_gold_sft.json")
        self.config.update(route_count=1, combat_seed_groups_per_route=1, output_dir="outputs/run")
        self.config["arms"] = [self.config["arms"][1]]
        self.config["arms"][0].update(id="arm", checkpoint="checkpoint")
        for path in (self.config["scope"], "configs/env/d_expansion_v1_scope.json",
                     self.config["arms"][0]["model_runtime"], self.config["arms"][0]["execution_profile"]):
            self.write(path, read_json_object(ROOT / path))
        self.runtime_path = self.config["arms"][0]["model_runtime"]
        self.runtime = read_json_object(self.root / self.runtime_path)
        self.write("checkpoint/adapter_config.json", {
            "schema_version": 1, "adapter_type": "project_lora_v1",
            "base_model_id": self.runtime["model"]["id"], "base_revision": self.runtime["model"]["revision"],
            "spec": {"target_module_suffixes": ["q_proj"], "rank": 1, "alpha": 1.0, "dropout": 0.0},
            "replaced_modules": ["q_proj"], "weights_file": "adapter.safetensors",
        })
        self.change_adapter(b"content inspected without loading tensors")
        picker = self.root / self.config["picker_database"]
        picker.parent.mkdir(parents=True)
        picker.write_bytes(b"picker fixture")
        self.bridge = self.root / "native"
        self.bridge.write_bytes(b"native fixture")
        installation = SimpleNamespace(revision="simulator-revision", bridge_executable=self.bridge)
        execution = SimpleNamespace(validate=lambda: installation, describe=lambda: {})
        self.addCleanup(patch.stopall)
        patch.object(plan, "resolve_simulator", return_value=execution).start()
        self.validated = self.validate()
        self.output = self.validated["output"] / "formal"
        self.trajectory = self.output / "arm/route-work/combat.jsonl"
        self.trajectory.parent.mkdir(parents=True)
        step = next(step for step in self.validated["routes"][0]["steps"] if step["kind"] == "combat")
        record = {
            "schema_version": "trajectory_v1", "record_type": "transition", "done": True,
            "episode_id": "fixture", "step_index": 0, "game_seed": step["combat_seed"],
            "policy_seed": step["policy_seed"], "scenario_id": step["scenario_id"], "policy_name": "arm",
            "terminal_outcome": "defeat", "turn": 1, "policy_result": {},
            "next_raw_state": {"source": "simulator", "state": {"terminal": True, "outcome": "PLAYER_LOSS",
                "player": {"current_hp": 0, "max_hp": 80}, "relics": [],
                "combat_accounting": {"starting_hp": 80, "total_hp_loss": 80}}},
        }
        self.trajectory.write_text(json.dumps(record) + "\n", encoding="utf-8")
        snapshot = {"deck": []}
        self.item = {
            "schema_version": "continuous_combat_route_v1", "configuration_identity": self.validated["identity"],
            "arm": "arm", "route": self.validated["routes"][0], "status": "defeated",
            "death_combat_index": step["combat_index"], "combats_started": 1, "combats_won": 0, "test_data_read": False,
            "combats": [{**{key: step[key] for key in ("combat_index", "floor", "encounter_family", "scenario_id", "combat_seed", "policy_seed")},
                         "input_snapshot": snapshot, "input_snapshot_sha256": artifacts.canonical_sha(snapshot),
                         "trajectory": "arm/route-work/combat.jsonl", "trajectory_sha256": sha256_file(self.trajectory),
                         "summary": artifacts.summarize_combat([record], "defeat")}],
        }
        self.relative_route = "arm/routes/route-000-combat-seed-group-00.json"
        self.write(self.output / self.relative_route, self.item)

    def write(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def change_adapter(self, value):
        path = self.root / "checkpoint/adapter.safetensors"
        path.write_bytes(value)
        metadata = read_json_object(path.parent / "adapter_config.json")
        metadata["weights_sha256"] = sha256_file(path)
        self.write(path.parent / "adapter_config.json", metadata)

    def validate(self):
        self.write("run.json", self.config)
        return plan.prepare_run(self.root / "run.json", project_root=self.root, target="auto")

    def consume(self, validated, output=None):
        output = output or self.output
        return artifacts.load_completed_route(output / self.relative_route, identity=validated["identity"],
                                          arm="arm", route=validated["routes"][0], output_root=output)

    def test_reformatted_relocated_inputs_and_complete_output_are_reused_by_runner(self):
        # Relocate scope and its base, runtime, execution, checkpoint, assets and output.
        scope = read_json_object(self.root / self.config["scope"])
        self.write("moved/base.json", read_json_object(self.root / scope["extends"]))
        scope["extends"] = "moved/base.json"
        self.write("moved/scope.json", scope)
        self.config["scope"] = "moved/scope.json"
        for name in ("model_runtime", "execution_profile"):
            original = self.root / self.config["arms"][0][name]
            value = read_json_object(original)
            if name == "model_runtime":
                value["model"]["snapshot_path"] = "moved/models/" + value["model"]["revision"]
            self.config["arms"][0][name] = f"moved/{name}.json"
            self.write(self.config["arms"][0][name], value)
        shutil.copytree(self.root / "checkpoint", self.root / "moved/checkpoint")
        self.config["arms"][0]["checkpoint"] = "moved/checkpoint"
        shutil.copyfile(self.root / self.config["picker_database"], self.root / "moved/picker.db")
        self.config["picker_database"] = "moved/picker.db"
        self.config.update(run_id="renamed", output_dir="outputs/moved")
        moved_output = self.root / "outputs/moved/formal"
        shutil.copytree(self.output, moved_output)
        for path in (self.root / "moved").rglob("*.json"):
            path.write_text(json.dumps(read_json_object(path), sort_keys=True), encoding="utf-8")
        self.validate()
        with (
            patch("sts1_llm_policy.data.card_pick_metrics.SqliteCardRateBackend"),
            patch("sts1_llm_policy.data.card_pick_metrics.MassNormalizedCardRewardPicker"),
            patch("sts1_llm_policy.eval.counterfactual_reward_picker_v2.StrategicRateTable"),
            patch.object(continuous.TransformersGenerationBackend, "from_config") as backend,
            patch.object(continuous, "_run_route") as execute,
        ):
            report = continuous.run(self.root / "run.json", project_root=self.root)
        backend.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(report["timing"]["arm"]["new_routes"], 0)
        self.assertEqual(report["metrics"]["arm"]["combat_count"], 1)
        self.assertEqual(read_json_object(moved_output / "inputs.json")["arms"]["arm"], [self.item])
        # A summary without its trajectory is insufficient even with matching identity.
        (moved_output / "arm/route-work/combat.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "trajectory is unavailable"):
            self.consume(self.validate(), moved_output)

    def test_behavior_changes_reject_cached_routes(self):
        original = deepcopy(self.config)
        for path, value in (
            (("seeds", "combat"), original["seeds"]["combat"] + 1),
            (("picker", "epsilon"), 0.3),
            (("max_decisions",), original["max_decisions"] + 1),
            (("observation_version",), "observation_v6"),
            (("upgrade_policy", "teacher_protocol", "search_budget"), 8),
            (("route_count",), 2),
        ):
            with self.subTest(path=path):
                self.config = deepcopy(original)
                node = self.config
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    self.consume(self.validate())

    def test_runtime_and_asset_changes_reject_cached_routes(self):
        for field, filename in (("model", "model-00001-of-00004.safetensors"), ("tokenizer", "tokenizer.json")):
            with self.subTest(field=field):
                runtime = deepcopy(self.runtime)
                key = "weight_assets_sha256" if field == "model" else "asset_sha256"
                runtime[field][key][filename] = "0" * 64
                self.write(self.runtime_path, runtime)
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    self.consume(self.validate())
        self.write(self.runtime_path, self.runtime)
        profile_path = self.config["arms"][0]["execution_profile"]
        profile = read_json_object(self.root / profile_path)
        changed_profile = deepcopy(profile)
        changed_profile["dependencies"]["transformers"] = "==0.0"
        self.write(profile_path, changed_profile)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.consume(self.validate())
        self.write(profile_path, profile)
        for path in (self.bridge, self.root / self.config["picker_database"]):
            with self.subTest(asset=path.name):
                original = path.read_bytes()
                path.write_bytes(b"changed asset")
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    self.consume(self.validate())
                path.write_bytes(original)
        self.change_adapter(b"different adapter weights")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.consume(self.validate())

    def test_frozen_runtime_still_rejects_unsupported_dtype_and_decoding(self):
        for section, key, value in (("model_loading", "dtype", "float32"), ("generation", "do_sample", True)):
            with self.subTest(field=key):
                runtime = deepcopy(self.runtime)
                runtime[section][key] = value
                self.write(self.runtime_path, runtime)
                with self.assertRaisesRegex(ValueError, "frozen"):
                    self.validate()

    def test_candidate_source_isolation_binds_resolved_sources_not_paths(self):
        self.config = read_json_object(ROOT / "configs/generation/continuous_teacher_pool.json")
        self.config.update(route_count=1, combat_seed_groups_per_route=1)
        excluded = self.config["data_source"]["excluded_sources"]
        for path in excluded:
            previous = read_json_object(ROOT / path)
            self.write(path, previous)
        original = self.validate()["identity"]
        moved = []
        for index, path in enumerate(excluded):
            destination = f"moved/source-{index}.json"
            self.write(destination, read_json_object(self.root / path))
            moved.append(destination)
        self.config["data_source"]["excluded_sources"] = list(reversed(moved))
        self.assertEqual(self.validate()["identity"], original)
        run_id = self.config["run_id"]
        self.config["run_id"] += "-renamed"
        self.assertNotEqual(self.validate()["identity"], original)
        self.config["run_id"] = run_id
        previous = read_json_object(self.root / moved[0])
        previous["seed_intervals"][0][0] += 1
        self.write(moved[0], previous)
        self.assertNotEqual(self.validate()["identity"], original)

    def test_execution_profile_validation_remains_required(self):
        path = self.config["arms"][0]["execution_profile"]
        profile = read_json_object(self.root / path)
        profile["schema_version"] = "unsupported"
        self.write(path, profile)
        with self.assertRaises(ValueError):
            self.validate()

    def test_legacy_or_unidentified_outputs_are_rejected_without_writes(self):
        for relative in ("report.json", "inputs-teacher.json", "teacher/routes/route-000.json"):
            with self.subTest(artifact=relative):
                output = self.root / relative.replace("/", "-")
                artifact = output / relative
                self.write(artifact, {"configuration_identity": {"schema_version": "continuous_combat_configuration_identity_v1"}})
                original = artifact.read_bytes()
                with self.assertRaisesRegex(ValueError, "legacy or missing"):
                    artifacts.prepare_output(output, self.validated["identity"])
                self.assertEqual(artifact.read_bytes(), original)
                self.assertFalse((output / "configuration_identity.json").exists())
        output = self.root / "partial"
        self.write(output / "arm/route-work/incomplete.json", {})
        with self.assertRaisesRegex(ValueError, "no resume identity"):
            artifacts.prepare_output(output, self.validated["identity"])
        legacy = deepcopy(self.item)
        legacy["configuration_identity"]["schema_version"] = "continuous_combat_configuration_identity_v1"
        self.write(self.output / self.relative_route, legacy)
        with self.assertRaisesRegex(ValueError, "legacy or missing"):
            self.consume(self.validated)

    def test_interrupted_run_marker_rejects_changed_configuration(self):
        output = self.root / "fresh"
        artifacts.prepare_output(output, self.validated["identity"])
        self.config["seeds"]["reward"] += 1
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            artifacts.prepare_output(output, self.validate()["identity"])
