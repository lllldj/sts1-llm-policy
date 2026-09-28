"""Tiny native sources exercise mixed export through the actual training loader."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sts1_llm_policy.data.trajectory import TrajectoryLogger, iter_trajectory_records
from sts1_llm_policy.data.gold.mixed_export import demonstration_record, export_mixed_sft
from sts1_llm_policy.env.combat_snapshot import with_snapshot_relics
from sts1_llm_policy.env.action_equivalence import build_model_action_space
from sts1_llm_policy.env.simulator_env import StsLightspeedEnv
from sts1_llm_policy.eval.episode_runner import run_combat_episode
from sts1_llm_policy.env.route_context import public_route_context
from sts1_llm_policy.policy.random_policy import RandomLegalPolicy
from sts1_llm_policy.train.configured_training import load_configured_training_run
from sts1_llm_policy.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    gzip_jsonl_bytes,
    iter_jsonl,
    read_json_object,
    sha256_file,
)
from sts1_llm_policy.data.manifest import build_dataset_manifest

from tests.simulator.native_support import NativeSelectionTestCase
from tests.training.model_fixture import _StubTokenizer
from tests.training.runtime_fixture import fixture as training_fixture


class AttackFixturePolicy(RandomLegalPolicy):
    def select_action(self, state, actions):
        result = super().select_action(state, actions)
        space = build_model_action_space(state, actions)
        selected = next((c for c in space.classes if c.card), space.classes[0])
        return replace(result, action=selected.representative,
                       selected_model_action_id=selected.model_action_id,
                       equivalent_internal_action_ids=selected.internal_action_ids)


class MixedExportNativeTests(NativeSelectionTestCase):
    def source(self, root, name, card="Carnage"):
        base = root / name
        lineage = {"route_index": 3, "combat_seed_group_index": 1,
                   "run_id": name, "arm": "teacher", "source_route_id": name + "-route-3"}
        identity = {"semantics_sha256": name}
        steps = [{"kind": "combat", "combat_index": 1, "encounter_family": "normal_weak",
                  "scenario_id": "jaw_worm"},
                 {"kind": "combat", "combat_index": 2, "encounter_family": "boss",
                  "scenario_id": "the_guardian"}]
        context = public_route_context(steps, 0)
        with self.execution.create_client() as client:
            client.reward_reset(712, ascension=0, act=1, relics=[])
            snapshot = client.export_combat_snapshot()["combat_snapshot"]
            snapshot["deck"] = [{"string_id": card, "upgraded": False}] * 5
            snapshot = with_snapshot_relics(snapshot, [])
            env = StsLightspeedEnv(client, allow_card_selection=True)
            env.reset("jaw_worm", 712, combat_snapshot=snapshot, route_context=context)
            trajectory = base / "combat.jsonl"
            logger = TrajectoryLogger(trajectory, episode_id=name, game_seed=712, policy_seed=101,
                                      policy_name="fixture", evidence_class="teacher_candidate_pool",
                                      observation_serializer_version="observation_v7",
                                      route_lineage={**lineage, "combat_index": 1})
            run_combat_episode(env, AttackFixturePolicy(101), logger, max_steps=30)
            records = list(iter_trajectory_records(trajectory))
            self.assertEqual(records[-1]["terminal_outcome"], "victory")
            env.reset("jaw_worm", 712, combat_snapshot=snapshot, route_context=context)
            gold = demonstration_record(env.public_state(), env.legal_actions(), records[0], dataset_id=name + "-gold",
                                        sample={"id": "gold-root", "route": 3, "group": 1, "combat": 1, "step": 0})
        pool_path = f"{name}/report.json"
        partition = {"role": "train_candidate", "excluded_routes": [1], "reserved_routes": [2]}
        route = {"status": "defeated", "configuration_identity": identity, "route_lineage": lineage,
                 "route": {"steps": steps}, "combats": [{
                     "combat_index": 1, "encounter_family": "normal_weak", "scenario_id": "jaw_worm",
                     "combat_seed": 712, "input_snapshot": snapshot, "trajectory": "combat.jsonl",
                     "trajectory_sha256": sha256_file(trajectory),
                     "summary": {"outcome": "victory", "decisions": len(records)}}]}
        atomic_write_json(base / "teacher/routes/route-003-combat-seed-group-01.json", route)
        atomic_write_json(root / pool_path, {
            "status": "completed", "run_id": name, "configuration_identity": identity,
            "configuration": {"data_source": {"evidence_class": "teacher_candidate_pool"}},
            "source_validation": {"excluded_sources": [{"overlap": False}]}})
        atomic_write_json(base / "selection.json", {"source_report": pool_path,
                          "source_partition": partition, "samples": [gold["source"]["sample"]]})
        payload = gzip_jsonl_bytes([gold])
        digest = atomic_write_bytes(base / "gold/train.jsonl.gz", payload)
        manifest = build_dataset_manifest(
            dataset_id=gold["dataset_id"], task_type="sft", observation_version="observation_v7",
            identity_fields=["record_id"], splits={"train": {"path": f"{name}/gold/train.jsonl.gz",
                "format": "jsonl", "compression": "gzip", "records": 1, "bytes": len(payload), "sha256": digest}},
            lineage={"source_pool": pool_path, "source_partition": partition, "source_routes": [3]},
            semantics={"state_weight": "equal", "record_schema": "decision_sft_group_v1"})
        atomic_write_json(base / "gold/manifest.json", manifest)
        return dict(schema_version="mixed_sft_export_v1", dataset_id=name + "-mixed",
                    gold_manifest=f"{name}/gold/manifest.json", source_report=pool_path,
                    source_selection=f"{name}/selection.json", output=f"export-{name}",
                    combat_indices=[1], encounter_families=["normal_weak"],
                    source_weights={"gold": .7, "teacher_strategy": .25, "forced_end": .05},
                    smoke_combats=[{"route": 3, "group": 1, "combat": 1}])

    def export(self, root, cfg, *, smoke=False):
        # Only the installed native profile belongs to the test host; all source
        # selection, replay, record construction and writes run through production code.
        path = root / (cfg["dataset_id"] + ".json")
        atomic_write_json(path, cfg)
        with patch("sts1_llm_policy.data.gold.mixed_export.resolve_simulator", return_value=self.execution):
            return export_mixed_sft(root, path, smoke=smoke)

    def load_for_training(self, root, manifest):
        consumer = root / ("consumer-" + manifest["dataset_id"])
        consumer.mkdir()
        run = training_fixture(consumer)
        artifact = consumer / manifest["splits"]["train"]["path"]
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes((root / manifest["splits"]["train"]["path"]).read_bytes())
        (consumer / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
        recipe = deepcopy(run.recipe)
        recipe["tokenization"]["max_sequence_tokens"] = 20000
        (consumer / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
        module = "sts1_llm_policy.train.configured_training."
        with patch(module + "load_verified_local_tokenizer", return_value=_StubTokenizer()):
            return load_configured_training_run(consumer, "run.json", mode="preflight")

    def test_source_and_weight_substitution_reach_native_replay_and_training(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            a = self.source(root, "first")
            b = self.source(root, "second", "Bludgeon")
            changed = {**a, "dataset_id": "reweighted", "output": "export-reweighted",
                       "source_weights": {"gold": .5, "teacher_strategy": .4, "forced_end": .1}}
            outputs = []
            for config in (a, changed, b):
                manifest = self.export(root, config)
                self.assertEqual(manifest, self.export(root, config))
                rows = list(iter_jsonl(root / manifest["splits"]["train"]["path"]))
                self.assertEqual(manifest["export_counts"]["gold_overlap_skipped"], 1)
                self.assertEqual(sum(r["source"]["sample"]["step"] == 0 for r in rows), 1)
                loaded = self.load_for_training(root, manifest)
                self.assertEqual([u["record_id"] for u in loaded.prepared.units], [r["record_id"] for r in rows])
                self.assertEqual([u["loss_weight"] for u in loaded.prepared.units], [r["loss_weight"] for r in rows])
                for source, mass in config["source_weights"].items():
                    self.assertAlmostEqual(manifest["mixture"][source]["loss_mass"], mass)
                outputs.append(rows)
            self.assertEqual([r["observation"] for r in outputs[0]], [r["observation"] for r in outputs[1]])
            self.assertNotEqual([r["loss_weight"] for r in outputs[0]], [r["loss_weight"] for r in outputs[1]])
            self.assertNotEqual([r["observation"] for r in outputs[0]], [r["observation"] for r in outputs[2]])

    def test_smoke_destination_and_late_manifest_conflict(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.source(root, "source")
            smoke = self.export(root, config, smoke=True)
            self.assertTrue(smoke["smoke"])
            self.assertIn("/smoke/", smoke["splits"]["train"]["path"])
            self.assertTrue(smoke["dataset_id"].endswith("_smoke"))
            output = root / config["output"] / "formal"
            output.mkdir()
            (output / "manifest.json").write_bytes(b"retained manifest")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                self.export(root, config)
            self.assertEqual(list(output.iterdir()), [output / "manifest.json"])
            self.assertEqual((output / "manifest.json").read_bytes(), b"retained manifest")

    def test_replay_mismatch_produces_no_dataset(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.source(root, "source")
            path = root / "source/combat.jsonl"
            rows = list(iter_trajectory_records(path))
            rows[0]["next_raw_state"]["state"]["player"]["current_hp"] -= 1
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            route_path = root / "source/teacher/routes/route-003-combat-seed-group-01.json"
            route = read_json_object(route_path)
            route["combats"][0]["trajectory_sha256"] = sha256_file(path)
            route_path.write_text(json.dumps(route), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "transition mismatch"):
                self.export(root, config)
            self.assertFalse((root / config["output"]).exists())
