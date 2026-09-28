import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

from sts1_llm_policy.data.gold.configuration import validate_storage
from unittest.mock import Mock, patch

from sts1_llm_policy.data.gold import storage as compact
from sts1_llm_policy.data.gold import collection as probe
from sts1_llm_policy.data.gold import verification


POLICY = {"mode": "compact_v1", "full_trace_probability": 0.,
          "keep_first_trial": True, "keep_boundary_trials": True}
SAMPLE = {"id": "example", "route": 0, "group": 0, "combat": 0, "step": 0}


class CompactStorageTests(unittest.TestCase):
    def rows(self):
        return [{"sample_id": "example", "trial": t, "root_action": "ACTION_0", "outcome": "victory",
                 "ending_hp": 40 + t, "max_hp": 80, "relic_counters": [["PEN_NIB", t]],
                 "decisions": 2, "search_calls": 4, "trace": f"example/trial-{t:03d}-ACTION_0.jsonl.gz"}
                for t in range(2)]

    def archive(self, output):
        rows = self.rows()
        tapes = {(t, "ACTION_0"): ["ACTION_0", "ACTION_1"] for t in range(2)}
        receipt = {"status": "completed", "verified_executions": 2, "verified_decisions": 4,
                   "full_search_trace_executions": 2}
        return compact.compact_verified_state(output, SAMPLE, rows, [], {"seed": 1, "continuation_storage": POLICY}, tapes, receipt)

    def test_compaction_preserves_each_terminal_and_tape_and_cleanup_is_scoped(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            state = output / "example"
            state.mkdir()
            for t in range(2):
                for ext in ("json", "jsonl.gz"):
                    (state / f"trial-{t:03d}-ACTION_0.{ext}").write_bytes(b"original")
            unrelated = state / "keep.json"
            unrelated.write_text("keep")
            archive = self.archive(output)
            for original, row in zip(self.rows(), archive["executions"]):
                self.assertEqual({k: row[k] for k in original}, original)
            compact.cleanup_compacted_state(output, SAMPLE, archive)
            compact.cleanup_compacted_state(output, SAMPLE, archive)
            self.assertTrue((state / "trial-000-ACTION_0.jsonl.gz").exists())
            self.assertFalse((state / "trial-001-ACTION_0.jsonl.gz").exists())
            self.assertEqual(unrelated.read_text(), "keep")
            self.assertEqual(compact.read_archive(state / "continuations.json.gz", SAMPLE), archive)

    def test_corrupt_archive_blocks_cleanup_and_partial_verification_blocks_publish(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            archive = self.archive(output)
            path = output / "example/continuations.json.gz"
            bad = json.loads(json.dumps(archive))
            bad["executions"][0]["model_actions"][0] = "ACTION_9"
            with gzip.open(path, "wt") as stream:
                json.dump(bad, stream)
            with self.assertRaisesRegex(ValueError, "sequence"):
                compact.cleanup_compacted_state(output, SAMPLE, archive)
            with self.assertRaisesRegex(ValueError, "Every original search"):
                compact.compact_verified_state(output, SAMPLE, self.rows(), [], {"seed": 1, "continuation_storage": POLICY}, {},
                    {"status": "completed", "verified_executions": 2, "verified_decisions": 4, "full_search_trace_executions": 1})

    def test_retention_controls_change_actual_archived_evidence(self):
        rows = [{"trial": t, "root_action": a} for t in range(64) for a in ("ACTION_0", "ACTION_1")]
        stages = [{"trials": 32, "hp_boundary_actions": ["ACTION_1"], "win_boundary_actions": [], "candidates": ["ACTION_0"]},
                  {"trials": 64, "hp_boundary_actions": [], "win_boundary_actions": [], "candidates": ["ACTION_0", "ACTION_1"]}]
        cfg = {"seed": 1, "continuation_storage": POLICY}
        self.assertEqual(compact.retained_trials(SAMPLE, rows, stages, cfg),
                         {(0, "ACTION_0"), (0, "ACTION_1"), (31, "ACTION_1"), (63, "ACTION_1")})
        cfg["continuation_storage"] = {**POLICY, "full_trace_probability": 1.}
        self.assertEqual(len(compact.retained_trials(SAMPLE, rows, stages, cfg)), 128)

    def test_resume_rejects_algorithm_changes_and_keeps_source_immutable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "old", root / "new"
            cfg = {"seed": 1, "samples": [SAMPLE], "search_budget": 2048}
            identity = {"protocol": probe.PROTOCOL, "configuration": cfg, "native": {"bridge_sha256": "native"},
                        "source": {"fixture": "unchanged-pool"}}
            probe.write_json(source / "identity.json", identity)
            probe.write_json(source / "report.json", {"status": "partial", "git_revision": "original"})
            newcfg = {**cfg, "continuation_storage": POLICY, "resume_from": "old"}
            saved = source / "example/trial-000-ACTION_0.json"
            probe.write_json(saved, self.rows()[0])
            saved.with_suffix(".jsonl.gz").write_bytes(b"trace")
            manifest = compact.prepare_resume(root, newcfg, output, {**identity, "configuration": newcfg})
            with self.assertRaisesRegex(ValueError, "changes execution inputs"):
                compact.prepare_resume(root, newcfg, output, {**identity, "configuration": {**newcfg, "search_budget": 8192}})
            compact.import_sample(root, newcfg, SAMPLE, output, threading.Event(), manifest)
            imported = output / "example/trial-000-ACTION_0.json"
            self.assertEqual(saved.read_bytes(), imported.read_bytes())
            imported.unlink()  # Completed-state compaction already consumed it.
            compact.import_sample(root, newcfg, SAMPLE, output, threading.Event(), manifest)
            self.assertFalse(imported.exists())
            self.assertTrue(saved.exists())
            (source / "report.json").unlink()
            (source / "identity.json").unlink()
            self.assertEqual(compact.prepare_resume(root, newcfg, output, {**identity, "configuration": newcfg}), manifest)

    def test_resume_requires_explicit_storage_and_rejects_bad_policy(self):
        for cfg in ({"resume_from": "old"}, {"continuation_storage": {**POLICY, "full_trace_probability": float("nan")}},
                    {"continuation_storage": {**POLICY, "keep_first_trial": 1}}):
            with self.assertRaises(ValueError):
                validate_storage(cfg)

    def test_cancelled_sample_never_loads_source_or_imports(self):
        stop = threading.Event()
        stop.set()
        with patch.object(probe, "load_source") as load, patch.object(probe, "import_sample") as importer:
            with self.assertRaises(InterruptedError):
                probe.run_sample(Path("."), {"resume_from": "old"}, SAMPLE, None, Path("."), stop)
            load.assert_not_called()
            importer.assert_not_called()

    def test_compact_replay_rejects_wrong_terminal_illegal_and_trailing_actions(self):
        terminal = {"state": {"outcome": "PLAYER_VICTORY", "player": {"current_hp": 40, "max_hp": 80}, "relics": []}}
        result = {**self.rows()[0], "model_actions": ["ACTION_0"], "decisions": 1,
                  "search_calls": 0, "relic_counters": []}
        config = {"seed": 1, "hidden_samples": 4, "max_decisions": 1000}
        env = Mock()
        env.step.side_effect = probe.SimulatorCombatEndedError(terminal, terminal_state=None, reward=1, outcome="victory")
        with patch.object(verification, "restore"), patch.object(verification, "build_model_action_space") as space:
            self.assertEqual(verification.verify_action_tape(env, None, SAMPLE, config, result), ["ACTION_0"])
            with self.assertRaisesRegex(ValueError, "score differs"):
                verification.verify_action_tape(env, None, SAMPLE, config, {**result, "ending_hp": 41})
            with self.assertRaisesRegex(ValueError, "beyond combat end"):
                verification.verify_action_tape(env, None, SAMPLE, config,
                                        {**result, "model_actions": ["ACTION_0", "ACTION_1"], "decisions": 2, "search_calls": 4})
            space.return_value.resolve.return_value = None
            with self.assertRaisesRegex(ValueError, "Illegal compact action"):
                verification.verify_action_tape(env, None, SAMPLE, config, result)


if __name__ == "__main__":
    unittest.main()
