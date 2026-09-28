from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
import torch
from unittest.mock import patch

from sts1_llm_policy.train.configured_adapters import (
    DpoAdapter,
    SftAdapter,
    adapter_for_algorithm,
    select_smoke_unit_indices,
)
from sts1_llm_policy.train.configured_runner import _reference_values, _training_order

from .model_fixture import _StubTokenizer


def _actions() -> list[dict[str, object]]:
    return [
        {"model_action_id": "ACTION_0", "action_type": "end_turn", "card_name": None},
        {"model_action_id": "ACTION_1", "action_type": "play_card", "card_name": "Strike"},
        {"model_action_id": "ACTION_2", "action_type": "play_card", "card_name": "Defend"},
    ]


OBSERVATION = "STATE\nLEGAL_ACTIONS:\nACTION_0: END_TURN\nACTION_1: PLAY Strike\nACTION_2: PLAY Defend"


class ConfiguredTrainingTests(unittest.TestCase):
    def test_training_order_is_seeded_and_covers_every_unit_once(self) -> None:
        first = _training_order(25, epochs=1, seed=123)
        self.assertEqual(first, _training_order(25, epochs=1, seed=123))
        self.assertNotEqual(first, _training_order(25, epochs=1, seed=124))
        self.assertEqual(sorted(first), list(range(25)))

    def test_sft_adapter_prepares_manifest_record(self) -> None:
        prepared = SftAdapter().prepare(
            _StubTokenizer(),
            [{
                "schema_version": "decision_sft_record_v1",
                "dataset_id": "gold-v1",
                "split": "train",
                "record_id": "record-1",
                "observation_sha256": sha256(OBSERVATION.encode()).hexdigest(),
                "observation": OBSERVATION,
                "model_actions": _actions(),
                "teacher_action_id": "ACTION_1",
            }],
            dataset_id="gold-v1",
            max_sequence_tokens=4096,
        )
        self.assertEqual(prepared.algorithm, "sft")
        self.assertEqual(len(prepared.units), 1)
        self.assertFalse(prepared.tokenization["truncated"])
        self.assertGreater(prepared.units[0]["sequence_tokens_max"], 0)

    def test_dpo_adapter_preserves_group_and_normalized_edges(self) -> None:
        prepared = DpoAdapter().prepare(
            _StubTokenizer(),
            [{
                "schema_version": "decision_preference_group_v1",
                "dataset_id": "silver-v1",
                "split": "train",
                "record_id": "record-1",
                "observation_sha256": sha256(OBSERVATION.encode()).hexdigest(),
                "observation": OBSERVATION,
                "model_actions": _actions(),
                "edges": [
                    {"chosen_action_id": "ACTION_0", "rejected_action_id": "ACTION_1", "weight": 0.5},
                    {"chosen_action_id": "ACTION_0", "rejected_action_id": "ACTION_2", "weight": 0.5},
                ],
            }],
            dataset_id="silver-v1",
            max_sequence_tokens=4096,
        )
        self.assertEqual(prepared.algorithm, "dpo")
        self.assertEqual(prepared.tokenization["edges"], 2)
        self.assertTrue(math.isclose(sum(edge["weight"] for edge in prepared.units[0]["edges"]), 1.0))
        chosen = prepared.units[0]["edges"][0]["chosen"]
        rejected = prepared.units[0]["edges"][0]["rejected"]
        prompt = int(chosen["prompt_token_count"])
        self.assertEqual(chosen["input_ids"][:prompt], rejected["input_ids"][:prompt])

    def test_smoke_selection_always_includes_longest(self) -> None:
        units = [
            {"record_id": "short", "sequence_tokens_max": 10},
            {"record_id": "long", "sequence_tokens_max": 100},
            {"record_id": "middle", "sequence_tokens_max": 50},
        ]
        self.assertEqual(select_smoke_unit_indices(units, count=1, seed=7), (1,))
        selected = select_smoke_unit_indices(units, count=2, seed=7)
        self.assertIn(1, selected)
        self.assertEqual(selected, select_smoke_unit_indices(units, count=2, seed=7))


    def test_adapter_registry_is_explicit(self) -> None:
        self.assertIsInstance(adapter_for_algorithm("sft"), SftAdapter)
        self.assertIsInstance(adapter_for_algorithm("dpo"), DpoAdapter)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            adapter_for_algorithm("ppo")

    def test_dpo_reference_groups_are_checkpointed_and_reused(self) -> None:
        with TemporaryDirectory() as directory:
            run = SimpleNamespace(
                adapter=DpoAdapter(),
                output_dir=Path(directory),
                runtime=SimpleNamespace(),
            )
            units = ({"record_id": "group-1", "edges": [{}]},)
            model = torch.nn.Linear(1, 1)
            computed = {"group-1": [{"chosen": -1.0, "rejected": -2.0}]}
            with (
                patch(
                    "sts1_llm_policy.train.configured_runner.load_initial_model",
                    return_value=(model, None, ()),
                ) as load_model,
                patch.object(DpoAdapter, "reference_values", return_value=computed),
            ):
                first, first_summary = _reference_values(
                    run, units=units, binding={"run": "fixture"}
                )
            self.assertEqual(first, computed)
            self.assertEqual(first_summary, {"groups_reused": 0, "groups_computed": 1})
            load_model.assert_called_once()
            with patch(
                "sts1_llm_policy.train.configured_runner.load_initial_model",
                side_effect=AssertionError("cache should avoid model loading"),
            ):
                second, second_summary = _reference_values(
                    run, units=units, binding={"run": "fixture"}
                )
            self.assertEqual(second, computed)
            self.assertEqual(second_summary, {"groups_reused": 1, "groups_computed": 0})


if __name__ == "__main__":
    unittest.main()
