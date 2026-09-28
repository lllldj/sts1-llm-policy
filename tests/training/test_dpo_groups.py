"""DPO tokenization, loss, state weights, and streamed gradients."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import torch

from sts1_llm_policy.data.gold.dpo_export import preference_records
from sts1_llm_policy.train.configured_adapters import DpoAdapter
from sts1_llm_policy.train.configured_runner import _backward_unit
from sts1_llm_policy.train.dpo import (
    dpo_loss,
    response_sequence_log_probs,
    validate_preference_weight_semantics,
)
from sts1_llm_policy.train.sft_data import IGNORE_INDEX

from .gold_fixture import PARAMS, sources
from .model_fixture import _StubTokenizer


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(123)
        self.embedding = torch.nn.Embedding(256, 4)
        self.head = torch.nn.Linear(4, 256)
        self.events = []

    def forward(self, input_ids, **kwargs):
        self.events.append("forward")
        logits = self.head(self.embedding(input_ids))
        if logits.requires_grad:
            logits.register_hook(lambda grad: self.events.append("backward"))
        return SimpleNamespace(logits=logits)


class DpoGroupTests(unittest.TestCase):
    def setUp(self):
        self.rows = preference_records(*sources(), **PARAMS)[0]["c"]
        self.adapter = DpoAdapter()
        self.runtime = SimpleNamespace(device="cpu", generation_pad_token_id=0)
        self.recipe = {"dpo": {"beta": .1}}

    def prepare(self, rows=None):
        return self.adapter.prepare(_StubTokenizer(), self.rows if rows is None else rows,
                                    dataset_id="test-c", max_sequence_tokens=4096)

    def test_streamed_pair_gradients_equal_whole_group_and_preserve_state_weight(self):
        unit = self.prepare().units[0]
        run = SimpleNamespace(adapter=self.adapter, runtime=self.runtime, recipe=self.recipe)
        reference = self.adapter.reference_values(TinyLM(), [unit], runtime=self.runtime)
        a, b, c = TinyLM(), TinyLM(), TinyLM()
        full = self.adapter.unit_loss(a, unit, runtime=self.runtime, reference=reference, recipe=self.recipe)
        full.backward()
        with patch("sts1_llm_policy.train.configured_runner.torch.autocast", return_value=nullcontext()):
            streamed = _backward_unit(run, b, unit, reference, scale=.125)
            changed = _backward_unit(run, c, {**unit, "loss_weight": unit["loss_weight"] * 2}, reference, scale=.125)
        self.assertAlmostEqual(streamed, full.item(), places=6)
        self.assertAlmostEqual(changed, streamed * 2, places=6)
        self.assertEqual(b.events[:3], ["forward", "forward", "backward"])
        for p, q, z in zip(a.parameters(), b.parameters(), c.parameters()):
            torch.testing.assert_close(q.grad, p.grad * .125, atol=1e-7, rtol=1e-5)
            torch.testing.assert_close(z.grad, q.grad * 2)
        self.assertNotEqual(self.adapter.input_identity(unit),
                            self.adapter.input_identity({**unit, "loss_weight": 2.}))

    def test_hp_edge_weights_change_gradient_direction(self):
        unit = self.prepare().units[0]
        reference = self.adapter.reference_values(TinyLM(), [unit], runtime=self.runtime)
        a, b = TinyLM(), TinyLM()
        other = deepcopy(unit)
        other["edges"][0]["weight"], other["edges"][1]["weight"] = .8, .2
        for model, row in ((a, unit), (b, other)):
            self.adapter.unit_loss(model, row, runtime=self.runtime, reference=reference, recipe=self.recipe).backward()
        self.assertFalse(torch.allclose(a.head.weight.grad, b.head.weight.grad))

    def test_explicit_weights_require_manifest_and_complete_mean_one_values(self):
        semantics = {"state_weight": "explicit_mean_one_loss_weight"}
        validate_preference_weight_semantics(self.rows, semantics)
        with self.assertRaises(ValueError): validate_preference_weight_semantics(self.rows, {"state_weight": "equal"})
        for value in (None, True, 0, -1, float("nan"), float("inf"), 8):
            rows = deepcopy(self.rows)
            if value is None: rows[0].pop("loss_weight")
            else: rows[0]["loss_weight"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.prepare(rows)
        for change in ("duplicate_state", "duplicate_edge", "same_action", "bool_weight"):
            rows = deepcopy(self.rows)
            if change == "duplicate_state": rows[1]["record_id"] = rows[0]["record_id"]
            if change == "duplicate_edge": rows[0]["edges"][1] = deepcopy(rows[0]["edges"][0])
            if change == "same_action": rows[0]["edges"][0]["chosen_action_id"] = rows[0]["edges"][0]["rejected_action_id"]
            if change == "bool_weight": rows[0]["edges"][0]["weight"] = True
            with self.subTest(change=change), self.assertRaises(ValueError): self.prepare(rows)

    def test_secondary_selection_survives_dpo_tokenization(self):
        rows = deepcopy(self.rows)
        rows[0]["model_actions"][0].update(action_type="select_card", card_name=None)
        unit = self.prepare(rows).units[0]
        self.assertEqual(unit["edges"][0]["chosen"]["selected_semantic"], "SELECT_CARD")
        self.assertGreater(unit["edges"][0]["chosen"]["assistant_terminator_token_count"], 0)


class DpoDataTests(unittest.TestCase):
    def test_group_edge_reuses_prompt_and_scores_both_terminators(self) -> None:
        rows = preference_records(*sources(), **PARAMS)[0]["c"]
        prepared = DpoAdapter().prepare(_StubTokenizer(), rows, dataset_id="test-c",
                                        max_sequence_tokens=4096)
        pair = prepared.units[0]["edges"][0]
        prompt_count = pair["chosen"]["prompt_token_count"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        self.assertEqual(
            chosen["input_ids"][:prompt_count],
            rejected["input_ids"][:prompt_count],
        )
        for branch in (chosen, rejected):
            self.assertTrue(
                all(value == IGNORE_INDEX for value in branch["labels"][:prompt_count])
            )
            self.assertEqual(
                branch["labels"][prompt_count:],
                branch["input_ids"][prompt_count:],
            )
            self.assertGreater(branch["assistant_terminator_token_count"], 0)

    def test_dpo_logprob_and_loss_are_response_only_with_expected_signs(self) -> None:
        logits = torch.zeros((1, 4, 3), dtype=torch.float32)
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2]])
        logps, counts = response_sequence_log_probs(logits, labels)
        self.assertEqual(counts.tolist(), [2])
        self.assertAlmostEqual(logps.item(), -2 * math.log(3), places=6)

        chosen = torch.tensor([-3.0], requires_grad=True)
        rejected = torch.tensor([-4.0], requires_grad=True)
        result = dpo_loss(
            chosen,
            rejected,
            torch.tensor([-3.0]),
            torch.tensor([-4.0]),
            beta=0.1,
        )
        loss = result["losses"].mean()
        loss.backward()
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertAlmostEqual(result["preference_logits"].item(), 0.0, places=6)
        self.assertLess(chosen.grad.item(), 0.0)
        self.assertGreater(rejected.grad.item(), 0.0)

        better = dpo_loss(
            torch.tensor([-2.0]),
            torch.tensor([-4.0]),
            torch.tensor([-3.0]),
            torch.tensor([-4.0]),
            beta=0.1,
        )["losses"]
        worse = dpo_loss(
            torch.tensor([-4.0]),
            torch.tensor([-4.0]),
            torch.tensor([-3.0]),
            torch.tensor([-4.0]),
            beta=0.1,
        )["losses"]
        self.assertLess(better.item(), math.log(2))
        self.assertGreater(worse.item(), math.log(2))


if __name__ == "__main__":
    unittest.main()
