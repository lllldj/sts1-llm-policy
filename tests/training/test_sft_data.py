from __future__ import annotations

from hashlib import sha256
import unittest

import torch

from sts1_llm_policy.train.sft_data import (
    IGNORE_INDEX,
    SftBatchCollator,
    legal_action_ids,
    tokenize_sft_record,
)

from tests.training.model_fixture import _StubTokenizer


def _record(record_id: str, semantic: str, action_id: str = "ACTION_0") -> dict[str, object]:
    action_type = "end_turn" if semantic == "END_TURN" else "play_card"
    observation = f"STATE {record_id}\nLEGAL_ACTIONS:\n" + "\n".join(
        f"ACTION_{index}: PLAY {semantic}" for index in range(int(action_id.split("_")[1]) + 1)
    )
    return {
        "record_id": record_id,
        "dataset_id": "teacher_dataset_v1_candidate",
        "split": "train",
        "observation_sha256": sha256(observation.encode()).hexdigest(),
        "observation": observation,
        "teacher_action_id": action_id,
        "model_actions": [
            {
                "model_action_id": f"ACTION_{index}",
                "action_type": action_type,
                "card_name": None if semantic == "END_TURN" else semantic,
            }
            for index in range(int(action_id.split("_")[1]) + 1)
        ],
    }


def _source_record(content: str) -> dict[str, object]:
    return {"observation": content}


class SftDataTest(unittest.TestCase):
    def test_extracts_contiguous_legal_action_ids(self) -> None:
        record = _source_record(
            "STATE\nLEGAL_ACTIONS:\nACTION_0: PLAY Defend\n"
            "ACTION_1: PLAY Strike\nACTION_2: END_TURN"
        )
        self.assertEqual(
            legal_action_ids(record), ("ACTION_0", "ACTION_1", "ACTION_2")
        )

    def test_rejects_gapped_or_duplicate_legal_action_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            legal_action_ids(
                _source_record("ACTION_0: PLAY Defend\nACTION_2: END_TURN")
            )
        with self.assertRaisesRegex(ValueError, "non-empty and unique"):
            legal_action_ids(
                _source_record("ACTION_0: PLAY Defend\nACTION_0: END_TURN")
            )

    def test_tokenization_masks_prompt_and_supervises_response_terminator(self) -> None:
        example = tokenize_sft_record(
            _StubTokenizer(),
            _record("one", "Strike", "ACTION_3"),
            max_sequence_tokens=1000,
            artifact_id="custom_sft_artifact",
        )
        prompt_count = example["prompt_token_count"]
        labels = example["labels"]
        input_ids = example["input_ids"]

        self.assertEqual(example["artifact_id"], "custom_sft_artifact")
        self.assertTrue(all(value == IGNORE_INDEX for value in labels[:prompt_count]))
        self.assertEqual(labels[prompt_count:], input_ids[prompt_count:])
        self.assertGreater(example["assistant_content_token_count"], 0)
        self.assertGreater(example["assistant_terminator_token_count"], 0)

    def test_collator_masks_padding(self) -> None:
        tokenizer = _StubTokenizer()
        examples = [
            tokenize_sft_record(
                tokenizer,
                _record(str(index), "Strike", f"ACTION_{index}"),
                max_sequence_tokens=1000,
            )
            for index in range(5)
        ]
        examples[0]["input_ids"] = examples[0]["input_ids"][:-2]
        examples[0]["attention_mask"] = examples[0]["attention_mask"][:-2]
        examples[0]["labels"] = examples[0]["labels"][:-2]
        batch = SftBatchCollator(pad_token_id=tokenizer.pad_token_id)(examples[:2])

        self.assertIsInstance(batch["input_ids"], torch.Tensor)
        self.assertEqual(batch["attention_mask"][0, -1].item(), 0)
        self.assertEqual(batch["labels"][0, -1].item(), IGNORE_INDEX)


if __name__ == "__main__":
    unittest.main()
