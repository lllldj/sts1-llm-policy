from __future__ import annotations

from hashlib import sha256
import re
from typing import Mapping, Sequence

import torch

from sts1_llm_policy.policy.chat_rendering import render_generation_request
from sts1_llm_policy.policy.protocol import (
    SYSTEM_PROMPT,
    build_initial_request_from_observation,
    parse_action_id,
)


IGNORE_INDEX = -100
_LEGAL_ACTION_PATTERN = re.compile(r"^ACTION_(\d+):", re.MULTILINE)


def legal_action_ids(record: Mapping[str, object]) -> tuple[str, ...]:
    observation = record.get("observation")
    if not isinstance(observation, str) or not observation:
        raise ValueError("Decision observation must be non-empty text")
    action_ids = tuple(f"ACTION_{value}" for value in _LEGAL_ACTION_PATTERN.findall(observation))
    if not action_ids or len(action_ids) != len(set(action_ids)):
        raise ValueError("Decision legal actions must be non-empty and unique")
    if action_ids != tuple(f"ACTION_{index}" for index in range(len(action_ids))):
        raise ValueError("Decision legal actions must be contiguous from ACTION_0")
    return action_ids


def validate_decision_record(record: Mapping[str, object]) -> tuple[str, ...]:
    """Bind public prompt, action metadata and selected labels at the training boundary."""
    legal = legal_action_ids(record)
    observation = record["observation"]
    if sha256(observation.encode("utf-8")).hexdigest() != record.get("observation_sha256"):
        raise ValueError("Decision observation hash differs from its prompt")
    actions = record.get("model_actions")
    if not isinstance(actions, list) or any(not isinstance(a, dict) for a in actions):
        raise ValueError("Decision model_actions must be an array of objects")
    ids = [a.get("model_action_id") for a in actions]
    if (any(not isinstance(i, str) for i in ids) or len(ids) != len(legal)
            or set(ids) != set(legal)
            or any(a.get("action_type") not in {"play_card", "end_turn", "select_card"} for a in actions)):
        raise ValueError("Decision action metadata differs from its prompt")
    return legal


def selected_semantic(record: Mapping[str, object]) -> str:
    teacher_action_id = record.get("teacher_action_id")
    if not isinstance(teacher_action_id, str):
        raise ValueError("teacher_action_id must be a string")
    actions = record.get("model_actions")
    if not isinstance(actions, list):
        raise ValueError("model_actions must be an array")
    selected = [
        item
        for item in actions
        if isinstance(item, dict)
        and item.get("model_action_id") == teacher_action_id
    ]
    if len(selected) != 1:
        raise ValueError("teacher_action_id must resolve to exactly one model action")
    action = selected[0]
    if action.get("action_type") == "end_turn":
        return "END_TURN"
    if action.get("action_type") == "select_card":
        if record.get("schema_version") not in {"decision_sft_group_v1", "decision_preference_group_v1"}:
            raise ValueError("Secondary selection requires a supported decision group")
        return "SELECT_CARD"
    card_name = action.get("card_name")
    if action.get("action_type") != "play_card" or not isinstance(card_name, str):
        raise ValueError("Selected model action has invalid semantics")
    return card_name


def _token_ids(value: object, name: str) -> list[int]:
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return list(value)
    raise ValueError(f"Tokenizer {name} must be a flat integer list")


def tokenize_sft_record(
    tokenizer: object,
    record: Mapping[str, object],
    *,
    max_sequence_tokens: int,
    ignore_index: int = IGNORE_INDEX,
    artifact_id: str = "sft_smoke_data_v1",
) -> dict[str, object]:
    if max_sequence_tokens <= 0:
        raise ValueError("max_sequence_tokens must be positive")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("artifact_id must be a non-empty string")
    observation = record.get("observation")
    target = record.get("teacher_action_id")
    if not isinstance(observation, str) or not observation:
        raise ValueError("observation must be a non-empty string")
    if not isinstance(target, str) or parse_action_id(target) != target:
        raise ValueError("teacher_action_id must be one strict ACTION_n")

    legal = validate_decision_record(record)
    if target not in legal:
        raise ValueError("Training target is absent from the observation legal actions")

    request = build_initial_request_from_observation(observation)
    prompt_ids = _token_ids(
        render_generation_request(tokenizer, request),
        "generation prompt",
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation},
        {"role": "assistant", "content": target},
    ]
    full_ids = _token_ids(
        tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tools=None,
            add_generation_prompt=False,
            tokenize=True,
        ),
        "complete chat",
    )
    target_ids = _token_ids(
        tokenizer.encode(target, add_special_tokens=False),  # type: ignore[attr-defined]
        "assistant target",
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Generation prompt is not a prefix of the complete chat")
    response_ids = full_ids[len(prompt_ids) :]
    if response_ids[: len(target_ids)] != target_ids:
        raise ValueError("Assistant response does not begin with the target tokens")
    if len(full_ids) > max_sequence_tokens:
        raise ValueError(
            f"Sequence length {len(full_ids)} exceeds {max_sequence_tokens}; truncation is disabled"
        )
    if not response_ids:
        raise ValueError("Assistant response token span must not be empty")

    labels = [ignore_index] * len(prompt_ids) + response_ids
    if len(labels) != len(full_ids):
        raise AssertionError("Input and label lengths diverged")
    return {
        "schema_version": "sft_tokenized_record_v1",
        "artifact_id": artifact_id,
        "record_id": record["record_id"],
        "source_dataset_id": record["dataset_id"],
        "source_split": record["split"],
        "source_observation_sha256": record["observation_sha256"],
        "selected_semantic": selected_semantic(record),
        "target": target,
        "messages": messages,
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "prompt_token_count": len(prompt_ids),
        "assistant_content_token_count": len(target_ids),
        "assistant_terminator_token_count": len(response_ids) - len(target_ids),
        "loss_token_count": len(response_ids),
    }


class SftBatchCollator:
    def __init__(self, *, pad_token_id: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, records: Sequence[Mapping[str, object]]) -> dict[str, object]:
        if not records:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(record["input_ids"]) for record in records)  # type: ignore[arg-type]
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        record_ids: list[str] = []
        for record in records:
            current_ids = list(record["input_ids"])  # type: ignore[arg-type]
            current_mask = list(record["attention_mask"])  # type: ignore[arg-type]
            current_labels = list(record["labels"])  # type: ignore[arg-type]
            if not (
                len(current_ids) == len(current_mask) == len(current_labels)
            ):
                raise ValueError("Tokenized record fields have unequal lengths")
            padding = max_length - len(current_ids)
            input_ids.append(current_ids + [self.pad_token_id] * padding)
            attention_masks.append(current_mask + [0] * padding)
            labels.append(current_labels + [self.ignore_index] * padding)
            record_ids.append(str(record["record_id"]))
        return {
            "record_ids": tuple(record_ids),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def move_sft_batch(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "labels"}
        and isinstance(value, torch.Tensor)
    }
