from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Iterator, Mapping, Protocol, Sequence

import torch

from sts1_llm_policy.model_runtime import BaseModelRuntimeConfig
from sts1_llm_policy.train.dpo import dpo_loss, response_sequence_log_probs
from sts1_llm_policy.train.sft_data import (
    SftBatchCollator, move_sft_batch, tokenize_sft_record, validate_decision_record,
)


@dataclass(frozen=True)
class PreparedTrainingData:
    algorithm: str
    units: tuple[dict[str, Any], ...]
    tokenization: dict[str, int | bool]


class TrainingAdapter(Protocol):
    algorithm: str
    requires_initial_checkpoint: bool
    uses_reference: bool
    loss_contract: str

    def input_identity(self, unit: Mapping[str, Any]) -> dict[str, Any]: ...

    def generation_prompt(self, unit: Mapping[str, Any]) -> list[int]: ...

    def loss_configuration(self, recipe: Mapping[str, Any]) -> dict[str, Any]: ...

    def prepare(
        self,
        tokenizer: object,
        records: Sequence[Mapping[str, Any]],
        *,
        dataset_id: str,
        max_sequence_tokens: int,
    ) -> PreparedTrainingData: ...

    def unit_loss(
        self,
        model: torch.nn.Module,
        unit: Mapping[str, Any],
        *,
        runtime: BaseModelRuntimeConfig,
        reference: Mapping[str, Any] | None,
        recipe: Mapping[str, Any],
    ) -> torch.Tensor: ...

    def loss_terms(self, model: torch.nn.Module, unit: Mapping[str, Any], **kwargs: Any
                   ) -> Iterator[torch.Tensor]: ...


def select_smoke_unit_indices(
    units: Sequence[Mapping[str, Any]], *, count: int, seed: int
) -> tuple[int, ...]:
    if count <= 0 or count > len(units):
        raise ValueError("Smoke unit count is outside the prepared dataset")
    longest = max(
        range(len(units)),
        key=lambda index: (int(units[index]["sequence_tokens_max"]), str(units[index]["record_id"])),
    )
    selected = [longest]
    if len(selected) == count:
        return tuple(selected)
    ranked = sorted(
        range(len(units)),
        key=lambda index: sha256(
            f"{seed}|{units[index]['record_id']}".encode("utf-8")
        ).hexdigest(),
    )
    for index in ranked:
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    return tuple(selected)


def _token_summary(lengths: Sequence[int], responses: Sequence[int]) -> dict[str, int | bool]:
    if not lengths or not responses:
        raise ValueError("Tokenization produced no records")
    return {
        "units": len(lengths),
        "sequence_tokens_min": min(lengths),
        "sequence_tokens_max": max(lengths),
        "response_tokens_min": min(responses),
        "response_tokens_max": max(responses),
        "truncated": False,
    }


def _token_identity(tokens: Mapping[str, Any]) -> dict[str, Any]:
    return {key: tokens[key] for key in (
        "input_ids", "attention_mask", "labels", "prompt_token_count"
    )}


def _branch_logp(
    model: torch.nn.Module,
    branch: Mapping[str, object],
    runtime: BaseModelRuntimeConfig,
) -> torch.Tensor:
    collator = SftBatchCollator(pad_token_id=runtime.generation_pad_token_id)
    batch = move_sft_batch(collator([branch]), torch.device(runtime.device))
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logps, _ = response_sequence_log_probs(output.logits, batch["labels"])
    return logps.squeeze(0)


class SftAdapter:
    algorithm = "sft"
    requires_initial_checkpoint = False
    uses_reference = False
    loss_contract = "response_token_ce_weighted_groups_v1"

    def input_identity(self, unit: Mapping[str, Any]) -> dict[str, Any]:
        if "candidates" in unit:
            return {"record_id": unit["record_id"],
                    **({"loss_weight": unit["loss_weight"]} if "loss_weight" in unit else {}),
                    "candidates": [
                {"action_id": c["action_id"], "weight": c["weight"],
                 **_token_identity(c["tokenized"])} for c in unit["candidates"]]}
        return {"record_id": unit["record_id"], **_token_identity(unit["tokenized"])}

    def generation_prompt(self, unit: Mapping[str, Any]) -> list[int]:
        tokens = unit["candidates"][0]["tokenized"] if "candidates" in unit else unit["tokenized"]
        return list(tokens["input_ids"][:tokens["prompt_token_count"]])

    def loss_configuration(self, recipe: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def prepare(
        self,
        tokenizer: object,
        records: Sequence[Mapping[str, Any]],
        *,
        dataset_id: str,
        max_sequence_tokens: int,
    ) -> PreparedTrainingData:
        units: list[dict[str, Any]] = []
        lengths: list[int] = []
        responses: list[int] = []
        seen: set[str] = set()
        for record in records:
            if (
                record.get("schema_version") not in {"decision_sft_record_v1", "decision_sft_group_v1"}
                or record.get("dataset_id") != dataset_id
                or record.get("split") != "train"
            ):
                raise ValueError("SFT record does not match its dataset manifest")
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or not record_id or record_id in seen:
                raise ValueError("SFT record IDs must be unique nonempty strings")
            seen.add(record_id)
            if record["schema_version"] == "decision_sft_group_v1":
                from sts1_llm_policy.train.sft_groups import prepare_group
                unit = prepare_group(tokenizer, record, dataset_id=dataset_id,
                                     max_sequence_tokens=max_sequence_tokens)
                units.append(unit)
                lengths.append(unit["sequence_tokens_max"])
                responses.extend(int(c["tokenized"]["loss_token_count"]) for c in unit["candidates"])
                continue
            if "loss_weight" in record:
                raise ValueError("State loss weights require the SFT group schema")
            tokenized = tokenize_sft_record(
                tokenizer,
                record,
                max_sequence_tokens=max_sequence_tokens,
                artifact_id=dataset_id,
            )
            length = len(tokenized["input_ids"])
            response = int(tokenized["loss_token_count"])
            units.append({
                "record_id": record["record_id"],
                "sequence_tokens_max": length,
                "tokenized": tokenized,
            })
            lengths.append(length)
            responses.append(response)
        weighted = [u for u in units if "loss_weight" in u]
        if weighted and (len(weighted) != len(units) or not math.isclose(
                math.fsum(u["loss_weight"] for u in units), len(units), rel_tol=1e-9)):
            raise ValueError("Explicit SFT state loss weights must cover all states and have mean one")
        return PreparedTrainingData(
            algorithm=self.algorithm,
            units=tuple(units),
            tokenization={**_token_summary(lengths, responses),
                          "candidate_answers": sum(len(u.get("candidates", [None])) for u in units)},
        )

    def unit_loss(
        self,
        model: torch.nn.Module,
        unit: Mapping[str, Any],
        *,
        runtime: BaseModelRuntimeConfig,
        reference: Mapping[str, Any] | None,
        recipe: Mapping[str, Any],
    ) -> torch.Tensor:
        if "candidates" in unit:
            return torch.stack(list(self.loss_terms(model, unit, runtime=runtime,
                                                    reference=reference, recipe=recipe))).sum()
        del reference, recipe
        collator = SftBatchCollator(pad_token_id=runtime.generation_pad_token_id)
        batch = move_sft_batch(
            collator([unit["tokenized"]]), torch.device(runtime.device)
        )
        return model(**batch).loss

    def loss_terms(self, model: torch.nn.Module, unit: Mapping[str, Any], **kwargs: Any
                   ) -> Iterator[torch.Tensor]:
        if "candidates" not in unit:
            yield self.unit_loss(model, unit, **kwargs)
            return
        for candidate in unit["candidates"]:
            yield (self.unit_loss(model, {"tokenized": candidate["tokenized"]}, **kwargs)
                   * candidate["weight"] * unit.get("loss_weight", 1.0))


class DpoAdapter:
    algorithm = "dpo"
    requires_initial_checkpoint = True
    uses_reference = True
    loss_contract = "frozen_reference_weighted_dpo_fp32_v2"

    def input_identity(self, unit: Mapping[str, Any]) -> dict[str, Any]:
        return {"record_id": unit["record_id"],
                **({"loss_weight": unit["loss_weight"]} if "loss_weight" in unit else {}), "edges": [
            {"edge_id": edge["edge_id"], "weight": edge["weight"],
             "chosen": _token_identity(edge["chosen"]),
             "rejected": _token_identity(edge["rejected"])}
            for edge in unit["edges"]
        ]}

    def generation_prompt(self, unit: Mapping[str, Any]) -> list[int]:
        branches = [edge[name] for edge in unit["edges"] for name in ("chosen", "rejected")]
        tokens = max(branches, key=lambda branch: len(branch["input_ids"]))
        return list(tokens["input_ids"][:tokens["prompt_token_count"]])

    def loss_configuration(self, recipe: Mapping[str, Any]) -> dict[str, Any]:
        return {"dpo": dict(recipe["dpo"])}

    def validate_reference_values(self, unit: Mapping[str, Any], values: object) -> bool:
        return bool(
            isinstance(values, list) and len(values) == len(unit["edges"])
            and all(isinstance(item, dict)
                    and math.isfinite(float(item.get("chosen", float("nan"))))
                    and math.isfinite(float(item.get("rejected", float("nan"))))
                    for item in values)
        )

    def prepare(
        self,
        tokenizer: object,
        records: Sequence[Mapping[str, Any]],
        *,
        dataset_id: str,
        max_sequence_tokens: int,
    ) -> PreparedTrainingData:
        units: list[dict[str, Any]] = []
        lengths: list[int] = []
        responses: list[int] = []
        for record in records:
            if (
                record.get("schema_version") != "decision_preference_group_v1"
                or record.get("dataset_id") != dataset_id
                or record.get("split") != "train"
            ):
                raise ValueError("DPO group does not match its dataset manifest")
            validate_decision_record(record)
            edges = record.get("edges")
            if not isinstance(edges, list) or not edges:
                raise ValueError("DPO group must contain preference edges")
            loss_weight = record.get("loss_weight", 1.0)
            if (isinstance(loss_weight, bool) or not isinstance(loss_weight, (int, float))
                    or not math.isfinite(loss_weight) or loss_weight <= 0):
                raise ValueError("DPO state loss_weight must be positive and finite")
            pairs = [(e.get("chosen_action_id"), e.get("rejected_action_id"))
                     for e in edges if isinstance(e, dict)]
            if len(set(pairs)) != len(edges) or any(c == r for c, r in pairs):
                raise ValueError("DPO edges must be unique and compare distinct actions")
            tokenized_edges: list[dict[str, Any]] = []
            weights: list[float] = []
            group_max = 0
            for edge_index, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    raise ValueError("DPO edge must be an object")
                weight = edge.get("weight")
                if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                        or not math.isfinite(weight) or weight <= 0):
                    raise ValueError("DPO edge weight must be positive and finite")
                branches: dict[str, dict[str, object]] = {}
                for branch, key in (("chosen", "chosen_action_id"), ("rejected", "rejected_action_id")):
                    branches[branch] = tokenize_sft_record(
                        tokenizer,
                        {
                            "schema_version": record["schema_version"],
                            "dataset_id": dataset_id,
                            "split": "train",
                            "record_id": f"{record['record_id']}:{edge_index}:{branch}",
                            "observation_sha256": record["observation_sha256"],
                            "observation": record["observation"],
                            "model_actions": record["model_actions"],
                            "teacher_action_id": edge[key],
                        },
                        max_sequence_tokens=max_sequence_tokens,
                        artifact_id=dataset_id,
                    )
                chosen_prompt = int(branches["chosen"]["prompt_token_count"])
                rejected_prompt = int(branches["rejected"]["prompt_token_count"])
                if any(int(b["assistant_terminator_token_count"]) <= 0 for b in branches.values()):
                    raise ValueError("DPO responses must include the assistant terminator")
                if (
                    chosen_prompt != rejected_prompt
                    or branches["chosen"]["input_ids"][:chosen_prompt]
                    != branches["rejected"]["input_ids"][:rejected_prompt]
                ):
                    raise ValueError("DPO branches do not share an exact prompt")
                branch_lengths = [len(branches[name]["input_ids"]) for name in ("chosen", "rejected")]
                branch_responses = [int(branches[name]["loss_token_count"]) for name in ("chosen", "rejected")]
                group_max = max(group_max, *branch_lengths)
                lengths.extend(branch_lengths)
                responses.extend(branch_responses)
                weights.append(weight)
                tokenized_edges.append({
                    "edge_id": f"{record['record_id']}:{edge_index}",
                    "weight": weight,
                    "chosen": branches["chosen"],
                    "rejected": branches["rejected"],
                })
            if not math.isclose(math.fsum(weights), 1.0):
                raise ValueError("DPO group weights must sum to one")
            units.append({
                "record_id": record["record_id"],
                **({"loss_weight": float(loss_weight)} if "loss_weight" in record else {}),
                "sequence_tokens_max": group_max,
                "edges": tokenized_edges,
            })
        ids = [u["record_id"] for u in units]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("DPO state IDs must be unique")
        weighted = [u for u in units if "loss_weight" in u]
        if weighted and (len(weighted) != len(units) or not math.isclose(
                math.fsum(u["loss_weight"] for u in units), len(units), rel_tol=1e-9)):
            raise ValueError("Explicit DPO state loss weights must cover all states and have mean one")
        summary = _token_summary(lengths, responses)
        summary["units"] = len(units)
        summary["edges"] = sum(len(unit["edges"]) for unit in units)
        return PreparedTrainingData(
            algorithm=self.algorithm,
            units=tuple(units),
            tokenization=summary,
        )

    def reference_values(
        self,
        model: torch.nn.Module,
        units: Sequence[Mapping[str, Any]],
        *,
        runtime: BaseModelRuntimeConfig,
    ) -> dict[str, Any]:
        model.eval()
        result: dict[str, Any] = {}
        with torch.inference_mode():
            for unit in units:
                result[str(unit["record_id"])] = [
                    {
                        "chosen": float(_branch_logp(model, edge["chosen"], runtime).cpu()),
                        "rejected": float(_branch_logp(model, edge["rejected"], runtime).cpu()),
                    }
                    for edge in unit["edges"]
                ]
        return result

    def unit_loss(
        self,
        model: torch.nn.Module,
        unit: Mapping[str, Any],
        *,
        runtime: BaseModelRuntimeConfig,
        reference: Mapping[str, Any] | None,
        recipe: Mapping[str, Any],
    ) -> torch.Tensor:
        return torch.stack(list(self.loss_terms(model, unit, runtime=runtime,
                                                reference=reference, recipe=recipe))).sum()

    def loss_terms(
        self, model: torch.nn.Module, unit: Mapping[str, Any], *,
        runtime: BaseModelRuntimeConfig, reference: Mapping[str, Any] | None,
        recipe: Mapping[str, Any],
    ) -> Iterator[torch.Tensor]:
        """Backpropagate each pair before allocating the next pair's graphs."""
        if reference is None:
            raise ValueError("DPO requires frozen reference log probabilities")
        values = reference.get(str(unit["record_id"]))
        if not isinstance(values, list) or len(values) != len(unit["edges"]):
            raise ValueError("DPO reference values do not match the preference group")
        for edge, fixed in zip(unit["edges"], values):
            chosen = _branch_logp(model, edge["chosen"], runtime).unsqueeze(0)
            rejected = _branch_logp(model, edge["rejected"], runtime).unsqueeze(0)
            outcome = dpo_loss(
                chosen,
                rejected,
                torch.tensor([fixed["chosen"]], device=chosen.device),
                torch.tensor([fixed["rejected"]], device=rejected.device),
                beta=float(recipe["dpo"]["beta"]),
            )
            yield outcome["losses"].mean() * float(edge["weight"]) * unit.get("loss_weight", 1.0)
            del chosen, rejected, outcome


def adapter_for_algorithm(algorithm: str) -> TrainingAdapter:
    if algorithm == "sft":
        return SftAdapter()
    if algorithm == "dpo":
        return DpoAdapter()
    raise ValueError(f"Unsupported training algorithm: {algorithm}")
