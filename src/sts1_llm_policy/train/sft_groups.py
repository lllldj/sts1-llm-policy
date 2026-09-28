"""State-normalized, explicit candidate answers for action-only SFT."""
import math

from .sft_data import tokenize_sft_record, validate_decision_record


def validate_weight_semantics(records, semantics):
    explicit = semantics.get("state_weight") == "explicit_mean_one_loss_weight"
    if not explicit:
        if any("loss_weight" in r for r in records):
            raise ValueError("State loss weights require explicit manifest semantics")
        return
    masses = semantics.get("source_weights")
    if (not isinstance(masses, dict) or not masses or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in masses.values()) or not math.isclose(math.fsum(masses.values()), 1., abs_tol=1e-12)):
        raise ValueError("Invalid SFT manifest source weights")
    actual = {source: [] for source in masses}
    for record in records:
        source, weight = record.get("supervision_source"), record.get("loss_weight")
        if (source not in actual or isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight <= 0):
            raise ValueError("Invalid SFT source or state loss weight")
        actual[source].append(weight)
    if any(not weights or not math.isclose(math.fsum(weights) / len(records), masses[source], rel_tol=1e-9)
           for source, weights in actual.items()):
        raise ValueError("SFT state loss weights differ from declared source masses")


def prepare_group(tokenizer, record, *, dataset_id, max_sequence_tokens):
    loss_weight = record.get("loss_weight", 1.0)
    if (isinstance(loss_weight, bool) or not isinstance(loss_weight, (int, float))
            or not math.isfinite(loss_weight) or loss_weight <= 0):
        raise ValueError("SFT state loss_weight must be positive and finite")
    if record.get("observation_version") != "observation_v7":
        raise ValueError("SFT group requires a bound observation_v7 prompt")
    legal = validate_decision_record(record)
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("SFT group requires nonempty candidates")
    ids = [c.get("action_id") for c in candidates]
    if len(set(ids)) != len(ids) or not set(ids) <= set(legal):
        raise ValueError("SFT candidate actions must be unique and legal")
    weights = [c.get("weight") for c in candidates]
    if (any(isinstance(w, bool) or not isinstance(w, (int, float))
            or not math.isfinite(w) or w <= 0 for w in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("SFT candidate weights must be positive, finite and sum to one")
    prepared = []
    for candidate in candidates:
        branch = {**record, "teacher_action_id": candidate["action_id"]}
        tokens = tokenize_sft_record(tokenizer, branch, max_sequence_tokens=max_sequence_tokens,
                                    artifact_id=dataset_id)
        prepared.append({**candidate, "tokenized": tokens})
    return {"record_id": record["record_id"], "candidates": prepared,
            **({"loss_weight": float(loss_weight)} if "loss_weight" in record else {}),
            "sequence_tokens_max": max(len(c["tokenized"]["input_ids"]) for c in prepared)}
