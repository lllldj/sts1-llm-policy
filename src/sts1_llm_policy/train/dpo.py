from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sts1_llm_policy.train.sft_data import IGNORE_INDEX


def validate_preference_weight_semantics(records, semantics):
    """Require explicit manifest opt-in for cross-state loss weights."""
    explicit = semantics.get("state_weight") == "explicit_mean_one_loss_weight"
    if not explicit:
        if any("loss_weight" in r for r in records):
            raise ValueError("DPO state loss weights require explicit manifest semantics")
        return
    weights = [r.get("loss_weight") for r in records]
    if (not weights or any(isinstance(w, bool) or not isinstance(w, (int, float))
                          or not math.isfinite(w) or w <= 0 for w in weights)
            or not math.isclose(math.fsum(weights), len(weights), rel_tol=1e-9)):
        raise ValueError("Explicit DPO state loss weights must be positive, finite and have mean one")


def response_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed causal response log-probabilities and response token counts."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape or logits.shape[1] < 2:
        raise ValueError("Logit and label sequence dimensions are incompatible")

    # Reference inference and autocast training must normalize in the same dtype.
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels.ne(ignore_index)
    token_counts = loss_mask.sum(dim=-1)
    if bool(token_counts.eq(0).any().item()):
        raise ValueError("Every sequence must contain at least one response token")

    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    selected_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_log_probs = selected_logits - torch.logsumexp(
        shifted_logits, dim=-1
    )
    sequence_log_probs = (token_log_probs * loss_mask).sum(dim=-1)
    return sequence_log_probs, token_counts


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Compute the standard reference-relative DPO objective without label smoothing."""

    shapes = {
        tuple(policy_chosen_logps.shape),
        tuple(policy_rejected_logps.shape),
        tuple(reference_chosen_logps.shape),
        tuple(reference_rejected_logps.shape),
    }
    if len(shapes) != 1:
        raise ValueError("All DPO log-probability tensors must have the same shape")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("DPO beta must be positive and finite")

    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    reference_log_ratios = reference_chosen_logps - reference_rejected_logps
    preference_logits = beta * (policy_log_ratios - reference_log_ratios)
    losses = -F.logsigmoid(preference_logits)
    return {
        "losses": losses,
        "preference_logits": preference_logits,
        "policy_log_ratios": policy_log_ratios,
        "reference_log_ratios": reference_log_ratios,
        "chosen_rewards": beta
        * (policy_chosen_logps - reference_chosen_logps),
        "rejected_rewards": beta
        * (policy_rejected_logps - reference_rejected_logps),
    }
