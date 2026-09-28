from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sts1_llm_policy.data.card_pick_metrics import (
    WeightedRewardChoice,
)

from sts1_llm_policy.env.combat_snapshot import resign_snapshot

from .card_profiles import CARD_DEMAND_PROFILES


OPENING_REWARD_COUNT = 3
ACT_2_TARGET_START_REWARD = 8


@dataclass(frozen=True)
class CapabilityScore:
    hidden_order_seed: int
    win_rate: float
    victory_ending_hp_mean: float
    mean_evaluation: float
    suggested_action_id: str

    def ordering_key(self) -> tuple[float, float, float]:
        return (
            self.win_rate,
            self.victory_ending_hp_mean,
            self.mean_evaluation,
        )


def reward_state_snapshot(
    reward: Mapping[str, Any],
    choice_id: str,
    *,
    promote_to_act_2: bool = False,
    boss_relic_id: str | None = None,
) -> dict[str, Any]:
    """Build and sign one counterfactual snapshot without mutating reward state."""
    state = reward.get("reward_state")
    choices = reward.get("choices")
    reward_id = reward.get("reward_id")
    if not isinstance(state, Mapping) or not isinstance(choices, list):
        raise ValueError("Malformed native reward state")
    if isinstance(reward_id, bool) or not isinstance(reward_id, int):
        raise ValueError("Malformed native reward id")
    deck = state.get("deck")
    relics = state.get("relics")
    if not isinstance(deck, list) or not isinstance(relics, list):
        raise ValueError("Malformed native reward deck or relics")
    snapshot: dict[str, Any] = {
        "schema_version": state.get("combat_snapshot_schema", "combat_snapshot_v1"),
        "character": "IRONCLAD",
        "source_reward_seed": int(state["seed"]),
        "source_reward_id": reward_id,
        "ascension": int(state["ascension"]),
        "act": int(state["act"]),
        "floor": int(state["floor"]),
        "current_hp": int(state["current_hp"]),
        "max_hp": int(state["max_hp"]),
        "deck": deepcopy(deck),
        "relics": [
            {"id": str(item["id"]), "counter": int(item.get("counter", 0))}
            for item in relics
            if isinstance(item, Mapping)
        ],
    }
    if choice_id.startswith("CARD_"):
        selected = next(
            (
                item
                for item in choices
                if isinstance(item, Mapping) and item.get("choice") == choice_id
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"Unknown reward card choice: {choice_id}")
        snapshot["deck"].append({
            "enum_id": str(selected["enum_id"]),
            "name": str(selected["name"]),
            "string_id": str(selected["string_id"]),
            "upgraded": bool(selected["upgraded"]),
        })
        if snapshot["schema_version"] == "combat_snapshot_v2" and selected["string_id"] == "Searing Blow":
            snapshot["deck"][-1]["upgrade_count"] = selected["upgrade_count"]
    elif choice_id == "SINGING_BOWL":
        if reward.get("can_singing_bowl") is not True:
            raise ValueError("Singing Bowl choice is unavailable")
        snapshot["current_hp"] += 2
        snapshot["max_hp"] += 2
    elif choice_id != "SKIP":
        raise ValueError(f"Unsupported reward choice: {choice_id}")

    if promote_to_act_2:
        if not boss_relic_id:
            raise ValueError("Act 2 promotion requires a boss relic")
        snapshot["act"] = 2
        snapshot["floor"] = 17
        if boss_relic_id not in {item["id"] for item in snapshot["relics"]}:
            snapshot["relics"].append({"id": boss_relic_id, "counter": 0})
    return resign_snapshot(snapshot)


def opening_weighted_argmax(
    distribution: Sequence[WeightedRewardChoice],
    *,
    damage_multiplier: float,
) -> tuple[WeightedRewardChoice, dict[str, float]]:
    if damage_multiplier <= 0:
        raise ValueError("damage_multiplier must be positive")
    cards = [item for item in distribution if item.card is not None]
    if not cards:
        raise ValueError("Opening forced pick has no card candidate")
    scores = {
        item.choice_id: item.probability * (
            damage_multiplier
            if "attack" in CARD_DEMAND_PROFILES[item.card.enum_id].traits
            else 1.0
        )
        for item in cards
    }
    selected = sorted(cards, key=lambda item: (-scores[item.choice_id], item.choice_id))[0]
    return selected, {key: round(value, 8) for key, value in scores.items()}
