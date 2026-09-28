"""Native reward operations for the frozen single-combat input panel."""
from __future__ import annotations

import random
from typing import Any, Protocol, Sequence

from .generation_strategies import (
    CardSelectionContext, GenerationStrategies, RewardSelector, RouteContext,
)


class RewardGenerationClient(Protocol):
    def reward_reset(self, seed: int, **kwargs: object) -> dict[str, Any]: ...
    def sample_card_reward(self, *, room: str, floor: int) -> dict[str, Any]: ...
    def apply_card_reward_choice(self, reward_id: int, choice: str) -> dict[str, Any]: ...
    def advance_reward_act(self, act: int) -> dict[str, Any]: ...
    def obtain_reward_relic(self, relic_id: str) -> dict[str, Any]: ...
    def remove_reward_card(self, card_id: str) -> dict[str, Any]: ...
    def upgrade_reward_card(self, deck_index: int) -> dict[str, Any]: ...
    def export_combat_snapshot(self) -> dict[str, Any]: ...



def generate_frozen_reward_snapshot(
    client: RewardGenerationClient,
    *,
    reward_seed: int,
    ascension: int,
    target_act: int,
    stage: str,
    act_1_regular_rewards: int,
    act_2_rewards: int,
    ordinary_relic_ids: Sequence[str],
    boss_relic_id: str | None,
    cumulative_upgrades: int,
    cumulative_removals: int,
    picker_seed: int,
    reward_selector: RewardSelector,
    strategies: GenerationStrategies,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the selected route and decisions through native reward mutations."""
    context = RouteContext(
        target_act, stage, act_1_regular_rewards, act_2_rewards, ordinary_relic_ids,
        boss_relic_id, cumulative_upgrades, cumulative_removals,
    )
    # TODO(new dataset): give each policy its own stream. Preserve the shared
    # picker/upgrade stream and historical draw order for the frozen panel.
    rng = random.Random(picker_seed)
    reward_events: list[dict[str, Any]] = []
    topology_events: list[dict[str, Any]] = []
    upgrade_count = removal_count = 0
    for step in strategies.route(context, rng):
        kind, act = step["kind"], int(step["act"])
        if kind == "reset":
            client.reward_reset(reward_seed, ascension=ascension, act=act, relics=step["relics"])
        elif kind == "reward":
            floor, room = int(step["floor"]), str(step["room"])
            reward = client.sample_card_reward(room=room, floor=floor)
            choice, probabilities, picker_details = reward_selector(reward, act, floor, rng)
            before = reward["reward_state"]
            applied = client.apply_card_reward_choice(reward["reward_id"], choice)
            reward_events.append({
                "act": act, "floor": floor, "room": room, "source": step["source"],
                "floor_reward_index": reward["floor_reward_index"],
                "reward_id": reward["reward_id"], "offered_cards": reward["choices"],
                "choice": choice, "choice_probabilities": probabilities,
                "picker_details": picker_details, "deck_size_before": len(before["deck"]),
                "deck_size_after": len(applied["reward_state"]["deck"]),
                "max_hp_after": applied["reward_state"]["max_hp"],
            })
        elif kind in {"ordinary_relic", "boss_relic"}:
            client.obtain_reward_relic(step["relic_id"])
            topology_events.append({"kind": kind, "act": act, "checkpoint": step["checkpoint"],
                                    "relic_id": step["relic_id"]})
        elif kind == "upgrade":
            state = client.export_combat_snapshot()["combat_snapshot"]
            selection = CardSelectionContext(act, step["checkpoint"], upgrade_count + 1)
            deck_index, priority = strategies.card_upgrade(state["deck"], selection, rng)
            result = client.upgrade_reward_card(deck_index)
            topology_events.append({
                "kind": "campfire_upgrade", "act": act, "checkpoint": step["checkpoint"],
                "priority": priority, "card_before": result["upgraded_card_before"],
                "card_after": result["upgraded_card_after"],
            })
            upgrade_count += 1
        elif kind == "remove":
            state = client.export_combat_snapshot()["combat_snapshot"]
            selection = CardSelectionContext(act, step["checkpoint"], removal_count + 1)
            card_id, reason = strategies.card_remove(state["deck"], selection, rng)
            result = client.remove_reward_card(card_id)
            topology_events.append({
                "kind": "card_removal", "act": act, "checkpoint": step["checkpoint"],
                "card_id": card_id, "removed_card": result["removed_card"], "reason": reason,
            })
            removal_count += 1
        elif kind == "advance_act":
            client.advance_reward_act(act)
        else:
            raise ValueError(f"Unknown route operation: {kind!r}")
    snapshot = client.export_combat_snapshot().get("combat_snapshot")
    if not isinstance(snapshot, dict):
        raise RuntimeError("Native bridge did not export combat_snapshot_v1")
    return snapshot, reward_events, topology_events
