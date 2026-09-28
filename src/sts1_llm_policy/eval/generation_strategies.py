"""Small, explicitly selected policies for combat-input construction.

Keep new variants as functions here. The runner applies decisions to native state.
The continuous runner supplies separate strategy streams; the frozen panel keeps
its historical shared RNG and draw order.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import random
from statistics import mean
from typing import Any

from sts1_llm_policy.data.card_pick_metrics import MassNormalizedCardRewardPicker
from sts1_llm_policy.env.combat_snapshot import resign_snapshot

from .reward_support import CapabilityScore
from .counterfactual_reward_picker_v2 import (
    CounterfactualBossRewardPickerV2, StrategicBranchEvaluator, StrategicRateLookup,
    StrategicWeights,
)


ACT_1_REWARD_FLOORS = (1, 3, 5, 7, 9, 11, 13, 15)
ACT_2_REWARD_FLOORS = (18, 20, 22, 24, 26, 28, 30, 32)

RewardSelector = Callable[
    [Mapping[str, Any], int, int, random.Random],
    tuple[str, dict[str, float], dict[str, Any]],
]


@dataclass(frozen=True)
class CardSelectionContext:
    act: int
    stage: str
    selection_index: int  # One-based count within this route and operation type.
    combat_snapshot: Mapping[str, Any] | None = None
    evaluator: StrategicBranchEvaluator | None = None
    decision_recorder: Callable[[Mapping[str, Any]], None] | None = None
    evaluation_targets: Sequence[Mapping[str, Any]] = ()


@dataclass(frozen=True)
class RouteContext:
    target_act: int
    stage: str
    act_1_regular_rewards: int
    act_2_rewards: int
    ordinary_relic_ids: Sequence[str]
    boss_relic_id: str | None
    cumulative_upgrades: int
    cumulative_removals: int
    operations: Sequence[str] = ()
    encounter_pools: Mapping[str, Sequence[str]] | None = None
    relic_pool: Sequence[str] = ()
    combat_floors: Sequence[int] = ()
    initial_relics: Sequence[str] = ()
    encounter_seed: int | None = None
    relic_seed: int | None = None


def teacher_counterfactual(
    *, picker: MassNormalizedCardRewardPicker, evaluator: StrategicBranchEvaluator,
    rate_lookup: StrategicRateLookup, spec: Mapping[str, Any],
    allowed_card_ids: frozenset[str], epsilon: float, total_rewards: int,
    protocol: Mapping[str, Any],
) -> RewardSelector:
    """Create a fresh stateful picker for one route; reuse the frozen algorithm."""
    return CounterfactualBossRewardPickerV2(
        picker, evaluator, rate_lookup,
        final_boss_scenario_id=str(spec["final_boss_scenario_id"]),
        act_1_boss_scenario_id=str(spec["act_1_boss_scenario_id"]),
        boss_relic_id=spec.get("boss_relic_id"), allowed_card_ids=allowed_card_ids,
        epsilon=epsilon, total_rewards=total_rewards,
        conflict_combat_seed_offsets=tuple(protocol["conflict_combat_seed_offsets"]),
        opening_damage_multiplier=float(protocol["opening_damage_multiplier"]),
        required_pair_wins=int(protocol["required_pair_wins_vs_skip"]),
        weights=StrategicWeights(**protocol["strategic_weights"]),
    )


def starter_alternating(
    deck: Sequence[Mapping[str, Any]], context: CardSelectionContext, rng: random.Random,
) -> tuple[str, str]:
    """Return the native card ID and reason, preserving the old removal order."""
    order = ("Strike_R", "Defend_R", "Strike_R", "Defend_R")
    if not 1 <= context.selection_index <= len(order):
        raise ValueError("Starter removal schedule supports at most four removals")
    return order[context.selection_index - 1], "abstracted_shop_or_event_removal"


_COST_REDUCTION_UPGRADES = frozenset({
    "BLOOD_FOR_BLOOD", "BODY_SLAM", "DARK_EMBRACE", "ENTRENCH",
    "HAVOC", "SEEING_RED", "BARRICADE", "CORRUPTION",
})
_FUNCTIONAL_VALUE_UPGRADES = frozenset({
    "BASH", "BERSERK", "BLOODLETTING", "BRUTALITY", "DISARM",
    "DOUBLE_TAP", "EVOLVE", "FEEL_NO_PAIN", "FLEX", "INFLAME",
    "INTIMIDATE", "LIMIT_BREAK", "METALLICIZE", "OFFERING",
    "POMMEL_STRIKE", "RAGE", "RUPTURE", "SHOCKWAVE", "SPOT_WEAKNESS",
    "UPPERCUT",
})
_STARTER_FALLBACK_UPGRADES = frozenset({"STRIKE_RED", "DEFEND_RED"})


def value_priority(
    deck: Sequence[Mapping[str, Any]], context: CardSelectionContext, rng: random.Random,
) -> tuple[int, str]:
    """Return a current deck index and the frozen v3 upgrade priority."""
    candidates: list[tuple[int, int]] = []
    for index, card in enumerate(deck):
        enum_id = card.get("enum_id")
        if not isinstance(enum_id, str) or card.get("upgraded") is True:
            continue
        if enum_id in {"ASCENDERS_BANE", "BURN"}:
            continue
        if enum_id in _COST_REDUCTION_UPGRADES:
            tier = 1
        elif enum_id in _FUNCTIONAL_VALUE_UPGRADES:
            tier = 2
        elif enum_id in _STARTER_FALLBACK_UPGRADES:
            tier = 4
        else:
            tier = 3
        candidates.append((index, tier))
    if not candidates:
        raise ValueError(f"No upgradeable card remains at act={context.act} stage={context.stage}")
    best_tier = min(tier for _, tier in candidates)
    tied = [index for index, tier in candidates if tier == best_tier]
    return rng.choice(tied), {
        1: "cost_reduction", 2: "functional_value", 3: "damage_block_efficiency",
        4: "starter_strike_defend_fallback",
    }[best_tier]


def teacher_combat_priority(
    deck: Sequence[Mapping[str, Any]], context: CardSelectionContext, rng: random.Random,
) -> tuple[int, str]:
    """Choose the upgrade with the strongest paired Teacher target-panel evidence."""
    del rng  # Candidate comparison is paired and deterministic.
    if (
        context.combat_snapshot is None
        or context.evaluator is None
        or not context.evaluation_targets
    ):
        raise ValueError("Teacher upgrade requires a snapshot and an evaluation target panel")
    candidates: list[
        tuple[
            int,
            str,
            str,
            tuple[float, float, float],
            tuple[tuple[dict[str, Any], tuple[CapabilityScore, ...]], ...],
        ]
    ] = []
    seen: set[tuple[str, str, int]] = set()
    for index, card in enumerate(deck):
        enum_id = card.get("enum_id")
        string_id = card.get("string_id", card.get("id"))
        counted_searing = (
            enum_id == "SEARING_BLOW"
            and context.combat_snapshot.get("schema_version") == "combat_snapshot_v2"
        )
        if (
            not isinstance(enum_id, str)
            or not isinstance(string_id, str)
            or (card.get("upgraded") is True and not counted_searing)
            or enum_id in {"ASCENDERS_BANE", "BURN"}
        ):
            continue
        count = card["upgrade_count"] if counted_searing else int(card.get("upgraded") is True)
        if counted_searing and (type(count) is not int or not 0 <= count < 32767):
            raise ValueError("Searing Blow upgrade_count is invalid or cannot be incremented")
        identity = (enum_id, string_id, count)
        if identity in seen:
            continue
        seen.add(identity)
        snapshot = deepcopy(dict(context.combat_snapshot))
        snapshot_deck = snapshot.get("deck")
        if not isinstance(snapshot_deck, list) or index >= len(snapshot_deck):
            raise ValueError("Teacher upgrade snapshot deck does not match current deck")
        snapshot_deck[index] = deepcopy(snapshot_deck[index])
        snapshot_deck[index]["upgraded"] = True
        if counted_searing:
            snapshot_deck[index]["upgrade_count"] = count + 1
        signed = resign_snapshot(snapshot)
        target_evidence: list[tuple[dict[str, Any], tuple[CapabilityScore, ...]]] = []
        evidence: list[CapabilityScore] = []
        for target in context.evaluation_targets:
            scenario_id = target.get("scenario_id")
            combat_seed = target.get("combat_seed")
            if (
                not isinstance(scenario_id, str)
                or isinstance(combat_seed, bool)
                or not isinstance(combat_seed, int)
            ):
                raise ValueError("Teacher upgrade evaluation target is malformed")
            scores = tuple(context.evaluator(signed, scenario_id, combat_seed))
            if not scores:
                raise ValueError("Teacher upgrade produced no capability evidence")
            evidence.extend(scores)
            target_evidence.append((dict(target), scores))
        score = (
            mean(item.win_rate for item in evidence),
            mean(item.victory_ending_hp_mean for item in evidence),
            mean(item.mean_evaluation for item in evidence),
        )
        candidates.append((index, enum_id, string_id, score, tuple(target_evidence)))
    if not candidates:
        raise ValueError(
            f"No upgradeable card remains at act={context.act} stage={context.stage}"
        )
    selected = sorted(
        candidates,
        key=lambda item: (-item[3][0], -item[3][1], -item[3][2], item[1], item[0]),
    )[0]
    if context.decision_recorder is not None:
        context.decision_recorder({
            "schema_version": "teacher_upgrade_decision_v1",
            "selected_deck_index": selected[0],
            "selected_enum_id": selected[1],
            "candidates": [
                {
                    "deck_index": index,
                    "enum_id": enum_id,
                    "string_id": string_id,
                    "mean_score": {
                        "win_rate": score[0],
                        "victory_ending_hp": score[1],
                        "evaluation": score[2],
                    },
                    "target_evidence": [
                        {
                            **target,
                            "search_evidence": [
                                {
                                    "hidden_order_seed": item.hidden_order_seed,
                                    "win_rate": item.win_rate,
                                    "victory_ending_hp": item.victory_ending_hp_mean,
                                    "evaluation": item.mean_evaluation,
                                    "suggested_action_id": item.suggested_action_id,
                                }
                                for item in target_scores
                            ],
                        }
                        for target, target_scores in evidence
                    ],
                }
                for index, enum_id, string_id, score, evidence in candidates
            ],
        })
    return selected[0], "teacher_multi_target_capability"


def configured_act1_survival(
    context: RouteContext, rng: random.Random,
) -> Iterable[dict[str, Any]]:
    """Resolve one configured Act-1 route with independent shuffled category bags."""
    if context.target_act != 1 or not context.operations:
        raise ValueError("Configured survival route requires Act 1 operations")
    if context.encounter_pools is None:
        raise ValueError("Configured survival route requires encounter pools")
    operations = tuple(context.operations)
    combat_count = sum(item.startswith("combat:") for item in operations)
    if combat_count != len(context.combat_floors):
        raise ValueError("Configured combat floors must match combat operations")

    if context.encounter_seed is None or context.relic_seed is None:
        raise ValueError("Configured survival route requires separate encounter and relic seeds")
    encounter_rng = random.Random(context.encounter_seed)
    relic_rng = random.Random(context.relic_seed)
    bags: dict[str, list[str]] = {}

    def draw(category: str) -> str:
        raw_pool = context.encounter_pools.get(category)
        if not raw_pool or len(set(raw_pool)) != len(raw_pool):
            raise ValueError(f"Encounter pool {category!r} must be nonempty and unique")
        if not bags.get(category):
            bags[category] = [str(item).lower() for item in raw_pool]
            encounter_rng.shuffle(bags[category])
        return bags[category].pop()

    relics = list(map(str, context.relic_pool))
    if len(set(relics)) != len(relics):
        raise ValueError("Configured relic pool contains duplicates")
    relic_rng.shuffle(relics)
    if len(set(context.initial_relics)) != len(context.initial_relics):
        raise ValueError("Configured initial relics contain duplicates")
    yield {
        "kind": "reset", "act": 1,
        "relics": [{"id": str(relic_id)} for relic_id in context.initial_relics],
    }
    combat_index = 0
    last_family: str | None = None
    for route_index, operation in enumerate(operations, start=1):
        checkpoint = f"route_step_{route_index:02d}"
        if operation.startswith("combat:"):
            family = operation.split(":", 1)[1]
            if family not in {"normal_weak", "normal_strong", "elite", "boss"}:
                raise ValueError(f"Unknown combat family: {family!r}")
            combat_index += 1
            last_family = family
            yield {
                "kind": "combat", "act": 1, "checkpoint": checkpoint,
                "combat_index": combat_index,
                "floor": int(context.combat_floors[combat_index - 1]),
                "encounter_family": family, "scenario_id": draw(family),
            }
        elif operation == "card_pick":
            if last_family not in {"normal_weak", "normal_strong", "elite"}:
                raise ValueError("Card pick must explicitly follow a non-Boss combat")
            yield {
                "kind": "reward", "act": 1, "checkpoint": checkpoint,
                "floor": int(context.combat_floors[combat_index - 1]),
                "room": "ELITE" if last_family == "elite" else "MONSTER",
                "source": "explicit_post_combat_card_reward",
            }
        elif operation == "upgrade":
            yield {"kind": "upgrade", "act": 1, "checkpoint": checkpoint}
        elif operation == "remove":
            yield {"kind": "remove", "act": 1, "checkpoint": checkpoint}
        elif operation == "random_relic":
            if not relics:
                raise ValueError("Configured route exhausted its relic pool")
            yield {
                "kind": "ordinary_relic", "act": 1, "checkpoint": checkpoint,
                "relic_id": relics.pop(),
            }
        elif operation.startswith("heal:"):
            amount = int(operation.split(":", 1)[1])
            if amount <= 0:
                raise ValueError("Configured healing must be positive")
            yield {
                "kind": "heal", "act": 1, "checkpoint": checkpoint,
                "amount": amount,
            }
        else:
            raise ValueError(f"Unknown configured route operation: {operation!r}")


def act_topology(context: RouteContext, rng: random.Random) -> Iterable[dict[str, Any]]:
    """Yield the existing route in order, without consuming strategy RNG.

    Stage-specific counts belong to this policy, not to the operation executor.
    A future native random-route policy can yield the same operation vocabulary.
    """
    c = context
    if c.target_act not in (1, 2) or c.stage not in {"entry", "mid", "pre-boss"}:
        raise ValueError("v3 target_act/stage is invalid")
    expected_a1 = {"entry": 3, "mid": 5, "pre-boss": 7}[c.stage] if c.target_act == 1 else 7
    expected_a2 = {"entry": 0, "mid": 4, "pre-boss": 7}[c.stage] if c.target_act == 2 else 0
    expected_upgrades = {(1, "entry"): 1, (1, "mid"): 2, (1, "pre-boss"): 4,
                         (2, "entry"): 4, (2, "mid"): 5, (2, "pre-boss"): 7}
    expected_removals = {(1, "entry"): 0, (1, "mid"): 1, (1, "pre-boss"): 2,
                        (2, "entry"): 2, (2, "mid"): 3, (2, "pre-boss"): 4}
    expected_ordinary = {(1, "entry"): 1, (1, "mid"): 2, (1, "pre-boss"): 3,
                        (2, "entry"): 3, (2, "mid"): 4, (2, "pre-boss"): 5}
    key = (c.target_act, c.stage)
    if c.act_1_regular_rewards != expected_a1 or c.act_2_rewards != expected_a2:
        raise ValueError("v3 reward count disagrees with stage topology")
    if (c.cumulative_upgrades != expected_upgrades[key]
            or c.cumulative_removals != expected_removals[key]):
        raise ValueError("v3 upgrade/removal count disagrees with stage topology")
    if len(c.ordinary_relic_ids) != expected_ordinary[key]:
        raise ValueError("v3 ordinary relic count disagrees with stage topology")
    if (c.target_act == 2) != (c.boss_relic_id is not None):
        raise ValueError("v3 Act 2 snapshots require exactly one boss relic")

    yield {"kind": "reset", "act": 1, "relics": [{"id": "BURNING_BLOOD"}]}
    relic_count = upgrade_count = removal_count = 0

    def checkpoint_steps(act: int, stage: str, relics: int, upgrades: int, removals: int):
        nonlocal relic_count, upgrade_count, removal_count
        while relic_count < relics:
            yield {"kind": "ordinary_relic", "act": act, "checkpoint": stage,
                   "relic_id": c.ordinary_relic_ids[relic_count]}
            relic_count += 1
        while upgrade_count < upgrades:
            yield {"kind": "upgrade", "act": act, "checkpoint": stage}
            upgrade_count += 1
        while removal_count < removals:
            yield {"kind": "remove", "act": act, "checkpoint": stage}
            removal_count += 1

    a1_checkpoints = {3: ("entry", 1, 1, 0), 5: ("mid", 2, 2, 1), 7: ("pre-boss", 3, 4, 2)}
    for index, floor in enumerate(ACT_1_REWARD_FLOORS[:c.act_1_regular_rewards], start=1):
        yield {"kind": "reward", "act": 1, "floor": floor,
               "room": "ELITE" if index == 4 else "MONSTER", "source": "regular_or_elite"}
        if index in a1_checkpoints:
            stage, relics, upgrades, removals = a1_checkpoints[index]
            yield from checkpoint_steps(1, stage, min(relics, len(c.ordinary_relic_ids)),
                                        min(upgrades, c.cumulative_upgrades), min(removals, c.cumulative_removals))
    if c.target_act == 2:
        yield {"kind": "reward", "act": 1, "floor": 16, "room": "BOSS", "source": "act_1_boss_rare_reward"}
        yield {"kind": "boss_relic", "act": 1, "checkpoint": "post-boss", "relic_id": c.boss_relic_id}
        yield {"kind": "advance_act", "act": 2}
        a2_checkpoints = {4: ("mid", 4, 5, 3), 7: ("pre-boss", 5, 7, 4)}
        for index, floor in enumerate(ACT_2_REWARD_FLOORS[:c.act_2_rewards], start=1):
            yield {"kind": "reward", "act": 2, "floor": floor,
                   "room": "ELITE" if index == 4 else "MONSTER", "source": "regular_or_elite"}
            if index in a2_checkpoints:
                yield from checkpoint_steps(2, *a2_checkpoints[index])


@dataclass(frozen=True)
class GenerationStrategies:
    card_pick: Callable[..., RewardSelector]
    card_remove: Callable[[Sequence[Mapping[str, Any]], CardSelectionContext, random.Random], tuple[str, str]]
    card_upgrade: Callable[[Sequence[Mapping[str, Any]], CardSelectionContext, random.Random], tuple[int, str]]
    route: Callable[[RouteContext, random.Random], Iterable[dict[str, Any]]]


STRATEGIES = {
    "card_pick": {"teacher_counterfactual": teacher_counterfactual},
    "card_remove": {"starter_alternating": starter_alternating},
    "card_upgrade": {
        "value_priority": value_priority,
        "teacher_combat_priority": teacher_combat_priority,
    },
    "route": {
        "act_topology": act_topology,
        "configured_act1_survival": configured_act1_survival,
    },
}


def resolve_generation_strategies(names: Mapping[str, str]) -> GenerationStrategies:
    if not isinstance(names, Mapping) or set(names) != set(STRATEGIES):
        raise ValueError("Select card_pick, card_remove, card_upgrade and route strategies")
    selected = {}
    for role, choices in STRATEGIES.items():
        name = names[role]
        if not isinstance(name, str) or name not in choices:
            raise ValueError(f"Unknown {role} strategy: {name!r}")
        selected[role] = choices[name]
    return GenerationStrategies(**selected)
