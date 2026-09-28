from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from statistics import mean
from typing import Any, Literal

from sts1_llm_policy.data.card_pick_metrics import (
    MassNormalizedCardRewardPicker,
    OfferedCard,
    WeightedRewardChoice,
    SqliteCardRateBackend, act_rate_metric,
)

from sts1_llm_policy.env.simulator_env import StsLightspeedEnv

from .card_profiles import CARD_DEMAND_PROFILES
from .reward_support import (
    ACT_2_TARGET_START_REWARD,
    OPENING_REWARD_COUNT,
    CapabilityScore,
    opening_weighted_argmax,
    reward_state_snapshot,
)


COUNTERFACTUAL_PICKER_V2_VERSION = "counterfactual_boss_reward_picker_v2"
StrategicTier = Literal["premium", "strong", "reserve", "unprotected"]
CardPersistence = Literal["power", "exhaust", "reusable"]


@dataclass(frozen=True)
class StrategicPickRates:
    overall: float
    first: float
    duplicate: float
    sample_size: int

    def validate(self) -> None:
        if not all(0.0 <= value <= 1.0 for value in (
            self.overall, self.first, self.duplicate
        )):
            raise ValueError("Strategic pick rates must stay inside [0, 1]")
        if self.sample_size <= 0:
            raise ValueError("Strategic pick rates require positive sample size")


class StrategicRateTable:
    def __init__(self, backend: SqliteCardRateBackend) -> None:
        self.backend = backend
        self.cache: dict[tuple[str, bool, int], StrategicPickRates] = {}

    def __call__(self, enum_id: str, upgraded: bool, act: int) -> StrategicPickRates:
        key = (enum_id, upgraded, act)
        if key not in self.cache:
            overall = self.backend.lookup(enum_id, upgraded, "pick_rate")
            first = self.backend.lookup(enum_id, upgraded, act_rate_metric(act, duplicate=False))
            duplicate = self.backend.lookup(enum_id, upgraded, act_rate_metric(act, duplicate=True))
            if any(item is None or item.probability is None for item in (overall, first, duplicate)):
                raise ValueError(f"Missing strategic pick rates for {key}")
            assert overall is not None and first is not None and duplicate is not None
            self.cache[key] = StrategicPickRates(
                overall=overall.probability,
                first=first.probability,
                duplicate=duplicate.probability,
                sample_size=min(overall.sample_size, first.sample_size, duplicate.sample_size),
            )
        return self.cache[key]


class TeacherCapabilityEvaluatorV2:
    def __init__(self, env: StsLightspeedEnv, *, combat_seed: int, protocol: Mapping[str, Any]) -> None:
        self.env = env
        self.combat_seed = combat_seed
        self.protocol = protocol
        self.search_calls = 0
        self.search_calls_by_offset: Counter[int] = Counter()

    def __call__(self, snapshot: Mapping[str, Any], scenario_id: str, combat_seed_offset: int) -> Sequence[CapabilityScore]:
        result: list[CapabilityScore] = []
        for hidden_seed in self.protocol["hidden_order_seeds"]:
            self.env.reset(scenario_id, self.combat_seed + combat_seed_offset, combat_snapshot=snapshot)
            search = self.env.search_model_actions(
                simulations=int(self.protocol["search_budget"]),
                search_seed=int(self.protocol["search_seed"]),
                hidden_order_seed=int(hidden_seed),
            ).native_result
            selected = next(item for item in search.root_actions if item.action_id == search.suggested_action_id)
            if selected.win_rate is None or selected.mean_evaluation is None:
                raise ValueError("Suggested Teacher action has no search evidence")
            result.append(CapabilityScore(
                hidden_order_seed=int(hidden_seed),
                win_rate=selected.win_rate,
                victory_ending_hp_mean=selected.victory_ending_hp_mean or 0.0,
                mean_evaluation=selected.mean_evaluation,
                suggested_action_id=selected.action_id,
            ))
            self.search_calls += 1
            self.search_calls_by_offset[combat_seed_offset] += 1
        return tuple(result)


StrategicRateLookup = Callable[[str, bool, int], StrategicPickRates]
StrategicBranchEvaluator = Callable[
    [Mapping[str, Any], str, int], Sequence[CapabilityScore]
]


@dataclass(frozen=True)
class StrategicCardPrior:
    tier: StrategicTier
    persistence: CardPersistence
    intrinsic: float
    first_rate: float
    duplicate_rate: float
    overall_rate: float
    existing_copies: int
    top_fraction: float


@dataclass(frozen=True)
class StrategicWeights:
    search: float = 0.50
    intrinsic: float = 0.35
    optionality: float = 0.15
    current_target: float = 0.70
    future_target: float = 0.30
    duplicate_penalty: float = 0.08
    premium_override_margin: float = 0.15
    strong_override_margin: float = 0.08

    def validate(self) -> None:
        if round(self.search + self.intrinsic + self.optionality, 8) != 1.0:
            raise ValueError("Strategic utility weights must sum to one")
        if round(self.current_target + self.future_target, 8) != 1.0:
            raise ValueError("Target weights must sum to one")
        if min(
            self.search,
            self.intrinsic,
            self.optionality,
            self.current_target,
            self.future_target,
            self.duplicate_penalty,
            self.premium_override_margin,
            self.strong_override_margin,
        ) < 0.0:
            raise ValueError("Strategic weights cannot be negative")


_EXHAUST_CARDS = frozenset({
    "DISARM",
    "FEED",
    "FIEND_FIRE",
    "HAVOC",
    "IMPERVIOUS",
    "INTIMIDATE",
    "OFFERING",
    "PUMMEL",
    "SEEING_RED",
    "SHOCKWAVE",
})


def card_persistence(enum_id: str, *, upgraded: bool) -> CardPersistence:
    if enum_id not in CARD_DEMAND_PROFILES:
        raise ValueError(f"Missing strategic profile source: {enum_id}")
    if "power" in CARD_DEMAND_PROFILES[enum_id].traits:
        return "power"
    if enum_id == "LIMIT_BREAK":
        return "reusable" if upgraded else "exhaust"
    if enum_id in _EXHAUST_CARDS:
        return "exhaust"
    return "reusable"


def strategic_card_prior(
    enum_id: str,
    *,
    upgraded: bool,
    existing_copies: int,
    rates: StrategicPickRates,
    universe: Mapping[str, StrategicPickRates],
) -> StrategicCardPrior:
    rates.validate()
    if existing_copies < 0:
        raise ValueError("Existing card copies cannot be negative")
    if enum_id not in universe or len(universe) < 2:
        raise ValueError("Strategic rate universe must cover the candidate card")

    def score(card_id: str, card_rates: StrategicPickRates) -> float:
        card_rates.validate()
        copies = existing_copies
        persistence = card_persistence(card_id, upgraded=upgraded)
        if copies == 0:
            return (
                0.60 * card_rates.first + 0.40 * card_rates.overall
                if persistence == "exhaust"
                else card_rates.first
            )
        decay = 0.70 ** max(0, copies - 1)
        return (
            0.60 * card_rates.duplicate * decay + 0.40 * card_rates.overall
            if persistence == "exhaust"
            else card_rates.duplicate * decay
        )

    persistence = card_persistence(enum_id, upgraded=upgraded)
    intrinsic = score(enum_id, rates)
    ordered = sorted(
        universe,
        key=lambda card_id: (-score(card_id, universe[card_id]), card_id),
    )
    top_fraction = (ordered.index(enum_id) + 1) / len(ordered)
    if intrinsic >= 0.65 and top_fraction <= 0.10:
        tier: StrategicTier = "premium"
    elif intrinsic >= 0.40 and top_fraction <= 0.25:
        tier = "strong"
    elif intrinsic >= 0.25 and top_fraction <= 0.40:
        tier = "reserve"
    else:
        tier = "unprotected"
    return StrategicCardPrior(
        tier=tier,
        persistence=persistence,
        intrinsic=round(intrinsic, 8),
        first_rate=rates.first,
        duplicate_rate=rates.duplicate,
        overall_rate=rates.overall,
        existing_copies=existing_copies,
        top_fraction=round(top_fraction, 8),
    )


def _deck_providers(deck_card_ids: Sequence[str]) -> Counter[str]:
    providers: Counter[str] = Counter()
    traits: Counter[str] = Counter()
    for enum_id in deck_card_ids:
        profile = CARD_DEMAND_PROFILES.get(enum_id)
        if profile is None:
            continue
        providers.update(profile.provides)
        traits.update(profile.traits)
    if traits["attack"] >= 5:
        providers["attack_density"] += 1
    if traits["strike"] >= 3:
        providers["strike_density"] += 1
    if len(deck_card_ids) - traits["attack"] <= 4:
        providers["skill_light"] += 1
    if len(deck_card_ids) >= 5:
        providers["exhaust_fuel"] += 1
    return providers


def strategic_optionality(
    enum_id: str,
    *,
    deck_card_ids: Sequence[str],
    allowed_card_ids: Sequence[str],
    remaining_rewards: int,
    total_rewards: int,
) -> dict[str, float]:
    if not 0 <= remaining_rewards <= total_rewards or total_rewards <= 0:
        raise ValueError("Invalid reward horizon")
    profile = CARD_DEMAND_PROFILES[enum_id]
    allowed = tuple(sorted(set(allowed_card_ids)))
    if not allowed:
        raise ValueError("Strategic optionality requires an allowed card pool")
    providers = _deck_providers(deck_card_ids)
    unmet = tuple(req for req in profile.requires if providers[req] <= 0)
    backward_matches = sum(
        bool(CARD_DEMAND_PROFILES[item].provides.intersection(unmet))
        for item in allowed
    )
    forward_matches = sum(
        bool(CARD_DEMAND_PROFILES[item].requires.intersection(profile.provides))
        for item in allowed
    )
    horizon = remaining_rewards / total_rewards
    raw = (
        0.55 * backward_matches / len(allowed)
        + 0.45 * forward_matches / len(allowed)
    )
    optionality = min(1.0, 2.5 * raw * horizon)
    return {
        "horizon": round(horizon, 8),
        "backward_compatibility": round(backward_matches / len(allowed), 8),
        "forward_compatibility": round(forward_matches / len(allowed), 8),
        "optionality": round(optionality, 8),
    }


def _aggregate(scores: Sequence[CapabilityScore]) -> tuple[float, float, float]:
    if not scores:
        raise ValueError("Strategic capability evidence cannot be empty")
    return (
        mean(item.win_rate for item in scores),
        mean(item.victory_ending_hp_mean for item in scores),
        mean(item.mean_evaluation for item in scores),
    )


def _search_components(
    evidence: Mapping[str, Sequence[CapabilityScore]],
    *,
    skip_choice_id: str,
) -> dict[str, dict[str, float | int]]:
    if skip_choice_id not in evidence:
        raise ValueError("Strategic evidence requires an explicit Skip-like branch")
    skip = tuple(evidence[skip_choice_id])
    aggregates = {choice: _aggregate(scores) for choice, scores in evidence.items()}
    ordered = sorted(aggregates, key=lambda choice: (aggregates[choice], choice))
    denominator = max(1, len(ordered) - 1)
    rank = {choice: index / denominator for index, choice in enumerate(ordered)}
    result: dict[str, dict[str, float | int]] = {}
    for choice, scores in evidence.items():
        paired = tuple(scores)
        if len(paired) != len(skip):
            raise ValueError("Every strategic branch must use the same paired count")
        pair_wins = sum(
            item.ordering_key() > baseline.ordering_key()
            for item, baseline in zip(paired, skip, strict=True)
        )
        result[choice] = {
            "pair_wins_vs_skip": pair_wins,
            "paired_count": len(skip),
            "aggregate_rank": round(rank[choice], 8),
            "search_score": round(
                0.60 * pair_wins / len(skip) + 0.40 * rank[choice], 8
            ),
        }
    return result


def select_strategic_choice(
    distribution: Sequence[WeightedRewardChoice],
    current_evidence: Mapping[str, Sequence[CapabilityScore]],
    *,
    future_evidence: Mapping[str, Sequence[CapabilityScore]] | None,
    strategic_rates: Mapping[str, StrategicPickRates],
    strategic_rate_universe: Mapping[bool, Mapping[str, StrategicPickRates]],
    deck_card_ids: Sequence[str],
    allowed_card_ids: Sequence[str],
    reward_index: int,
    total_rewards: int,
    required_pair_wins: int,
    weights: StrategicWeights = StrategicWeights(),
) -> tuple[WeightedRewardChoice, dict[str, Any]]:
    weights.validate()
    cards = [item for item in distribution if item.card is not None]
    skip = next((item for item in distribution if item.card is None), None)
    if skip is None or not cards:
        raise ValueError("Strategic selection requires cards and explicit Skip")
    current = _search_components(current_evidence, skip_choice_id=skip.choice_id)
    future = (
        _search_components(future_evidence, skip_choice_id=skip.choice_id)
        if future_evidence else None
    )
    remaining = total_rewards - reward_index
    counts = Counter(deck_card_ids)
    details: dict[str, Any] = {}
    eligible: list[WeightedRewardChoice] = []
    audit_required: list[str] = []
    for item in cards:
        assert item.card is not None
        enum_id = item.card.enum_id
        if item.choice_id not in strategic_rates:
            raise ValueError(f"Missing strategic rates for {item.choice_id}")
        prior = strategic_card_prior(
            enum_id,
            upgraded=item.card.upgraded,
            existing_copies=counts[enum_id],
            rates=strategic_rates[item.choice_id],
            universe=strategic_rate_universe[item.card.upgraded],
        )
        optionality = strategic_optionality(
            enum_id,
            deck_card_ids=deck_card_ids,
            allowed_card_ids=allowed_card_ids,
            remaining_rewards=remaining,
            total_rewards=total_rewards,
        )
        current_search = float(current[item.choice_id]["search_score"])
        if future is None:
            combined_search = current_search
        else:
            combined_search = (
                weights.current_target * current_search
                + weights.future_target
                * float(future[item.choice_id]["search_score"])
            )
        duplicate_penalty = (
            weights.duplicate_penalty * max(0, counts[enum_id] - 1)
        )
        utility = (
            weights.search * combined_search
            + weights.intrinsic * prior.intrinsic
            + weights.optionality * float(optionality["optionality"])
            - duplicate_penalty
        )
        pair_wins = int(current[item.choice_id]["pair_wins_vs_skip"])
        is_eligible = pair_wins >= required_pair_wins
        if is_eligible:
            eligible.append(item)
        elif prior.tier in {"premium", "strong"}:
            audit_required.append(item.choice_id)
        details[item.choice_id] = {
            "enum_id": enum_id,
            "tier": prior.tier,
            "persistence": prior.persistence,
            "intrinsic": prior.intrinsic,
            "first_rate": prior.first_rate,
            "duplicate_rate": prior.duplicate_rate,
            "overall_rate": prior.overall_rate,
            "existing_copies": prior.existing_copies,
            "top_fraction": prior.top_fraction,
            "current_search": current[item.choice_id],
            "future_search": future[item.choice_id] if future else None,
            "combined_search_score": round(combined_search, 8),
            "optionality": optionality,
            "duplicate_penalty": round(duplicate_penalty, 8),
            "strategic_utility": round(utility, 8),
            "eligible": is_eligible,
        }
    if not eligible:
        return skip, {
            "candidates": details,
            "audit_required_choices": sorted(audit_required),
            "selection_reason": "no_eligible_card",
        }
    selected = sorted(
        eligible,
        key=lambda item: (-details[item.choice_id]["strategic_utility"], item.choice_id),
    )[0]
    selected_utility = float(details[selected.choice_id]["strategic_utility"])
    protected = [
        item
        for item in eligible
        if details[item.choice_id]["tier"] in {"premium", "strong"}
    ]
    override = None
    if protected:
        protected_best = sorted(
            protected,
            key=lambda item: (
                -details[item.choice_id]["strategic_utility"], item.choice_id
            ),
        )[0]
        tier = details[protected_best.choice_id]["tier"]
        margin = (
            weights.premium_override_margin
            if tier == "premium"
            else weights.strong_override_margin
        )
        protected_utility = float(
            details[protected_best.choice_id]["strategic_utility"]
        )
        if selected_utility - protected_utility <= margin:
            override = protected_best.choice_id if protected_best != selected else None
            selected = protected_best
    return selected, {
        "candidates": details,
        "audit_required_choices": sorted(audit_required),
        "selection_reason": (
            "protected_prior_override" if override else "maximum_strategic_utility"
        ),
        "protected_override_from": override and next(
            item.choice_id
            for item in eligible
            if float(details[item.choice_id]["strategic_utility"])
            == selected_utility
        ),
    }


def _rate_context(
    distribution: Sequence[WeightedRewardChoice],
    *,
    act: int,
    allowed_card_ids: Sequence[str],
    rate_lookup: StrategicRateLookup,
) -> tuple[
    dict[str, StrategicPickRates],
    dict[bool, dict[str, StrategicPickRates]],
]:
    offered_rates = {
        item.choice_id: rate_lookup(item.card.enum_id, item.card.upgraded, act)
        for item in distribution
        if item.card is not None
    }
    universes = {
        upgraded: {
            enum_id: rate_lookup(enum_id, upgraded, act)
            for enum_id in allowed_card_ids
        }
        for upgraded in (False, True)
    }
    return offered_rates, universes


def select_strategic_opening(
    distribution: Sequence[WeightedRewardChoice],
    *,
    strategic_rates: Mapping[str, StrategicPickRates],
    strategic_rate_universe: Mapping[bool, Mapping[str, StrategicPickRates]],
    deck_card_ids: Sequence[str],
    allowed_card_ids: Sequence[str],
    reward_index: int,
    total_rewards: int,
    damage_multiplier: float,
    weights: StrategicWeights,
) -> tuple[WeightedRewardChoice, dict[str, Any]]:
    weighted_selected, weighted = opening_weighted_argmax(
        distribution, damage_multiplier=damage_multiplier
    )
    cards = [item for item in distribution if item.card is not None]
    maximum = max(weighted.values())
    counts = Counter(deck_card_ids)
    details: dict[str, Any] = {}
    for item in cards:
        assert item.card is not None
        prior = strategic_card_prior(
            item.card.enum_id,
            upgraded=item.card.upgraded,
            existing_copies=counts[item.card.enum_id],
            rates=strategic_rates[item.choice_id],
            universe=strategic_rate_universe[item.card.upgraded],
        )
        optionality = strategic_optionality(
            item.card.enum_id,
            deck_card_ids=deck_card_ids,
            allowed_card_ids=allowed_card_ids,
            remaining_rewards=total_rewards - reward_index,
            total_rewards=total_rewards,
        )
        historical_score = weighted[item.choice_id] / maximum
        utility = (
            weights.search * historical_score
            + weights.intrinsic * prior.intrinsic
            + weights.optionality * float(optionality["optionality"])
        )
        details[item.choice_id] = {
            "enum_id": item.card.enum_id,
            "tier": prior.tier,
            "persistence": prior.persistence,
            "historical_damage_weighted_score": weighted[item.choice_id],
            "historical_normalized_score": round(historical_score, 8),
            "intrinsic": prior.intrinsic,
            "top_fraction": prior.top_fraction,
            "existing_copies": prior.existing_copies,
            "optionality": optionality,
            "strategic_utility": round(utility, 8),
        }
    selected = sorted(
        cards,
        key=lambda item: (-details[item.choice_id]["strategic_utility"], item.choice_id),
    )[0]
    return selected, {
        "candidates": details,
        "historical_damage_argmax": weighted_selected.choice_id,
        "selection_reason": "forced_non_skip_maximum_strategic_utility",
    }


def _direct_win_fraction(
    candidate: Sequence[CapabilityScore], baseline: Sequence[CapabilityScore]
) -> float:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("Conflict audit evidence must be paired")
    return sum(
        left.ordering_key() > right.ordering_key()
        for left, right in zip(candidate, baseline, strict=True)
    ) / len(candidate)


class CounterfactualBossRewardPickerV2:
    def __init__(
        self,
        picker: MassNormalizedCardRewardPicker,
        evaluator: StrategicBranchEvaluator,
        rate_lookup: StrategicRateLookup,
        *,
        final_boss_scenario_id: str,
        act_1_boss_scenario_id: str,
        boss_relic_id: str | None,
        allowed_card_ids: frozenset[str],
        epsilon: float,
        total_rewards: int,
        conflict_combat_seed_offsets: Sequence[int],
        opening_damage_multiplier: float = 1.5,
        required_pair_wins: int = 2,
        weights: StrategicWeights = StrategicWeights(),
    ) -> None:
        if total_rewards not in (7, 15):
            raise ValueError("Strategic picker supports seven or fifteen rewards")
        if len(tuple(conflict_combat_seed_offsets)) != 2:
            raise ValueError("Strategic conflicts require exactly two extra seeds")
        weights.validate()
        self.picker = picker
        self.evaluator = evaluator
        self.rate_lookup = rate_lookup
        self.final_boss_scenario_id = final_boss_scenario_id
        self.act_1_boss_scenario_id = act_1_boss_scenario_id
        self.boss_relic_id = boss_relic_id
        self.allowed_card_ids = allowed_card_ids
        self.epsilon = float(epsilon)
        self.total_rewards = total_rewards
        self.conflict_combat_seed_offsets = tuple(conflict_combat_seed_offsets)
        self.opening_damage_multiplier = float(opening_damage_multiplier)
        self.required_pair_wins = int(required_pair_wins)
        self.weights = weights
        self.reward_index = 0

    def __call__(
        self,
        reward: Mapping[str, Any],
        act: int,
        floor: int,
        rng,
    ) -> tuple[str, dict[str, float], dict[str, Any]]:
        self.reward_index += 1
        choices = reward.get("choices")
        state = reward.get("reward_state")
        if not isinstance(choices, list) or not isinstance(state, Mapping):
            raise ValueError("Malformed native reward")
        deck = state.get("deck")
        if not isinstance(deck, list):
            raise ValueError("Malformed native reward deck")
        offered = []
        for raw in choices:
            if not isinstance(raw, Mapping):
                raise ValueError("Malformed native reward choice")
            if (
                isinstance(raw.get("enum_id"), str)
                and raw["enum_id"] in self.allowed_card_ids
                and isinstance(raw.get("choice"), str)
                and isinstance(raw.get("upgraded"), bool)
            ):
                offered.append(OfferedCard(
                    raw["enum_id"], raw["upgraded"], raw["choice"]
                ))
        if not offered:
            selected = "SINGING_BOWL" if reward.get("can_singing_bowl") else "SKIP"
            return selected, {selected: 1.0}, {
                "version": COUNTERFACTUAL_PICKER_V2_VERSION,
                "reward_index": self.reward_index,
                "selection_rule": "no_allowed_cards",
                "selected_choice": selected,
            }
        deck_ids = [
            str(item["enum_id"])
            for item in deck
            if isinstance(item, Mapping) and isinstance(item.get("enum_id"), str)
        ]
        distribution = self.picker.distribution(
            offered,
            act=act,
            deck_card_ids=deck_ids,
            epsilon=self.epsilon,
            singing_bowl_available=reward.get("can_singing_bowl") is True,
        )
        probabilities = {
            item.choice_id: round(item.probability, 8) for item in distribution
        }
        discarded = rng.random()
        rates, universes = _rate_context(
            distribution,
            act=act,
            allowed_card_ids=tuple(self.allowed_card_ids),
            rate_lookup=self.rate_lookup,
        )
        common = {
            "version": COUNTERFACTUAL_PICKER_V2_VERSION,
            "reward_index": self.reward_index,
            "act": act,
            "floor": floor,
            "base_probabilities": probabilities,
            "discarded_sample_threshold": round(discarded, 8),
        }
        if self.reward_index <= OPENING_REWARD_COUNT:
            selected, details = select_strategic_opening(
                distribution,
                strategic_rates=rates,
                strategic_rate_universe=universes,
                deck_card_ids=deck_ids,
                allowed_card_ids=tuple(self.allowed_card_ids),
                reward_index=self.reward_index,
                total_rewards=self.total_rewards,
                damage_multiplier=self.opening_damage_multiplier,
                weights=self.weights,
            )
            return selected.choice_id, probabilities, {
                **common,
                "selection_rule": "forced_non_skip_strategic_opening",
                "damage_multiplier": self.opening_damage_multiplier,
                **details,
                "selected_choice": selected.choice_id,
            }

        promote = self.reward_index == ACT_2_TARGET_START_REWARD and act == 1
        current_target = (
            self.final_boss_scenario_id
            if self.reward_index >= ACT_2_TARGET_START_REWARD
            else self.act_1_boss_scenario_id
        )
        dual_target = (
            self.final_boss_scenario_id != self.act_1_boss_scenario_id
            and OPENING_REWARD_COUNT < self.reward_index < ACT_2_TARGET_START_REWARD
        )
        current_snapshots = {
            item.choice_id: reward_state_snapshot(
                reward,
                item.choice_id,
                promote_to_act_2=promote,
                boss_relic_id=self.boss_relic_id,
            )
            for item in distribution
        }
        current_evidence = {
            choice_id: tuple(self.evaluator(snapshot, current_target, 0))
            for choice_id, snapshot in current_snapshots.items()
        }
        future_snapshots = None
        future_evidence = None
        if dual_target:
            future_snapshots = {
                item.choice_id: reward_state_snapshot(
                    reward,
                    item.choice_id,
                    promote_to_act_2=True,
                    boss_relic_id=self.boss_relic_id,
                )
                for item in distribution
            }
            future_evidence = {
                choice_id: tuple(
                    self.evaluator(snapshot, self.final_boss_scenario_id, 0)
                )
                for choice_id, snapshot in future_snapshots.items()
            }
        selected, details = select_strategic_choice(
            distribution,
            current_evidence,
            future_evidence=future_evidence,
            strategic_rates=rates,
            strategic_rate_universe=universes,
            deck_card_ids=deck_ids,
            allowed_card_ids=tuple(self.allowed_card_ids),
            reward_index=self.reward_index,
            total_rewards=self.total_rewards,
            required_pair_wins=self.required_pair_wins,
            weights=self.weights,
        )
        provisional = selected
        conflict_rows: list[dict[str, Any]] = []
        for choice_id in details["audit_required_choices"]:
            current_fractions = []
            future_fractions = []
            for offset in self.conflict_combat_seed_offsets:
                candidate_scores = self.evaluator(
                    current_snapshots[choice_id], current_target, offset
                )
                baseline_scores = self.evaluator(
                    current_snapshots[provisional.choice_id], current_target, offset
                )
                current_fractions.append(
                    _direct_win_fraction(candidate_scores, baseline_scores)
                )
                if dual_target and future_snapshots is not None:
                    candidate_future = self.evaluator(
                        future_snapshots[choice_id], self.final_boss_scenario_id, offset
                    )
                    baseline_future = self.evaluator(
                        future_snapshots[provisional.choice_id],
                        self.final_boss_scenario_id,
                        offset,
                    )
                    future_fractions.append(
                        _direct_win_fraction(candidate_future, baseline_future)
                    )
            current_fraction = mean(current_fractions)
            combined = current_fraction
            if future_fractions:
                combined = (
                    self.weights.current_target * current_fraction
                    + self.weights.future_target * mean(future_fractions)
                )
            conflict_rows.append({
                "choice_id": choice_id,
                "against_choice_id": provisional.choice_id,
                "current_target_win_fraction": round(current_fraction, 8),
                "future_target_win_fraction": (
                    round(mean(future_fractions), 8) if future_fractions else None
                ),
                "combined_win_fraction": round(combined, 8),
                "passed": combined >= 0.5,
            })
        passed = [item for item in conflict_rows if item["passed"]]
        if passed:
            winner = sorted(
                passed,
                key=lambda item: (
                    -item["combined_win_fraction"],
                    -details["candidates"][item["choice_id"]]["strategic_utility"],
                    item["choice_id"],
                ),
            )[0]
            selected = next(
                item for item in distribution if item.choice_id == winner["choice_id"]
            )
            details["selection_reason"] = "cross_seed_protected_prior_override"
        return selected.choice_id, probabilities, {
            **common,
            "selection_rule": "strategic_teacher_capability_v2",
            "current_target_scenario_id": current_target,
            "future_target_scenario_id": (
                self.final_boss_scenario_id if dual_target else None
            ),
            "target_weights": {
                "current": self.weights.current_target,
                "future": self.weights.future_target if dual_target else 0.0,
            },
            "promoted_to_act_2_for_evaluation": promote,
            "required_pair_wins": self.required_pair_wins,
            **deepcopy(details),
            "conflict_audit": conflict_rows,
            "selected_choice": selected.choice_id,
        }
