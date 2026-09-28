from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from .action_equivalence import ModelActionSpace
from .action_schema import CanonicalAction


EXPECTED_SEARCH_SOURCE = "BattleScumSearcher2"
EXPECTED_SELECTION_RULE = (
    "max_win_rate_then_victory_hp_then_visits_then_mean_evaluation_then_action_id"
)
EXPECTED_OBJECTIVE = "upstream_battle_scum_evaluate_end_state_v1"


@dataclass(frozen=True)
class RootActionSearchStats:
    action_id: str
    kind: str
    visits: int
    evaluation_sum: float
    evaluation_square_sum: float
    mean_evaluation: float | None
    terminal_wins: int
    terminal_losses: int
    win_rate: float | None
    ending_hp_sum: int
    ending_hp_mean: float | None
    victory_ending_hp_sum: int
    victory_ending_hp_mean: float | None


@dataclass(frozen=True)
class BattleScumSearchResult:
    decision_id: int
    simulations: int
    search_seed: int
    search_quality_normalization: str
    root_action_coverage_policy: str
    minimum_root_action_visits: int
    hidden_order_seed: int | None
    hidden_order_policy: str
    hidden_order_fingerprint: str
    root_simulation_count: int
    root_actions: tuple[RootActionSearchStats, ...]
    suggested_action_id: str
    best_sequence_first_action_id: str | None
    best_sequence_length: int
    best_action_value: float
    min_action_value: float
    best_outcome_player_hp: int
    elapsed_ms: float
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class ModelActionSearchStats:
    model_action_id: str
    native_action_ids: tuple[str, ...]
    visits: int
    evaluation_sum: float
    evaluation_square_sum: float
    mean_evaluation: float | None
    terminal_wins: int
    terminal_losses: int
    win_rate: float | None
    ending_hp_sum: int
    ending_hp_mean: float | None
    victory_ending_hp_sum: int
    victory_ending_hp_mean: float | None


@dataclass(frozen=True)
class ProjectedBattleScumSearchResult:
    native_result: BattleScumSearchResult
    model_actions: tuple[ModelActionSearchStats, ...]
    suggested_model_action_id: str
    best_sequence_first_model_action_id: str | None


def parse_battle_scum_search_result(
    response: Mapping[str, Any],
    *,
    current_decision_id: int,
    current_native_action_ids: Sequence[str],
    requested_hidden_order_seed: int | None = None,
    requested_public_state_seed: int | None = None,
    requested_root_action_coverage: bool = False,
    requested_minimum_root_action_visits: int = 0,
) -> BattleScumSearchResult:
    """Validate privileged native-search evidence for the current decision."""

    if response.get("source") != EXPECTED_SEARCH_SOURCE:
        raise ValueError("Search response source mismatch")
    if response.get("selection_rule") != EXPECTED_SELECTION_RULE:
        raise ValueError("Search response selection rule mismatch")
    if response.get("objective") != EXPECTED_OBJECTIVE:
        raise ValueError("Search response objective mismatch")

    decision_id = _integer(response.get("decision_id"), "decision_id")
    if decision_id != current_decision_id:
        raise ValueError("Search response decision_id is stale")
    simulations = _positive_integer(response.get("simulations"), "simulations")
    search_seed = _integer(response.get("search_seed"), "search_seed")
    if search_seed < 0 or search_seed > 2**32 - 1:
        raise ValueError("search_seed must be an unsigned 32-bit integer")
    search_quality_normalization = _string(
        response.get("search_quality_normalization"),
        "search_quality_normalization",
    )
    if search_quality_normalization != "min_max_guarded_v1":
        raise ValueError("Search response quality normalization mismatch")
    root_action_coverage_policy = _string(
        response.get(
            "root_action_coverage_policy",
            "legacy_ucb_no_root_coverage_guarantee",
        ),
        "root_action_coverage_policy",
    )
    effective_minimum_root_visits = max(
        requested_minimum_root_action_visits,
        1 if requested_root_action_coverage else 0,
    )
    expected_coverage_policy = (
        "minimum_root_edge_visits_before_ucb_v1"
        if effective_minimum_root_visits > 1
        else (
            "visit_every_root_edge_before_ucb_v1"
            if effective_minimum_root_visits == 1
            else "legacy_ucb_no_root_coverage_guarantee"
        )
    )
    if root_action_coverage_policy != expected_coverage_policy:
        raise ValueError("Search response root-action coverage policy mismatch")
    minimum_root_action_visits = _integer(
        response.get("minimum_root_action_visits", effective_minimum_root_visits),
        "minimum_root_action_visits",
    )
    if minimum_root_action_visits != effective_minimum_root_visits:
        raise ValueError("Search response minimum root-action visits mismatch")
    hidden_order_seed_raw = response.get("hidden_order_seed")
    hidden_order_seed = None
    if hidden_order_seed_raw is not None:
        hidden_order_seed = _integer(hidden_order_seed_raw, "hidden_order_seed")
        if hidden_order_seed < 0 or hidden_order_seed > 2**32 - 1:
            raise ValueError("hidden_order_seed must be an unsigned 32-bit integer")
    if hidden_order_seed != requested_hidden_order_seed:
        raise ValueError("Search response hidden_order_seed mismatch")
    hidden_order_policy = _string(
        response.get("hidden_order_policy", "exact_privileged_order_v1"),
        "hidden_order_policy",
    )
    if response.get("public_state_seed") != requested_public_state_seed:
        raise ValueError("Search public-state seed mismatch")
    expected_hidden_policy = (
        "fresh_future_rng_known_draw_top_v1" if requested_public_state_seed is not None else
        "shuffle_current_draw_pile_and_reseed_shuffle_rng_v1"
        if hidden_order_seed is not None
        else "exact_privileged_order_v1"
    )
    if hidden_order_policy != expected_hidden_policy:
        raise ValueError("Search response hidden-order policy mismatch")
    hidden_order_fingerprint = response.get("hidden_order_fingerprint", "")
    if not isinstance(hidden_order_fingerprint, str):
        raise ValueError("hidden_order_fingerprint must be a string")
    root_simulation_count = _positive_integer(
        response.get("root_simulation_count"),
        "root_simulation_count",
    )
    if root_simulation_count != simulations:
        raise ValueError("Root simulation count must equal the requested budget")

    native_action_ids = tuple(current_native_action_ids)
    if not native_action_ids or len(native_action_ids) != len(set(native_action_ids)):
        raise ValueError("Current native action IDs must be non-empty and unique")
    native_action_set = set(native_action_ids)

    raw_actions = response.get("root_actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("Search response root_actions must be a non-empty array")
    parsed_actions: list[RootActionSearchStats] = []
    seen_action_ids: set[str] = set()
    visit_total = 0
    for index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, Mapping):
            raise ValueError(f"root_actions[{index}] must be an object")
        action_id = _string(raw_action.get("action_id"), f"root_actions[{index}].action_id")
        if action_id not in native_action_set:
            raise ValueError("Search returned an action outside the current legal set")
        if action_id in seen_action_ids:
            raise ValueError("Search returned duplicate root action IDs")
        seen_action_ids.add(action_id)
        kind = _string(raw_action.get("kind"), f"root_actions[{index}].kind")
        if kind not in {"PLAY_CARD", "END_TURN", "SELECT_CARD"}:
            raise ValueError("Search returned an unsupported root action kind")
        visits = _integer(raw_action.get("visits"), f"root_actions[{index}].visits")
        if visits < 0:
            raise ValueError("Root action visits must not be negative")
        evaluation_sum = _finite_number(
            raw_action.get("evaluation_sum"),
            f"root_actions[{index}].evaluation_sum",
        )
        evaluation_square_sum = _nonnegative_number(
            raw_action.get("evaluation_square_sum"),
            f"root_actions[{index}].evaluation_square_sum",
        )
        terminal_wins = _integer(
            raw_action.get("terminal_wins"),
            f"root_actions[{index}].terminal_wins",
        )
        terminal_losses = _integer(
            raw_action.get("terminal_losses"),
            f"root_actions[{index}].terminal_losses",
        )
        if terminal_wins < 0 or terminal_losses < 0:
            raise ValueError("Terminal outcome counts must not be negative")
        if terminal_wins + terminal_losses != visits:
            raise ValueError("Terminal outcome counts must equal root visits")
        ending_hp_sum = _integer(
            raw_action.get("ending_hp_sum"),
            f"root_actions[{index}].ending_hp_sum",
        )
        victory_ending_hp_sum = _integer(
            raw_action.get("victory_ending_hp_sum"),
            f"root_actions[{index}].victory_ending_hp_sum",
        )
        if ending_hp_sum < 0 or victory_ending_hp_sum < 0:
            raise ValueError("Terminal HP sums must not be negative")
        raw_mean = raw_action.get("mean_evaluation")
        if visits == 0:
            if raw_mean is not None:
                raise ValueError("Unvisited root actions must have null mean evaluation")
            mean_evaluation = None
        else:
            mean_evaluation = _finite_number(
                raw_mean,
                f"root_actions[{index}].mean_evaluation",
            )
            expected_mean = evaluation_sum / visits
            if not math.isclose(mean_evaluation, expected_mean, rel_tol=1e-12):
                raise ValueError("Root action mean evaluation is inconsistent")
        win_rate = _optional_mean(
            raw_action.get("win_rate"),
            terminal_wins,
            visits,
            f"root_actions[{index}].win_rate",
        )
        ending_hp_mean = _optional_mean(
            raw_action.get("ending_hp_mean"),
            ending_hp_sum,
            visits,
            f"root_actions[{index}].ending_hp_mean",
        )
        victory_ending_hp_mean = _optional_mean(
            raw_action.get("victory_ending_hp_mean"),
            victory_ending_hp_sum,
            terminal_wins,
            f"root_actions[{index}].victory_ending_hp_mean",
        )
        visit_total += visits
        parsed_actions.append(
            RootActionSearchStats(
                action_id=action_id,
                kind=kind,
                visits=visits,
                evaluation_sum=evaluation_sum,
                evaluation_square_sum=evaluation_square_sum,
                mean_evaluation=mean_evaluation,
                terminal_wins=terminal_wins,
                terminal_losses=terminal_losses,
                win_rate=win_rate,
                ending_hp_sum=ending_hp_sum,
                ending_hp_mean=ending_hp_mean,
                victory_ending_hp_sum=victory_ending_hp_sum,
                victory_ending_hp_mean=victory_ending_hp_mean,
            )
        )
    if visit_total != root_simulation_count:
        raise ValueError("Root action visits must sum to root_simulation_count")

    suggested_action_id = _string(
        response.get("suggested_action_id"),
        "suggested_action_id",
    )
    if suggested_action_id not in seen_action_ids:
        raise ValueError("Suggested action is absent from root search actions")
    suggested = next(
        action for action in parsed_actions if action.action_id == suggested_action_id
    )
    if suggested.visits <= 0:
        raise ValueError("Suggested action must have at least one visit")

    best_sequence_first = response.get("best_sequence_first_action_id")
    if best_sequence_first is not None:
        best_sequence_first = _string(
            best_sequence_first,
            "best_sequence_first_action_id",
        )
        if best_sequence_first not in seen_action_ids:
            raise ValueError("Best sequence starts with an unknown root action")

    best_sequence_length = _integer(
        response.get("best_sequence_length"),
        "best_sequence_length",
    )
    if best_sequence_length < 0:
        raise ValueError("best_sequence_length must not be negative")
    if (best_sequence_length == 0) != (best_sequence_first is None):
        raise ValueError("Best sequence length/action fields are inconsistent")
    best_action_value = _finite_number(
        response.get("best_action_value"),
        "best_action_value",
    )
    min_action_value = _finite_number(
        response.get("min_action_value"),
        "min_action_value",
    )
    if best_action_value < min_action_value:
        raise ValueError("Best action value must not be below minimum action value")
    best_outcome_player_hp = _integer(
        response.get("best_outcome_player_hp"),
        "best_outcome_player_hp",
    )
    if best_outcome_player_hp < 0:
        raise ValueError("best_outcome_player_hp must not be negative")

    privileged = response.get("privileged_state")
    if not isinstance(privileged, Mapping) or any(
        privileged.get(field) is not True
        for field in (
            "battle_context_copy",
            "ordered_draw_pile",
            "future_rng_state",
        )
    ):
        raise ValueError("Search response must declare privileged state access")
    semantics = response.get("transition_semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("Search response transition semantics are missing")
    if semantics.get("bridge_corrections_applied_inside_rollouts") is not True:
        raise ValueError("Search response transition semantics are unexpected")
    if semantics.get("lagavulin_rejected") is not False:
        raise ValueError("Search response unexpectedly rejects Lagavulin")
    if semantics.get("upgraded_disarm_decks_rejected") is not False:
        raise ValueError("Search response unexpectedly rejects Disarm+")
    corrections = semantics.get("corrections_applied")
    if not isinstance(corrections, Mapping) or any(
        _integer(corrections.get(field), f"transition_semantics.corrections_applied.{field}") < 0
        for field in (
            "lagavulin_natural_wake",
            "upgraded_disarm",
            "red_slaver_entangle_once",
            "philosopher_bronze_orb_strength",
            "burning_blood_victory_heal",
        )
    ):
        raise ValueError("Search response correction counts are invalid")

    return BattleScumSearchResult(
        decision_id=decision_id,
        simulations=simulations,
        search_seed=search_seed,
        search_quality_normalization=search_quality_normalization,
        root_action_coverage_policy=root_action_coverage_policy,
        minimum_root_action_visits=minimum_root_action_visits,
        hidden_order_seed=hidden_order_seed,
        hidden_order_policy=hidden_order_policy,
        hidden_order_fingerprint=hidden_order_fingerprint,
        root_simulation_count=root_simulation_count,
        root_actions=tuple(parsed_actions),
        suggested_action_id=suggested_action_id,
        best_sequence_first_action_id=best_sequence_first,
        best_sequence_length=best_sequence_length,
        best_action_value=best_action_value,
        min_action_value=min_action_value,
        best_outcome_player_hp=best_outcome_player_hp,
        elapsed_ms=_nonnegative_number(response.get("elapsed_ms"), "elapsed_ms"),
        raw_response=deepcopy(dict(response)),
    )


def project_battle_scum_search_result(
    result: BattleScumSearchResult,
    *,
    native_action_ids: Sequence[str],
    canonical_actions: Sequence[CanonicalAction],
    action_space: ModelActionSpace,
) -> ProjectedBattleScumSearchResult:
    """Project native root statistics onto model-visible action classes."""

    if len(native_action_ids) != len(canonical_actions):
        raise ValueError("Native and canonical action arrays must align")
    native_to_internal: dict[str, str] = {}
    for native_action_id, canonical_action in zip(
        native_action_ids,
        canonical_actions,
        strict=True,
    ):
        if native_action_id in native_to_internal:
            raise ValueError("Native action IDs must be unique")
        native_to_internal[native_action_id] = canonical_action.action_id

    grouped: dict[str, list[RootActionSearchStats]] = {}
    native_to_model: dict[str, str] = {}
    for stats in result.root_actions:
        internal_action_id = native_to_internal.get(stats.action_id)
        if internal_action_id is None:
            raise ValueError("Search action is absent from canonical action mapping")
        model_action_id = action_space.model_id_for_internal_action(
            internal_action_id
        )
        if model_action_id is None:
            raise ValueError("Search action is absent from model action classes")
        grouped.setdefault(model_action_id, []).append(stats)
        native_to_model[stats.action_id] = model_action_id

    missing_model_actions = set(action_space.action_ids) - set(grouped)
    if missing_model_actions:
        raise ValueError(
            "Search did not cover every model action class: "
            + ", ".join(sorted(missing_model_actions))
        )

    model_actions: list[ModelActionSearchStats] = []
    for model_action_id in action_space.action_ids:
        members = grouped[model_action_id]
        visits = sum(member.visits for member in members)
        evaluation_sum = sum(member.evaluation_sum for member in members)
        evaluation_square_sum = sum(
            member.evaluation_square_sum for member in members
        )
        terminal_wins = sum(member.terminal_wins for member in members)
        terminal_losses = sum(member.terminal_losses for member in members)
        ending_hp_sum = sum(member.ending_hp_sum for member in members)
        victory_ending_hp_sum = sum(
            member.victory_ending_hp_sum for member in members
        )
        model_actions.append(
            ModelActionSearchStats(
                model_action_id=model_action_id,
                native_action_ids=tuple(member.action_id for member in members),
                visits=visits,
                evaluation_sum=evaluation_sum,
                evaluation_square_sum=evaluation_square_sum,
                mean_evaluation=(evaluation_sum / visits if visits else None),
                terminal_wins=terminal_wins,
                terminal_losses=terminal_losses,
                win_rate=(terminal_wins / visits if visits else None),
                ending_hp_sum=ending_hp_sum,
                ending_hp_mean=(ending_hp_sum / visits if visits else None),
                victory_ending_hp_sum=victory_ending_hp_sum,
                victory_ending_hp_mean=(
                    victory_ending_hp_sum / terminal_wins
                    if terminal_wins
                    else None
                ),
            )
        )

    if native_to_model.get(result.suggested_action_id) is None:
        raise ValueError("Suggested search action has no model action class")
    suggested_model_action_id = min(
        model_actions,
        key=lambda item: (
            -(item.win_rate if item.win_rate is not None else -1.0),
            -(
                item.victory_ending_hp_mean
                if item.victory_ending_hp_mean is not None
                else -1.0
            ),
            -item.visits,
            -(
                item.mean_evaluation
                if item.mean_evaluation is not None
                else -math.inf
            ),
            item.model_action_id,
        ),
    ).model_action_id
    best_sequence_first_model_action_id = None
    if result.best_sequence_first_action_id is not None:
        best_sequence_first_model_action_id = native_to_model.get(
            result.best_sequence_first_action_id
        )
        if best_sequence_first_model_action_id is None:
            raise ValueError("Best search sequence has no model action class")

    return ProjectedBattleScumSearchResult(
        native_result=result,
        model_actions=tuple(model_actions),
        suggested_model_action_id=suggested_model_action_id,
        best_sequence_first_model_action_id=(
            best_sequence_first_model_action_id
        ),
    )


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _positive_integer(value: object, path: str) -> int:
    result = _integer(value, path)
    if result <= 0:
        raise ValueError(f"{path} must be positive")
    return result


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _nonnegative_number(value: object, path: str) -> float:
    result = _finite_number(value, path)
    if result < 0:
        raise ValueError(f"{path} must not be negative")
    return result


def _optional_mean(
    value: object,
    numerator: int,
    denominator: int,
    path: str,
) -> float | None:
    if denominator == 0:
        if value is not None:
            raise ValueError(f"{path} must be null when its denominator is zero")
        return None
    result = _finite_number(value, path)
    expected = numerator / denominator
    if not math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{path} is inconsistent with its sum and count")
    return result
