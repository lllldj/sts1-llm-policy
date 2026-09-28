from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .simulator_parity import load_supported_mechanics


PREFLIGHT_SCHEMA_VERSION = "real_game_capture_preflight_v1"


class RealGamePreflightError(RuntimeError):
    """Raised before policy execution when a real-game capture is unsafe."""

    def __init__(self, report: Mapping[str, object]) -> None:
        self.report = dict(report)
        scenario_id = self.report.get("scenario_id")
        mismatches = self.report.get("mismatches")
        details: list[str] = []

        if isinstance(mismatches, list):
            for mismatch in mismatches[:5]:
                if not isinstance(mismatch, Mapping):
                    continue
                details.append(
                    f"{mismatch.get('field')}: expected "
                    f"{mismatch.get('expected')!r}, observed "
                    f"{mismatch.get('observed')!r}"
                )

        suffix = "; ".join(details) or "unknown mismatch"
        super().__init__(
            f"P0 clean preflight rejected scenario {scenario_id!r}: {suffix}"
        )


@dataclass(frozen=True)
class CaptureScenarioConfig:
    monster_count: int
    allowed_monster_ids: tuple[str, ...]
    duplicates_allowed: bool


@dataclass(frozen=True)
class P0CleanCaptureConfig:
    profile_id: str
    evidence_class: str
    mechanics_manifest: Path
    requirements: Mapping[str, object]
    scenarios: Mapping[str, CaptureScenarioConfig]
    deck: tuple[tuple[str, int, int], ...]
    character: str
    ascension: int
    act: int
    floor: int
    starter_relic: str


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} {path}: {error}") from error

    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")

    return value


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_p0_clean_capture_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> P0CleanCaptureConfig:
    config_path = Path(path).resolve()
    raw = _json_object(config_path, "real-game capture config")

    if raw.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("Unsupported real-game capture preflight schema")

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    manifest_value = _required_str(
        raw.get("mechanics_manifest"),
        "mechanics_manifest",
    )
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = load_supported_mechanics(manifest_path)

    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("mechanics manifest scope must be an object")

    raw_deck = manifest.get("deck")
    if not isinstance(raw_deck, list) or not raw_deck:
        raise ValueError("mechanics manifest deck must be a non-empty array")

    deck: list[tuple[str, int, int]] = []
    for index, value in enumerate(raw_deck):
        if not isinstance(value, dict):
            raise ValueError(f"mechanics manifest deck[{index}] must be an object")
        deck.append(
            (
                _required_str(value.get("card_id"), f"deck[{index}].card_id"),
                _required_int(value.get("count"), f"deck[{index}].count"),
                _required_int(
                    value.get("upgrades"),
                    f"deck[{index}].upgrades",
                ),
            )
        )

    raw_scenarios = raw.get("scenarios")
    if not isinstance(raw_scenarios, dict) or not raw_scenarios:
        raise ValueError("scenarios must be a non-empty object")

    scenarios: dict[str, CaptureScenarioConfig] = {}
    for scenario_id, value in raw_scenarios.items():
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("scenario IDs must be non-empty strings")
        if not isinstance(value, dict):
            raise ValueError(f"scenarios.{scenario_id} must be an object")
        monster_count = _required_int(
            value.get("monster_count"),
            f"scenarios.{scenario_id}.monster_count",
        )
        if monster_count <= 0:
            raise ValueError(
                f"scenarios.{scenario_id}.monster_count must be positive"
            )
        monster_ids = value.get("allowed_monster_ids")
        if (
            not isinstance(monster_ids, list)
            or not monster_ids
            or not all(isinstance(item, str) and item for item in monster_ids)
        ):
            raise ValueError(
                f"scenarios.{scenario_id}.allowed_monster_ids must be a non-empty "
                "string array"
            )
        duplicates_allowed = value.get("duplicates_allowed", False)
        if not isinstance(duplicates_allowed, bool):
            raise ValueError(
                f"scenarios.{scenario_id}.duplicates_allowed must be a boolean"
            )
        if monster_count > len(monster_ids) and not duplicates_allowed:
            raise ValueError(
                f"scenarios.{scenario_id} needs duplicate monster IDs but "
                "duplicates_allowed is false"
            )
        scenarios[scenario_id] = CaptureScenarioConfig(
            monster_count=monster_count,
            allowed_monster_ids=tuple(monster_ids),
            duplicates_allowed=duplicates_allowed,
        )

    manifest_encounters = manifest.get("encounters")
    if not isinstance(manifest_encounters, list):
        raise ValueError("mechanics manifest encounters must be an array")
    manifest_scenarios: dict[str, tuple[str, ...]] = {}
    for index, value in enumerate(manifest_encounters):
        if not isinstance(value, dict):
            raise ValueError(
                f"mechanics manifest encounters[{index}] must be an object"
            )
        manifest_scenario_id = _required_str(
            value.get("scenario_id"),
            f"encounters[{index}].scenario_id",
        )
        manifest_monsters = value.get("monsters")
        if (
            not isinstance(manifest_monsters, list)
            or not manifest_monsters
            or not all(
                isinstance(item, str) and item
                for item in manifest_monsters
            )
        ):
            raise ValueError(
                f"encounters[{index}].monsters must be a string array"
            )
        manifest_scenarios[manifest_scenario_id] = tuple(manifest_monsters)
    if set(scenarios) != set(manifest_scenarios):
        raise ValueError("capture scenarios must match the mechanics manifest")
    for scenario_id, scenario in scenarios.items():
        if set(scenario.allowed_monster_ids) != set(
            manifest_scenarios[scenario_id]
        ):
            raise ValueError(
                f"capture scenario {scenario_id} monster types must match "
                "the mechanics manifest"
            )

    requirements = raw.get("requirements")
    if not isinstance(requirements, dict):
        raise ValueError("requirements must be an object")

    for field in (
        "turn",
        "player_current_hp",
        "player_max_hp",
        "player_block",
        "player_energy",
        "hand_size",
        "draw_pile_size",
    ):
        _required_int(requirements.get(field), f"requirements.{field}")
    _required_str(requirements.get("room_type"), "requirements.room_type")
    for field in (
        "require_no_player_powers",
        "require_empty_potion_slots",
    ):
        if not isinstance(requirements.get(field), bool):
            raise ValueError(f"requirements.{field} must be a boolean")
    empty_piles = requirements.get("empty_piles")
    if (
        not isinstance(empty_piles, list)
        or not empty_piles
        or not all(isinstance(item, str) and item for item in empty_piles)
    ):
        raise ValueError("requirements.empty_piles must be a string array")

    evidence_class = _required_str(
        raw.get("evidence_class"),
        "evidence_class",
    )
    if evidence_class != "parity_clean":
        raise ValueError("P0 clean capture evidence_class must be parity_clean")

    return P0CleanCaptureConfig(
        profile_id=_required_str(raw.get("profile_id"), "profile_id"),
        evidence_class=evidence_class,
        mechanics_manifest=manifest_path,
        requirements=dict(requirements),
        scenarios=scenarios,
        deck=tuple(deck),
        character=_required_str(scope.get("character"), "scope.character"),
        ascension=_required_int(scope.get("ascension"), "scope.ascension"),
        act=_required_int(scope.get("act"), "scope.act"),
        floor=_required_int(scope.get("floor"), "scope.floor"),
        starter_relic=_required_str(
            scope.get("starter_relic"),
            "scope.starter_relic",
        ),
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _card_counter_rows(
    value: Counter[tuple[object, object]],
) -> list[dict[str, object]]:
    rows = [
        {
            "card_id": key[0],
            "upgrades": key[1],
            "count": count,
        }
        for key, count in value.items()
    ]
    return sorted(
        rows,
        key=lambda row: (
            str(row["card_id"]),
            str(row["upgrades"]),
        ),
    )


def evaluate_p0_clean_preflight(
    raw_state: Mapping[str, object],
    config: P0CleanCaptureConfig,
    scenario_id: str,
) -> dict[str, object]:
    if scenario_id not in config.scenarios:
        raise ValueError(f"Unknown P0 clean scenario: {scenario_id}")

    mismatches: list[dict[str, object]] = []

    def check(field: str, expected: object, observed: object) -> None:
        if observed != expected:
            mismatches.append(
                {
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                }
            )

    def read_list(
        value: Mapping[str, object],
        key: str,
        field: str,
    ) -> list[object]:
        observed = value.get(key)
        if not isinstance(observed, list):
            mismatches.append(
                {
                    "field": field,
                    "expected": "array",
                    "observed": observed,
                }
            )
            return []
        return observed

    game_state = _mapping(raw_state.get("game_state"))
    combat = _mapping(game_state.get("combat_state"))
    player = _mapping(combat.get("player"))
    requirements = config.requirements

    check("ready_for_command", True, raw_state.get("ready_for_command"))
    check("in_game", True, raw_state.get("in_game"))
    check("game_state.room_phase", "COMBAT", game_state.get("room_phase"))
    check(
        "game_state.action_phase",
        "WAITING_ON_USER",
        game_state.get("action_phase"),
    )
    check("game_state.class", config.character, game_state.get("class"))
    check(
        "game_state.ascension_level",
        config.ascension,
        game_state.get("ascension_level"),
    )
    check("game_state.act", config.act, game_state.get("act"))
    check("game_state.floor", config.floor, game_state.get("floor"))
    check(
        "game_state.current_hp",
        requirements.get("player_current_hp"),
        game_state.get("current_hp"),
    )
    check(
        "game_state.max_hp",
        requirements.get("player_max_hp"),
        game_state.get("max_hp"),
    )
    check(
        "game_state.room_type",
        requirements.get("room_type"),
        game_state.get("room_type"),
    )
    check("combat_state.turn", requirements.get("turn"), combat.get("turn"))
    check(
        "combat_state.player.current_hp",
        requirements.get("player_current_hp"),
        player.get("current_hp"),
    )
    check(
        "combat_state.player.max_hp",
        requirements.get("player_max_hp"),
        player.get("max_hp"),
    )
    check(
        "combat_state.player.block",
        requirements.get("player_block"),
        player.get("block"),
    )
    check(
        "combat_state.player.energy",
        requirements.get("player_energy"),
        player.get("energy"),
    )

    if requirements.get("require_no_player_powers") is True:
        check(
            "combat_state.player.powers",
            [],
            read_list(player, "powers", "combat_state.player.powers.type"),
        )

    monsters = read_list(combat, "monsters", "combat_state.monsters.type")
    monster_ids = [
        _mapping(monster).get("id")
        for monster in monsters
    ]
    scenario = config.scenarios[scenario_id]
    check(
        "combat_state.monsters.count",
        scenario.monster_count,
        len(monster_ids),
    )
    unexpected_monsters = [
        monster_id
        for monster_id in monster_ids
        if monster_id not in scenario.allowed_monster_ids
    ]
    check(
        "combat_state.monsters[].id",
        {
            "allowed": list(scenario.allowed_monster_ids),
            "duplicates_allowed": scenario.duplicates_allowed,
        },
        (
            {"unexpected": unexpected_monsters}
            if unexpected_monsters
            else {
                "allowed": list(scenario.allowed_monster_ids),
                "duplicates_allowed": scenario.duplicates_allowed,
            }
        ),
    )
    if not scenario.duplicates_allowed and len(monster_ids) != len(
        set(monster_ids)
    ):
        mismatches.append(
            {
                "field": "combat_state.monsters.duplicates",
                "expected": False,
                "observed": True,
            }
        )

    deck = read_list(game_state, "deck", "game_state.deck.type")
    observed_deck = Counter(
        (
            _mapping(card).get("id"),
            _mapping(card).get("upgrades"),
        )
        for card in deck
    )
    expected_deck = Counter(
        {
            (card_id, upgrades): count
            for card_id, count, upgrades in config.deck
        }
    )
    check(
        "game_state.deck",
        _card_counter_rows(expected_deck),
        _card_counter_rows(observed_deck),
    )

    relic_ids = [
        _mapping(relic).get("id")
        for relic in read_list(
            game_state,
            "relics",
            "game_state.relics.type",
        )
    ]
    check("game_state.relics[].id", [config.starter_relic], relic_ids)

    hand = read_list(combat, "hand", "combat_state.hand.type")
    draw_pile = read_list(
        combat,
        "draw_pile",
        "combat_state.draw_pile.type",
    )
    check("combat_state.hand.length", requirements.get("hand_size"), len(hand))
    check(
        "combat_state.draw_pile.length",
        requirements.get("draw_pile_size"),
        len(draw_pile),
    )

    empty_piles = requirements.get("empty_piles")
    if isinstance(empty_piles, list):
        for pile_name in empty_piles:
            if isinstance(pile_name, str):
                check(
                    f"combat_state.{pile_name}",
                    [],
                    read_list(
                        combat,
                        pile_name,
                        f"combat_state.{pile_name}.type",
                    ),
                )

    combat_cards = hand + draw_pile
    for pile_name in ("discard_pile", "exhaust_pile"):
        pile = combat.get(pile_name)
        if isinstance(pile, list):
            combat_cards.extend(pile)
    observed_combat_deck = Counter(
        (
            _mapping(card).get("id"),
            _mapping(card).get("upgrades"),
        )
        for card in combat_cards
    )
    check(
        "combat_state.card_piles",
        _card_counter_rows(expected_deck),
        _card_counter_rows(observed_combat_deck),
    )

    if requirements.get("require_empty_potion_slots") is True:
        potion_ids = [
            _mapping(potion).get("id")
            for potion in read_list(
                game_state,
                "potions",
                "game_state.potions.type",
            )
        ]
        actual_potions = [
            potion_id
            for potion_id in potion_ids
            if potion_id != "Potion Slot"
        ]
        check("game_state.actual_potions", [], actual_potions)

    observed = {
        "seed": game_state.get("seed"),
        "character": game_state.get("class"),
        "ascension": game_state.get("ascension_level"),
        "act": game_state.get("act"),
        "floor": game_state.get("floor"),
        "turn": combat.get("turn"),
        "player_current_hp": player.get("current_hp"),
        "player_max_hp": player.get("max_hp"),
        "monster_ids": monster_ids,
        "deck": _card_counter_rows(observed_deck),
        "relic_ids": relic_ids,
    }
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "profile_id": config.profile_id,
        "evidence_class": config.evidence_class,
        "scenario_id": scenario_id,
        "accepted": not mismatches,
        "mismatches": mismatches,
        "observed": observed,
    }


def require_p0_clean_preflight(
    raw_state: Mapping[str, object],
    config: P0CleanCaptureConfig,
    scenario_id: str,
) -> dict[str, object]:
    report = evaluate_p0_clean_preflight(raw_state, config, scenario_id)
    if report["accepted"] is not True:
        raise RealGamePreflightError(report)
    return report
