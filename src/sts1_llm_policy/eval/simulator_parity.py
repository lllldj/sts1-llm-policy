from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from sts1_llm_policy.env.action_schema import CanonicalAction
from sts1_llm_policy.env.simulator_env import (
    SimulatorCombatEndedError,
    StsLightspeedEnv,
)
from sts1_llm_policy.env.state_schema import CanonicalState


class SimulatorParityError(RuntimeError):
    """Raised when a parity fixture or comparison contract is invalid."""


@dataclass(frozen=True)
class SimulatorParityConfig:
    gate_id: str
    backend_revision: str
    mechanics_manifest: Path
    fixture_path: Path
    required_real_encounters: tuple[str, ...]
    all_executable_transitions_match: bool
    all_required_real_encounters_executable: bool
    no_unresolved_supported_mechanic_mismatch: bool


class ParityEnvironment(Protocol):
    def reset(
        self,
        scenario_id: str,
        seed: int,
        *,
        ascension: int = 0,
        player_current_hp: int | None = None,
    ) -> CanonicalState:
        ...

    def legal_actions(self) -> tuple[CanonicalAction, ...]:
        ...

    def step(self, action: CanonicalAction) -> CanonicalState:
        ...

    def close(self) -> None:
        ...


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SimulatorParityError(f"Unable to read {label}: {path}") from error
    if not isinstance(value, dict):
        raise SimulatorParityError(f"{label} must contain a JSON object")
    return value


def _resolved_project_path(project_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SimulatorParityError(f"{label} must be a non-empty path")
    path = (project_root / value).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise SimulatorParityError(f"{label} escapes the project root") from error
    return path


def load_parity_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> SimulatorParityConfig:
    config_path = Path(path).resolve()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    raw = _json_object(config_path, "simulator parity config")
    if raw.get("schema_version") != 1:
        raise SimulatorParityError("Unsupported simulator parity config schema")
    if raw.get("backend") != "sts_lightspeed":
        raise SimulatorParityError("Parity backend must be sts_lightspeed")

    gate_id = raw.get("gate_id")
    revision = raw.get("backend_revision")
    encounters = raw.get("required_real_encounters")
    requirements = raw.get("promotion_requirements")
    if not isinstance(gate_id, str) or not gate_id:
        raise SimulatorParityError("gate_id must be a non-empty string")
    if not isinstance(revision, str) or len(revision) != 40:
        raise SimulatorParityError("backend_revision must be a 40-character revision")
    if (
        not isinstance(encounters, list)
        or not encounters
        or not all(isinstance(item, str) and item for item in encounters)
        or len(encounters) != len(set(encounters))
    ):
        raise SimulatorParityError(
            "required_real_encounters must be a non-empty unique string array"
        )
    if not isinstance(requirements, dict):
        raise SimulatorParityError("promotion_requirements must be an object")
    requirement_names = (
        "all_executable_transitions_match",
        "all_required_real_encounters_executable",
        "no_unresolved_supported_mechanic_mismatch",
    )
    if any(not isinstance(requirements.get(name), bool) for name in requirement_names):
        raise SimulatorParityError("Every promotion requirement must be boolean")

    return SimulatorParityConfig(
        gate_id=gate_id,
        backend_revision=revision,
        mechanics_manifest=_resolved_project_path(
            root,
            raw.get("mechanics_manifest"),
            "mechanics_manifest",
        ),
        fixture_path=_resolved_project_path(
            root,
            raw.get("fixture_path"),
            "fixture_path",
        ),
        required_real_encounters=tuple(encounters),
        all_executable_transitions_match=requirements[
            "all_executable_transitions_match"
        ],
        all_required_real_encounters_executable=requirements[
            "all_required_real_encounters_executable"
        ],
        no_unresolved_supported_mechanic_mismatch=requirements[
            "no_unresolved_supported_mechanic_mismatch"
        ],
    )


def load_supported_mechanics(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    raw = _json_object(manifest_path, "supported-mechanics manifest")
    if raw.get("schema_version") != "supported_mechanics_v1":
        raise SimulatorParityError("Unsupported mechanics manifest schema")
    if raw.get("backend") != "sts_lightspeed":
        raise SimulatorParityError("Mechanics manifest backend mismatch")
    for required in (
        "scope",
        "deck",
        "encounters",
        "powers",
        "action_types",
        "parity_contract",
        "explicit_exclusions",
    ):
        if required not in raw:
            raise SimulatorParityError(f"Mechanics manifest is missing {required}")
    return raw


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seed_bit_pattern_u64(value: object) -> object:
    """Normalize Java signed long and native uint64 seed representations."""

    if isinstance(value, bool) or not isinstance(value, int):
        return value
    if not -(1 << 63) <= value < (1 << 64):
        raise SimulatorParityError("Seed is outside signed-long/uint64 range")
    return value % (1 << 64)


def _repository_display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[3]
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _plain(value: object) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _power_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    card_value = value.get("card")
    card = None
    if isinstance(card_value, Mapping):
        card = {
            "card_id": card_value.get("card_id"),
            "name": card_value.get("name"),
            "upgrades": card_value.get("upgrades"),
        }
    return {
        "power_id": value.get("power_id"),
        "name": value.get("name"),
        "amount": value.get("amount"),
        "damage": value.get("damage", 0),
        "misc": value.get("misc", 0),
        "just_applied": value.get("just_applied", False),
        "card": card,
    }


def _powers_projection(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    powers = [_power_projection(value) for value in values]
    powers.sort(key=canonical_json_sha256)
    return powers


def _card_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    # UUIDs are source-private. CommunicationMod descriptions are dynamic
    # display text while the current simulator descriptions are static.
    return {
        "card_id": value.get("card_id"),
        "name": value.get("name"),
        "card_type": value.get("card_type"),
        "cost": value.get("cost"),
        "upgrades": value.get("upgrades"),
        "exhausts": value.get("exhausts"),
        "ethereal": value.get("ethereal"),
        "has_target": value.get("has_target"),
        "is_playable": value.get("is_playable"),
    }


def project_canonical_state(value: CanonicalState | Mapping[str, Any]) -> dict[str, Any]:
    raw_value = _plain(value)
    if not isinstance(raw_value, Mapping):
        raise SimulatorParityError("Canonical state must be an object")
    combat = raw_value.get("combat")
    if not isinstance(combat, Mapping):
        raise SimulatorParityError("Canonical state combat must be an object")
    player = combat.get("player")
    monsters = combat.get("monsters")
    if not isinstance(player, Mapping) or not isinstance(monsters, (list, tuple)):
        raise SimulatorParityError("Canonical combat player/monsters are malformed")

    projected_monsters = []
    for monster_value in monsters:
        if not isinstance(monster_value, Mapping):
            raise SimulatorParityError("Canonical monster must be an object")
        power_values = monster_value.get("powers", [])
        if not isinstance(power_values, (list, tuple)):
            raise SimulatorParityError("Canonical monster powers must be an array")
        projected_monsters.append(
            {
                "monster_id": monster_value.get("monster_id"),
                "name": monster_value.get("name"),
                "current_hp": monster_value.get("current_hp"),
                "max_hp": monster_value.get("max_hp"),
                "block": monster_value.get("block"),
                "intent": monster_value.get("intent"),
                "move_id": monster_value.get("move_id"),
                "move_hits": monster_value.get("move_hits"),
                "move_base_damage": monster_value.get("move_base_damage"),
                "move_adjusted_damage": monster_value.get("move_adjusted_damage"),
                "powers": _powers_projection(power_values),
                "is_gone": monster_value.get("is_gone"),
                "half_dead": monster_value.get("half_dead"),
            }
        )

    player_power_values = player.get("powers", [])
    if not isinstance(player_power_values, (list, tuple)):
        raise SimulatorParityError("Canonical player powers must be an array")
    piles: dict[str, list[dict[str, Any]]] = {}
    for name in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        cards = combat.get(name)
        if not isinstance(cards, (list, tuple)):
            raise SimulatorParityError(f"Canonical {name} must be an array")
        if not all(isinstance(card, Mapping) for card in cards):
            raise SimulatorParityError(f"Canonical {name} cards must be objects")
        piles[name] = [_card_projection(card) for card in cards]

    return {
        "seed": seed_bit_pattern_u64(raw_value.get("seed")),
        "character": raw_value.get("character"),
        "ascension_level": raw_value.get("ascension_level"),
        "act": raw_value.get("act"),
        "floor": raw_value.get("floor"),
        "turn": combat.get("turn"),
        "player": {
            "current_hp": player.get("current_hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "energy": player.get("energy"),
            "powers": _powers_projection(player_power_values),
        },
        "monsters": projected_monsters,
        "piles": piles,
    }


def load_parity_fixtures(path: str | Path) -> tuple[dict[str, Any], ...]:
    fixture_path = Path(path)
    try:
        lines = fixture_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise SimulatorParityError(f"Unable to read parity fixtures: {fixture_path}") from error
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SimulatorParityError(
                f"Malformed parity fixture JSON at line {line_number}"
            ) from error
        if not isinstance(value, dict):
            raise SimulatorParityError(f"Parity fixture line {line_number} is not an object")
        if value.get("schema_version") != "parity_fixture_v1":
            raise SimulatorParityError(
                f"Unsupported parity fixture schema at line {line_number}"
            )
        fixture_id = value.get("fixture_id")
        step_index = value.get("step_index")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise SimulatorParityError(f"fixture_id is invalid at line {line_number}")
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise SimulatorParityError(f"step_index is invalid at line {line_number}")
        key = (fixture_id, step_index)
        if key in seen:
            raise SimulatorParityError(f"Duplicate fixture transition: {key}")
        seen.add(key)
        raw_snapshot = value.get("real_raw_snapshot")
        expected_hash = value.get("real_raw_snapshot_sha256")
        if not isinstance(raw_snapshot, dict) or not isinstance(expected_hash, str):
            raise SimulatorParityError(f"Raw evidence is invalid at line {line_number}")
        if canonical_json_sha256(raw_snapshot) != expected_hash:
            raise SimulatorParityError(f"Raw evidence hash mismatch at line {line_number}")
        project_canonical_state(value.get("canonical_input", {}))
        records.append(value)
    if not records:
        raise SimulatorParityError("Parity fixture file is empty")
    return tuple(records)


def _action_projection(value: CanonicalAction | Mapping[str, Any]) -> dict[str, Any]:
    raw_value = _plain(value)
    if not isinstance(raw_value, Mapping):
        raise SimulatorParityError("Canonical action must be an object")
    action_type = raw_value.get("action_type")
    if hasattr(action_type, "value"):
        action_type = action_type.value
    return {
        "action_id": raw_value.get("action_id"),
        "action_type": action_type,
        "hand_index": raw_value.get("hand_index"),
        "card_name": raw_value.get("card_name"),
        "target_index": raw_value.get("target_index"),
    }


def _diff(expected: object, actual: object, path: str = "$") -> list[dict[str, Any]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in expected:
                result.append({"path": child, "expected": "<absent>", "actual": actual[key]})
            elif key not in actual:
                result.append({"path": child, "expected": expected[key], "actual": "<absent>"})
            else:
                result.extend(_diff(expected[key], actual[key], child))
        return result
    if isinstance(expected, list) and isinstance(actual, list):
        result = []
        if len(expected) != len(actual):
            result.append(
                {"path": f"{path}.length", "expected": len(expected), "actual": len(actual)}
            )
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            result.extend(_diff(expected_item, actual_item, f"{path}[{index}]"))
        return result
    if expected != actual:
        return [{"path": path, "expected": expected, "actual": actual}]
    return []


def _terminal_outcome(value: str) -> str:
    try:
        return {
            "PLAYER_VICTORY": "victory",
            "PLAYER_LOSS": "defeat",
        }[value]
    except KeyError as error:
        raise SimulatorParityError(f"Unsupported simulator terminal outcome: {value}") from error


def evaluate_simulator_parity(
    config: SimulatorParityConfig,
    *,
    env: ParityEnvironment | None = None,
) -> dict[str, Any]:
    mechanics = load_supported_mechanics(config.mechanics_manifest)
    if mechanics.get("backend_revision") != config.backend_revision:
        raise SimulatorParityError("Parity config/mechanics revision mismatch")
    records = load_parity_fixtures(config.fixture_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["fixture_id"]].append(record)
    for fixture_records in grouped.values():
        fixture_records.sort(key=lambda item: item["step_index"])

    owns_env = env is None
    simulator_env = env if env is not None else StsLightspeedEnv()
    fixture_results: list[dict[str, Any]] = []
    executable_encounters: set[str] = set()
    matched_transitions = 0
    compared_transitions = 0
    try:
        for fixture_id, fixture_records in grouped.items():
            first = fixture_records[0]
            encounter = first.get("encounter")
            execution = first.get("execution")
            if not isinstance(encounter, str) or not isinstance(execution, dict):
                raise SimulatorParityError(f"Fixture metadata is malformed: {fixture_id}")
            status = execution.get("status")
            if status == "excluded":
                blockers = execution.get("blockers")
                if not isinstance(blockers, list) or not blockers:
                    raise SimulatorParityError(
                        f"Excluded fixture must document blockers: {fixture_id}"
                    )
                fixture_results.append(
                    {
                        "fixture_id": fixture_id,
                        "encounter": encounter,
                        "status": "excluded",
                        "transitions_preserved": len(fixture_records),
                        "blockers": blockers,
                    }
                )
                continue
            if status != "executable":
                raise SimulatorParityError(f"Unknown fixture execution status: {status}")
            reset = execution.get("simulator_reset")
            if not isinstance(reset, dict):
                raise SimulatorParityError(f"Executable fixture lacks reset: {fixture_id}")
            executable_encounters.add(encounter)
            state = simulator_env.reset(
                reset["scenario_id"],
                reset["seed"],
                ascension=reset.get("ascension", 0),
                player_current_hp=reset.get("player_current_hp"),
            )
            fixture_mismatches: list[dict[str, Any]] = []
            fixture_matched = 0
            terminal_seen = False
            for record in fixture_records:
                compared_transitions += 1
                transition_mismatches: list[dict[str, Any]] = []
                expected_input = project_canonical_state(record["canonical_input"])
                transition_mismatches.extend(
                    {
                        **mismatch,
                        "stage": "input_state",
                        "step_index": record["step_index"],
                    }
                    for mismatch in _diff(expected_input, project_canonical_state(state))
                )

                expected_action = _action_projection(record["chosen_action"])
                actions = simulator_env.legal_actions()
                selected = next(
                    (
                        action
                        for action in actions
                        if action.action_id == expected_action["action_id"]
                    ),
                    None,
                )
                if selected is None:
                    transition_mismatches.append(
                        {
                            "stage": "action",
                            "step_index": record["step_index"],
                            "path": "$.action_id",
                            "expected": expected_action["action_id"],
                            "actual": [action.action_id for action in actions],
                        }
                    )
                    fixture_mismatches.extend(transition_mismatches)
                    break
                transition_mismatches.extend(
                    {
                        **mismatch,
                        "stage": "action",
                        "step_index": record["step_index"],
                    }
                    for mismatch in _diff(expected_action, _action_projection(selected))
                )

                expected_next = record.get("expected_next")
                if not isinstance(expected_next, dict):
                    raise SimulatorParityError("Fixture expected_next must be an object")
                expected_done = expected_next.get("done")
                try:
                    next_state = simulator_env.step(selected)
                    if expected_done is True:
                        transition_mismatches.append(
                            {
                                "stage": "terminal",
                                "step_index": record["step_index"],
                                "path": "$.done",
                                "expected": True,
                                "actual": False,
                            }
                        )
                    else:
                        expected_fields = expected_next.get("state_fields")
                        transition_mismatches.extend(
                            {
                                **mismatch,
                                "stage": "next_state",
                                "step_index": record["step_index"],
                            }
                            for mismatch in _diff(
                                expected_fields,
                                project_canonical_state(next_state),
                            )
                        )
                        state = next_state
                except SimulatorCombatEndedError as terminal:
                    terminal_seen = True
                    actual_outcome = _terminal_outcome(terminal.outcome)
                    if expected_done is not True:
                        transition_mismatches.append(
                            {
                                "stage": "terminal",
                                "step_index": record["step_index"],
                                "path": "$.done",
                                "expected": expected_done,
                                "actual": True,
                            }
                        )
                    if expected_next.get("terminal_outcome") != actual_outcome:
                        transition_mismatches.append(
                            {
                                "stage": "terminal",
                                "step_index": record["step_index"],
                                "path": "$.terminal_outcome",
                                "expected": expected_next.get("terminal_outcome"),
                                "actual": actual_outcome,
                            }
                        )
                    expected_terminal_hp = expected_next.get("player_current_hp")
                    actual_terminal_hp = terminal.terminal_state.combat.player.current_hp
                    if expected_terminal_hp != actual_terminal_hp:
                        transition_mismatches.append(
                            {
                                "stage": "terminal",
                                "step_index": record["step_index"],
                                "path": "$.player_current_hp",
                                "expected": expected_terminal_hp,
                                "actual": actual_terminal_hp,
                            }
                        )

                if transition_mismatches:
                    fixture_mismatches.extend(transition_mismatches)
                else:
                    fixture_matched += 1
                    matched_transitions += 1
            fixture_results.append(
                {
                    "fixture_id": fixture_id,
                    "encounter": encounter,
                    "status": "passed" if not fixture_mismatches else "failed",
                    "compared_transitions": len(fixture_records),
                    "matched_transitions": fixture_matched,
                    "terminal_seen": terminal_seen,
                    "mismatches": fixture_mismatches,
                }
            )
    finally:
        if owns_env:
            simulator_env.close()

    required_encounters = set(config.required_real_encounters)
    missing_executable = sorted(required_encounters - executable_encounters)
    executable_passed = all(
        result["status"] == "passed"
        for result in fixture_results
        if result["status"] != "excluded"
    )
    unresolved_mismatches = sum(
        len(result.get("mismatches", [])) for result in fixture_results
    )
    gates = {
        "executable_transition_parity": {
            "required": config.all_executable_transitions_match,
            "passed": executable_passed and matched_transitions == compared_transitions,
            "compared_transitions": compared_transitions,
            "matched_transitions": matched_transitions,
        },
        "required_real_encounter_coverage": {
            "required": config.all_required_real_encounters_executable,
            "passed": not missing_executable,
            "required_encounters": list(config.required_real_encounters),
            "executable_encounters": sorted(executable_encounters),
            "missing_executable_encounters": missing_executable,
        },
        "supported_mechanic_mismatches": {
            "required": config.no_unresolved_supported_mechanic_mismatch,
            "passed": unresolved_mismatches == 0,
            "unresolved_mismatch_count": unresolved_mismatches,
        },
    }
    promoted = all(not gate["required"] or gate["passed"] for gate in gates.values())
    return {
        "schema_version": 1,
        "report_type": "sts_lightspeed_real_parity",
        "gate_id": config.gate_id,
        "status": "promoted" if promoted else "no_go",
        "promotion_decision": "promote" if promoted else "no_go",
        "source": "real_game_fixtures_and_simulator_replay",
        "backend": "sts_lightspeed",
        "backend_revision": config.backend_revision,
        "mechanics_manifest": {
            "manifest_id": mechanics.get("manifest_id"),
            "sha256": canonical_json_sha256(mechanics),
        },
        "fixture_set": {
            "path": _repository_display_path(config.fixture_path),
            "sha256": hashlib.sha256(config.fixture_path.read_bytes()).hexdigest(),
            "records": len(records),
            "fixtures": len(grouped),
        },
        "comparison_contract": mechanics["parity_contract"],
        "gates": gates,
        "fixtures": fixture_results,
    }
