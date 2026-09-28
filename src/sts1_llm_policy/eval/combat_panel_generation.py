from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
from statistics import mean
from time import perf_counter
from typing import Any, Mapping, Sequence

from sts1_llm_policy.data.card_pick_metrics import (
    MassNormalizedCardRewardPicker,
    SqliteCardRateBackend,
)
from sts1_llm_policy.env.simulator_execution import CORRECTED_MECHANICS, LEGACY_MECHANICS, resolve_simulator
from sts1_llm_policy.artifacts import (
    replace_json as _atomic_json,
    resolve_repository_path,
    read_json_object as _read_object,
    sha256_file as _sha,
    canonical_json_bytes,
    sha256_bytes,
)
from sts1_llm_policy.eval.card_profiles import validate_reward_profile_coverage
from sts1_llm_policy.eval.counterfactual_reward_picker_v2 import (
    COUNTERFACTUAL_PICKER_V2_VERSION,
    StrategicRateTable, TeacherCapabilityEvaluatorV2,
)
from sts1_llm_policy.eval.reward_generation import generate_frozen_reward_snapshot
from sts1_llm_policy.eval.generation_strategies import resolve_generation_strategies


ACT_1_BOSSES = ("hexaghost", "slime_boss", "the_guardian")
ACT_2_BOSSES = ("automaton", "champ", "collector")


@dataclass(frozen=True)
class StageRule:
    act_1_rewards: int
    act_2_rewards: int
    cumulative_upgrades: int
    cumulative_removals: int


@dataclass(frozen=True)
class SourceRules:
    reward_card_ids: frozenset[str]
    picker_database_path: Path
    picker_temperature: float
    picker_exploration: float
    picker_epsilon: float
    stages: Mapping[str, StageRule]


def _canonical_sha(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _build_source_matrix(
    plan: Mapping[str, Any],
    scope: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stage_by_key = {
        (int(item["act"]), str(item["stage"])): item
        for item in plan["deck_stages"]
    }
    boss_relics = tuple(map(str, plan["boss_relic_pool"]))
    ordinary_relics = tuple(
        str(item)
        for item in plan["relic_pool"]
        if item != "BURNING_BLOOD" and item not in boss_relics
    )
    encounters: list[tuple[str, int, str]] = []
    for act in (1, 2):
        act_scope = scope["encounters"][f"act_{act}"]
        for family in ("normal_weak", "normal_strong", "elite", "boss"):
            encounters.extend(
                (str(value).lower(), act, family) for value in act_scope[family]
            )
    seeds = matrix["seed_bases"]
    result: list[dict[str, Any]] = []
    for scenario_id, act, family in encounters:
        stage_names = (
            ("pre-boss",)
            if family == "boss"
            else ("entry", "mid", "pre-boss")
        )
        for stage_name in stage_names:
            stage = stage_by_key[(act, stage_name)]
            for ascension in matrix["ascensions"]:
                for replicate_index in range(int(matrix["combat_seeds_per_cell"])):
                    index = len(result)
                    relic_seed = int(seeds["relic"]) + index
                    rng = random.Random(relic_seed)
                    ordinary = rng.sample(
                        ordinary_relics, int(stage["ordinary_relic_count"])
                    )
                    boss = (
                        rng.choice(boss_relics)
                        if stage["requires_boss_relic"] is True
                        else None
                    )
                    result.append({
                        "act": act,
                        "ascension": int(ascension),
                        "boss_relic_id": boss,
                        "combat_seed": int(seeds["combat"]) + index,
                        "encounter_family": family,
                        "episode_index": index,
                        "ordinary_relic_ids": ordinary,
                        "picker_seed": int(seeds["picker"]) + index,
                        "policy_seed": int(seeds["policy"]) + index,
                        "relic_seed": relic_seed,
                        "replicate_index": replicate_index,
                        "reward_seed": int(seeds["reward"]) + index,
                        "scenario_id": scenario_id,
                        "stage": stage_name,
                        "stratum_id": str(stage["stratum_id"]),
                    })
    return result


def _build_route_specs(
    source_specs: Sequence[Mapping[str, Any]],
    panel: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bosses = sorted(
        (item for item in source_specs if item["encounter_family"] == "boss"),
        key=lambda item: int(item["episode_index"]),
    )
    ascensions = sorted({int(item["ascension"]) for item in bosses})
    by_ascension = {
        ascension: [item for item in bosses if int(item["ascension"]) == ascension]
        for ascension in ascensions
    }
    routes_per_ascension = int(panel["routes_per_ascension"])
    source_count = len(bosses) // len(ascensions) if ascensions else 0
    if (source_count == 0 or routes_per_ascension <= 0
            or routes_per_ascension % source_count
            or any(len(items) != source_count for items in by_ascension.values())):
        raise ValueError("Routes must evenly repeat equal Boss groups per ascension")
    seeds = panel["seed_bases"]
    result: list[dict[str, Any]] = []
    act_2_offsets: Counter[tuple[int, str]] = Counter()
    stratum_offsets: Counter[tuple[int, str]] = Counter()
    for ascension in ascensions:
        for repeat in range(routes_per_ascension // source_count):
            for source in by_ascension[ascension]:
                final_boss = str(source["scenario_id"])
                stratum_key = (ascension, final_boss)
                panel_replicate = stratum_offsets[stratum_key]
                stratum_offsets[stratum_key] += 1
                if int(source["act"]) == 1:
                    act_1_boss = final_boss
                else:
                    final_index = ACT_2_BOSSES.index(final_boss)
                    within_final = act_2_offsets[stratum_key]
                    act_2_offsets[stratum_key] += 1
                    act_1_boss = ACT_1_BOSSES[(final_index + within_final) % len(ACT_1_BOSSES)]
                index = len(result)
                spec = dict(source)
                spec.update({
                    "route_index": index,
                    "source_episode_index": int(source["episode_index"]),
                    "source_replicate_index": int(source["replicate_index"]),
                    "panel_repeat_index": repeat,
                    "panel_replicate_index": panel_replicate,
                    "episode_index": index,
                    "act_1_boss_scenario_id": act_1_boss,
                    "final_boss_scenario_id": final_boss,
                    "reward_seed": int(seeds["reward"]) + index,
                    "combat_seed": int(seeds["combat"]) + index,
                    "picker_seed": int(seeds["picker"]) + index,
                })
                result.append(spec)
    return result


def _build_evaluation_specs(
    routes: Sequence[Mapping[str, Any]], panel: Mapping[str, Any]
) -> list[dict[str, Any]]:
    original_count = int(panel["original_seeds_per_route"])
    total_count = int(panel["seeds_per_route"])
    additional_start = int(panel["additional_episode_index_start"])
    if not 0 < original_count <= total_count or additional_start < 0:
        raise ValueError("Invalid evaluation seed layout")
    result: list[dict[str, Any]] = []
    for route_position, route in enumerate(routes):
        route_index = int(route["route_index"])
        for seed_index in range(total_count):
            spec = dict(route)
            if seed_index < original_count:
                index = route_index * original_count + seed_index
                spec.update({
                    "episode_index": index,
                    "heldout_seed_index": seed_index,
                    "combat_seed": int(panel["original_combat_seed_base"]) + index,
                    "policy_seed": int(panel["original_policy_seed_base"]) + index,
                })
            else:
                addition_index = route_position * (total_count - original_count) + seed_index - original_count
                spec.update({
                    "episode_index": additional_start + addition_index,
                    "heldout_seed_index": seed_index,
                    "combat_seed": int(panel["additional_combat_seed_base"]) + addition_index,
                    "policy_seed": int(panel["additional_policy_seed_base"]) + addition_index,
                })
            result.append(spec)
    if len({item["episode_index"] for item in result}) != len(result):
        raise ValueError("Evaluation episode indices overlap")
    return sorted(result, key=lambda item: int(item["episode_index"]))


def _validate(config_path: Path, *, project_root: Path, target: str = "auto") -> dict[str, Any]:
    config_path = resolve_repository_path(project_root, config_path, expected_kind="file")
    config = _read_object(config_path)
    allowed = {
        "schema_version", "run_id", "purpose", "panel_id", "observation_version",
        "simulator_revision", "simulator_mechanics", "source_rules", "scope",
        "picker_database", "expected_picker_database_sha256", "source_matrix",
        "route_panel", "evaluation_panel", "protocol", "strategies", "output_dir",
        "test_data_read", "sealed_test_run",
    }
    if (
        config.keys() - allowed
        or config.get("schema_version") != "combat_panel_generation_v2"
        or config.get("simulator_mechanics") not in (CORRECTED_MECHANICS, LEGACY_MECHANICS)
        or config.get("observation_version") != "observation_v5"
        or not isinstance(config.get("panel_id"), str) or not config["panel_id"]
        or config.get("test_data_read") is not False
        or config.get("sealed_test_run") is not False
    ):
        raise ValueError("Unsupported combat panel generation config")
    strategies = resolve_generation_strategies(config.get("strategies"))
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id
    ) or run_id in {".", ".."}:
        raise ValueError("Invalid generation run_id")
    output = resolve_repository_path(
        project_root, config.get("output_dir", f"outputs/generation/{run_id}"),
        must_exist=False, expected_kind="directory",
    )
    paths = {
        name: resolve_repository_path(project_root, config[name], expected_kind="file")
        for name in ("scope", "picker_database")
    }
    database_sha = _sha(paths["picker_database"])
    if database_sha != config["expected_picker_database_sha256"]:
        raise ValueError("Combat panel generation picker database changed")
    plan = config["source_rules"]
    scope = _read_object(paths["scope"])
    source_specs = _build_source_matrix(plan, scope, config["source_matrix"])
    all_routes = _build_route_specs(source_specs, config["route_panel"])
    selected_routes = [
        item for item in all_routes
        if int(item["act"]) == int(config["route_panel"]["selected_act"])
        and int(item["ascension"]) == int(config["route_panel"]["selected_ascension"])
    ]
    if len(selected_routes) != int(config["route_panel"]["expected_selected_routes"]):
        raise ValueError("Focused route count changed")
    evaluation_specs = _build_evaluation_specs(selected_routes, config["evaluation_panel"])
    if len(evaluation_specs) != int(config["evaluation_panel"]["expected_combats"]):
        raise ValueError("Evaluation combat count changed")
    expected_panel = config["evaluation_panel"].get("expected_panel_sha256")
    if expected_panel is not None and (
        not isinstance(expected_panel, str) or len(expected_panel) != 64
        or any(char not in "0123456789abcdef" for char in expected_panel)
    ):
        raise ValueError("expected_panel_sha256 must be a lowercase SHA-256 digest when supplied")
    picker = plan["picker"]
    stages = {
        str(item["stratum_id"]): StageRule(
            act_1_rewards=int(item["act_1_regular_rewards"]),
            act_2_rewards=int(item["act_2_rewards"]),
            cumulative_upgrades=int(item["cumulative_upgrades"]),
            cumulative_removals=int(item["cumulative_removals"]),
        )
        for item in plan["deck_stages"]
    }
    source = SourceRules(
        reward_card_ids=frozenset(map(str, scope["reward_cards"]["included_enum_ids"])),
        picker_database_path=paths["picker_database"],
        picker_temperature=float(picker["temperature"]),
        picker_exploration=float(picker["exploration"]),
        picker_epsilon=float(picker["epsilon"]),
        stages=stages,
    )
    validate_reward_profile_coverage(tuple(sorted(source.reward_card_ids)))
    backend = SqliteCardRateBackend(source.picker_database_path)
    rates = StrategicRateTable(backend)
    for upgraded in (False, True):
        for enum_id in source.reward_card_ids:
            rates(enum_id, upgraded, 1)
    execution = resolve_simulator(project_root=project_root, target=target, mechanics=config["simulator_mechanics"])
    installation = execution.validate()
    if installation.revision != config["simulator_revision"]:
        raise ValueError("Simulator revision changed")
    return {
        "config": config,
        "config_path": config_path.resolve(),
        "output": output,
        "picker_database_sha256": database_sha,
        "source": source,
        "routes": selected_routes,
        "evaluation_specs": evaluation_specs,
        "installation": installation,
        "execution": execution,
        "strategies": strategies,
    }


def _binding(validated: Mapping[str, Any]) -> dict[str, Any]:
    config = validated["config"]
    semantics = {name: config[name] for name in (
        "source_rules", "source_matrix", "route_panel", "evaluation_panel", "protocol", "strategies",
        "panel_id", "observation_version",
    )}
    # Scope descriptions/formatting are not rules. Encounter effects are captured
    # by the resolved route specs; card membership also affects picker decisions.
    semantics["reward_card_ids"] = sorted(validated["source"].reward_card_ids)
    return {
        "schema_version": "combat_panel_generation_binding_v4",
        "semantics_sha256": _canonical_sha(semantics),
        "picker_database_sha256": validated["picker_database_sha256"],
        "simulator_revision": validated["installation"].revision,
        "simulator_mechanics": config["simulator_mechanics"],
        # Native bytes must match before route caches can be reused.
        "native_bridge_sha256": _sha(validated["installation"].bridge_executable),
        "picker_version": COUNTERFACTUAL_PICKER_V2_VERSION,
        "route_specs_sha256": _canonical_sha(validated["routes"]),
    }


def _load_route(path: Path, *, binding: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    item = _read_object(path)
    if (item.get("binding") != binding or item.get("spec") != spec
            or item.get("snapshot_sha256") != _canonical_sha(item.get("snapshot"))):
        raise ValueError(f"Completed route binding changed: {path}")
    return item


def run(
    config_path: str | Path, *, project_root: str | Path,
    target: str = "auto", preflight_only: bool = False, smoke_routes: int | None = None,
    arm_id: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the configured generator; the caller supplies workspace and execution host."""
    root = Path(project_root).resolve()
    if workers != 1:
        raise ValueError("--workers is supported only by continuous Teacher generation")
    if arm_id is not None:
        raise ValueError("--arm is supported only by continuous combat generation")
    if preflight_only and smoke_routes is not None:
        raise ValueError("Preflight and smoke are separate modes")
    project_root = root
    validated = _validate(Path(config_path), project_root=project_root, target=target)
    config = validated["config"]
    routes = list(validated["routes"])
    if smoke_routes is not None:
        if not 1 <= smoke_routes <= len(routes):
            raise ValueError(f"--smoke-routes must be between 1 and {len(routes)}")
        routes = routes[:smoke_routes]
    if preflight_only:
        return {
            "status": "preflight_passed",
            "routes": len(validated["routes"]),
            "combats": len(validated["evaluation_specs"]),
            "binding": _binding(validated),
            "execution": validated["execution"].describe(),
            "simulator_started": False,
        }
    output = validated["output"]
    if smoke_routes is not None:
        output = output / "smoke"
    route_dir = output / "routes"
    binding = _binding(validated)
    completed: dict[int, dict[str, Any]] = {}
    for spec in routes:
        index = int(spec["route_index"])
        item = _load_route(route_dir / f"route-{index:03d}.json", binding=binding, spec=spec)
        if item is not None:
            completed[index] = item
    pending = [spec for spec in routes if int(spec["route_index"]) not in completed]
    print(f"validated {len(routes) - len(pending)} completed routes; {len(pending)} pending", flush=True)
    source: SourceRules = validated["source"]
    backend = SqliteCardRateBackend(source.picker_database_path)
    picker = MassNormalizedCardRewardPicker(backend)
    rate_table = StrategicRateTable(backend)
    protocol = config["protocol"]
    strategies = validated["strategies"]
    started = perf_counter()
    if pending:
        reward_client = validated["execution"].create_client()
        env = validated["execution"].create_environment()
        try:
            for position, spec in enumerate(pending, 1):
                evaluator = TeacherCapabilityEvaluatorV2(
                    env, combat_seed=int(spec["combat_seed"]), protocol=protocol
                )
                stage = source.stages[str(spec["stratum_id"])]
                selector = strategies.card_pick(
                    picker=picker, evaluator=evaluator, rate_lookup=rate_table, spec=spec,
                    allowed_card_ids=source.reward_card_ids,
                    epsilon=source.picker_epsilon,
                    total_rewards=stage.act_1_rewards,
                    protocol=protocol,
                )
                snapshot, reward_events, topology_events = generate_frozen_reward_snapshot(
                    reward_client,
                    reward_seed=int(spec["reward_seed"]),
                    ascension=int(spec["ascension"]),
                    target_act=int(spec["act"]),
                    stage=str(spec["stage"]),
                    act_1_regular_rewards=stage.act_1_rewards,
                    act_2_rewards=stage.act_2_rewards,
                    ordinary_relic_ids=tuple(spec["ordinary_relic_ids"]),
                    boss_relic_id=spec.get("boss_relic_id"),
                    cumulative_upgrades=stage.cumulative_upgrades,
                    cumulative_removals=stage.cumulative_removals,
                    picker_seed=int(spec["picker_seed"]),
                    reward_selector=selector,
                    strategies=strategies,
                )
                item = {
                    "schema_version": "generated_counterfactual_route_v1",
                    "binding": binding,
                    "spec": spec,
                    "snapshot": snapshot,
                    "snapshot_sha256": _canonical_sha(snapshot),
                    "reward_events": reward_events,
                    "topology_events": topology_events,
                    "teacher_search_calls": evaluator.search_calls,
                    "teacher_search_calls_by_offset": dict(evaluator.search_calls_by_offset),
                }
                index = int(spec["route_index"])
                _atomic_json(route_dir / f"route-{index:03d}.json", item)
                completed[index] = item
                elapsed = perf_counter() - started
                eta = elapsed / position * (len(pending) - position) / 60
                print(f"[{position:02d}/{len(pending):02d}] route={index:02d} searches={evaluator.search_calls} eta~{eta:.1f}m", flush=True)
        finally:
            reward_client.close()
            env.close()
    selected = [completed[int(spec["route_index"])] for spec in routes]
    decks = {
        str(int(item["spec"]["route_index"])): {
            "snapshot_sha256": item["snapshot_sha256"],
            "snapshot": item["snapshot"],
        }
        for item in selected
    }
    selected_route_ids = {int(spec["route_index"]) for spec in routes}
    evaluation_specs = [
        spec for spec in validated["evaluation_specs"]
        if int(spec["route_index"]) in selected_route_ids
    ]
    panel_sha = _canonical_sha({
        "specs": evaluation_specs,
        "decks": {key: value["snapshot_sha256"] for key, value in decks.items()},
    })
    generated = {
        "schema_version": "frozen_combat_panel_inputs_v1",
        "panel_id": config["panel_id"],
        "observation_version": config["observation_version"],
        "simulator_revision": validated["installation"].revision,
        "simulator_mechanics": config["simulator_mechanics"],
        "specs": evaluation_specs,
        "decks": decks,
        "panel_sha256": panel_sha,
        "test_data_read": False,
        "sealed_test_run": False,
    }
    generated_path = output / "inputs.json"
    _atomic_json(generated_path, generated)
    is_formal = smoke_routes is None
    checks = {
        "all_routes_generated": len(selected) == len(routes),
    }
    if is_formal and config["evaluation_panel"].get("expected_panel_sha256") is not None:
        checks["expected_panel_sha256"] = panel_sha == config["evaluation_panel"]["expected_panel_sha256"]
    if not all(checks.values()):
        raise ValueError(f"Generated panel differs from frozen input: {checks}")
    report = {
        "schema_version": "combat_panel_generation_report_v2",
        "run_id": config["run_id"],
        "status": "completed" if is_formal else "smoke_completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding": binding,
        "configuration": config,
        "execution": validated["execution"].describe(),
        "scope": {"routes": len(routes), "combats": len(evaluation_specs)},
        "metrics": {
            "teacher_search_calls": sum(int(item["teacher_search_calls"]) for item in selected),
            "final_deck_size_mean": round(mean(len(item["snapshot"]["deck"]) for item in selected), 3),
        },
        "timing": {"run_seconds": round(perf_counter() - started, 3)},
        "artifact": {
            "path": str(generated_path.relative_to(project_root)).replace("\\", "/"),
            "panel_sha256": panel_sha,
        },
        "checks": checks,
        "test_data_read": False,
        "sealed_test_run": False,
    }
    _atomic_json(output / "report.json", report)
    return report
