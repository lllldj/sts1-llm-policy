"""Resolve continuous-route configuration, schedules, source isolation and identity."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from sts1_llm_policy.env.card_selection import CARD_SELECTION_VERSION, SELECTION_OBSERVATION_VERSION
from sts1_llm_policy.env.serializer import SUPPORTED_OBSERVATION_SERIALIZER_VERSIONS
from sts1_llm_policy.env.route_context import ROUTE_OBSERVATION_VERSION, public_route_context
from sts1_llm_policy.env.simulator_execution import resolve_simulator
from sts1_llm_policy.eval.card_profiles import validate_reward_profile_coverage
from sts1_llm_policy.eval.generation_strategies import RouteContext, resolve_generation_strategies
from sts1_llm_policy.model_runtime import load_base_model_runtime_config
from sts1_llm_policy.workflows.continuous_config import load_continuous_config
from sts1_llm_policy.artifacts import resolve_repository_path, read_json_object, sha256_file
from .continuous_artifacts import CANDIDATE_EVIDENCE_CLASS, IDENTITY_SCHEMA_VERSION, canonical_sha


SCHEMA_VERSION = "continuous_combat_panel_generation_v1"


def load_scope(path: Path, *, project_root: Path) -> dict[str, Any]:
    scope = read_json_object(path)
    extension = scope.get("extends")
    if extension is None:
        return scope
    base_path = resolve_repository_path(project_root, extension, expected_kind="file")
    base = read_json_object(base_path)
    additions = scope.get("reward_cards", {}).get("add_enum_ids")
    if not isinstance(additions, list) or not all(isinstance(item, str) for item in additions):
        raise ValueError("Extended scope reward_cards.add_enum_ids must be a string array")
    included = list(base["reward_cards"]["included_enum_ids"])
    if set(included).intersection(additions) or len(set(additions)) != len(additions):
        raise ValueError("Extended scope card additions must be new and unique")
    result = deepcopy(base)
    for key, value in scope.items():
        if key != "reward_cards":
            result[key] = deepcopy(value)
    result["reward_cards"] = deepcopy(base["reward_cards"])
    result["reward_cards"]["included_enum_ids"] = included + additions
    result["reward_cards"]["included_count"] = len(included) + len(additions)
    result["reward_cards"]["added_enum_ids"] = additions
    result["reward_cards"]["excluded"] = scope.get("reward_cards", {}).get("excluded", [])
    return result


def require_positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def operation_counts(operations: Sequence[str]) -> Counter[str]:
    return Counter(
        "combat" if operation.startswith("combat:")
        else "heal" if operation.startswith("heal:")
        else operation
        for operation in operations
    )


def build_route_steps(
    config: Mapping[str, Any],
    scope: Mapping[str, Any],
    route_index: int,
    combat_seed_group_index: int,
    route_policy,
) -> list[dict[str, Any]]:
    route = config["route"]
    declared_counts = operation_counts(tuple(map(str, route["operations"])))
    pools = scope["encounters"][f"act_{int(route['act'])}"]
    context = RouteContext(
        target_act=int(route["act"]), stage=str(route["stage"]),
        act_1_regular_rewards=int(declared_counts["card_pick"]),
        act_2_rewards=0, ordinary_relic_ids=(), boss_relic_id=None,
        cumulative_upgrades=int(declared_counts["upgrade"]),
        cumulative_removals=int(declared_counts["remove"]),
        operations=tuple(route["operations"]), encounter_pools=pools,
        relic_pool=tuple(route["relic_pool"]), combat_floors=tuple(route["combat_floors"]),
        initial_relics=tuple(route["initial_relics"]),
        encounter_seed=int(config["seeds"]["encounter"]) + route_index,
        relic_seed=int(config["seeds"]["relic"]) + route_index,
    )
    rng = random.Random(int(config["seeds"]["route"]) + route_index)
    steps = [dict(item) for item in route_policy(context, rng)]
    counts = Counter(
        "card_pick" if item["kind"] == "reward"
        else "random_relic" if item["kind"] == "ordinary_relic"
        else item["kind"]
        for item in steps if item["kind"] != "reset"
    )
    expected = dict(declared_counts)
    if dict(counts) != expected:
        raise ValueError(
            f"Resolved route counts differ from operation list: {dict(counts)} != {expected}"
        )
    combat_steps = [item for item in steps if item["kind"] == "combat"]
    seed_groups = int(config["combat_seed_groups_per_route"])
    for item in combat_steps:
        route_group_index = route_index * seed_groups + combat_seed_group_index
        offset = route_group_index * len(combat_steps) + int(item["combat_index"]) - 1
        item["combat_seed"] = int(config["seeds"]["combat"]) + offset
        item["policy_seed"] = int(config["seeds"]["policy"]) + offset
    upgrade_index = 0
    for item in steps:
        if item["kind"] == "upgrade":
            upgrade_index += 1
            item["evaluation_targets"] = _upgrade_evaluation_targets(
                config, scope, steps, route_index=route_index,
                upgrade_index=upgrade_index,
            )
    return steps


def _upgrade_evaluation_targets(
    config: Mapping[str, Any],
    scope: Mapping[str, Any],
    route_steps: Sequence[Mapping[str, Any]],
    *,
    route_index: int,
    upgrade_index: int,
) -> list[dict[str, Any]]:
    policy = config.get("upgrade_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("upgrade_policy must be an object")
    groups = policy.get("target_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("upgrade_policy.target_groups must be nonempty")
    act = int(config["route"]["act"])
    pools = scope["encounters"][f"act_{act}"]
    scenarios: list[tuple[str, str, str]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("Upgrade target group must be an object")
        family = group.get("encounter_family")
        selection = group.get("selection")
        if not isinstance(family, str) or family not in pools:
            raise ValueError(f"Unknown upgrade target encounter family: {family!r}")
        if selection == "all":
            selected = [str(item).lower() for item in pools[family]]
        elif selection == "route_selected":
            selected = [
                str(item["scenario_id"])
                for item in route_steps
                if item.get("kind") == "combat"
                and item.get("encounter_family") == family
            ]
            if len(selected) != 1:
                raise ValueError(
                    "route_selected upgrade target requires exactly one route encounter"
                )
        else:
            raise ValueError(f"Unknown upgrade target selection: {selection!r}")
        scenarios.extend((family, str(selection), scenario) for scenario in selected)
    scenario_ids = [scenario for _, _, scenario in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Upgrade evaluation target groups overlap")
    seeds_per_target = require_positive_int(
        policy.get("combat_seeds_per_target"),
        "upgrade_policy.combat_seeds_per_target",
    )
    upgrades_per_route = operation_counts(
        tuple(map(str, config["route"]["operations"]))
    )["upgrade"]
    panel_size = len(scenarios) * seeds_per_target
    decision_offset = (
        route_index * upgrades_per_route * panel_size
        + (upgrade_index - 1) * panel_size
    )
    seed_base = int(config["seeds"]["upgrade_combat"]) + decision_offset
    return [
        {
            "encounter_family": family,
            "selection": selection,
            "scenario_id": scenario,
            "combat_seed": seed_base + target_index * seeds_per_target + replicate,
            "replicate_index": replicate,
        }
        for target_index, (family, selection, scenario) in enumerate(scenarios)
        for replicate in range(seeds_per_target)
    ]


def prepare_run(config_path: Path, *, project_root: Path, target: str) -> dict[str, Any]:
    config_path = resolve_repository_path(project_root, config_path, expected_kind="file")
    config = load_continuous_config(project_root, config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported continuous combat generation schema")
    if config.keys() & {"test_data_read", "sealed_test_run", "training_started", "stable_claim"}:
        raise ValueError("Execution outcomes belong in reports, not continuous run settings")
    run_id = config.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in run_id
        )
        or run_id in {".", ".."}
    ):
        raise ValueError("Invalid continuous generation run_id")
    observation = config.get("observation_version")
    if observation not in SUPPORTED_OBSERVATION_SERIALIZER_VERSIONS:
        raise ValueError("Unsupported configured observation version")
    capabilities = config.get("required_simulator_capabilities")
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise ValueError("required_simulator_capabilities must be a string array")
    interaction = config.get("interaction_contract")
    protocol_requirements = {
        CARD_SELECTION_VERSION: {
            "observations": {SELECTION_OBSERVATION_VERSION, ROUTE_OBSERVATION_VERSION},
            "capability": CARD_SELECTION_VERSION,
        }
    }
    if interaction is not None:
        requirements = protocol_requirements.get(interaction)
        if requirements is None:
            raise ValueError(f"Unsupported interaction contract: {interaction!r}")
        if observation not in requirements["observations"] or requirements["capability"] not in capabilities:
            raise ValueError("Configured observation/capabilities do not implement the interaction contract")

    strategies = resolve_generation_strategies(config.get("strategies"))
    paths = {
        name: resolve_repository_path(project_root, config[name], expected_kind="file")
        for name in ("scope", "picker_database")
    }
    scope = load_scope(paths["scope"], project_root=project_root)
    scope_interaction = scope.get("interaction_contract")
    if scope_interaction is not None and scope_interaction != interaction:
        raise ValueError("Configured scope and runner select different interaction contracts")
    card_ids = frozenset(map(str, scope["reward_cards"]["included_enum_ids"]))
    validate_reward_profile_coverage(tuple(sorted(card_ids)))
    route = config.get("route")
    if (
        not isinstance(route, dict)
        or not isinstance(route.get("operations"), list)
        or any(not isinstance(item, str) for item in route["operations"])
    ):
        raise ValueError("route.operations must be a string array")
    route_count = require_positive_int(config.get("route_count"), "route_count")
    combat_seed_groups = require_positive_int(
        config.get("combat_seed_groups_per_route"),
        "combat_seed_groups_per_route",
    )
    seeds = config.get("seeds")
    required_seeds = {
        "route", "encounter", "relic", "reward", "combat", "policy",
        "upgrade_combat",
        "picker", "upgrade", "remove",
    }
    if not isinstance(seeds, dict) or not required_seeds.issubset(seeds):
        raise ValueError(f"seeds must configure {sorted(required_seeds)}")
    for name in required_seeds:
        if isinstance(seeds[name], bool) or not isinstance(seeds[name], int):
            raise ValueError(f"seeds.{name} must be an integer")
    upgrade_policy = config.get("upgrade_policy")
    teacher_protocol = (
        upgrade_policy.get("teacher_protocol")
        if isinstance(upgrade_policy, Mapping)
        else None
    )
    if not isinstance(teacher_protocol, Mapping):
        raise ValueError("upgrade_policy.teacher_protocol must be an object")
    require_positive_int(
        teacher_protocol.get("search_budget"),
        "upgrade_policy.teacher_protocol.search_budget",
    )
    search_seed = teacher_protocol.get("search_seed")
    hidden_order_seeds = teacher_protocol.get("hidden_order_seeds")
    if isinstance(search_seed, bool) or not isinstance(search_seed, int):
        raise ValueError("upgrade_policy.teacher_protocol.search_seed must be an integer")
    if (
        not isinstance(hidden_order_seeds, list)
        or not hidden_order_seeds
        or any(isinstance(item, bool) or not isinstance(item, int) for item in hidden_order_seeds)
        or len(set(hidden_order_seeds)) != len(hidden_order_seeds)
    ):
        raise ValueError(
            "upgrade_policy.teacher_protocol.hidden_order_seeds must be unique integers"
        )
    routes = [
        {
            "route_index": route_index,
            "combat_seed_group_index": group_index,
            "steps": build_route_steps(
                config,
                scope,
                route_index,
                group_index,
                strategies.route,
            ),
        }
        for route_index in range(route_count)
        for group_index in range(combat_seed_groups)
    ]
    if observation == ROUTE_OBSERVATION_VERSION:
        for resolved_route in routes:
            for position, step in enumerate(resolved_route["steps"]):
                if step["kind"] == "combat":
                    public_route_context(resolved_route["steps"], position)

    arms = config.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("arms must be a nonempty array")
    resolved_arms = []
    for index, arm in enumerate(arms):
        if not isinstance(arm, dict) or not isinstance(arm.get("id"), str):
            raise ValueError(f"arms[{index}] is invalid")
        configured_arm_id = arm["id"]
        if (
            not configured_arm_id
            or configured_arm_id in {".", ".."}
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in configured_arm_id
            )
        ):
            raise ValueError(f"arms[{index}].id is invalid")
        kind = arm.get("policy")
        if kind == "llm":
            required = {"id", "policy", "model_runtime", "execution_profile"}
            allowed = required | {"checkpoint"}
        elif kind == "teacher_search":
            required = allowed = {"id", "policy", "search_budget", "search_seed"}
        else:
            raise ValueError(f"Unknown arm policy: {kind!r}")
        if required - arm.keys() or arm.keys() - allowed:
            raise ValueError(
                f"arms[{index}] fields: missing={sorted(required - arm.keys())}, "
                f"unsupported={sorted(arm.keys() - allowed)}"
            )
        for field in {"model_runtime", "execution_profile", "checkpoint"} & arm.keys():
            if not isinstance(arm[field], str) or not arm[field].strip():
                raise ValueError(f"arms[{index}].{field} must be a nonempty path")
        item = {"config": arm, "runtime": None, "execution_profile": None}
        if kind == "llm":
            runtime_path = resolve_repository_path(project_root, arm["model_runtime"], expected_kind="file")
            execution_path = resolve_repository_path(project_root, arm["execution_profile"], expected_kind="file")
            execution_profile = read_json_object(execution_path)
            item.update(
                runtime=load_base_model_runtime_config(
                    runtime_path, project_root=project_root,
                    execution_profile=execution_profile,
                ),
                execution_profile=execution_profile,
            )
            if "checkpoint" in arm:
                from sts1_llm_policy.train.lora import inspect_lora_checkpoint
                checkpoint = resolve_repository_path(
                    project_root, arm["checkpoint"], expected_kind="directory",
                )
                item["checkpoint"] = checkpoint
                item["checkpoint_metadata"] = inspect_lora_checkpoint(
                    checkpoint, expected_base_model_id=item["runtime"].model_id,
                    expected_base_revision=item["runtime"].revision,
                )
        elif kind == "teacher_search":
            require_positive_int(arm.get("search_budget"), f"arms[{index}].search_budget")
            teacher_seed = arm.get("search_seed")
            if isinstance(teacher_seed, bool) or not isinstance(teacher_seed, int):
                raise ValueError(f"arms[{index}].search_seed must be an integer")
        resolved_arms.append(item)
    if len({item["config"]["id"] for item in resolved_arms}) != len(resolved_arms):
        raise ValueError("Arm IDs must be unique")
    require_positive_int(config.get("max_decisions"), "max_decisions")
    source_validation = validate_source_isolation(config, routes, project_root=project_root)

    execution = resolve_simulator(
        project_root=project_root, target=target, required_capabilities=capabilities,
    )
    installation = execution.validate()
    output = resolve_repository_path(
        project_root, config.get("output_dir", f"outputs/generation/{config['run_id']}"),
        must_exist=False, expected_kind="directory",
    )
    identity = _configuration_identity(config, scope, resolved_arms, paths, installation, source_validation)
    return {
        "config": config, "config_path": config_path, "scope": scope,
        "card_ids": card_ids, "strategies": strategies, "routes": routes,
        "arms": resolved_arms, "execution": execution, "installation": installation,
        "output": output, "identity": identity, "paths": paths,
        "source_validation": source_validation,
    }


def _configuration_identity(config, scope, arms, paths, installation, source_validation):
    """Bind resolved behavior and asset content, independently of file locations."""
    semantics = {
        key: config[key] for key in (
            "strategies", "picker", "route", "route_count",
            "combat_seed_groups_per_route", "seeds", "protocol",
            "upgrade_policy",
            "observation_version", "interaction_contract",
            "required_simulator_capabilities", "max_decisions",
        )
    }
    semantics["arms"] = {
        item["config"]["id"]: {
            key: value for key, value in item["config"].items()
            if key not in {"model_runtime", "execution_profile", "checkpoint"}
        } for item in arms
    }
    semantics["data_source"] = {
        **source_validation,
        "excluded_sources": sorted(
            ({key: value for key, value in check.items() if key != "config"}
             for check in source_validation.get("excluded_sources", [])),
            key=canonical_sha,
        ),
    }
    if source_validation["evidence_class"] == CANDIDATE_EVIDENCE_CLASS:
        # Exporters require trajectory lineage.run_id to match the pool report.
        semantics["source_run_id"] = config["run_id"]
    models = {}
    for item in arms:
        if item["config"]["policy"] != "llm":
            continue
        model = asdict(item["runtime"])
        # Hardware requirements remain enforced by the runtime loader/backend.
        for name in (
            "runtime_id", "experiment_protocol_id", "snapshot_path", "device",
            "validated_device_name", "validated_compute_capability", "validated_cuda_runtime",
            "validated_bf16_supported", "tokenizer_local_only_verified", "model_weights_loaded",
            "generation_smoke_completed", "execution_requirements",
        ):
            model.pop(name)
        for name in ("weight_assets_sha256", "tokenizer_asset_sha256", "dependencies"):
            model[name] = dict(model[name])
        models[item["config"]["id"]] = {
            "model_runtime": canonical_sha(model),
            **({"checkpoint": canonical_sha({
                key: value for key, value in item["checkpoint_metadata"].items()
                if key != "weights_file"
            })} if "checkpoint_metadata" in item else {}),
        }
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "semantics_sha256": canonical_sha(semantics),
        # Match the consumers: reward eligibility is a set; encounter order
        # affects seeded selection. Scope descriptions are not execution inputs.
        "scope_sha256": canonical_sha({
            "reward_card_ids": sorted(set(map(str, scope["reward_cards"]["included_enum_ids"]))),
            "encounter_pools": scope["encounters"][f"act_{int(config['route']['act'])}"],
            "interaction_contract": scope.get("interaction_contract"),
        }),
        "picker_database_sha256": sha256_file(paths["picker_database"]),
        "arm_runtime_sha256": models,
        "simulator_revision": installation.revision,
        "native_bridge_sha256": sha256_file(installation.bridge_executable),
    }


def build_route_lineage(config: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    # All arms and formal seed groups of a source route share one split group.
    source = {
        "steps": [{key: value for key, value in step.items()
                   if key not in {"combat_seed", "policy_seed", "evaluation_targets"}}
                  for step in route["steps"]],
        "reward_seed": int(config["seeds"]["reward"]) + int(route["route_index"]),
    }
    return {
        "source_route_id": canonical_sha(source),
        "route_index": route["route_index"],
        "combat_seed_group_index": route["combat_seed_group_index"],
        "run_id": config["run_id"],
    }


def _source_seed_sets(config, routes):
    """Enumerate configured RNG inputs, including reward counterfactual offsets."""
    count = int(config["route_count"])
    result = {
        name: set(range(int(base), int(base) + count))
        for name, base in config["seeds"].items()
        if name not in {"combat", "policy", "upgrade_combat"}
    }
    for name in ("combat", "policy", "upgrade_combat"):
        result[name] = set()
    for route in routes:
        for step in route["steps"]:
            if step["kind"] == "combat":
                result["combat"].add(step["combat_seed"])
                result["policy"].add(step["policy_seed"])
            elif step["kind"] == "upgrade":
                result["upgrade_combat"].update(t["combat_seed"] for t in step["evaluation_targets"])
    result["reward_counterfactual"] = {
        seed + int(offset)
        for seed in result["combat"]
        for offset in (0, *config["protocol"].get("conflict_combat_seed_offsets", []))
    }
    return result


def validate_source_isolation(config, routes, *, project_root):
    source = config.get("data_source")
    if source is None:
        return {"evidence_class": "development_evaluation"}
    if (not isinstance(source, dict)
            or set(source) != {"evidence_class", "excluded_sources"}
            or source["evidence_class"] != CANDIDATE_EVIDENCE_CLASS):
        raise ValueError("data_source must declare teacher_candidate_pool and excluded_sources")
    if any(arm["policy"] != "teacher_search" for arm in config["arms"]):
        raise ValueError("Teacher candidate collection requires Teacher-only arms")
    excluded = source["excluded_sources"]
    if (not isinstance(excluded, list) or not excluded
            or any(not isinstance(p, str) for p in excluded) or len(set(excluded)) != len(excluded)):
        raise ValueError("excluded_sources must be nonempty unique paths")
    current_ids = {build_route_lineage(config, route)["source_route_id"] for route in routes}
    current_seeds = _source_seed_sets(config, routes)
    all_current_seeds = set().union(*current_seeds.values())
    checks = []
    for relative in excluded:
        path = resolve_repository_path(project_root, relative, expected_kind="file")
        previous = read_json_object(path)
        if (previous.get("schema_version") != "continuous_source_exclusions_v1"
                or set(previous) != {"schema_version", "origins", "source_route_ids", "seed_intervals"}):
            raise ValueError("Source isolation requires explicit continuous source exclusions")
        ids = previous["source_route_ids"]
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(v, str) or len(v) != 64
                       or any(c not in "0123456789abcdef" for c in v) for v in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Excluded source_route_ids must be unique SHA-256 identities")
        old_ids = set(ids)
        if current_ids.intersection(old_ids):
            raise ValueError(f"Source route overlap with {relative}")
        intervals = previous["seed_intervals"]
        if not isinstance(intervals, list) or not intervals:
            raise ValueError("Excluded seed_intervals must be nonempty closed integer intervals")
        old_seeds = set()
        last = -1
        for pair in intervals:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(v) is not int for v in pair)
                    or not last < pair[0] <= pair[1]):
                raise ValueError("Excluded seed_intervals must be ordered, nonoverlapping and nonnegative")
            old_seeds.update(range(pair[0], pair[1] + 1))
            last = pair[1]
        if all_current_seeds.intersection(old_seeds):
            raise ValueError(f"Source RNG seed overlap with {relative}")
        origins = previous["origins"]
        if not isinstance(origins, list) or not origins:
            raise ValueError("Excluded sources require origin identities")
        for origin in origins:
            if (not isinstance(origin, dict)
                    or set(origin) != {"configuration", "run_id", "output_dir", "route_executions"}
                    or any(not isinstance(origin[k], str) or not origin[k]
                           for k in ("configuration", "run_id", "output_dir"))):
                raise ValueError("Invalid excluded source origin")
            old_output = resolve_repository_path(project_root, origin["output_dir"], must_exist=False)
            if (origin["run_id"] == config["run_id"] or old_output ==
                    resolve_repository_path(project_root, config["output_dir"], must_exist=False)):
                raise ValueError("Candidate source must have a new run ID and output directory")
            executions = require_positive_int(origin["route_executions"], "origin.route_executions")
            # The old configuration label is provenance only; never open it.
            checks.append({"config": origin["configuration"], "source_routes": len(old_ids),
                           "semantics_sha256": canonical_sha({
                               "source_route_ids": sorted(old_ids), "seeds": sorted(old_seeds),
                           }),
                           "route_executions": executions, "overlap": 0})
    return {"evidence_class": CANDIDATE_EVIDENCE_CLASS,
            "source_routes": len(current_ids), "route_executions": len(routes),
            "excluded_sources": checks,
            "seed_ranges": {k: [min(v), max(v)] for k, v in current_seeds.items() if v},
            "split_unit": "source_route_id", "labels_certified": False}
