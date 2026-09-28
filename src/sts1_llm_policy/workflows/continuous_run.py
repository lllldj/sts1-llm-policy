"""Execute configured continuous routes with shared Teacher and student behavior."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import gc
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Mapping

from sts1_llm_policy.data.trajectory import TrajectoryLogger, iter_trajectory_records
from sts1_llm_policy.env.route_context import ROUTE_OBSERVATION_VERSION, public_route_context
from sts1_llm_policy.eval.episode_runner import run_combat_episode
from sts1_llm_policy.env.combat_snapshot import resign_snapshot
from sts1_llm_policy.eval.reward_generation import RewardGenerationClient
from sts1_llm_policy.eval.generation_strategies import CardSelectionContext
from sts1_llm_policy.policy.llm_policy import LLMPolicy
from sts1_llm_policy.policy.teacher_search_policy import TeacherSearchPolicy
from sts1_llm_policy.policy.transformers_backend import TransformersGenerationBackend
from sts1_llm_policy.artifacts import replace_json, sha256_file
from .continuous_plan import (
    require_positive_int,
    operation_counts,
    prepare_run,
    build_route_lineage,
)
from .continuous_artifacts import (
    canonical_sha,
    summarize_combat,
    prepare_output,
    load_completed_route,
    summarize_arm,
    OUTPUT_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    CANDIDATE_EVIDENCE_CLASS,
)


def _stateful_snapshot(
    snapshot: Mapping[str, Any], run_state: Mapping[str, Any], *, floor: int,
) -> dict[str, Any]:
    result = deepcopy(dict(snapshot))
    result["floor"] = floor
    result["current_hp"] = int(run_state["current_hp"])
    result["max_hp"] = int(run_state["max_hp"])
    counters = {str(item["id"]): int(item.get("counter", 0)) for item in run_state["relics"]}
    for relic in result.get("relics", []):
        relic_id = str(relic["id"])
        if relic_id in counters:
            relic["counter"] = counters[relic_id]
    return resign_snapshot(result)


def _reward_with_run_state(
    reward: Mapping[str, Any], run_state: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(reward))
    state = result.get("reward_state")
    if not isinstance(state, dict):
        raise ValueError("Native reward response has no reward_state")
    state["current_hp"] = int(run_state["current_hp"])
    state["max_hp"] = int(run_state["max_hp"])
    counters = {str(item["id"]): int(item.get("counter", 0)) for item in run_state["relics"]}
    for relic in state.get("relics", []):
        relic_id = str(relic["id"])
        if relic_id in counters:
            relic["counter"] = counters[relic_id]
    return result


def _run_route(
    validated: Mapping[str, Any], arm: Mapping[str, Any], route: Mapping[str, Any],
    *, backend: Any, picker: Any, rate_table: Any, route_root: Path,
) -> dict[str, Any]:
    from sts1_llm_policy.eval.counterfactual_reward_picker_v2 import TeacherCapabilityEvaluatorV2

    config = validated["config"]
    arm_config = arm["config"]
    arm_id = str(arm_config["id"])
    route_index = int(route["route_index"])
    group_index = int(route["combat_seed_group_index"])
    steps = route["steps"]
    final_combat = [item for item in steps if item["kind"] == "combat"][-1]
    reward_client: RewardGenerationClient = validated["execution"].create_client()
    env = validated["execution"].create_environment()
    trajectory_root = (
        route_root
        / f"route-{route_index:03d}"
        / f"combat-seed-group-{group_index:02d}"
    )
    reward_events: list[dict[str, Any]] = []
    topology_events: list[dict[str, Any]] = []
    combats: list[dict[str, Any]] = []
    run_state: dict[str, Any] | None = None
    card_rng = random.Random(int(config["seeds"]["picker"]) + route_index)
    upgrade_rng = random.Random(int(config["seeds"]["upgrade"]) + route_index)
    picker_evaluator = TeacherCapabilityEvaluatorV2(
        env, combat_seed=int(final_combat["combat_seed"]), protocol=config["protocol"],
    )
    selector = validated["strategies"].card_pick(
        picker=picker, evaluator=picker_evaluator, rate_lookup=rate_table,
        spec={
            "final_boss_scenario_id": final_combat["scenario_id"],
            "act_1_boss_scenario_id": final_combat["scenario_id"],
            "boss_relic_id": None,
        },
        allowed_card_ids=validated["card_ids"],
        epsilon=float(config["picker"]["epsilon"]),
        total_rewards=operation_counts(
            tuple(map(str, config["route"]["operations"]))
        )["card_pick"],
        protocol=config["protocol"],
    )
    upgrade_count = removal_count = 0
    route_lineage = build_route_lineage(config, route)
    remove_rng = random.Random(int(config["seeds"]["remove"]) + route_index)
    try:
        for position, step in enumerate(steps):
            kind = str(step["kind"])
            if kind == "reset":
                reset = reward_client.reward_reset(
                    int(config["seeds"]["reward"]) + route_index,
                    ascension=int(config["route"]["ascension"]), act=int(step["act"]),
                    relics=step["relics"],
                )
                state = reset["reward_state"]
                run_state = {
                    "current_hp": int(state["current_hp"]),
                    "max_hp": int(state["max_hp"]),
                    "relics": [
                        {"id": str(item["id"]), "counter": int(item.get("counter", 0))}
                        for item in state["relics"]
                    ],
                }
                continue
            if run_state is None:
                raise ValueError("Continuous route did not begin with reset")
            if kind == "combat":
                exported = reward_client.export_combat_snapshot()["combat_snapshot"]
                snapshot = _stateful_snapshot(exported, run_state, floor=int(step["floor"]))
                combat_index = int(step["combat_index"])
                partial = trajectory_root / f"combat-{combat_index:02d}.partial.jsonl"
                final = trajectory_root / f"combat-{combat_index:02d}.jsonl"
                partial.parent.mkdir(parents=True, exist_ok=True)
                for stale in (partial, final):
                    if stale.exists():
                        stale.unlink()
                logger = TrajectoryLogger(
                    partial,
                    episode_id=(
                        f"{config['run_id']}-{arm_id}-{route_index:03d}-"
                        f"g{group_index:02d}-{combat_index:02d}"
                    ),
                    game_seed=int(step["combat_seed"]), policy_seed=int(step["policy_seed"]),
                    policy_name=arm_id, encounter=str(step["scenario_id"]),
                    scenario_id=str(step["scenario_id"]),
                    evidence_class=config.get("data_source", {}).get(
                        "evidence_class", "development_evaluation",
                    ),
                    observation_serializer_version=str(config["observation_version"]),
                    route_lineage={**route_lineage, "arm": arm_id, "combat_index": combat_index},
                )
                env.reset(
                    str(step["scenario_id"]), int(step["combat_seed"]),
                    combat_snapshot=snapshot,
                    **({"route_context": public_route_context(steps, position)}
                       if config["observation_version"] == ROUTE_OBSERVATION_VERSION else {}),
                )
                if arm_config["policy"] == "llm":
                    policy = LLMPolicy(
                        backend, int(step["policy_seed"]),
                        observation_version=str(config["observation_version"]),
                    )
                else:
                    policy = TeacherSearchPolicy(
                        env, simulations=int(arm_config["search_budget"]),
                        search_seed=int(arm_config["search_seed"]),
                    )
                result = run_combat_episode(
                    env, policy, logger, max_steps=int(config["max_decisions"]),
                )
                records = tuple(iter_trajectory_records(partial))
                summary = summarize_combat(records, result.outcome.value)
                if config.get("data_source", {}).get("evidence_class") == CANDIDATE_EVIDENCE_CLASS:
                    from sts1_llm_policy.data.trajectory_replay import replay_combat
                    summary["replay"] = replay_combat(
                        env, scenario_id=str(step["scenario_id"]),
                        combat_seed=int(step["combat_seed"]), snapshot=snapshot,
                        records=records,
                        route_context=(public_route_context(steps, position)
                                       if config["observation_version"] == ROUTE_OBSERVATION_VERSION
                                       else None),
                    )
                partial.replace(final)
                combat_record = {
                    **{key: step[key] for key in (
                        "combat_index", "floor", "encounter_family", "scenario_id",
                        "combat_seed", "policy_seed",
                    )},
                    "input_snapshot": snapshot,
                    "input_snapshot_sha256": canonical_sha(snapshot),
                    "trajectory": str(final.relative_to(validated["output"])).replace("\\", "/"),
                    "trajectory_sha256": sha256_file(final), "summary": summary,
                }
                combats.append(combat_record)
                if summary["outcome"] == "aborted":
                    if result.termination_reason != "max_steps":
                        raise RuntimeError("Continuous combat aborted unexpectedly")
                    break
                run_state.update(
                    current_hp=summary["ending_hp"], max_hp=summary["max_hp"],
                    relics=summary["relics"],
                )
                if summary["outcome"] == "defeat":
                    break
            elif kind == "reward":
                reward = reward_client.sample_card_reward(
                    room=str(step["room"]), floor=int(step["floor"]),
                )
                visible_reward = _reward_with_run_state(reward, run_state)
                choice, probabilities, details = selector(
                    visible_reward, int(step["act"]), int(step["floor"]), card_rng,
                )
                before_size = len(visible_reward["reward_state"]["deck"])
                applied = reward_client.apply_card_reward_choice(reward["reward_id"], choice)
                if choice == "SINGING_BOWL":
                    run_state["current_hp"] = min(
                        int(run_state["max_hp"]) + 2, int(run_state["current_hp"]) + 2
                    )
                    run_state["max_hp"] = int(run_state["max_hp"]) + 2
                reward_events.append({
                    "kind": "card_pick", "checkpoint": step["checkpoint"],
                    "floor": step["floor"], "room": step["room"],
                    "offered_cards": reward["choices"], "choice": choice,
                    "choice_probabilities": probabilities, "picker_details": details,
                    "deck_size_before": before_size,
                    "deck_size_after": len(applied["reward_state"]["deck"]),
                })
            elif kind == "ordinary_relic":
                reward_client.obtain_reward_relic(str(step["relic_id"]))
                topology_events.append(dict(step))
            elif kind == "remove":
                exported = reward_client.export_combat_snapshot()["combat_snapshot"]
                snapshot = _stateful_snapshot(exported, run_state, floor=int(exported["floor"]))
                removal_count += 1
                context = CardSelectionContext(1, str(step["checkpoint"]), removal_count)
                card_id, reason = validated["strategies"].card_remove(
                    snapshot["deck"], context, remove_rng,
                )
                removed = reward_client.remove_reward_card(card_id)
                topology_events.append({
                    "kind": "card_removal", "checkpoint": step["checkpoint"],
                    "card_id": card_id, "reason": reason,
                    "removed_card": removed["removed_card"],
                })
            elif kind == "upgrade":
                exported = reward_client.export_combat_snapshot()["combat_snapshot"]
                snapshot = _stateful_snapshot(exported, run_state, floor=int(exported["floor"]))
                evaluator = TeacherCapabilityEvaluatorV2(
                    env, combat_seed=0,
                    protocol=config["upgrade_policy"]["teacher_protocol"],
                )
                upgrade_count += 1
                upgrade_decisions: list[Mapping[str, Any]] = []
                context = CardSelectionContext(
                    1, str(step["checkpoint"]), upgrade_count,
                    combat_snapshot=snapshot, evaluator=evaluator,
                    decision_recorder=upgrade_decisions.append,
                    evaluation_targets=tuple(step["evaluation_targets"]),
                )
                deck_index, reason = validated["strategies"].card_upgrade(
                    snapshot["deck"], context, upgrade_rng,
                )
                upgraded = reward_client.upgrade_reward_card(deck_index)
                topology_events.append({
                    "kind": "card_upgrade", "checkpoint": step["checkpoint"],
                    "reason": reason,
                    "card_before": upgraded["upgraded_card_before"],
                    "card_after": upgraded["upgraded_card_after"],
                    "teacher_search_calls": evaluator.search_calls,
                    "decision_evidence": upgrade_decisions[0] if upgrade_decisions else None,
                })
            elif kind == "heal":
                before = int(run_state["current_hp"])
                run_state["current_hp"] = min(
                    int(run_state["max_hp"]), before + int(step["amount"])
                )
                topology_events.append({
                    **dict(step), "current_hp_before": before,
                    "current_hp_after": run_state["current_hp"],
                })
            else:
                raise ValueError(f"Unsupported continuous route operation: {kind!r}")
    finally:
        reward_client.close()
        env.close()

    defeated = bool(combats and combats[-1]["summary"]["outcome"] == "defeat")
    truncated = bool(combats and combats[-1]["summary"]["outcome"] == "aborted")
    return {
        "schema_version": "continuous_combat_route_v1",
        "configuration_identity": validated["identity"],
        "arm": arm_id, "route": route,
        "route_lineage": {**route_lineage, "arm": arm_id},
        "status": "truncated" if truncated else ("defeated" if defeated else "completed"),
        **({"truncation_reason": "decision_limit",
            "truncation_combat_index": int(combats[-1]["combat_index"])} if truncated else {}),
        "combats_started": len(combats),
        "combats_won": sum(item["summary"]["outcome"] == "victory" for item in combats),
        "death_combat_index": int(combats[-1]["combat_index"]) if defeated else None,
        "combats": combats, "reward_events": reward_events,
        "topology_events": topology_events, "test_data_read": False,
    }


def _run_teacher_route(validated, arm, route):
    """Each concurrent route owns its picker, RNGs and native processes."""
    from sts1_llm_policy.data.card_pick_metrics import (
        MassNormalizedCardRewardPicker,
        SqliteCardRateBackend,
    )
    from sts1_llm_policy.eval.counterfactual_reward_picker_v2 import StrategicRateTable
    database = SqliteCardRateBackend(validated["paths"]["picker_database"])
    return _run_route(
        validated, arm, route, backend=None,
        picker=MassNormalizedCardRewardPicker(database), rate_table=StrategicRateTable(database),
        route_root=validated["output"] / arm["config"]["id"] / "route-work",
    )


def _parallel_route_results(validated, arm, routes, workers):
    # Heavy search runs in independent native processes, outside Python's GIL.
    # Bound in-flight work so interruption does not drain a queue of all routes.
    remaining = iter(routes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}

        def submit_next():
            route = next(remaining, None)
            if route is not None:
                active[pool.submit(_run_teacher_route, validated, arm, route)] = route

        for _ in range(workers):
            submit_next()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                route = active.pop(future)
                yield route, future.result()
                submit_next()


def run(
    config_path: str | Path, *, project_root: str | Path, target: str = "auto",
    preflight_only: bool = False, smoke_routes: int | None = None,
    arm_id: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    from sts1_llm_policy.data.card_pick_metrics import (
        MassNormalizedCardRewardPicker,
        SqliteCardRateBackend,
    )
    from sts1_llm_policy.eval.counterfactual_reward_picker_v2 import StrategicRateTable

    root = Path(project_root).resolve()
    requested_arm_id = arm_id
    require_positive_int(workers, "workers")
    if preflight_only and smoke_routes is not None:
        raise ValueError("Preflight and smoke are separate modes")
    validated = prepare_run(Path(config_path), project_root=root, target=target)
    routes = list(validated["routes"])
    selected_arms = list(validated["arms"])
    if requested_arm_id is not None:
        selected_arms = [
            item for item in selected_arms
            if item["config"]["id"] == requested_arm_id
        ]
        if not selected_arms:
            available = [item["config"]["id"] for item in validated["arms"]]
            raise ValueError(
                f"Unknown --arm {requested_arm_id!r}; choose one of {available}"
            )
    if smoke_routes is not None:
        configured_routes = int(validated["config"]["route_count"])
        if not 1 <= smoke_routes <= configured_routes:
            raise ValueError(f"--smoke-routes must be between 1 and {configured_routes}")
        routes = [route for route in routes if int(route["route_index"]) < smoke_routes]
    if workers > 1 and any(arm["config"]["policy"] != "teacher_search" for arm in selected_arms):
        raise ValueError("Concurrent generation supports Teacher-only arms")
    if preflight_only:
        return {
            "status": "preflight_passed",
            "routes": int(validated["config"]["route_count"]),
            "combat_seed_groups_per_route": int(
                validated["config"]["combat_seed_groups_per_route"]
            ),
            "route_executions": len(validated["routes"]),
            "arms": [item["config"]["id"] for item in selected_arms],
            "configuration_identity": validated["identity"],
            "source_validation": validated.get("source_validation"),
            "execution": validated["execution"].describe(),
            "model_weights_loaded": False, "simulator_started": False,
            "workers": workers,
        }
    output = validated["output"] / ("smoke" if smoke_routes is not None else "formal")
    prepare_output(output, validated["identity"])
    validated = {**validated, "output": output}
    backend_db = SqliteCardRateBackend(validated["paths"]["picker_database"])
    picker = MassNormalizedCardRewardPicker(backend_db)
    rate_table = StrategicRateTable(backend_db)
    results: dict[str, list[dict[str, Any]]] = {}
    timing: dict[str, Any] = {}
    started = perf_counter()
    for arm in selected_arms:
        arm_started = perf_counter()
        current_arm_id = str(arm["config"]["id"])
        route_dir = output / current_arm_id / "routes"
        completed: list[dict[str, Any]] = []
        pending: list[Mapping[str, Any]] = []
        for route in routes:
            route_index = int(route["route_index"])
            group_index = int(route["combat_seed_group_index"])
            route_name = f"route-{route_index:03d}-combat-seed-group-{group_index:02d}.json"
            item = load_completed_route(
                route_dir / route_name,
                identity=validated["identity"], arm=current_arm_id, route=route,
                output_root=output,
                evidence_class=validated["config"].get("data_source", {}).get("evidence_class"),
            )
            (pending if item is None else completed).append(route if item is None else item)
        print(
            f"{current_arm_id}: validated {len(completed)} completed routes; "
            f"{len(pending)} pending",
            flush=True,
        )
        backend = None
        loaded = perf_counter()
        if pending and arm["config"]["policy"] == "llm":
            backend = TransformersGenerationBackend.from_config(
                arm["runtime"], project_root=root,
                lora_checkpoint_dir=arm.get("checkpoint"),
                verified_lora_metadata=arm.get("checkpoint_metadata"),
            )
            loaded = perf_counter()
        if workers == 1:
            route_results = (
                (route, _run_route(
                    validated, arm, route, backend=backend, picker=picker,
                    rate_table=rate_table,
                    route_root=output / current_arm_id / "route-work",
                )) for route in pending
            )
        else:
            route_results = _parallel_route_results(validated, arm, pending, workers)
        for position, (route, item) in enumerate(route_results, start=1):
            route_index = int(route["route_index"])
            group_index = int(route["combat_seed_group_index"])
            route_name = f"route-{route_index:03d}-combat-seed-group-{group_index:02d}.json"
            replace_json(route_dir / route_name, item)
            completed.append(item)
            print(
                f"[{position:03d}/{len(pending):03d}] {current_arm_id} route={route_index} "
                f"combat_seed_group={group_index} "
                f"combats={item['combats_started']} status={item['status']}", flush=True,
            )
        timing[current_arm_id] = {
            "model_load_seconds": round(loaded - arm_started, 3) if backend is not None else 0.0,
            "new_routes": len(pending),
        }
        if backend is not None:
            del backend.model
            del backend
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        by_index = {
            (
                int(item["route"]["route_index"]),
                int(item["route"]["combat_seed_group_index"]),
            ): item
            for item in completed
        }
        results[current_arm_id] = [
            by_index[(
                int(route["route_index"]),
                int(route["combat_seed_group_index"]),
            )]
            for route in routes
        ]

    inputs = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": validated["config"]["run_id"],
        "observation_version": validated["config"]["observation_version"],
        "interaction_contract": validated["config"].get("interaction_contract"),
        "configuration_identity": validated["identity"],
        "routes": routes,
        "source_validation": validated.get("source_validation"),
        "arms": results,
        "test_data_read": False, "sealed_test_run": False,
    }
    artifact_suffix = f"-{requested_arm_id}" if requested_arm_id is not None else ""
    artifact_name = f"inputs{artifact_suffix}.json"
    report_name = f"report{artifact_suffix}.json"
    replace_json(output / artifact_name, inputs)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": validated["config"]["run_id"],
        "status": "smoke_completed" if smoke_routes is not None else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": validated["config"],
        "configuration_identity": validated["identity"],
        "execution": validated["execution"].describe(),
        "source_validation": validated.get("source_validation"),
        "scope": {
            "routes": len({int(route["route_index"]) for route in routes}),
            "combat_seed_groups_per_route": int(
                validated["config"]["combat_seed_groups_per_route"]
            ),
            "route_executions": len(routes),
            "arms": len(results),
        },
        "metrics": {arm: summarize_arm(items) for arm, items in results.items()},
        "timing": {**timing, "workers": workers, "run_seconds": round(perf_counter() - started, 3)},
        "artifact": artifact_name, "test_data_read": False,
        "sealed_test_run": False, "training_started": False, "stable_claim": False,
    }
    replace_json(output / report_name, report)
    return report
