"""Prepare explicit GOLD collection inputs from a declared Teacher source pool."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import random

from sts1_llm_policy.data.trajectory import iter_trajectory_records
from sts1_llm_policy.data.gold.candidate_statistics import STATISTICS_VERSION
from sts1_llm_policy.data.gold.state_sampling import DEFAULT_WEIGHTS, VERSION, sample_decisions
from sts1_llm_policy.data.gold.configuration import (
    validate_config,
    validate_execution_settings,
    seed_for,
)
from sts1_llm_policy.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_json_object as read_json,
    resolve_repository_path,
    sha256_file,
)


def teacher_floors(report, margin):
    """Use the declared fixed route and aggregate death positions, not hidden states."""
    metrics = report["metrics"]["teacher"]
    families = [op.split(":", 1)[1] for op in report["configuration"]["route"]["operations"]
                if op.startswith("combat:")]
    if report["status"] != "completed" or metrics["truncated_routes"]:
        raise ValueError("A completed, untruncated fixed-route pool is required")
    arrivals = metrics["route_execution_count"]
    totals = defaultdict(lambda: [0, 0])
    for index, family in enumerate(families, 1):
        wins = arrivals - metrics["death_combat_index_counts"].get(str(index), 0)
        key = "normal" if family.startswith("normal_") else family
        totals[key][0] += wins
        totals[key][1] += arrivals
        arrivals = wins
    if (sum(v[0] for v in totals.values()), sum(v[1] for v in totals.values())) != (metrics["combat_victories"], metrics["combat_count"]):
        raise ValueError("Fixed-route death counts do not reproduce the combat totals")
    rates = {key: {"wins": w, "battles": n, "win_rate": w / n} for key, (w, n) in totals.items()}
    floors = {f: rates["normal" if f.startswith("normal_") else f]["win_rate"] - margin for f in set(families)}
    if any(not 0 < v < 1 for v in floors.values()):
        raise ValueError("Invalid reference win-rate margin")
    return dict(sorted(floors.items())), rates


def select_panel(root, report_path, *, route_count, groups_per_route, seed, states_per_combat,
                 sampling=VERSION, sampling_weights=None, excluded_routes=None, progress=False):
    if sampling not in {VERSION, "uniform_decisions_v1"}:
        raise ValueError("Unsupported decision sampling strategy")
    report = read_json(root / report_path)
    if report["status"] != "completed" or report["configuration"]["data_source"]["evidence_class"] != "teacher_candidate_pool":
        raise ValueError("A completed fresh candidate pool is required")
    scope = report["scope"]
    if not 1 <= route_count <= scope["routes"] or not 1 <= groups_per_route <= scope["combat_seed_groups_per_route"]:
        raise ValueError("Requested route/group count exceeds the declared pool")
    if not states_per_combat or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in states_per_combat.values()):
        raise ValueError("Per-combat counts must be positive integers")
    excluded = set(excluded_routes or [])
    if any(isinstance(r, bool) or not isinstance(r, int) or not 0 <= r < scope["routes"] for r in excluded):
        raise ValueError("Excluded route outside source pool")
    available_routes = [r for r in range(scope["routes"]) if r not in excluded]
    if route_count > len(available_routes):
        raise ValueError("Not enough routes after excluding tuning sources")
    rng = random.Random(seed)
    routes = sorted(rng.sample(available_routes, route_count))
    slots = [{"route": r, "groups": sorted(rng.sample(range(scope["combat_seed_groups_per_route"]), groups_per_route))}
             for r in routes]
    samples, selections, combats = [], [], []
    base = (root / report_path).parent
    for slot_index, slot in enumerate(slots):
        for group in slot["groups"]:
            route = read_json(base / f"teacher/routes/route-{slot['route']:03d}-combat-seed-group-{group:02d}.json")
            for combat in route["combats"]:
                family = combat["encounter_family"]
                if family not in states_per_combat:
                    raise ValueError(f"Missing decision count for {family}")
                records = list(iter_trajectory_records(base / combat["trajectory"]))
                if len(records) != combat["summary"]["decisions"] or not records:
                    raise ValueError("Source decision count differs from combat summary")
                if len({r["step_index"] for r in records}) != len(records):
                    raise ValueError("Duplicate source decision positions")
                combat_rng = random.Random(seed_for(seed, "combat-states", slot["route"], group, combat["combat_index"]))
                if sampling == VERSION:
                    choices, coverage = sample_decisions(records, states_per_combat[family], combat_rng,
                                                        weights=sampling_weights)
                else:
                    choices = {i: {"conditional_decision_inclusion_probability": min(states_per_combat[family], len(records)) / len(records)}
                               for i in combat_rng.sample(range(len(records)), min(states_per_combat[family], len(records)))}
                    coverage = {}
                chosen = sorted(choices)
                combats.append({"route": slot["route"], "group": group, "combat": combat["combat_index"],
                                "encounter_family": family, "available_decisions": len(records),
                                "selected_decisions": len(chosen), **coverage})
                for index in chosen:
                    record = records[index]
                    sample = {"id": f"collect-r{slot['route']:03d}-g{group:02d}-c{combat['combat_index']:02d}-s{record['step_index']:03d}",
                              "route": slot["route"], "group": group, "combat": combat["combat_index"], "step": record["step_index"]}
                    samples.append(sample)
                    selections.append({"sample": sample, "encounter_family": family, "scenario": combat["scenario_id"],
                                       "trajectory": combat["trajectory"], "available_decisions_in_combat": len(records),
                                       **choices[index],
                                       "source_remaining_decisions": len(records) - index,
                                       "native_root_count": len(record["legal_actions"]),
                                       "model_root_count": len(record["model_legal_actions"])})
        if progress:
            print(f"[select] routes {slot_index + 1}/{len(slots)} ({100 * (slot_index + 1) / len(slots):.2f}%) | states {len(samples)}", flush=True)
    return samples, {"schema_version": "gold_collection_selection_v2" if sampling == VERSION else "gold_collection_selection_v1", "status": "completed",
                     "source_report": str(report_path).replace("\\", "/"), "seed": seed,
                     "sampling": sampling,
                     "sampling_weights": dict(DEFAULT_WEIGHTS if sampling_weights is None else sampling_weights) if sampling == VERSION else None,
                     "route_count": route_count, "groups_per_route": groups_per_route,
                     "route_inclusion_probability": route_count / len(available_routes),
                     "conditional_group_inclusion_probability": groups_per_route / scope["combat_seed_groups_per_route"],
                     "states_per_combat": states_per_combat, "route_groups": slots,
                     "combats": combats, "selections": selections,
                     "state_count": len(samples), "model_root_count": sum(s["model_root_count"] for s in selections),
                     "states_by_family": dict(sorted(Counter(s["encounter_family"] for s in selections).items())),
                     "states_by_sampling_channel": dict(sorted(Counter(s.get("sampling_channel", "uniform") for s in selections).items())),
                     "selected_event_counts": dict(sorted(Counter(t for s in selections for t in s.get("events", [])).items())),
                     **({"excluded_routes": sorted(excluded), "eligible_route_count": len(available_routes),
                         "unselected_eligible_routes": sorted(set(available_routes) - set(routes))} if excluded_routes is not None else {})}


def smoke_samples(selection, *, include_mechanisms=False):
    choices = [max((s for s in selection["selections"] if s["encounter_family"] == f),
                   key=lambda s: s["source_remaining_decisions"])["sample"]["id"]
               for f in sorted(selection["states_by_family"])]
    if include_mechanisms:
        for event in ("secondary_selection", "hand_access"):
            available = [s for s in selection["selections"] if event in s.get("events", []) and s["sample"]["id"] not in choices]
            if available:
                choices.append(min(available, key=lambda s: (s["model_root_count"], s["sample"]["id"]))["sample"]["id"])
    return choices


def prepare_collection(root, *, template_config, config_output, output,
                       route_count, groups_per_route, seed, states_per_combat,
                       sampling=VERSION, sampling_weights=None, workers=4,
                       exclusions=None, training_candidates=False, adaptive=False,
                       hp_tolerance=5.0, trace_observation="full", compact_storage=False,
                       route_progress=False):
    """Select a panel and validate both documents before publishing new inputs."""
    root = Path(root).resolve()
    template_path = resolve_repository_path(root, template_config, expected_kind="file")
    config_path = resolve_repository_path(root, config_output, must_exist=False, expected_kind="file")
    output_path = resolve_repository_path(root, output, must_exist=False, expected_kind="directory")
    selection_path = output_path / "selection.json"
    if config_path == selection_path:
        raise ValueError("Config and selection must have separate destinations")
    template = read_json(template_path)
    source_path = resolve_repository_path(root, template["source_report"], expected_kind="file")
    excluded = set()
    if exclusions is not None:
        previous = read_json(resolve_repository_path(root, exclusions, expected_kind="file"))
        if (set(previous) != {"source_report", "source_report_sha256", "excluded_routes"}
                or previous["source_report"] != template["source_report"]
                or previous["source_report_sha256"] != sha256_file(source_path)):
            raise ValueError("Exclusions belong to a different source pool or report content")
        routes = previous["excluded_routes"]
        if (not isinstance(routes, list) or any(type(r) is not int or r < 0 for r in routes)
                or len(set(routes)) != len(routes)):
            raise ValueError("Excluded routes must be distinct nonnegative integers")
        excluded.update(routes)
    if training_candidates and exclusions is None:
        raise ValueError("Training candidate collection must explicitly exclude tuning routes")

    # Only execution settings are inherited. Panel membership, partitions and
    # import/storage settings belong to this newly prepared collection.
    config = deepcopy(template)
    for key in ("selection", "samples", "smoke_samples", "panel_groups", "source_partition", "sampling_ladder", "resume_from",
                "continuation_storage", "route_progress"):
        config.pop(key, None)
    config.update(seed=seed, output=output_path.relative_to(root).as_posix(), workers=workers,
                  win_floor_rounding="floor", expected_hp_tolerance=hp_tolerance,
                  card_statistics=STATISTICS_VERSION, trace_observation=trace_observation)
    if adaptive:
        config.update(outer_trials=128, sampling_ladder={"mode": "adaptive", "stages": [32, 64, 128],
                      "hp_boundary_band": .25, "paired_se_multiplier_at_64": 2.,
                      "audit_probability": .1, "audit_seed": seed})
    if compact_storage:
        config["continuation_storage"] = {"mode": "compact_v1", "full_trace_probability": .01,
                                          "keep_first_trial": True, "keep_boundary_trials": True}
    if route_progress:
        config["route_progress"] = True
    validate_execution_settings(config)

    samples, selection = select_panel(
        root, Path(template["source_report"]), route_count=route_count,
        groups_per_route=groups_per_route, seed=seed, states_per_combat=states_per_combat,
        sampling=sampling, sampling_weights=sampling_weights,
        excluded_routes=sorted(excluded) if exclusions is not None else None, progress=route_progress,
    )
    floors, references = teacher_floors(read_json(source_path), .01)
    config.update(samples=samples, smoke_samples=smoke_samples(selection, include_mechanisms=training_candidates),
                  minimum_win_rate=floors)
    if training_candidates:
        config["source_partition"] = {"role": "train_candidate", "excluded_routes": sorted(excluded),
                                      "exclusions": str(exclusions),
                                      "reserved_routes": selection["unselected_eligible_routes"]}
    selection.update(configuration=config_path.relative_to(root).as_posix(),
                     template_config=template_path.relative_to(root).as_posix(),
                     reference_win_rates=references, reference_margin=.01, minimum_win_rate=floors,
                     win_floor_rounding="floor", expected_hp_tolerance=hp_tolerance,
                     planned_executions=selection["model_root_count"] * config["outer_trials"])
    if training_candidates:
        selection["source_partition"] = config["source_partition"]
    selection["samples"] = samples
    if adaptive:
        selection.update(planned_executions_is_maximum=True,
                         minimum_executions=selection["model_root_count"] * 32)
    validate_config(config)
    # Execution consumes this one selection; reports/identities still use its
    # expanded contents, never the location of the selection file.
    config.pop("samples")
    config.pop("source_partition", None)
    config["selection"] = selection_path.relative_to(root).as_posix()
    destinations = [(config_path, config), (selection_path, selection)]
    for path, content in destinations:
        canonical_json_bytes(content)
        if path.exists() and read_json(path) != content:
            raise ValueError(f"Existing collection inputs differ: {path}; use separate output/config paths")
        if path.with_suffix(path.suffix + ".tmp").exists():
            raise ValueError(f"Conflicting temporary artifact exists: {path}")
    for path, content in destinations:
        if not path.exists():
            atomic_write_json(path, content, allow_identical=False)
    return selection
