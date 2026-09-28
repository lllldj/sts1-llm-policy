"""Combine existing GOLD groups with replayed, explicitly separate demonstrations."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import math
import time

from sts1_llm_policy.data.gold.candidate_statistics import root_records
from sts1_llm_policy.data.trajectory import iter_trajectory_records
from sts1_llm_policy.env.action_equivalence import build_model_action_space
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.env.simulator_env import StsLightspeedEnv
from sts1_llm_policy.env.simulator_execution import resolve_simulator, SELECTION
from sts1_llm_policy.env.route_context import public_route_context
from sts1_llm_policy.data.trajectory_replay import replay_combat
from sts1_llm_policy.data.export_artifacts import (
    json_artifact_bytes,
    validate_export_destinations,
    write_export_artifacts,
)
from sts1_llm_policy.data.manifest import build_dataset_manifest, validate_dataset_manifest
from sts1_llm_policy.artifacts import (
    gzip_jsonl_bytes,
    iter_jsonl,
    read_json_object,
    repository_relative,
    resolve_repository_path,
    sha256_bytes,
    sha256_file,
)
from sts1_llm_policy.configuration import load_config_document


SOURCES = {"gold", "teacher_strategy", "forced_end"}


def resolve_source_pool(root, cfg, gold_manifest, selection):
    """Locate immutable pool reports, retaining historical provenance references.

    Relocation is explicit and content-bound. The report's sibling route tree
    retains its published relative layout; no historical directory search occurs.
    """
    references = (cfg["source_report"], gold_manifest["lineage"]["source_pool"],
                  selection["source_report"])
    if any(not isinstance(value, str) or not value for value in references):
        raise ValueError("Source pool references must be non-empty paths")
    locations = cfg.get("source_locations", {})
    if not isinstance(locations, dict) or set(locations) - set(references):
        raise ValueError("source_locations must map only declared source pool reports")
    paths, digests = [], {}
    for reference in references:
        location = locations.get(reference)
        if location is not None:
            if (not isinstance(location, dict) or set(location) != {"path", "sha256"}
                    or not isinstance(location["path"], str) or not location["path"]
                    or not isinstance(location["sha256"], str) or len(location["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in location["sha256"])):
                raise ValueError("Source location requires a path and lowercase SHA-256")
        elif reference in locations:
            raise ValueError("Source location cannot be null")
        path = resolve_repository_path(root, location["path"] if location else reference,
                                       expected_kind="file")
        if path not in digests:
            digests[path] = sha256_file(path)
        if location and digests[path] != location["sha256"]:
            raise ValueError("Relocated source pool content mismatch")
        paths.append(path)
    if len(set(digests.values())) != 1:
        raise ValueError("Teacher source pool content differs from GOLD provenance")
    return paths[0], digests[paths[0]]


def sample_key(sample):
    return tuple(sample[k] for k in ("route", "group", "combat", "step"))


def validate_source_route(route, pool, route_index, group):
    # A defeated route is a finished execution; selected combats are checked
    # separately. Conditioning early demonstrations on a later Boss win biases them.
    if route["status"] not in {"completed", "defeated"}:
        raise ValueError(f"Teacher route {route_index}/g{group} is unfinished: {route['status']}")
    lineage = route["route_lineage"]
    if (route["configuration_identity"] != pool["configuration_identity"]
            or lineage["route_index"] != route_index or lineage["combat_seed_group_index"] != group
            or lineage["run_id"] != pool["run_id"] or lineage["arm"] != "teacher"):
        raise ValueError(f"Teacher route identity mismatch: route {route_index}/g{group}")


def validate_mixture_weights(masses):
    if (not isinstance(masses, dict) or set(masses) != SOURCES
            or any(isinstance(v, bool) or not isinstance(v, (float, int))
            or not math.isfinite(v) or v <= 0 for v in masses.values())
            or not math.isclose(math.fsum(masses.values()), 1., abs_tol=1e-12)):
        raise ValueError("Mixture source weights must be positive and sum to one")


def apply_mixture_weights(records, masses):
    """Mean-one state coefficients make the runner's mean CE the desired mixture."""
    validate_mixture_weights(masses)
    by_source = defaultdict(list)
    for record in records:
        by_source[record["supervision_source"]].append(record)
    if set(by_source) != SOURCES:
        raise ValueError("Every configured mixture source must have records")
    for source, rows in by_source.items():
        combats = Counter(sample_key(r["source"]["sample"])[:3] for r in rows)
        for row in rows:
            denominator = (len(combats) * combats[sample_key(row["source"]["sample"])[:3]]
                           if source == "teacher_strategy" else len(rows))
            row["loss_weight"] = len(records) * masses[source] / denominator
    return {source: {"states": len(rows), "loss_mass": math.fsum(r["loss_weight"] for r in rows) / len(records)}
            for source, rows in sorted(by_source.items())}


def training_route_groups(selection, gold_manifest):
    partition = selection["source_partition"]
    if partition != gold_manifest["lineage"]["source_partition"] or partition.get("role") != "train_candidate":
        raise ValueError("Teacher selection differs from GOLD training partition")
    excluded = set(partition["excluded_routes"]) | set(partition["reserved_routes"])
    groups = {}
    for sample in selection["samples"]:
        route, group = sample["route"], sample["group"]
        if route in excluded or (route in groups and groups[route] != group):
            raise ValueError("Teacher routes must use one declared seed group and exclude reserved routes")
        groups[route] = group
    if sorted(groups) != gold_manifest["lineage"]["source_routes"]:
        raise ValueError("Teacher routes differ from GOLD training routes")
    return groups


def demonstration_record(state, actions, original, *, dataset_id, sample):
    space = build_model_action_space(state, actions)
    selected = next((a for a in space.classes if asdict(a.representative) == original["action"]), None)
    if selected is None or selected.model_action_id != original["policy_result"]["selected_model_action_id"]:
        raise ValueError("Rebuilt Teacher action differs from executed model action")
    if original["policy_result"]["fallback_used"]:
        raise ValueError("Fallback is not a Teacher demonstration")
    forced = len(space.classes) == 1 and selected.action_type.value == "end_turn"
    observation = serialize_observation(state, actions, version="observation_v7")
    return {
        "schema_version": "decision_sft_group_v1", "dataset_id": dataset_id,
        "record_id": sample["id"], "split": "train", "observation_version": "observation_v7",
        "observation": observation, "observation_sha256": sha256(observation.encode()).hexdigest(),
        "source": {"sample": sample, "route_lineage": original["route_lineage"],
                   "encounter_family": state.route_context.current_encounter_family,
                   "encounter": original["encounter"], "evidence_class": "teacher_demonstration"},
        "supervision_source": "forced_end" if forced else "teacher_strategy",
        "model_actions": [{"model_action_id": a["id"], "action_type": a["action_type"],
                           "card_name": a["card"], "target": a["target"]}
                          for a in root_records(space, extended=True)],
        "candidates": [{"action_id": selected.model_action_id, "weight": 1.0}],
    }


def export_mixed_sft(root, config_path, *, smoke=False):
    started = time.monotonic()
    cfg = load_config_document(root, config_path, supported_schemas={"mixed_sft_export_v1"}).value
    required = {"schema_version", "dataset_id", "gold_manifest", "source_report", "source_selection",
                "output", "combat_indices", "encounter_families", "source_weights", "smoke_combats"}
    if set(cfg) - {"source_locations"} != required:
        raise ValueError("Unexpected or missing mixed SFT export fields")
    indices = cfg["combat_indices"]
    if (not isinstance(indices, list) or not indices or len(indices) != len(set(indices))
            or any(type(i) is not int or i < 1 for i in indices)):
        raise ValueError("combat_indices must be distinct positive integers")
    validate_mixture_weights(cfg["source_weights"])
    if not isinstance(cfg["dataset_id"], str) or not cfg["dataset_id"]:
        raise ValueError("dataset_id must be a non-empty string")
    families = cfg["encounter_families"]
    if (not isinstance(families, list) or not families
            or any(not isinstance(f, str) or not f for f in families)
            or len(set(families)) != len(families)):
        raise ValueError("encounter_families must be distinct non-empty strings")
    output = resolve_repository_path(root, cfg["output"], must_exist=False, expected_kind="directory")
    output = output / ("smoke" if smoke else "formal")
    artifact, manifest_path = output / "train.jsonl.gz", output / "manifest.json"
    validate_export_destinations([artifact, manifest_path])
    gold_path = resolve_repository_path(root, cfg["gold_manifest"], expected_kind="file")
    gold_manifest = read_json_object(gold_path)
    if (gold_manifest.get("task_type") != "sft" or gold_manifest.get("observation_version") != "observation_v7"
            or set(gold_manifest.get("splits", {})) != {"train"}):
        raise ValueError("Mixed SFT requires a train-only V7 SFT source")
    gold_manifest = validate_dataset_manifest(gold_manifest, project_root=root, verify_artifacts=True)
    selection_path = resolve_repository_path(root, cfg["source_selection"], expected_kind="file")
    selection = read_json_object(selection_path)
    source_path, source_digest = resolve_source_pool(root, cfg, gold_manifest, selection)
    pool = read_json_object(source_path)
    if (pool.get("status") != "completed"
            or pool["configuration"]["data_source"]["evidence_class"] != "teacher_candidate_pool"
            or any(s["overlap"] for s in pool["source_validation"]["excluded_sources"])):
        raise ValueError("Teacher source must be completed, isolated and match GOLD provenance")
    groups = training_route_groups(selection, gold_manifest)
    panel = [(r, g, c) for r, g in sorted(groups.items()) for c in indices]
    if smoke:
        chosen = [tuple(s[k] for k in ("route", "group", "combat")) for s in cfg["smoke_combats"]]
        if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(panel):
            raise ValueError("Smoke combats must be a distinct subset of the declared panel")
        panel = chosen
    dataset_id = cfg["dataset_id"] + ("_smoke" if smoke else "")
    gold = list(iter_jsonl(resolve_repository_path(root, gold_manifest["splits"]["train"]["path"], expected_kind="file")))
    if len(gold) != gold_manifest["splits"]["train"]["records"]:
        raise ValueError("GOLD artifact record count mismatch")
    overlap = {sample_key(r["source"]["sample"]): r for r in gold}
    if len(overlap) != len(gold):
        raise ValueError("Duplicate GOLD source states")
    for key in overlap:
        if groups.get(key[0]) != key[1]:
            raise ValueError("GOLD record falls outside its source partition")
    records = deepcopy(gold)
    for row in records:
        row["dataset_id"] = dataset_id
        row["supervision_source"] = "gold"
        row["source"]["dataset_id"] = gold_manifest["dataset_id"]
    counts = Counter()
    encounters = Counter()
    route_ids = set()
    execution = resolve_simulator(project_root=root, required_capabilities={SELECTION})
    native = execution.describe(verify=True)
    with execution.create_client() as client:
        env = StsLightspeedEnv(client, allow_card_selection=True)
        for position, (r, g, c) in enumerate(panel, 1):
            route_path = source_path.parent / f"teacher/routes/route-{r:03d}-combat-seed-group-{g:02d}.json"
            route = read_json_object(route_path)
            lineage = route["route_lineage"]
            validate_source_route(route, pool, r, g)
            route_ids.add(lineage["source_route_id"])
            combat = next(item for item in route["combats"] if item["combat_index"] == c)
            if combat["encounter_family"] not in cfg["encounter_families"]:
                raise ValueError("Selected combat has an undeclared encounter family")
            trajectory = resolve_repository_path(root, repository_relative(root, source_path.parent / combat["trajectory"]), expected_kind="file")
            if sha256_file(trajectory) != combat["trajectory_sha256"]:
                raise ValueError("Teacher trajectory differs from its source binding")
            original = list(iter_trajectory_records(trajectory))
            if (not original or not original[-1]["done"] or original[-1]["terminal_outcome"] != "victory"
                    or combat["summary"]["outcome"] != "victory"
                    or len(original) != combat["summary"]["decisions"]):
                raise ValueError("Teacher demonstrations require complete victorious combats")
            steps = route["route"]["steps"]
            step_position = next(i for i, s in enumerate(steps) if s["kind"] == "combat" and s["combat_index"] == c)
            def consume(index, state, actions):
                old = original[index]
                if old["route_lineage"] != {**lineage, "combat_index": c} or old["evidence_class"] != "teacher_candidate_pool":
                    raise ValueError("Teacher transition lineage mismatch")
                sample = {"id": f"teacher-r{r:03d}-g{g:02d}-c{c:02d}-s{index:03d}",
                          "route": r, "group": g, "combat": c, "step": index}
                row = demonstration_record(state, actions, old, dataset_id=dataset_id, sample=sample)
                counts["replayed_states"] += 1
                counts["known_top_states"] += bool(state.known_draw_top)
                counts["secondary_selection_states"] += row["model_actions"][0]["action_type"] == "select_card"
                prior = overlap.get(sample_key(sample))
                if prior is not None:
                    if row["model_actions"] != prior["model_actions"] or row["observation"] != prior["observation"]:
                        raise ValueError(f"Rebuilt GOLD overlap differs at {sample['id']}")
                    counts["gold_overlap_skipped"] += 1
                    counts["overlap_teacher_outside_gold"] += row["candidates"][0]["action_id"] not in {a["action_id"] for a in prior["candidates"]}
                else:
                    records.append(row)
            verified = replay_combat(env, scenario_id=combat["scenario_id"], combat_seed=combat["combat_seed"],
                                     snapshot=combat["input_snapshot"], records=original,
                                     route_context=public_route_context(steps, step_position),
                                     public_observation_consumer=consume)
            counts["verified_transitions"] += verified["verified_transitions"]
            counts["combats"] += 1
            encounters[combat["scenario_id"]] += 1
            if position == 1 or position % 25 == 0 or position == len(panel):
                print(f"Rebuilt combats {position}/{len(panel)} ({position / len(panel):.1%})", flush=True)
    records.sort(key=lambda r: r["record_id"])
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Mixed SFT record IDs must be unique")
    mixture = apply_mixture_weights(records, cfg["source_weights"])
    counts.update(states=len(records), candidate_answers=sum(len(r["candidates"]) for r in records))
    payload = gzip_jsonl_bytes(records)
    digest = sha256_bytes(payload)
    manifest = build_dataset_manifest(
        dataset_id=dataset_id, task_type="sft", observation_version="observation_v7", identity_fields=["record_id"],
        splits={"train": {"path": repository_relative(root, artifact), "format": "jsonl", "compression": "gzip",
                          "records": len(records), "bytes": len(payload), "sha256": digest}},
        lineage={"gold_manifest": cfg["gold_manifest"], "gold_manifest_sha256": sha256_file(gold_path),
                 "source_report": repository_relative(root, source_path), "source_report_sha256": source_digest,
                 "source_selection": cfg["source_selection"],
                 "source_selection_sha256": sha256_file(selection_path),
                 "source_partition": selection["source_partition"],
                 "source_route_ids": sorted(route_ids), "selected_combats": panel,
                 "teacher_labels_certified": False, "native": native},
        semantics={"record_schema": "decision_sft_group_v1", "state_weight": "explicit_mean_one_loss_weight",
                   "candidate_loss": "weighted_mean_response_token_cross_entropy",
                   "source_weights": cfg["source_weights"], "gold_semantics": gold_manifest["semantics"],
                   "teacher_strategy_normalization": "combat_equal_then_state_equal",
                   "forced_end_normalization": "state_equal", "overlap": "gold_only",
                   "observation_rebuild": "executed_actions_public_memory_no_rng_resampling"})
    manifest["export_counts"] = dict(counts)
    manifest["mixture"] = mixture
    manifest["encounter_combats"] = dict(encounters)
    manifest["smoke"] = smoke
    write_export_artifacts([(artifact, payload), (manifest_path, json_artifact_bytes(manifest))])
    print(f"Export completed in {time.monotonic() - started:.1f}s", flush=True)
    return manifest
