"""Execute resumable GOLD continuation collection."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import subprocess
import threading
import time

from sts1_llm_policy.artifacts import read_json
from sts1_llm_policy.data.gold.candidate_statistics import collection_statistics, root_records
from sts1_llm_policy.data.gold.configuration import seed_for
from sts1_llm_policy.env.action_equivalence import build_model_action_space
from sts1_llm_policy.env.simulator_env import StsLightspeedEnv, SimulatorCombatEndedError
from sts1_llm_policy.env.simulator_execution import resolve_simulator, SELECTION
from sts1_llm_policy.data.gold.sampling_ladder import stage_decision, ladder_statistics
from sts1_llm_policy.data.gold.storage import (
    write_json,
    read_archive,
    compact_verified_state,
    cleanup_compacted_state,
    prepare_resume,
    import_sample,
)
from .continuation import (
    PROTOCOL,
    RouteProgress,
    actual_trials,
    choose,
    continuation_policy,
    load_source,
    source_identity,
    observation,
    restore,
    scoring_config,
    summarize,
)
from .verification import verify_state


def execute_trial(env, source, sample, config, output, trial, root_action, stop):
    destination = output / sample["id"] / f"trial-{trial:03d}-{root_action}.json"
    trace = destination.with_suffix(".jsonl.gz")
    if destination.exists():
        result = read_json(destination)
        if result["root_action"] != root_action or result["trial"] != trial or not trace.is_file():
            raise ValueError("Incomplete or mismatched saved trial")
        return result
    restore(env, source, sample)
    outer_seed = seed_for(config["seed"], sample["id"], "outer-world", trial)
    env.public_state(seed=outer_seed)
    trace.parent.mkdir(parents=True, exist_ok=True)
    temporary = trace.with_suffix(trace.suffix + ".partial")
    started = time.monotonic()
    outcome, raw, search_calls = "aborted", None, 0
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        def emit(value):
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        emit({"type": "start", "sample": sample, "trial": trial, "root_action": root_action,
              "outer_seed": outer_seed, "protocol": PROTOCOL})
        for decision in range(config["max_decisions"]):
            if stop.is_set():
                raise InterruptedError("Probe interrupted; completed trials remain resumable")
            before = env.get_raw_state()
            if decision == 0:
                action_id, searches = root_action, []
            else:
                action_id, searches = choose(env, config, sample["id"], trial, decision)
                search_calls += len(searches)
            space = build_model_action_space(env.get_state(), env.legal_actions())
            chosen = space.resolve(action_id)
            if chosen is None:
                raise ValueError("Chosen action is not legal")
            record = {"type": "step", "decision": decision, "raw_state": before,
                      "model_action": action_id, "searches": searches}
            if config.get("trace_observation", "full") == "full":
                record["observation"] = observation(env)
            emit(record)
            try:
                env.step(chosen.representative)
            except SimulatorCombatEndedError as end:
                raw = end.raw_state
                outcome = "victory" if raw["state"]["outcome"] == "PLAYER_VICTORY" else "defeat"
                break
        if raw is None:
            raw = env.get_raw_state()
        emit({"type": "terminal", "outcome": outcome, "raw_state": raw})
    temporary.replace(trace)
    state = raw["state"]
    result = {"sample_id": sample["id"], "trial": trial, "root_action": root_action,
              "outcome": outcome, "ending_hp": state["player"]["current_hp"], "max_hp": state["player"]["max_hp"],
              "relic_counters": sorted([r["id"], r.get("counter", 0)] for r in state["relics"]),
              "decisions": decision + 1, "search_calls": search_calls,
              "elapsed_seconds": time.monotonic() - started, "trace": str(trace.relative_to(output)).replace("\\", "/")}
    write_json(destination, result)
    return result


def run_sample(root, config, sample, execution, output, stop, progress=None, resume_manifest=None):
    if stop is not None and stop.is_set():
        raise InterruptedError("Probe interrupted before starting another state")
    if config.get("resume_from"):
        import_sample(root, config, sample, output, stop, resume_manifest)
    saved_path = output / sample["id"] / "report.json"
    if saved_path.exists():
        saved = read_json(saved_path)
        if saved["sample"] != sample:
            raise ValueError("Saved sample differs from configuration")
        if saved["status"] == "completed":
            actual_trials(config, saved)
            if config.get("continuation_storage"):
                archive_path = saved_path.parent / "continuations.json.gz"
                if archive_path.exists():
                    archive = read_archive(archive_path, sample)
                else:
                    with execution.create_client() as client:
                        env = StsLightspeedEnv(client, allow_card_selection=True)
                        archive = compact_state(root, config, output, saved, env, stop)
                cleanup_compacted_state(output, sample, archive)
            return saved
    source = load_source(root, config, sample)
    scoring = scoring_config(config, source[0])
    execution_config = {**config, "continuation_policy": continuation_policy(config, source[0])}
    with execution.create_client() as client:
        env = StsLightspeedEnv(client, allow_card_selection=True)
        restore(env, source, sample)
        env.public_state()
        space = build_model_action_space(env.get_state(), env.legal_actions())
        rows, stages = [], []
        initial = {"sample": sample, "scenario": source[0]["scenario_id"],
                   "encounter_family": source[0]["encounter_family"], "minimum_win_rate": scoring["minimum_win_rate"],
                   "observation": observation(env),
                   "root_actions": root_records(space, extended=bool(config.get("card_statistics")))}
        for trial in range(config["outer_trials"]):
            for action in space.action_ids:
                rows.append(execute_trial(env, source, sample, execution_config, output, trial, action, stop))
            if progress:
                if (trial + 1) % 8 == 0:
                    progress.update(sample, detail=f"paired trials {trial + 1}/{config['outer_trials']} cap")
            else:
                print(f"{sample['id']}: {trial + 1}/{config['outer_trials']} paired trials", flush=True)
            if config.get("sampling_ladder") and trial + 1 in config["sampling_ladder"]["stages"]:
                count = trial + 1
                stage = stage_decision(summarize(rows, {**scoring, "outer_trials": count}),
                                       stages[-1] if stages else None, scoring, sample["id"], count)
                stage.update(executions=len(rows), search_calls=sum(r["search_calls"] for r in rows),
                             executed_decisions=sum(r["decisions"] for r in rows))
                stages.append(stage)
                write_json(output / sample["id"] / f"stage-{count:03d}.json", stage)
                detail = f"stage {count}, expand={stage['would_expand']}, reasons={stage['reasons']}, audit={stage['audit_selected']}"
                if progress:
                    progress.update(sample, detail=detail)
                else:
                    print(f"{sample['id']}: {detail}", flush=True)
                if config["sampling_ladder"]["mode"] == "adaptive" and not stage["would_expand"]:
                    break
        result = {**initial, **summarize(rows, {**scoring, "outer_trials": trial + 1}), "executions": len(rows),
                  "executed_decisions": sum(r["decisions"] for r in rows), "search_calls": sum(r["search_calls"] for r in rows)}
        if stages:
            result["sampling_ladder"] = stages
            if config["sampling_ladder"]["mode"] == "adaptive":
                result["outer_trials_completed"] = trial + 1
        write_json(output / sample["id"] / "report.json", result)
        if config.get("continuation_storage") and result["status"] == "completed":
            compact_state(root, config, output, result, env, stop)
        return result


def compact_state(root, config, output, state, env, stop):
    receipt, tapes, rows = verify_state(root, config, output, state, env, allow_archive=False, stop=stop)
    archive = compact_verified_state(output, state["sample"], rows, state.get("sampling_ladder", []), config, tapes, receipt)
    cleanup_compacted_state(output, state["sample"], archive)
    return archive


def _git_provenance(root):
    return {
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "git_provenance_at": "run_start",
    }


def run(root, config, *, smoke=False, preflight=False):
    provenance = _git_provenance(root)
    config = dict(config)
    if smoke:
        config.pop("resume_from", None)
        config.pop("sampling_ladder", None)
        config.pop("panel_groups", None)
        config["outer_trials"] = 1
        wanted = set(config.get("smoke_samples", [s["id"] for s in config["samples"][:4]]))
        config["samples"] = [s for s in config["samples"] if s["id"] in wanted]
        if {s["id"] for s in config["samples"]} != wanted or not wanted:
            raise ValueError("Smoke samples must be a nonempty subset of this panel")
    output = root / config["output"] / ("preflight" if preflight else "smoke" if smoke else "formal")
    execution = resolve_simulator(project_root=root, required_capabilities={SELECTION})
    native = execution.describe(verify=True)
    started = time.monotonic()
    if preflight:
        states = []
        progress = RouteProgress(config["samples"], "preflight") if config.get("route_progress") else None
        with execution.create_client() as client:
            env = StsLightspeedEnv(client, allow_card_selection=True)
            for sample in config["samples"]:
                source = load_source(root, config, sample)
                scoring = scoring_config(config, source[0])
                policy = continuation_policy(config, source[0])
                restore(env, source, sample)
                env.public_state()
                expected = observation(env)
                actions = build_model_action_space(env.get_state(), env.legal_actions()).action_ids
                env.public_state(seed=seed_for(config["seed"], sample["id"], "preflight"))
                if observation(env) != expected:
                    raise ValueError("Hidden sampling changed the public observation")
                if config["minimum_root_visits"] * len(env.legal_actions()) > config["search_budget"]:
                    raise ValueError("Search budget cannot cover this root")
                states.append({"sample": sample, "actions": list(actions), "replay_passed": True,
                               "native_root_count": len(env.legal_actions()),
                               "minimum_win_rate": scoring["minimum_win_rate"], "continuation_policy": policy})
                if progress:
                    progress.update(sample, finished=True)
        report = {"status": "completed", "states": states, "native": native, "configuration": config,
                  **provenance, "elapsed_seconds": time.monotonic() - started, "labels_certified": False}
        write_json(output / "report.json", report)
        return report
    source = source_identity(root, config)
    identity = {"protocol": PROTOCOL, "configuration": {k: v for k, v in config.items() if k not in {"workers", "output"}}, "native": native,
                "source": source}
    identity_path = output / "identity.json"
    if identity_path.exists():
        previous = read_json(identity_path)
        if previous.get("source") != source:
            raise ValueError("GOLD source content changed or its binding is missing; use a new output directory")
        if (previous["protocol"], previous["configuration"], previous["native"]["bridge_sha256"]) != (identity["protocol"], identity["configuration"], native["bridge_sha256"]):
            raise ValueError("Run inputs or native runtime changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("GOLD output has no resume identity; use a new output directory")
        write_json(identity_path, identity)
    resume_manifest = prepare_resume(root, config, output, identity) if config.get("resume_from") else None
    stop = threading.Event()
    results = []
    progress = RouteProgress(config["samples"], "collect") if config.get("route_progress") else None
    if progress:
        progress.update()
    try:
        with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
            futures = [pool.submit(run_sample, root, config, s, execution, output, stop, progress, resume_manifest) for s in config["samples"]]
            try:
                for future in as_completed(futures):
                    state = future.result()
                    results.append(state)
                    if progress:
                        progress.update(state["sample"], finished=state["status"] == "completed")
            except BaseException:
                stop.set()
                for queued in futures:
                    queued.cancel()
                raise
    except BaseException as error:
        write_json(output / "report.json", {"status": "partial", "error": str(error), "states": results,
                                            **provenance, "labels_certified": False, "elapsed_seconds": time.monotonic() - started})
        raise
    result = {"schema_version": "teacher_gold_probe_report_v3", "protocol": PROTOCOL,
              "status": "completed" if all(s["status"] == "completed" for s in results) else "partial",
              "configuration": config, "native": native, "source": source,
              **provenance,
              "states": sorted(results, key=lambda s: s["sample"]["id"]),
              "elapsed_seconds": time.monotonic() - started, "labels_certified": False, "training_exported": False}
    if config.get("card_statistics") and result["status"] == "completed":
        result["card_statistics"] = collection_statistics(result["states"])
    if config.get("sampling_ladder") and result["status"] == "completed":
        result["sampling_ladder_statistics"] = ladder_statistics(result["states"], config)
    write_json(output / "report.json", result)
    return result
