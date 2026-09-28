"""Replay GOLD evidence and validate collection receipts without new search."""
from __future__ import annotations

import gzip
import hashlib
import json
import time

from sts1_llm_policy.data.gold.candidate_statistics import collection_statistics, root_records
from sts1_llm_policy.data.gold.configuration import OBSERVATION_VERSION, seed_for
from sts1_llm_policy.data.gold.export_source import replay_selection
from sts1_llm_policy.env.action_equivalence import build_model_action_space
from sts1_llm_policy.env.simulator_env import StsLightspeedEnv, SimulatorCombatEndedError
from sts1_llm_policy.env.simulator_execution import resolve_simulator, SELECTION
from sts1_llm_policy.artifacts import canonical_json_bytes, read_json
from sts1_llm_policy.data.trajectory_replay import comparable_execution_state
from sts1_llm_policy.data.gold.sampling_ladder import ladder_records, ladder_statistics
from sts1_llm_policy.data.gold.storage import read_archive, retained_trials
from .continuation import (
    PROTOCOL,
    RouteProgress,
    actual_trials,
    aggregate_searches,
    continuation_policy,
    load_source,
    source_identity,
    observation,
    restore,
    scoring_config,
    summarize,
)


def verify_run(root, config, *, smoke=False, selection=None):
    """Replay complete states, optionally a declared subset, without new search."""
    started = time.monotonic()
    output = root / config["output"] / ("smoke" if smoke else "formal")
    report = read_json(output / "report.json")
    if report["status"] != "completed":
        raise ValueError("Cannot verify an incomplete run")
    effective = report["configuration"]
    if effective.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError(
            f"Recorded observation version {effective.get('observation_version')!r} differs from "
            f"this verifier's {OBSERVATION_VERSION!r}. Verify with the implementation matching "
            "the recorded observation contract; do not relabel or overwrite the original artifacts."
        )
    report_digest = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    identity = read_json(output / "identity.json")
    if identity["protocol"] != PROTOCOL or identity["configuration"] != {k: v for k, v in effective.items() if k not in {"workers", "output"}}:
        raise ValueError("Report differs from its declared execution inputs")
    if "source" in identity or "source" in report:
        if identity.get("source") != report.get("source") or identity.get("source") != source_identity(root, effective):
            raise ValueError("GOLD source content differs from the executed source binding")
    if sorted(s["sample"]["id"] for s in report["states"]) != sorted(s["id"] for s in effective["samples"]):
        raise ValueError("Report is missing or duplicating panel states")
    declared = {s["id"]: s for s in effective["samples"]}
    if (len(declared) != len(report["states"])
            or any(s["sample"] != declared[s["sample"]["id"]] for s in report["states"])):
        raise ValueError("Reported sample is outside the declared panel")
    states = (report["states"] if selection is None
              else replay_selection(report["states"], selection, report_digest))
    execution = resolve_simulator(project_root=root, required_capabilities={SELECTION})
    if identity["native"]["bridge_sha256"] != execution.describe(verify=True)["bridge_sha256"]:
        raise ValueError("Replay runtime differs from executed runtime")
    checked, decisions, full_checked = 0, 0, 0
    progress = RouteProgress([s["sample"] for s in states], "verify") if effective.get("route_progress") else None
    if progress:
        progress.update()
    with execution.create_client() as client:
        env = StsLightspeedEnv(client, allow_card_selection=True)
        for state_report in states:
            sample = state_report["sample"]
            if sample not in effective["samples"]:
                raise ValueError("Reported sample is outside the declared panel")
            receipt, _, _ = verify_state(root, effective, output, state_report, env)
            checked += receipt["verified_executions"]
            decisions += receipt["verified_decisions"]
            full_checked += receipt["full_search_trace_executions"]
            state_trials = actual_trials(effective, state_report)
            if progress:
                progress.update(sample, finished=True, detail=f"verified paired trials={state_trials}")
    if effective.get("card_statistics") and report["card_statistics"] != collection_statistics(report["states"]):
        raise ValueError("Card statistics differ from executed candidates")
    if effective.get("sampling_ladder") and report["sampling_ladder_statistics"] != ladder_statistics(report["states"], effective):
        raise ValueError("Sampling ladder statistics differ from executed states")
    if hashlib.sha256(canonical_json_bytes(read_json(output / "report.json"))).hexdigest() != report_digest:
        raise ValueError("Report changed during verification; rerun verification against the final report")
    return {"status": "completed", "verified_executions": checked, "verified_decisions": decisions,
            "full_search_trace_executions": full_checked, "report_canonical_sha256": report_digest,
            "scope": "full" if selection is None else "sampled", "selection": selection,
            "verified_states": len(states), "total_states": len(report["states"]),
            "elapsed_seconds": time.monotonic() - started}


def verify_full_trial(env, source, sample, effective, output, trial, root_action, result):
    restore(env, source, sample)
    env.public_state(seed=seed_for(effective["seed"], sample["id"], "outer-world", trial))
    with gzip.open(output / result["trace"], "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header["sample"] != sample or header["trial"] != trial or header["root_action"] != root_action:
            raise ValueError("Trace identity mismatch")
        if header["outer_seed"] != seed_for(effective["seed"], sample["id"], "outer-world", trial) or header["protocol"] != PROTOCOL:
            raise ValueError("Trace sampler identity mismatch")
        terminal = None
        count = 0
        tape = []
        for line in stream:
            record = json.loads(line)
            if record["type"] == "terminal":
                if terminal is None or record["outcome"] != result["outcome"] or comparable_execution_state(record["raw_state"]) != comparable_execution_state(terminal):
                    raise ValueError("Trace terminal mismatch")
                actual = terminal["state"]
                actual_outcome = "victory" if actual["outcome"] == "PLAYER_VICTORY" else "defeat"
                counters = sorted([r["id"], r.get("counter", 0)] for r in actual["relics"])
                if (result["outcome"], result["ending_hp"], result["max_hp"], result["relic_counters"]) != (actual_outcome, actual["player"]["current_hp"], actual["player"]["max_hp"], counters):
                    raise ValueError("Trial score differs from its executed terminal state")
                terminal = None
                break
            if terminal is not None or record["type"] != "step" or record["decision"] != count:
                raise ValueError("Malformed transition sequence")
            if comparable_execution_state(record["raw_state"]) != comparable_execution_state(env.get_raw_state()):
                raise ValueError("Replayed decision/observation mismatch")
            if effective.get("trace_observation", "full") == "full":
                if record["observation"] != observation(env):
                    raise ValueError("Replayed observation mismatch")
            elif "observation" in record:
                raise ValueError("Derived observation was unexpectedly stored")
            if count == 0:
                expected = root_action
                if record["searches"]:
                    raise ValueError("The root action must be forced")
            else:
                expected = aggregate_searches(record["searches"], continuation_policy(effective, source[0]))
                if len(record["searches"]) != effective["hidden_samples"]:
                    raise ValueError("Missing continuation search evidence")
                for h, search in enumerate(record["searches"]):
                    if "simulations" in search:
                        native_roots = sum(len(a["native_action_ids"]) for a in search["actions"])
                        if (search["simulations"], search["minimum_root_action_visits"], search["native_root_count"]) != (effective["search_budget"], effective["minimum_root_visits"], native_roots):
                            raise ValueError("Recorded search budget/root allocation differs from configuration")
                    if search["public_state_seed"] != seed_for(effective["seed"], sample["id"], "inner-world", trial, count, h):
                        raise ValueError("Inner sampler seed mismatch")
                    if search["search_seed"] != seed_for(effective["seed"], sample["id"], "search", trial, count, h):
                        raise ValueError("Search seed mismatch")
            if record["model_action"] != expected:
                raise ValueError("Continuation differs from the declared Teacher rule")
            chosen = build_model_action_space(env.get_state(), env.legal_actions()).resolve(expected)
            if chosen is None:
                raise ValueError("Illegal recorded action")
            try:
                env.step(chosen.representative)
            except SimulatorCombatEndedError as end:
                terminal = end.raw_state
            tape.append(expected)
            count += 1
        else:
            raise ValueError("Missing terminal trace record")
        if stream.read().strip() or count != result["decisions"] or result["search_calls"] != (count - 1) * effective["hidden_samples"]:
            raise ValueError("Trace contains extra or missing decisions")
    return tape


def verify_state(root, effective, output, state_report, env, *, allow_archive=True, stop=None):
    """Replay every execution; complete search evidence is checked where retained."""
    sample = state_report["sample"]
    source = load_source(root, effective, sample)
    restore(env, source, sample)
    env.public_state()
    if state_report["observation"] != observation(env):
        raise ValueError("Root observation differs from restored source")
    if [a["id"] for a in state_report["root_actions"]] != list(build_model_action_space(env.get_state(), env.legal_actions()).action_ids):
        raise ValueError("Report is missing or changing root actions")
    if effective.get("card_statistics"):
        expected_roots = root_records(build_model_action_space(env.get_state(), env.legal_actions()), extended=True)
        if state_report["root_actions"] != expected_roots or state_report["encounter_family"] != source[0]["encounter_family"]:
            raise ValueError("Card statistics metadata differs from restored source")
    state_trials = actual_trials(effective, state_report)
    archive_path = output / sample["id"] / "continuations.json.gz"
    if allow_archive and effective.get("continuation_storage") and not archive_path.is_file():
        raise ValueError("Completed compact state is missing its archive")
    archive = read_archive(archive_path, sample) if allow_archive and effective.get("continuation_storage") and archive_path.exists() else None
    summaries, tapes, full_count = [], {}, 0
    compact_rows = {} if archive is None else {(r["trial"], r["root_action"]): r for r in archive["executions"]}
    expected_keys = {(t, a["id"]) for t in range(state_trials) for a in state_report["root_actions"]}
    if archive is not None:
        if set(compact_rows) != expected_keys:
            raise ValueError("Compact archive has missing or extra trials/roots")
        keep = retained_trials(sample, archive["executions"], state_report.get("sampling_ladder", []), effective)
        if {k for k, r in compact_rows.items() if r["full_trace_retained"]} != keep:
            raise ValueError("Compact trace retention differs from configuration")
    for trial in range(state_trials):
        for action in state_report["root_actions"]:
            if stop is not None and stop.is_set():
                raise InterruptedError("Verification interrupted; completed files remain resumable")
            key = (trial, action["id"])
            result = compact_rows[key] if archive is not None else read_json(output / sample["id"] / f"trial-{trial:03d}-{action['id']}.json")
            if (result["sample_id"], result["trial"], result["root_action"], result["trace"]) != (sample["id"], trial, action["id"], f"{sample['id']}/trial-{trial:03d}-{action['id']}.jsonl.gz"):
                raise ValueError("Trial metadata identity mismatch")
            if archive is None or result["full_trace_retained"]:
                tape = verify_full_trial(env, source, sample, effective, output, trial, action["id"], result)
                full_count += 1
                if archive is not None and tape != result["model_actions"]:
                    raise ValueError("Retained trace differs from compact actions")
            else:
                tape = verify_action_tape(env, source, sample, effective, result)
            summaries.append(result)
            tapes[key] = tape
    recalculated = summarize(summaries, {**scoring_config(effective, source[0]), "outer_trials": state_trials})
    if state_report["executions"] != len(summaries) or state_report["executed_decisions"] != sum(r["decisions"] for r in summaries):
        raise ValueError("State execution counts do not match traces")
    for key, value in recalculated.items():
        if value != state_report[key]:
            raise ValueError(f"Reported statistic differs from trial evidence: {key}")
    if effective.get("sampling_ladder") and state_report["sampling_ladder"] != ladder_records(summaries, scoring_config(effective, source[0]), sample["id"], summarize):
        raise ValueError("Sampling ladder differs from paired trial evidence")
    receipt = {"status": "completed", "verified_executions": len(summaries),
               "verified_decisions": sum(r["decisions"] for r in summaries),
               "full_search_trace_executions": full_count}
    return receipt, tapes, summaries


def verify_action_tape(env, source, sample, config, result):
    restore(env, source, sample)
    env.public_state(seed=seed_for(config["seed"], sample["id"], "outer-world", result["trial"]))
    tape = result["model_actions"]
    if len(tape) > config["max_decisions"] or result["search_calls"] != (len(tape) - 1) * config["hidden_samples"]:
        raise ValueError("Compact decision/search count mismatch")
    terminal = None
    for action in tape:
        if terminal is not None:
            raise ValueError("Compact actions extend beyond combat end")
        chosen = build_model_action_space(env.get_state(), env.legal_actions()).resolve(action)
        if chosen is None:
            raise ValueError("Illegal compact action")
        try:
            env.step(chosen.representative)
        except SimulatorCombatEndedError as end:
            terminal = end.raw_state["state"]
    if terminal is None:
        raise ValueError("Compact actions do not finish combat")
    outcome = "victory" if terminal["outcome"] == "PLAYER_VICTORY" else "defeat"
    counters = sorted([r["id"], r.get("counter", 0)] for r in terminal["relics"])
    if (result["outcome"], result["ending_hp"], result["max_hp"], result["relic_counters"]) != (outcome, terminal["player"]["current_hp"], terminal["player"]["max_hp"], counters):
        raise ValueError("Compact score differs from replayed terminal")
    return tape
