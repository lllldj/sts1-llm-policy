"""Shared GOLD configuration validation and deterministic seed domains.

Preparation and execution use these rules without importing the GOLD executor.
"""
import hashlib
import json
import math

from sts1_llm_policy.artifacts import read_json_object, resolve_repository_path

from sts1_llm_policy.data.gold.candidate_statistics import STATISTICS_VERSION


OBSERVATION_VERSION = "observation_v7"


def seed_for(base, *parts):
    encoded = json.dumps([base, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def load_config(path, *, project_root):
    """Expand explicit selection inputs before validation and execution identity."""
    config = read_json_object(resolve_repository_path(project_root, path, expected_kind="file"))
    if "selection" in config:
        if "samples" in config or "source_partition" in config:
            raise ValueError("Selection inputs cannot be overridden by the execution config")
        selection = read_json_object(resolve_repository_path(
            project_root, config.pop("selection"), expected_kind="file"))
        if selection.get("source_report") != config.get("source_report"):
            raise ValueError("Selection belongs to a different source pool")
        if "samples" not in selection:
            raise ValueError("Selection must contain explicit samples")
        config["samples"] = selection["samples"]
        if "source_partition" in selection:
            config["source_partition"] = selection["source_partition"]
    return validate_config(config)


def validate_execution_settings(config):
    """Validate search/storage settings independently of panel membership."""
    if config.get("schema_version") != "teacher_gold_probe_v3" or config.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError("Expected the V3 probe with explicit observation_v7")
    for key in ("outer_trials", "hidden_samples", "search_budget", "minimum_root_visits", "max_decisions", "workers"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["minimum_root_visits"] > config["search_budget"]:
        raise ValueError("Root allocation exceeds budget")
    floor = config["minimum_win_rate"]
    floors = list(floor.values()) if isinstance(floor, dict) else [floor]
    if not floors or (isinstance(floor, dict) and any(not isinstance(k, str) or not k for k in floor)):
        raise ValueError("minimum_win_rate must declare nonempty encounter-family thresholds")
    for key, values in (("minimum_win_rate", floors), ("confidence", [config["confidence"]])):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1 for v in values):
            raise ValueError(f"{key} must be between zero and one")
    if not math.isfinite(config["expected_hp_tolerance"]) or config["expected_hp_tolerance"] < 0:
        raise ValueError("Invalid HP tolerance")
    if config.get("win_floor_rounding", "exact") not in {"exact", "floor"}:
        raise ValueError("win_floor_rounding must be exact or floor")
    if "card_statistics" in config and config["card_statistics"] != STATISTICS_VERSION:
        raise ValueError(f"card_statistics must be {STATISTICS_VERSION}")
    validate_continuation_policy(config.get("continuation_policy"))
    validate_ladder(config)
    validate_storage(config)
    if config.get("trace_observation", "full") not in {"full", "reconstruct"}:
        raise ValueError("Unsupported trace observation storage")
    if "route_progress" in config and not isinstance(config["route_progress"], bool):
        raise ValueError("route_progress must be boolean")
    return config


def validate_config(config):
    """Validate execution settings and the actual selected source positions."""
    validate_execution_settings(config)
    if not isinstance(config.get("samples"), list):
        raise ValueError("Samples must be an explicit list")
    identities = set()
    for sample in config["samples"]:
        if not isinstance(sample, dict):
            raise ValueError("Each sample must declare an ID and source position")
        identity = sample.get("id")
        if not isinstance(identity, str) or not identity or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in identity) or identity in identities:
            raise ValueError("Sample IDs must be unique safe path components")
        identities.add(identity)
        for key in ("route", "group", "combat", "step"):
            if type(sample.get(key)) is not int or sample[key] < 0:
                raise ValueError("Invalid source position")
    if not identities:
        raise ValueError("Empty panel")
    if "panel_groups" in config:
        grouped = [key for group in config["panel_groups"].values() for key in group]
        if len(grouped) != len(set(grouped)) or set(grouped) != identities:
            raise ValueError("Panel groups must partition the declared samples")
    if "source_partition" in config:
        partition = config["source_partition"]
        selected = {s["route"] for s in config["samples"]}
        excluded, reserved = set(partition["excluded_routes"]), set(partition["reserved_routes"])
        if partition["role"] != "train_candidate" or selected & (excluded | reserved) or excluded & reserved:
            raise ValueError("Training, tuning and reserved route partitions overlap or have unsupported roles")
    return config


def validate_continuation_policy(policy):
    if policy is None:
        return
    if not isinstance(policy, dict):
        raise ValueError("continuation_policy must be an object")
    mode = policy.get("mode")
    if mode == "win_rate_then_hp" and set(policy) == {"mode"}:
        return
    if mode != "win_floor_then_expected_hp" or set(policy) != {"mode", "minimum_search_win_rate"}:
        raise ValueError("Unsupported continuation_policy or policy fields")
    floor = policy["minimum_search_win_rate"]
    values = list(floor.values()) if isinstance(floor, dict) else [floor]
    if not values or (isinstance(floor, dict) and any(not isinstance(k, str) or not k for k in floor)):
        raise ValueError("minimum_search_win_rate must declare nonempty encounter-family thresholds")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1 for v in values):
        raise ValueError("minimum_search_win_rate must be greater than zero and at most one")


def validate_ladder(config):
    rule = config.get("sampling_ladder")
    if rule is None:
        return
    required = {"mode", "stages", "hp_boundary_band", "audit_probability", "audit_seed"}
    if not required <= set(rule) or set(rule) - required - {"paired_se_multiplier_at_64"}:
        raise ValueError("Unsupported sampling_ladder fields")
    if rule["mode"] not in {"shadow", "adaptive"} or rule["stages"] != [32, 64, 128]:
        raise ValueError("Only the 32/64/128 shadow or adaptive ladder is supported")
    if config["outer_trials"] != 128 or config.get("win_floor_rounding") != "floor":
        raise ValueError("Sampling ladder requires 128 trials and floor rounding")
    band, probability = rule["hp_boundary_band"], rule["audit_probability"]
    if not isinstance(band, (int, float)) or isinstance(band, bool) or not math.isfinite(band) or not 0 < band < config["expected_hp_tolerance"]:
        raise ValueError("HP boundary band must be positive and below tolerance")
    if not isinstance(probability, (int, float)) or isinstance(probability, bool) or not 0 <= probability <= 1:
        raise ValueError("Invalid audit probability")
    if not isinstance(rule["audit_seed"], int) or isinstance(rule["audit_seed"], bool):
        raise ValueError("Invalid audit seed")
    multiplier = rule.get("paired_se_multiplier_at_64", 0)
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError("Invalid paired SE multiplier")


def validate_storage(config):
    policy = config.get("continuation_storage")
    if "resume_from" in config and (not policy or not isinstance(config["resume_from"], str) or not config["resume_from"]):
        raise ValueError("resume_from requires compact storage and an explicit source directory")
    if policy is None:
        return
    if not isinstance(policy, dict) or set(policy) != {"mode", "full_trace_probability", "keep_first_trial", "keep_boundary_trials"}:
        raise ValueError("Unsupported continuation storage fields")
    if policy["mode"] != "compact_v1":
        raise ValueError("Unsupported continuation storage mode")
    probability = policy["full_trace_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid complete-trace sampling probability")
    if any(not isinstance(policy[k], bool) for k in ("keep_first_trial", "keep_boundary_trials")):
        raise ValueError("Trace retention switches must be boolean")
