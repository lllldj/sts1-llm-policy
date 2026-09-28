"""Prepare explicit experiment members without loading models or executing runs."""
from copy import deepcopy
import json
from pathlib import Path
import re

from sts1_llm_policy.artifacts import (
    atomic_write_bytes,
    read_json_object,
    repository_relative,
    resolve_repository_path,
)
from sts1_llm_policy.workflows.continuous_config import load_continuous_config


def prepare_experiments(project_root, configs, output):
    root = Path(project_root).resolve()
    destination = resolve_repository_path(root, output, must_exist=False, expected_kind="directory")
    if destination == root:
        raise ValueError("Preparation requires a separate output directory")
    members = {"training": [], "evaluation": []}
    seen = set()
    references = {}
    for config in configs:
        document = read_json_object(resolve_repository_path(root, config, expected_kind="file"))
        if (set(document) - {"schema_version", "training", "evaluation", "reference_checkpoints"}
                or not {"training", "evaluation"} <= document.keys()
                or document.get("schema_version") != "experiment_preparation_v1"):
            raise ValueError("Expected an experiment_preparation_v1 member list")
        for kind in members:
            paths = document[kind]
            if not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths):
                raise ValueError(f"{kind} must be an explicit list of config paths")
            for path in paths:
                resolved = resolve_repository_path(root, path, expected_kind="file")
                if resolved in seen:
                    raise ValueError(f"Duplicate experiment member: {path}")
                seen.add(resolved)
                members[kind].append(resolved)
        declared = document.get("reference_checkpoints", {})
        if not isinstance(declared, dict):
            raise ValueError("reference_checkpoints must map adapter directories to training configs")
        for adapter, training in declared.items():
            if not isinstance(adapter, str) or not adapter or not isinstance(training, str) or not training:
                raise ValueError("reference_checkpoints paths must be non-empty strings")
            adapter_path = resolve_repository_path(root, adapter, expected_kind="directory")
            training_path = resolve_repository_path(root, training, expected_kind="file")
            if adapter_path in references:
                raise ValueError(f"Duplicate reference checkpoint: {adapter}")
            references[adapter_path] = training_path
    if not seen:
        raise ValueError("No experiment members selected")
    if any(path not in members["training"] for path in references.values()):
        raise ValueError("Reference checkpoints require their training configs in the member list")

    pending, checkpoints, outputs = {}, {}, set()

    def location(path):
        return repository_relative(root, path)

    def output_for(cfg, kind):
        run_id = cfg.get("run_id")
        if (not isinstance(run_id, str) or re.fullmatch(r"[a-zA-Z0-9._-]+", run_id) is None
                or run_id in {".", ".."}):
            raise ValueError("Invalid run_id")
        target = resolve_repository_path(root, destination / kind / run_id,
                                         must_exist=False, expected_kind="directory")
        if target in outputs:
            raise ValueError(f"Duplicate output for run_id: {run_id}")
        outputs.add(target)
        return target

    def checkpoint(path):
        original = resolve_repository_path(root, path, must_exist=False, expected_kind="directory")
        if original not in checkpoints:
            raise ValueError(f"Missing upstream training member for checkpoint: {path}; list upstream training first")
        return checkpoints[original]

    for path in members["training"]:
        cfg = deepcopy(read_json_object(path))
        if cfg.get("schema_version") != "training_run_v2":
            raise ValueError(f"Preparation supports portable training_run_v2: {path}")
        for key in ("training_recipe", "model_runtime", "dataset_manifest", "execution_profile"):
            resolve_repository_path(root, cfg[key], expected_kind="file")
        recipe = read_json_object(root / cfg["training_recipe"])
        if recipe.get("algorithm") == "dpo" and "initial_checkpoint" not in cfg:
            raise ValueError("DPO requires an upstream initial checkpoint")
        if "initial_checkpoint" in cfg:
            cfg["initial_checkpoint"] = checkpoint(cfg["initial_checkpoint"])
        original = resolve_repository_path(root, cfg.get("output_dir", f"outputs/training/{cfg['run_id']}"),
                                            must_exist=False, expected_kind="directory")
        target = output_for(cfg, "training")
        if target == original or original / "checkpoint" in checkpoints:
            raise ValueError("Training outputs must be independent and unambiguous")
        aliases = [original / "checkpoint", *(adapter for adapter, source in references.items() if source == path)]
        for alias in aliases:
            if alias in checkpoints and checkpoints[alias] != location(target / "checkpoint"):
                raise ValueError(f"Ambiguous upstream checkpoint: {alias}")
            checkpoints[alias] = location(target / "checkpoint")
        cfg["output_dir"] = location(target)
        cfg["report"] = location(target / "report.json")
        for mode in ("backward", "smoke"):
            cfg[mode] = {"output_dir": location(target / mode), "report": location(target / mode / "report.json")}
        pending[path] = cfg

    for path in members["evaluation"]:
        cfg = deepcopy(read_json_object(path))
        if cfg.get("schema_version") == "continuous_combat_panel_generation_v1":
            cfg = load_continuous_config(root, path)
            arms = cfg["arms"]
        elif cfg.get("schema_version") == "frozen_policy_panel_evaluation_v1":
            arms = [cfg]
        else:
            raise ValueError(f"Unsupported evaluation config: {path}")
        original = resolve_repository_path(root, cfg["output_dir"], must_exist=False)
        target = output_for(cfg, "evaluation")
        if target == original:
            raise ValueError("Evaluation output must be independent")
        cfg["output_dir"] = location(target)
        for arm in arms:
            if "checkpoint" in arm:
                arm["checkpoint"] = checkpoint(arm["checkpoint"])
        pending[path] = cfg

    # Validate the entire batch before creating files; existing runs are never overwritten.
    files = []
    for source, cfg in pending.items():
        target = resolve_repository_path(root, destination / source.relative_to(root),
                                         must_exist=False, expected_kind="file")
        if target in seen:
            raise ValueError("Prepared config would replace an input config")
        if target.exists():
            if read_json_object(target) != cfg:
                raise ValueError(f"Conflicting config: {target}; choose a new output directory")
        else:
            run_output = root / cfg["output_dir"]
            if run_output.exists() and any(run_output.iterdir()):
                raise ValueError(f"Existing run output without its prepared config: {run_output}")
            if target.with_suffix(target.suffix + ".tmp").exists():
                raise ValueError(f"Conflicting temporary config: {target}")
            if any(parent.exists() and not parent.is_dir() for parent in target.parents):
                raise ValueError(f"Config parent is not a directory: {target}")
        payload = (json.dumps(cfg, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        files.append((target, payload))
    for target, payload in files:
        if not target.exists():
            atomic_write_bytes(target, payload)
    return {"training": len(members["training"]), "evaluation": len(members["evaluation"]),
            "configs": [location(path) for path, _ in files]}
