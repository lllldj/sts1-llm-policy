"""Resolve the shared panel for a continuous-route run."""
from pathlib import Path
from typing import Any

from sts1_llm_policy.artifacts import read_json_object, resolve_repository_path


CONTINUOUS_PANEL_FIELDS = frozenset({
    "strategies", "scope", "picker_database", "picker", "upgrade_policy",
    "observation_version", "interaction_contract", "required_simulator_capabilities",
    "route_count", "combat_seed_groups_per_route", "route", "seeds", "protocol",
    "max_decisions",
})


def load_continuous_config(project_root: str | Path, path: str | Path) -> dict[str, Any]:
    """Expand one shared panel into the existing continuous execution contract."""
    config = read_json_object(resolve_repository_path(project_root, path, expected_kind="file"))
    if config.get("schema_version") != "continuous_combat_panel_generation_v1":
        raise ValueError("Unsupported continuous combat generation schema")
    if "panel" not in config:
        return config
    run_fields = {"schema_version", "run_id", "purpose", "arms", "output_dir"}
    if set(config) - run_fields - {"panel"}:
        raise ValueError("Panel-based runs cannot override panel conditions or add unknown fields")
    if not isinstance(config["panel"], str) or not config["panel"]:
        raise ValueError("panel must be a non-empty repository path")
    panel = read_json_object(resolve_repository_path(project_root, config["panel"], expected_kind="file"))
    if (panel.get("schema_version") != "continuous_combat_panel_v1"
            or set(panel) != CONTINUOUS_PANEL_FIELDS | {"schema_version"}):
        raise ValueError("Panel must declare all conditions exactly once; nested panels are unsupported")
    return {**{key: value for key, value in config.items() if key != "panel"},
            **{key: value for key, value in panel.items() if key != "schema_version"}}
