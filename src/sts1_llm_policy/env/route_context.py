"""Public remaining-route state; never contains seeds or hidden reward draws."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ROUTE_OBSERVATION_VERSION = "observation_v7"
COMBAT_FAMILIES = {"normal_weak", "normal_strong", "elite", "boss"}


@dataclass(frozen=True)
class RouteOperation:
    kind: str
    encounter_family: str | None = None
    heal_amount: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"combat", "heal", "upgrade", "card_pick", "remove", "random_relic"}:
            raise ValueError(f"Unsupported public route operation: {self.kind}")
        if self.kind == "combat":
            if self.encounter_family not in COMBAT_FAMILIES:
                raise ValueError("Route combat requires an encounter family")
        elif self.encounter_family is not None:
            raise ValueError("Only combat nodes have an encounter family")
        if self.kind == "heal":
            if type(self.heal_amount) is not int or self.heal_amount <= 0:
                raise ValueError("Route healing requires a positive HP amount")
        elif self.heal_amount is not None:
            raise ValueError("Only healing nodes have a heal amount")


@dataclass(frozen=True)
class CombatRouteContext:
    combat_index: int
    total_combats: int
    current_encounter_family: str
    boss_scenario_id: str
    remaining_operations: tuple[RouteOperation, ...]

    def __post_init__(self) -> None:
        if (type(self.combat_index) is not int or type(self.total_combats) is not int
                or not 1 <= self.combat_index <= self.total_combats):
            raise ValueError("Invalid route combat position")
        if self.current_encounter_family not in COMBAT_FAMILIES:
            raise ValueError("Invalid current encounter family")
        if not isinstance(self.boss_scenario_id, str) or not self.boss_scenario_id.strip():
            raise ValueError("Route requires a public Boss identity")
        if (not isinstance(self.remaining_operations, tuple)
                or any(not isinstance(op, RouteOperation) for op in self.remaining_operations)):
            raise ValueError("Remaining route must contain public RouteOperations")
        combats = [op for op in self.remaining_operations if op.kind == "combat"]
        if len(combats) != self.total_combats - self.combat_index:
            raise ValueError("Remaining combat count disagrees with route position")
        if self.current_encounter_family == "boss":
            if self.remaining_operations or self.combat_index != self.total_combats:
                raise ValueError("Boss victory must finish the route")
        elif (not combats or combats[-1].encounter_family != "boss"
              or self.remaining_operations[-1] != combats[-1]
              or any(op.encounter_family == "boss" for op in combats[:-1])):
            raise ValueError("Remaining route must end at its only Boss")

    def serialize(self) -> str:
        lines = [
            "ROUTE_CONTEXT:",
            f"CURRENT_COMBAT: {self.combat_index}/{self.total_combats}",
            f"CURRENT_ENCOUNTER_TYPE: {self.current_encounter_family.upper()}",
            f"IS_FINAL_BOSS_COMBAT: {str(self.current_encounter_family == 'boss').upper()}",
            f"REMAINING_COMBATS_AFTER_CURRENT: {self.total_combats - self.combat_index}",
            f"VISIBLE_BOSS: {self.boss_scenario_id}",
            "REMAINING_ROUTE_AFTER_CURRENT_COMBAT (in execution order):",
        ]
        for index, op in enumerate(self.remaining_operations, 1):
            detail = op.kind.upper()
            if op.kind == "combat":
                detail += f" | TYPE {op.encounter_family.upper()}"
            elif op.kind == "heal":
                detail += f" | HP +{op.heal_amount} (capped at max HP)"
            lines.append(f"{index}. {detail}")
        if not self.remaining_operations:
            lines.append("NONE; winning this Boss combat completes the objective.")
        lines.append("Non-combat nodes are performed by the configured route policies.")
        return "\n".join(lines)


def public_route_context(
    steps: Sequence[Mapping[str, Any]], position: int,
) -> CombatRouteContext:
    """Project only public node semantics; never copy the internal route mapping."""
    current = steps[position]
    if current["kind"] != "combat":
        raise ValueError("Route context requires a current combat")
    combats = [step for step in steps if step["kind"] == "combat"]
    operations = []
    for step in steps[position + 1:]:
        kind = {"reward": "card_pick", "ordinary_relic": "random_relic"}.get(step["kind"], step["kind"])
        operations.append(RouteOperation(
            kind=kind,
            encounter_family=step["encounter_family"] if kind == "combat" else None,
            heal_amount=step["amount"] if kind == "heal" else None,
        ))
    return CombatRouteContext(
        combat_index=current["combat_index"], total_combats=len(combats),
        current_encounter_family=current["encounter_family"],
        boss_scenario_id=combats[-1]["scenario_id"],
        remaining_operations=tuple(operations),
    )
