from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CardDemandProfile:
    provides: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    traits: frozenset[str] = frozenset()


def _p(
    *,
    provides: Sequence[str] = (),
    requires: Sequence[str] = (),
    traits: Sequence[str] = (),
) -> CardDemandProfile:
    return CardDemandProfile(
        frozenset(provides),
        frozenset(requires),
        frozenset(traits),
    )


# Card tags used by opening selection, persistence and synergy checks.
CARD_DEMAND_PROFILES: dict[str, CardDemandProfile] = {
    "ANGER": _p(traits=("attack", "low_cost")),
    "ARMAMENTS": _p(provides=("block_engine", "future_upgrades")),
    "BURNING_PACT": _p(provides=("exhaust",)),
    "CLEAVE": _p(traits=("attack", "aoe")),
    "FLEX": _p(provides=("strength",)),
    "IRON_WAVE": _p(traits=("attack",)),
    "BODY_SLAM": _p(requires=("block_engine",), traits=("attack", "low_cost")),
    "SHRUG_IT_OFF": _p(provides=("block_engine",)),
    "CLASH": _p(requires=("skill_light",), traits=("attack", "low_cost")),
    "THUNDERCLAP": _p(provides=("vulnerable",), traits=("attack", "aoe")),
    "POMMEL_STRIKE": _p(traits=("attack",)),
    "TWIN_STRIKE": _p(traits=("attack", "multi_hit")),
    "CLOTHESLINE": _p(traits=("attack",)),
    "HAVOC": _p(provides=("exhaust",)),
    "WILD_STRIKE": _p(provides=("status",), traits=("attack",)),
    "HEAVY_BLADE": _p(requires=("strength",), traits=("attack", "strength_payoff")),
    "PERFECTED_STRIKE": _p(requires=("strike_density",), traits=("attack",)),
    "SWORD_BOOMERANG": _p(requires=("strength",), traits=("attack", "multi_hit")),
    "EVOLVE": _p(requires=("status",), traits=("power",)),
    "UPPERCUT": _p(provides=("vulnerable",), traits=("attack",)),
    "GHOSTLY_ARMOR": _p(provides=("block_engine",)),
    "HEADBUTT": _p(traits=("attack",)),
    "FIRE_BREATHING": _p(requires=("status",), traits=("power", "aoe")),
    "DROPKICK": _p(requires=("vulnerable",), traits=("attack",)),
    "DUAL_WIELD": _p(requires=("attack_density",)),
    "CARNAGE": _p(traits=("attack",)),
    "EXHUME": _p(requires=("exhaust",)),
    "BLOODLETTING": _p(provides=("self_damage",)),
    "RUPTURE": _p(provides=("strength",), requires=("self_damage",), traits=("power",)),
    "SECOND_WIND": _p(provides=("block_engine", "exhaust")),
    "SEARING_BLOW": _p(requires=("future_upgrades",), traits=("attack",)),
    "BATTLE_TRANCE": _p(),
    "SENTINEL": _p(requires=("exhaust",)),
    "TRUE_GRIT": _p(provides=("block_engine", "exhaust")),
    "ENTRENCH": _p(requires=("block_engine",)),
    "RAGE": _p(provides=("block_engine",), requires=("attack_density",)),
    "FEEL_NO_PAIN": _p(requires=("exhaust",), traits=("power",)),
    "DISARM": _p(provides=("exhaust",)),
    "SEEING_RED": _p(provides=("exhaust",)),
    "DARK_EMBRACE": _p(requires=("exhaust",), traits=("power",)),
    "COMBUST": _p(provides=("self_damage",), traits=("power", "aoe")),
    "WHIRLWIND": _p(requires=("energy_support",), traits=("attack", "aoe")),
    "WARCRY": _p(provides=("exhaust",)),
    "SEVER_SOUL": _p(provides=("exhaust",), traits=("attack",)),
    "RAMPAGE": _p(traits=("attack",)),
    "SHOCKWAVE": _p(provides=("exhaust", "vulnerable"), traits=("aoe",)),
    "METALLICIZE": _p(provides=("block_engine",), traits=("power",)),
    "PUMMEL": _p(requires=("strength",), traits=("attack", "multi_hit")),
    "FLAME_BARRIER": _p(provides=("block_engine",)),
    "BLOOD_FOR_BLOOD": _p(requires=("self_damage",), traits=("attack",)),
    "INTIMIDATE": _p(provides=("exhaust",)),
    "HEMOKINESIS": _p(provides=("self_damage",), traits=("attack",)),
    "RECKLESS_CHARGE": _p(provides=("status", "exhaust"), traits=("attack", "low_cost")),
    "POWER_THROUGH": _p(provides=("block_engine", "status")),
    "INFLAME": _p(provides=("strength",), traits=("power",)),
    "SPOT_WEAKNESS": _p(provides=("strength",)),
    "DOUBLE_TAP": _p(requires=("attack_density",)),
    "DEMON_FORM": _p(provides=("strength",), traits=("power",)),
    "BLUDGEON": _p(traits=("attack",)),
    "FEED": _p(traits=("attack",)),
    "LIMIT_BREAK": _p(requires=("strength",)),
    "CORRUPTION": _p(provides=("exhaust",), traits=("power",)),
    "BARRICADE": _p(requires=("block_engine",), traits=("power",)),
    "FIEND_FIRE": _p(provides=("exhaust",), requires=("exhaust_fuel",), traits=("attack",)),
    "BERSERK": _p(traits=("power",)),
    "IMPERVIOUS": _p(provides=("block_engine", "exhaust")),
    "JUGGERNAUT": _p(requires=("block_engine",), traits=("power",)),
    "BRUTALITY": _p(provides=("self_damage",), traits=("power",)),
    "REAPER": _p(requires=("strength",), traits=("attack", "aoe")),
    "OFFERING": _p(provides=("self_damage", "exhaust")),
    "IMMOLATE": _p(provides=("status",), traits=("attack", "aoe")),
    "STRIKE_RED": _p(traits=("attack", "strike")),
    "DEFEND_RED": _p(provides=("block_seed",)),
    "BASH": _p(provides=("vulnerable",), traits=("attack",)),
    "ASCENDERS_BANE": _p(provides=("status_seed",)),
}


def validate_reward_profile_coverage(card_ids: Sequence[str]) -> None:
    missing = sorted(set(card_ids) - set(CARD_DEMAND_PROFILES))
    if missing:
        raise ValueError(f"reward demand profiles are missing cards: {missing}")
