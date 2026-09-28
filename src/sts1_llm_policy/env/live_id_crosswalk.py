from __future__ import annotations

import re
from types import MappingProxyType
from typing import Collection, Mapping


COMMUNICATION_MOD_PROTOCOL_SOURCE = (
    "https://github.com/ForgottenArbiter/CommunicationMod/blob/master/"
    "src/main/java/communicationmod/GameStateConverter.java"
)
STS_LIGHTSPEED_REVISION = "7476a81954020087da31d41d16fddf475746ec2d"
LIVE_ID_CROSSWALK_ID = "communication_mod_to_sts_lightspeed_v7"


# CommunicationMod forwards the base game's runtime power.ID. Most values are
# text-equivalent to sts_lightspeed's display catalog and are handled by the
# normalized index below. Keep only true semantic-name exceptions here.
POWER_ID_EXCEPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "Anger": "Enrage",
        "Confusion": "Confused",
        "Flex": "Lose Strength",
        "Weakened": "Weak",
    }
)


MONSTER_ID_EXCEPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "Champ": "TheChamp",
        "Healer": "Mystic",
        "Shelled Parasite": "ShelledParasite",
        "SlaverBoss": "Taskmaster",
    }
)


# CommunicationMod forwards AbstractMonster.nextMove, whose byte values are
# monster implementation details. sts_lightspeed exposes project-normalized
# move IDs. Most supported monsters happen to use the same values, but the
# Guardian's native constants use a different ordering.
MOVE_ID_EXCEPTIONS: Mapping[str, Mapping[int, int]] = MappingProxyType(
    {
        "AcidSlime_S": MappingProxyType({2: 4}),
        "BronzeOrb": MappingProxyType({2: 3, 3: 2}),
        "GremlinFat": MappingProxyType({2: 1}),
        "Lagavulin": MappingProxyType({1: 2, 3: 1, 4: 3, 5: 3, 6: 3}),
        "Sentry": MappingProxyType(
            {
                3: 2,  # Bolt
                4: 1,  # Beam
            }
        ),
        "SlimeBoss": MappingProxyType({1: 2, 2: 1}),
        "Taskmaster": MappingProxyType({2: 1}),
        "TheCollector": MappingProxyType({1: 5}),
        "TheGuardian": MappingProxyType(
            {
                1: 5,  # Close Up / Defensive Mode
                2: 2,  # Fierce Bash
                3: 6,  # Roll Attack
                4: 7,  # Twin Slam
                5: 4,  # Whirlwind
                6: 1,  # Charging Up
                7: 3,  # Vent Steam
            }
        )
    }
)


# Native intent labels are UI categories and can differ from the canonical
# simulator label even when the move semantics are identical. Lagavulin's
# wake-up is presented as STUN but represented by the canonical no-action Sleep
# move. Red Slaver's Entangle is STRONG_DEBUFF natively and DEBUFF canonically.
# The Guardian's defensive-mode transition is BUFF natively and DEFEND
# canonically because the transition immediately grants Block. Its native Twin
# Slam is ATTACK_BUFF because it also raises the next Mode Shift threshold,
# while the canonical intent categorizes the visible two-hit action as ATTACK.
MOVE_INTENT_EXCEPTIONS: Mapping[tuple[str, int], str] = MappingProxyType(
    {
        ("Lagavulin", 4): "SLEEP",
        ("SlaverRed", 2): "DEBUFF",
        ("TheGuardian", 1): "DEFEND",
        ("TheGuardian", 4): "ATTACK",
    }
)


_ID_SEPARATOR = re.compile(r"[^A-Za-z0-9]+")


def normalized_id(value: str) -> str:
    """Return a locale-independent comparison key for native identifiers."""

    return _ID_SEPARATOR.sub("", value).casefold()


def canonicalize_power_id(
    native_id: str,
    canonical_ids: Collection[str],
) -> str:
    """Reconcile a CommunicationMod power.ID with the lightspeed catalog.

    Formatting-only differences are resolved for the complete current catalog;
    only identifiers whose semantics use genuinely different words need an
    explicit entry. Unknown identifiers are preserved so the observation
    catalog can still fail closed before an action is issued.
    """

    explicit = POWER_ID_EXCEPTIONS.get(native_id)
    if explicit is not None:
        if explicit not in canonical_ids:
            raise ValueError(
                f"Live power crosswalk target is outside the catalog: {explicit}"
            )
        return explicit

    wanted = normalized_id(native_id)
    matches = [item for item in canonical_ids if normalized_id(item) == wanted]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous normalized live power identifier: {native_id}"
        )
    return native_id


def canonicalize_move_id(monster_id: str, native_move_id: int) -> int:
    """Project a native monster move byte into the canonical move catalog."""

    return MOVE_ID_EXCEPTIONS.get(monster_id, {}).get(
        native_move_id,
        native_move_id,
    )


def canonicalize_monster_id(native_id: str) -> str:
    """Project a base-game monster ID into the canonical simulator catalog."""

    return MONSTER_ID_EXCEPTIONS.get(native_id, native_id)


def canonicalize_move_intent(
    monster_id: str,
    native_move_id: int,
    native_intent: str,
) -> str:
    """Project native intent only where canonical move semantics require it."""

    return MOVE_INTENT_EXCEPTIONS.get(
        (monster_id, native_move_id),
        native_intent,
    )
