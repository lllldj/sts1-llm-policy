from __future__ import annotations

from types import MappingProxyType
from typing import Mapping
import re

from .mechanics_catalog import describe_move, describe_move_by_id
from .state_schema import MonsterState


MECHANIC_GLOSSARY: Mapping[str, str] = MappingProxyType(
    {
        "BURN": "A Status card that is UNPLAYABLE and deals 2 damage to the player at end of turn (4 when upgraded).",
        "CONFUSED": "Whenever a card is drawn, randomize its cost to 0, 1, 2, or 3 Energy for this combat.",
        "DAZED": "A Status card that is UNPLAYABLE and ETHEREAL.",
        "DEXTERITY": "Modifies Block gained from cards by the shown amount.",
        "ENTANGLED": "Prevents the player from playing Attacks this turn.",
        "ETHEREAL": "If this card remains in hand at end of turn, EXHAUST it.",
        "EXHAUST": "Remove this card for the rest of this combat.",
        "FLIGHT": "Halves Attack damage received; lose 1 stack when hit by an Attack and become Stunned at 0.",
        "FRAIL": "Reduces Block gained from cards by 25% for the shown number of turns.",
        "HEX": "Whenever the player plays a non-Attack, shuffle 1 Dazed into the draw pile.",
        "METALLICIZE": "At the end of the owner's turn, gain Block equal to the shown amount.",
        "MINION": "A summoned enemy that leaves combat when its leader dies.",
        "NO_DRAW": "The player cannot draw additional cards this turn.",
        "RAGE": "Whenever the player plays an Attack this turn, gain the shown amount of Block.",
        "RITUAL": "At the end of the owner's turn, gain Strength equal to the Ritual amount.",
        "SHARP_HIDE": "Whenever the player plays an Attack, take damage equal to the shown amount.",
        "SLIMED": "A playable Status card that costs 1 Energy and EXHAUSTS.",
        "STASIS": "A Bronze Orb takes a highest-rarity card from the draw or discard pile; killing that Orb returns the visible held card to hand.",
        "STRENGTH": "Modifies Attack damage by the shown amount per hit.",
        "UNPLAYABLE": "This card cannot be played.",
        "VULNERABLE": "Increases damage received from Attacks by 50% for the shown number of turns.",
        "WEAK": "Reduces Attack damage dealt by 25% for the shown number of turns.",
        "WOUND": "A Status card that is UNPLAYABLE.",
    }
)


MOVE_EFFECT_OVERRIDES: Mapping[tuple[str, int], str] = MappingProxyType(
    {
        (
            "BronzeAutomaton",
            4,
        ): (
            "Summon two Bronze Orb minions into the open slots. Each Orb can "
            "use Stasis to take a highest-rarity card from the draw or discard "
            "pile; killing it returns its visible held card to hand. Orbs can "
            "also attack or give the Bronze Automaton Block."
        ),
        (
            "GremlinLeader",
            2,
        ): (
            "Summon Gremlin minions into up to two open slots. Their exact "
            "types are not known before the summon; they leave combat when "
            "the Gremlin Leader dies."
        ),
        (
            "TheCollector",
            5,
        ): (
            "Summon Torch Head minions into the open slots. Torch Heads "
            "repeatedly attack and leave combat when The Collector dies."
        ),
    }
)


def describe_move_v5(monster: MonsterState) -> str:
    """Return the v5 first-exposure description for the current move."""

    default = describe_move(monster)
    return MOVE_EFFECT_OVERRIDES.get(
        (monster.monster_id, monster.move_id),
        default,
    )


def describe_move_by_id_v5(
    monster_id: str,
    move_id: int,
) -> tuple[str, str, str]:
    """Return a behavior candidate with v5 first-exposure detail."""

    name, intent, default = describe_move_by_id(monster_id, move_id)
    return name, intent, MOVE_EFFECT_OVERRIDES.get((monster_id, move_id), default)


def select_mechanics(texts: list[str]) -> tuple[str, ...]:
    """Return the deterministic transitive glossary closure for rules text."""

    selected: set[str] = set()
    pending = list(texts)
    while pending:
        normalized = pending.pop().replace("_", " ").casefold()
        for keyword, definition in MECHANIC_GLOSSARY.items():
            term = keyword.replace("_", " ").casefold()
            if keyword not in selected and re.search(
                rf"\b{re.escape(term)}(?:es|s)?\b",
                normalized,
            ):
                selected.add(keyword)
                pending.append(definition)
    return tuple(sorted(selected))


def upgrade_v4_observation(observation: str) -> str:
    """Upgrade stored v4 text without changing state or action semantics."""

    if "COMBAT_OBJECTIVE:\n" not in observation or "\nCHARACTER: " not in observation:
        raise ValueError("Input is not a complete observation_v4 payload")

    upgraded = observation
    for key, replacement in MOVE_EFFECT_OVERRIDES.items():
        original = describe_move_by_id(*key)[2]
        upgraded = upgraded.replace(original, replacement)

    # Retain the old glossary as selection evidence (it captures mechanics of
    # unordered pile cards), but replace it in the emitted v5 payload.
    selection_text = re.sub(r"(?m)^EXHAUST: \d+ \|.*$", "", upgraded)
    mechanics = select_mechanics([selection_text])
    without_old = re.sub(
        r"\n\nKEYWORDS:\n.*?(?=\nCHARACTER: )",
        "",
        upgraded,
        count=1,
        flags=re.DOTALL,
    )
    marker = "\nCHARACTER: "
    prefix, suffix = without_old.split(marker, 1)
    if not mechanics:
        return without_old
    glossary = "\n".join(
        f"{keyword}: {MECHANIC_GLOSSARY[keyword]}" for keyword in mechanics
    )
    return f"{prefix}\n\nKEYWORDS:\n{glossary}{marker}{suffix}"
