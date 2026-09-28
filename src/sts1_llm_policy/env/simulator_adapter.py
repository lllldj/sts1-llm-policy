from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from .action_schema import ActionType, CanonicalAction
from .card_selection import (
    TASKS, CardSelectionState, SelectionCardState,
    build_selection_actions, selection_effect,
)
from .mechanics_catalog import SUPPORTED_CARD_EFFECTS
from .monster_behavior import build_monster_behavior
from .state_schema import (
    CanonicalState,
    CardState,
    CombatAccounting,
    CombatState,
    MonsterState,
    PlayerState,
    PowerState,
    RelicState,
)


class SimulatorAdapterError(ValueError):
    """Raised when a bridge record is outside the frozen P0 schema."""


@dataclass(frozen=True)
class SimulatorCanonicalDecision:
    state: CanonicalState
    actions: tuple[CanonicalAction, ...]
    native_action_ids: tuple[str, ...]
    decision_id: int
    scenario_id: str
    deck_preset: str
    loadout_id: str | None
    recipe_tier: str | None
    deck_hash: str | None
    terminal: bool
    outcome: str
    reward: float

    def native_action_id(self, action: CanonicalAction) -> str:
        """Resolve only an exact action object from this decision."""

        for index, candidate in enumerate(self.actions):
            if candidate is action:
                return self.native_action_ids[index]
        raise ValueError("Action is not from the current simulator legal set")


_CARD_ENUM_TO_ID = {
    "BURNING_PACT": "Burning Pact",
    "TRUE_GRIT": "True Grit",
    "WARCRY": "Warcry",
    "DUAL_WIELD": "Dual Wield",
    "ARMAMENTS": "Armaments",
    "HEADBUTT": "Headbutt",
    "EXHUME": "Exhume",
    "ANGER": "Anger",
    "ASCENDERS_BANE": "AscendersBane",
    "BASH": "Bash",
    "BATTLE_TRANCE": "Battle Trance",
    "BLOOD_FOR_BLOOD": "Blood for Blood",
    "BLOODLETTING": "Bloodletting",
    "BODY_SLAM": "Body Slam",
    "CARNAGE": "Carnage",
    "CLASH": "Clash",
    "CLEAVE": "Cleave",
    "CLOTHESLINE": "Clothesline",
    "COMBUST": "Combust",
    "DAZED": "Dazed",
    "DEFEND_RED": "Defend_R",
    "DISARM": "Disarm",
    "DROPKICK": "Dropkick",
    "ENTRENCH": "Entrench",
    "EVOLVE": "Evolve",
    "FEEL_NO_PAIN": "Feel No Pain",
    "FLAME_BARRIER": "Flame Barrier",
    "FIRE_BREATHING": "Fire Breathing",
    "FLEX": "Flex",
    "GHOSTLY_ARMOR": "Ghostly Armor",
    "HAVOC": "Havoc",
    "HEAVY_BLADE": "Heavy Blade",
    "HEMOKINESIS": "Hemokinesis",
    "INFLAME": "Inflame",
    "IRON_WAVE": "Iron Wave",
    "INTIMIDATE": "Intimidate",
    "METALLICIZE": "Metallicize",
    "POMMEL_STRIKE": "Pommel Strike",
    "POWER_THROUGH": "Power Through",
    "PERFECTED_STRIKE": "Perfected Strike",
    "PUMMEL": "Pummel",
    "RAGE": "Rage",
    "RECKLESS_CHARGE": "Reckless Charge",
    "RUPTURE": "Rupture",
    "SEARING_BLOW": "Searing Blow",
    "SEEING_RED": "Seeing Red",
    "DARK_EMBRACE": "Dark Embrace",
    "RAMPAGE": "Rampage",
    "DOUBLE_TAP": "Double Tap",
    "DEMON_FORM": "Demon Form",
    "BLUDGEON": "Bludgeon",
    "FEED": "Feed",
    "LIMIT_BREAK": "Limit Break",
    "CORRUPTION": "Corruption",
    "BARRICADE": "Barricade",
    "FIEND_FIRE": "Fiend Fire",
    "BERSERK": "Berserk",
    "IMPERVIOUS": "Impervious",
    "JUGGERNAUT": "Juggernaut",
    "BRUTALITY": "Brutality",
    "REAPER": "Reaper",
    "OFFERING": "Offering",
    "IMMOLATE": "Immolate",
    "BURN": "Burn",
    "SECOND_WIND": "Second Wind",
    "SENTINEL": "Sentinel",
    "SEVER_SOUL": "Sever Soul",
    "SHOCKWAVE": "Shockwave",
    "SHRUG_IT_OFF": "Shrug It Off",
    "SLIMED": "Slimed",
    "SPOT_WEAKNESS": "Spot Weakness",
    "STRIKE_RED": "Strike_R",
    "SWORD_BOOMERANG": "Sword Boomerang",
    "THUNDERCLAP": "Thunderclap",
    "TWIN_STRIKE": "Twin Strike",
    "UPPERCUT": "Uppercut",
    "WHIRLWIND": "Whirlwind",
    "WILD_STRIKE": "Wild Strike",
    "WOUND": "Wound",
}

_MONSTERS = {
    "INVALID = 0": ("OpenMonsterSlot", "Open monster slot"),
    "ACID_SLIME_M": ("AcidSlime_M", "Acid Slime (M)"),
    "ACID_SLIME_S": ("AcidSlime_S", "Acid Slime (S)"),
    "BLUE_SLAVER": ("SlaverBlue", "Blue Slaver"),
    "CULTIST": ("Cultist", "Cultist"),
    "FAT_GREMLIN": ("GremlinFat", "Fat Gremlin"),
    "FUNGI_BEAST": ("FungiBeast", "Fungi Beast"),
    "GREMLIN_NOB": ("GremlinNob", "Gremlin Nob"),
    "GREMLIN_WIZARD": ("GremlinWizard", "Gremlin Wizard"),
    "JAW_WORM": ("JawWorm", "Jaw Worm"),
    "LAGAVULIN": ("Lagavulin", "Lagavulin"),
    "LOOTER": ("Looter", "Looter"),
    "MAD_GREMLIN": ("GremlinWarrior", "Mad Gremlin"),
    "RED_LOUSE": ("FuzzyLouseNormal", "Louse"),
    "RED_SLAVER": ("SlaverRed", "Red Slaver"),
    "GREEN_LOUSE": ("FuzzyLouseDefensive", "Louse"),
    "SENTRY": ("Sentry", "Sentry"),
    "SHIELD_GREMLIN": ("GremlinTsundere", "Shield Gremlin"),
    "SNEAKY_GREMLIN": ("GremlinThief", "Sneaky Gremlin"),
    "SPIKE_SLIME_M": ("SpikeSlime_M", "Spike Slime (M)"),
    "SPIKE_SLIME_S": ("SpikeSlime_S", "Spike Slime (S)"),
    "ACID_SLIME_L": ("AcidSlime_L", "Acid Slime (L)"),
    "SPIKE_SLIME_L": ("SpikeSlime_L", "Spike Slime (L)"),
    "BYRD": ("Byrd", "Byrd"),
    "CENTURION": ("Centurion", "Centurion"),
    "CHOSEN": ("Chosen", "Chosen"),
    "MUGGER": ("Mugger", "Mugger"),
    "MYSTIC": ("Mystic", "Mystic"),
    "SHELLED_PARASITE": ("ShelledParasite", "Shelled Parasite"),
    "SNAKE_PLANT": ("SnakePlant", "Snake Plant"),
    "SNECKO": ("Snecko", "Snecko"),
    "SPHERIC_GUARDIAN": ("SphericGuardian", "Spheric Guardian"),
    "BOOK_OF_STABBING": ("BookOfStabbing", "Book of Stabbing"),
    "BRONZE_AUTOMATON": ("BronzeAutomaton", "Bronze Automaton"),
    "BRONZE_ORB": ("BronzeOrb", "Bronze Orb"),
    "GREMLIN_LEADER": ("GremlinLeader", "Gremlin Leader"),
    "HEXAGHOST": ("Hexaghost", "Hexaghost"),
    "SLIME_BOSS": ("SlimeBoss", "Slime Boss"),
    "TASKMASTER": ("Taskmaster", "Taskmaster"),
    "THE_CHAMP": ("TheChamp", "The Champ"),
    "THE_COLLECTOR": ("TheCollector", "The Collector"),
    "THE_GUARDIAN": ("TheGuardian", "The Guardian"),
    "TORCH_HEAD": ("TorchHead", "Torch Head"),
}

# (CommunicationMod-compatible intent, CommunicationMod move_id)
_MOVES = {
    "INVALID": ("UNKNOWN", 0),
    "ACID_SLIME_M_CORROSIVE_SPIT": ("ATTACK_DEBUFF", 1),
    "ACID_SLIME_M_LICK": ("DEBUFF", 4),
    "ACID_SLIME_M_TACKLE": ("ATTACK", 2),
    "ACID_SLIME_S_LICK": ("DEBUFF", 4),
    "ACID_SLIME_S_TACKLE": ("ATTACK", 1),
    "BLUE_SLAVER_STAB": ("ATTACK", 1),
    "BLUE_SLAVER_RAKE": ("ATTACK_DEBUFF", 4),
    "CULTIST_INCANTATION": ("BUFF", 3),
    "CULTIST_DARK_STRIKE": ("ATTACK", 1),
    "FAT_GREMLIN_SMASH": ("ATTACK_DEBUFF", 1),
    "FUNGI_BEAST_BITE": ("ATTACK", 1),
    "FUNGI_BEAST_GROW": ("BUFF", 2),
    "GREMLIN_NOB_BELLOW": ("BUFF", 3),
    "GREMLIN_NOB_RUSH": ("ATTACK", 1),
    "GREMLIN_NOB_SKULL_BASH": ("ATTACK_DEBUFF", 2),
    "GREMLIN_WIZARD_CHARGING": ("UNKNOWN", 2),
    "GREMLIN_WIZARD_ULTIMATE_BLAST": ("ATTACK", 1),
    "JAW_WORM_CHOMP": ("ATTACK", 1),
    "JAW_WORM_THRASH": ("ATTACK_DEFEND", 3),
    "JAW_WORM_BELLOW": ("DEFEND_BUFF", 2),
    "LAGAVULIN_ATTACK": ("ATTACK", 1),
    "LAGAVULIN_SIPHON_SOUL": ("STRONG_DEBUFF", 2),
    "LAGAVULIN_SLEEP": ("SLEEP", 3),
    "LOOTER_MUG": ("ATTACK", 1),
    "LOOTER_SMOKE_BOMB": ("DEFEND", 2),
    "LOOTER_ESCAPE": ("ESCAPE", 3),
    "LOOTER_LUNGE": ("ATTACK", 4),
    "MAD_GREMLIN_SCRATCH": ("ATTACK", 1),
    "GREEN_LOUSE_BITE": ("ATTACK", 3),
    "GREEN_LOUSE_SPIT_WEB": ("DEBUFF", 4),
    "RED_LOUSE_BITE": ("ATTACK", 3),
    "RED_LOUSE_GROW": ("BUFF", 4),
    "RED_SLAVER_STAB": ("ATTACK", 1),
    "RED_SLAVER_ENTANGLE": ("DEBUFF", 2),
    "RED_SLAVER_SCRAPE": ("ATTACK_DEBUFF", 3),
    "SENTRY_BEAM": ("ATTACK", 1),
    "SENTRY_BOLT": ("DEBUFF", 2),
    "SHIELD_GREMLIN_PROTECT": ("DEFEND", 1),
    "SHIELD_GREMLIN_SHIELD_BASH": ("ATTACK", 2),
    "SNEAKY_GREMLIN_PUNCTURE": ("ATTACK", 1),
    "SPIKE_SLIME_M_FLAME_TACKLE": ("ATTACK_DEBUFF", 1),
    "SPIKE_SLIME_M_LICK": ("DEBUFF", 4),
    "SPIKE_SLIME_S_TACKLE": ("ATTACK", 1),
    "ACID_SLIME_L_CORROSIVE_SPIT": ("ATTACK_DEBUFF", 1),
    "ACID_SLIME_L_TACKLE": ("ATTACK", 2),
    "ACID_SLIME_L_SPLIT": ("UNKNOWN", 3),
    "ACID_SLIME_L_LICK": ("DEBUFF", 4),
    "SPIKE_SLIME_L_FLAME_TACKLE": ("ATTACK_DEBUFF", 1),
    "SPIKE_SLIME_L_SPLIT": ("UNKNOWN", 3),
    "SPIKE_SLIME_L_LICK": ("DEBUFF", 4),
    "BYRD_PECK": ("ATTACK", 1),
    "BYRD_FLY": ("BUFF", 2),
    "BYRD_SWOOP": ("ATTACK", 3),
    "BYRD_STUNNED": ("STUN", 4),
    "BYRD_HEADBUTT": ("ATTACK", 5),
    "BYRD_CAW": ("BUFF", 6),
    "CENTURION_SLASH": ("ATTACK", 1),
    "CENTURION_DEFEND": ("DEFEND", 2),
    "CENTURION_FURY": ("ATTACK", 3),
    "CHOSEN_ZAP": ("ATTACK", 1),
    "CHOSEN_DRAIN": ("DEBUFF", 2),
    "CHOSEN_DEBILITATE": ("ATTACK_DEBUFF", 3),
    "CHOSEN_HEX": ("DEBUFF", 4),
    "CHOSEN_POKE": ("ATTACK", 5),
    "MUGGER_MUG": ("ATTACK", 1),
    "MUGGER_SMOKE_BOMB": ("DEFEND", 2),
    "MUGGER_ESCAPE": ("ESCAPE", 3),
    "MUGGER_LUNGE": ("ATTACK", 4),
    "MYSTIC_ATTACK_DEBUFF": ("ATTACK_DEBUFF", 1),
    "MYSTIC_HEAL": ("HEAL", 2),
    "MYSTIC_BUFF": ("BUFF", 3),
    "SHELLED_PARASITE_FELL": ("ATTACK_DEBUFF", 1),
    "SHELLED_PARASITE_DOUBLE_STRIKE": ("ATTACK", 2),
    "SHELLED_PARASITE_SUCK": ("ATTACK_BUFF", 3),
    "SHELLED_PARASITE_STUNNED": ("STUN", 4),
    "SNAKE_PLANT_CHOMP": ("ATTACK", 1),
    "SNAKE_PLANT_ENFEEBLING_SPORES": ("STRONG_DEBUFF", 2),
    "SNECKO_PERPLEXING_GLARE": ("STRONG_DEBUFF", 1),
    "SNECKO_BITE": ("ATTACK", 2),
    "SNECKO_TAIL_WHIP": ("ATTACK_DEBUFF", 3),
    "SPHERIC_GUARDIAN_SLAM": ("ATTACK", 1),
    "SPHERIC_GUARDIAN_ACTIVATE": ("DEFEND", 2),
    "SPHERIC_GUARDIAN_HARDEN": ("ATTACK_DEFEND", 3),
    "SPHERIC_GUARDIAN_ATTACK_DEBUFF": ("ATTACK_DEBUFF", 4),
    "BOOK_OF_STABBING_MULTI_STAB": ("ATTACK", 1),
    "BOOK_OF_STABBING_SINGLE_STAB": ("ATTACK", 2),
    "BRONZE_AUTOMATON_BOOST": ("DEFEND_BUFF", 5),
    "BRONZE_AUTOMATON_FLAIL": ("ATTACK", 1),
    "BRONZE_AUTOMATON_HYPER_BEAM": ("ATTACK", 2),
    "BRONZE_AUTOMATON_SPAWN_ORBS": ("UNKNOWN", 4),
    "BRONZE_AUTOMATON_STUNNED": ("STUN", 3),
    "BRONZE_ORB_BEAM": ("ATTACK", 1),
    "BRONZE_ORB_STASIS": ("UNKNOWN", 2),
    "BRONZE_ORB_SUPPORT_BEAM": ("DEFEND", 3),
    "GREMLIN_LEADER_ENCOURAGE": ("DEFEND_BUFF", 3),
    "GREMLIN_LEADER_RALLY": ("UNKNOWN", 2),
    "GREMLIN_LEADER_STAB": ("ATTACK", 4),
    "HEXAGHOST_ACTIVATE": ("UNKNOWN", 5),
    "HEXAGHOST_DIVIDER": ("ATTACK", 1),
    "HEXAGHOST_INFERNO": ("ATTACK", 6),
    "HEXAGHOST_SEAR": ("ATTACK_DEBUFF", 4),
    "HEXAGHOST_TACKLE": ("ATTACK", 2),
    "HEXAGHOST_INFLAME": ("DEFEND_BUFF", 3),
    "SLIME_BOSS_GOOP_SPRAY": ("STRONG_DEBUFF", 4),
    "SLIME_BOSS_PREPARING": ("UNKNOWN", 1),
    "SLIME_BOSS_SLAM": ("ATTACK", 2),
    "SLIME_BOSS_SPLIT": ("UNKNOWN", 3),
    "TASKMASTER_SCOURING_WHIP": ("ATTACK_DEBUFF", 1),
    "THE_CHAMP_DEFENSIVE_STANCE": ("DEFEND_BUFF", 2),
    "THE_CHAMP_FACE_SLAP": ("ATTACK_DEBUFF", 4),
    "THE_CHAMP_TAUNT": ("STRONG_DEBUFF", 6),
    "THE_CHAMP_HEAVY_SLASH": ("ATTACK", 1),
    "THE_CHAMP_GLOAT": ("BUFF", 5),
    "THE_CHAMP_EXECUTE": ("ATTACK", 3),
    "THE_CHAMP_ANGER": ("BUFF", 7),
    "THE_COLLECTOR_BUFF": ("DEFEND_BUFF", 3),
    "THE_COLLECTOR_FIREBALL": ("ATTACK", 2),
    "THE_COLLECTOR_MEGA_DEBUFF": ("STRONG_DEBUFF", 4),
    "THE_COLLECTOR_SPAWN": ("UNKNOWN", 5),
    "THE_GUARDIAN_CHARGING_UP": ("DEFEND", 1),
    "THE_GUARDIAN_FIERCE_BASH": ("ATTACK", 2),
    "THE_GUARDIAN_VENT_STEAM": ("STRONG_DEBUFF", 3),
    "THE_GUARDIAN_WHIRLWIND": ("ATTACK", 4),
    "THE_GUARDIAN_DEFENSIVE_MODE": ("DEFEND", 5),
    "THE_GUARDIAN_ROLL_ATTACK": ("ATTACK", 6),
    "THE_GUARDIAN_TWIN_SLAM": ("ATTACK", 7),
    "TORCH_HEAD_TACKLE": ("ATTACK", 1),
}

_SUPPORTED_POWERS = {
    "Angry",
    "Artifact",
    "Asleep",
    "Curl Up",
    "Dexterity",
    "Enrage",
    "Frail",
    "Metallicize",
    "Combust",
    "Evolve",
    "Feel No Pain",
    "Flame Barrier",
    "Fire Breathing",
    "Lose Strength",
    "No Draw",
    "Rage",
    "Ritual",
    "Spore Cloud",
    "Strength",
    "Thievery",
    "Entangled",
    "Vulnerable",
    "Weak",
    "Barricade",
    "Brutality",
    "Confused",
    "Corruption",
    "Dark Embrace",
    "Demon Form",
    "Double Tap",
    "Flight",
    "Hex",
    "Juggernaut",
    "Malleable",
    "Plated Armor",
    "Rupture",
    "Minion",
    "Minion Leader",
    "Painful Stabs",
    "Mode Shift",
    "Sharp Hide",
    "Stasis",
    "Thorns",
    "Vigor",
}

P0_SCENARIOS = ("cultist", "jaw_worm", "two_louse")
SUPPORTED_SCENARIOS = P0_SCENARIOS + (
    "gremlin_gang",
    "gremlin_nob",
    "lagavulin",
    "three_sentries",
)
D1_SCENARIOS = SUPPORTED_SCENARIOS + (
    "small_slimes",
    "lots_of_slimes",
    "blue_slaver",
    "red_slaver",
    "looter",
    "exordium_thugs",
    "exordium_wildlife",
    "two_fungi_beasts",
    "large_slime",
    "three_louse",
    "spheric_guardian",
    "chosen",
    "shell_parasite",
    "three_byrds",
    "two_thieves",
    "chosen_and_byrds",
    "sentry_and_sphere",
    "cultist_and_chosen",
    "three_cultist",
    "shelled_parasite_and_fungi",
    "snecko",
    "snake_plant",
    "centurion_and_healer",
    "gremlin_leader",
    "slavers",
    "book_of_stabbing",
    "slime_boss",
    "the_guardian",
    "hexaghost",
    "automaton",
    "collector",
    "champ",
)
SUPPORTED_DECK_PRESETS = ("starter", "elite_transition", "boss_ready")
COMBAT_SNAPSHOT_PRESET = "combat_snapshot_v1"
COMBAT_SNAPSHOT_PRESETS = (COMBAT_SNAPSHOT_PRESET, "combat_snapshot_v2")
SUPPORTED_FEATURE_RECIPES = ("feature_basic", "feature_elite", "feature_boss")


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SimulatorAdapterError(f"{path} must be an object")
    return value


def _array(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SimulatorAdapterError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise SimulatorAdapterError(f"{path} must be a string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimulatorAdapterError(f"{path} must be an integer")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise SimulatorAdapterError(f"{path} must be a boolean")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulatorAdapterError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SimulatorAdapterError(f"{path} must be finite")
    return result


def _field(value: dict[str, Any], key: str, path: str) -> object:
    if key not in value:
        raise SimulatorAdapterError(f"{path}.{key} is required")
    return value[key]


def _uuid_prefix(raw_state: dict[str, Any], decision_id: int) -> str:
    scenario_id = _string(_field(raw_state, "scenario_id", "state"), "state.scenario_id")
    seed = _integer(_field(raw_state, "seed", "state"), "state.seed")
    return f"sim:{scenario_id}:{seed}:{decision_id}"


def _parse_power(raw_value: object, path: str) -> PowerState:
    raw = _object(raw_value, path)
    name = _string(_field(raw, "name", path), f"{path}.name")
    if name not in _SUPPORTED_POWERS:
        raise SimulatorAdapterError(f"{path}.name is outside P0 support: {name}")
    amount = _integer(_field(raw, "amount", path), f"{path}.amount")
    just_applied = False
    if "just_applied" in raw:
        just_applied = _boolean(raw["just_applied"], f"{path}.just_applied")
    return PowerState(
        power_id=name,
        name=name,
        amount=amount,
        just_applied=just_applied,
    )


def _parse_powers(raw_value: object, path: str) -> tuple[PowerState, ...]:
    return tuple(
        _parse_power(item, f"{path}[{index}]")
        for index, item in enumerate(_array(raw_value, path))
    )


def _parse_card(
    raw_value: object,
    path: str,
    *,
    uuid: str,
) -> CardState:
    raw = _object(raw_value, path)
    enum_id = _string(_field(raw, "enum_id", path), f"{path}.enum_id")
    upgrades = _integer(_field(raw, "upgrades", path), f"{path}.upgrades")
    try:
        expected_card_id = _CARD_ENUM_TO_ID[enum_id]
    except KeyError as error:
        raise SimulatorAdapterError(
            f"{path} card is outside supported deck presets: "
            f"{enum_id} upgrade {upgrades}"
        ) from error
    description_key = (expected_card_id, upgrades)
    extended_effect = selection_effect(expected_card_id, upgrades)
    if description_key not in SUPPORTED_CARD_EFFECTS and extended_effect is None:
        raise SimulatorAdapterError(
            f"{path} card version is outside support: "
            f"{expected_card_id} upgrade {upgrades}"
        )
    card_id = _string(_field(raw, "string_id", path), f"{path}.string_id")
    if card_id != expected_card_id:
        raise SimulatorAdapterError(f"{path}.string_id disagrees with enum_id")

    card_class = SelectionCardState if extended_effect is not None else CardState
    return card_class(
        card_id=card_id,
        name=_string(_field(raw, "name", path), f"{path}.name"),
        card_type=_string(_field(raw, "type", path), f"{path}.type"),
        cost=_integer(_field(raw, "cost_for_turn", path), f"{path}.cost_for_turn"),
        upgrades=upgrades,
        description=(extended_effect if extended_effect is not None
                     else SUPPORTED_CARD_EFFECTS[description_key]),
        exhausts=_boolean(_field(raw, "exhausts", path), f"{path}.exhausts"),
        ethereal=_boolean(_field(raw, "ethereal", path), f"{path}.ethereal"),
        has_target=_boolean(_field(raw, "has_target", path), f"{path}.has_target"),
        is_playable=_boolean(
            _field(raw, "is_playable", path),
            f"{path}.is_playable",
        ),
        uuid=uuid,
        special_data=_integer(raw.get("special_data", 0), f"{path}.special_data"),
    )


def _parse_pile(
    raw_value: object,
    path: str,
    *,
    uuid_prefix: str,
) -> tuple[CardState, ...]:
    return tuple(
        _parse_card(item, f"{path}[{index}]", uuid=f"{uuid_prefix}:{index}")
        for index, item in enumerate(_array(raw_value, path))
    )


def _parse_monster(
    raw_value: object,
    path: str,
) -> tuple[MonsterState, int | None, bool, bool]:
    raw = _object(raw_value, path)
    native_id = _string(_field(raw, "id", path), f"{path}.id")
    try:
        monster_id, name = _MONSTERS[native_id]
    except KeyError as error:
        raise SimulatorAdapterError(
            f"{path}.id is outside P0 support: {native_id}"
        ) from error

    move_name = _string(_field(raw, "move_name", path), f"{path}.move_name")
    try:
        intent, move_id = _MOVES[move_name]
    except KeyError as error:
        raise SimulatorAdapterError(
            f"{path}.move_name is outside support: {move_name}"
        ) from error

    has_behavior_source = "previous_move_name" in raw
    previous_move_id: int | None = None
    if has_behavior_source:
        previous_move_name = _string(
            raw["previous_move_name"],
            f"{path}.previous_move_name",
        )
        if previous_move_name != "INVALID":
            try:
                _, previous_move_id = _MOVES[previous_move_name]
            except KeyError as error:
                raise SimulatorAdapterError(
                    f"{path}.previous_move_name is outside support: "
                    f"{previous_move_name}"
                ) from error

    attacking = _boolean(_field(raw, "is_attacking", path), f"{path}.is_attacking")
    hits = _integer(_field(raw, "hits", path), f"{path}.hits")
    base_damage = _integer(_field(raw, "base_damage", path), f"{path}.base_damage")
    adjusted_damage = _integer(
        _field(raw, "adjusted_damage", path),
        f"{path}.adjusted_damage",
    )
    if not attacking:
        hits = 1
        base_damage = -1
        adjusted_damage = -1

    monster = MonsterState(
        monster_id=monster_id,
        name=name,
        current_hp=_integer(_field(raw, "current_hp", path), f"{path}.current_hp"),
        max_hp=_integer(_field(raw, "max_hp", path), f"{path}.max_hp"),
        block=_integer(_field(raw, "block", path), f"{path}.block"),
        intent=intent,
        move_id=move_id,
        move_hits=hits,
        move_base_damage=base_damage,
        move_adjusted_damage=adjusted_damage,
        powers=_parse_powers(_field(raw, "powers", path), f"{path}.powers"),
        is_gone=_boolean(_field(raw, "is_gone", path), f"{path}.is_gone"),
        half_dead=_boolean(_field(raw, "half_dead", path), f"{path}.half_dead"),
        stasis_card=(
            _parse_card(
                raw["stasis_card"],
                f"{path}.stasis_card",
                uuid=f"sim:stasis:{path}",
            )
            if "stasis_card" in raw
            else None
        ),
    )
    has_used_entangle = False
    if "has_used_entangle" in raw:
        has_used_entangle = _boolean(
            raw["has_used_entangle"],
            f"{path}.has_used_entangle",
        )
    return monster, previous_move_id, has_behavior_source, has_used_entangle


def _parse_state(raw_state: dict[str, Any]) -> CanonicalState:
    if _integer(_field(raw_state, "schema_version", "state"), "state.schema_version") != 1:
        raise SimulatorAdapterError("Unsupported simulator state schema")
    decision_id = _integer(
        _field(raw_state, "decision_id", "state"),
        "state.decision_id",
    )
    prefix = _uuid_prefix(raw_state, decision_id)
    raw_player = _object(_field(raw_state, "player", "state"), "state.player")
    raw_monsters = _array(_field(raw_state, "monsters", "state"), "state.monsters")
    parsed_monsters = tuple(
        _parse_monster(item, f"state.monsters[{index}]")
        for index, item in enumerate(raw_monsters)
    )
    living_monster_count = sum(
        not monster.is_gone
        for monster, _previous_move_id, _has_behavior_source, _has_used_entangle in parsed_monsters
    )
    monsters = tuple(
        replace(
            monster,
            behavior=build_monster_behavior(
                monster,
                previous_move_id=previous_move_id,
                living_monster_count=living_monster_count,
                combat_turn=_integer(
                    _field(raw_state, "turn", "state"),
                    "state.turn",
                )
                + 1,
                ascension=_integer(
                    _field(raw_state, "ascension", "state"),
                    "state.ascension",
                ),
                has_used_entangle=has_used_entangle,
            ),
        )
        if has_behavior_source
        else monster
        for monster, previous_move_id, has_behavior_source, has_used_entangle in parsed_monsters
    )
    accounting: CombatAccounting | None = None
    if "combat_accounting" in raw_state:
        raw_accounting = _object(
            raw_state["combat_accounting"],
            "state.combat_accounting",
        )
        starting_hp = _integer(
            _field(raw_accounting, "starting_hp", "state.combat_accounting"),
            "state.combat_accounting.starting_hp",
        )
        enemy_damage_taken = _integer(
            _field(
                raw_accounting,
                "enemy_damage_taken",
                "state.combat_accounting",
            ),
            "state.combat_accounting.enemy_damage_taken",
        )
        self_hp_loss = _integer(
            _field(raw_accounting, "self_hp_loss", "state.combat_accounting"),
            "state.combat_accounting.self_hp_loss",
        )
        total_hp_loss = _integer(
            _field(raw_accounting, "total_hp_loss", "state.combat_accounting"),
            "state.combat_accounting.total_hp_loss",
        )
        if starting_hp <= 0 or enemy_damage_taken < 0 or self_hp_loss < 0:
            raise SimulatorAdapterError("state.combat_accounting is negative or invalid")
        if total_hp_loss != enemy_damage_taken + self_hp_loss:
            raise SimulatorAdapterError(
                "state.combat_accounting total disagrees with its sources"
            )
        accounting = CombatAccounting(
            starting_hp=starting_hp,
            enemy_damage_taken=enemy_damage_taken,
            self_hp_loss=self_hp_loss,
        )

    raw_relics = _array(raw_state.get("relics", []), "state.relics")
    relics: list[RelicState] = []
    for index, value in enumerate(raw_relics):
        path = f"state.relics[{index}]"
        relic = _object(value, path)
        counter_value = relic.get("counter")
        relics.append(
            RelicState(
                relic_id=_string(_field(relic, "id", path), f"{path}.id"),
                name=_string(_field(relic, "name", path), f"{path}.name"),
                counter=None
                if counter_value is None
                else _integer(counter_value, f"{path}.counter"),
            )
        )

    return CanonicalState(
        seed=_integer(_field(raw_state, "seed", "state"), "state.seed"),
        character="IRONCLAD",
        ascension_level=_integer(
            _field(raw_state, "ascension", "state"),
            "state.ascension",
        ),
        act=_integer(_field(raw_state, "act", "state"), "state.act"),
        floor=_integer(_field(raw_state, "floor", "state"), "state.floor"),
        relics=tuple(relics),
        combat=CombatState(
            # sts_lightspeed uses zero-based combat turns internally.
            turn=_integer(_field(raw_state, "turn", "state"), "state.turn") + 1,
            player=PlayerState(
                current_hp=_integer(
                    _field(raw_player, "current_hp", "state.player"),
                    "state.player.current_hp",
                ),
                max_hp=_integer(
                    _field(raw_player, "max_hp", "state.player"),
                    "state.player.max_hp",
                ),
                block=_integer(
                    _field(raw_player, "block", "state.player"),
                    "state.player.block",
                ),
                energy=_integer(
                    _field(raw_player, "energy", "state.player"),
                    "state.player.energy",
                ),
                powers=_parse_powers(
                    _field(raw_player, "powers", "state.player"),
                    "state.player.powers",
                ),
            ),
            monsters=monsters,
            hand=_parse_pile(
                _field(raw_state, "hand", "state"),
                "state.hand",
                uuid_prefix=f"{prefix}:hand",
            ),
            draw_pile=_parse_pile(
                _field(raw_state, "draw_pile", "state"),
                "state.draw_pile",
                uuid_prefix=f"{prefix}:draw",
            ),
            discard_pile=_parse_pile(
                _field(raw_state, "discard_pile", "state"),
                "state.discard_pile",
                uuid_prefix=f"{prefix}:discard",
            ),
            exhaust_pile=_parse_pile(
                _field(raw_state, "exhaust_pile", "state"),
                "state.exhaust_pile",
                uuid_prefix=f"{prefix}:exhaust",
            ),
            accounting=accounting,
        ),
    )


def _parse_actions(
    raw_value: object,
    state: CanonicalState,
    raw_monsters: list[Any],
    raw_hand: list[Any],
    *,
    terminal: bool,
) -> tuple[tuple[CanonicalAction, ...], tuple[str, ...]]:
    raw_actions = _array(raw_value, "legal_actions")
    if terminal and raw_actions:
        raise SimulatorAdapterError("Terminal state must have no legal actions")
    if not terminal and not raw_actions:
        raise SimulatorAdapterError("Non-terminal state must have legal actions")

    actions: list[CanonicalAction] = []
    native_ids: list[str] = []
    seen_native_ids: set[str] = set()
    end_count = 0
    for index, item in enumerate(raw_actions):
        path = f"legal_actions[{index}]"
        raw = _object(item, path)
        native_id = _string(_field(raw, "action_id", path), f"{path}.action_id")
        if native_id in seen_native_ids:
            raise SimulatorAdapterError(f"Duplicate native action_id: {native_id}")
        seen_native_ids.add(native_id)
        kind = _string(_field(raw, "kind", path), f"{path}.kind")
        action_id = f"ACTION_{index}"

        if kind == "END_TURN":
            if native_id != "END":
                raise SimulatorAdapterError(f"{path}.action_id is not canonical END")
            end_count += 1
            action = CanonicalAction(
                action_id=action_id,
                action_type=ActionType.END_TURN,
            )
        elif kind == "PLAY_CARD":
            hand_index = _integer(
                _field(raw, "hand_index", path),
                f"{path}.hand_index",
            )
            if not 0 <= hand_index < len(state.combat.hand):
                raise SimulatorAdapterError(f"{path}.hand_index is out of range")
            card = state.combat.hand[hand_index]
            raw_card = _object(raw_hand[hand_index], f"state.hand[{hand_index}]")
            if _string(_field(raw, "card_name", path), f"{path}.card_name") != card.name:
                raise SimulatorAdapterError(f"{path}.card_name does not match hand")
            if _string(_field(raw, "card_id", path), f"{path}.card_id") != _string(
                _field(raw_card, "enum_id", f"state.hand[{hand_index}]"),
                f"state.hand[{hand_index}].enum_id",
            ):
                raise SimulatorAdapterError(f"{path}.card_id does not match hand")
            if _integer(
                _field(raw, "cost_for_turn", path),
                f"{path}.cost_for_turn",
            ) != card.cost:
                raise SimulatorAdapterError(f"{path}.cost_for_turn does not match hand")
            if not card.is_playable:
                raise SimulatorAdapterError(f"{path} references an unplayable card")

            target_value = _field(raw, "target_index", path)
            if target_value is None:
                target_index = None
            else:
                target_index = _integer(target_value, f"{path}.target_index")
            if card.has_target:
                if target_index is None or not 0 <= target_index < len(raw_monsters):
                    raise SimulatorAdapterError(f"{path}.target_index is invalid")
                raw_target = _object(raw_monsters[target_index], f"state.monsters[{target_index}]")
                if _field(raw_target, "is_targetable", f"state.monsters[{target_index}]") is not True:
                    raise SimulatorAdapterError(f"{path}.target_index is not targetable")
            elif target_index is not None:
                raise SimulatorAdapterError(
                    f"{path}.target_index must be null for a non-targeted card"
                )
            expected_native_id = (
                f"PLAY:{hand_index}:"
                f"{target_index if target_index is not None else -1}"
            )
            if native_id != expected_native_id:
                raise SimulatorAdapterError(f"{path}.action_id does not match action fields")

            action = CanonicalAction(
                action_id=action_id,
                action_type=ActionType.PLAY_CARD,
                hand_index=hand_index,
                card_uuid=card.uuid,
                card_name=card.name,
                target_index=target_index,
            )
        else:
            raise SimulatorAdapterError(f"{path}.kind is unsupported: {kind}")

        actions.append(action)
        native_ids.append(native_id)

    if not terminal and (end_count != 1 or actions[-1].action_type != ActionType.END_TURN):
        raise SimulatorAdapterError("Non-terminal actions must end with one END_TURN")
    return tuple(actions), tuple(native_ids)


def from_sts_lightspeed_response(
    response_value: object,
    *,
    allow_card_selection: bool = False,
) -> SimulatorCanonicalDecision:
    """Convert one bridge reset/step response into project-owned schemas."""

    response = _object(response_value, "response")
    raw_state = _object(_field(response, "state", "response"), "state")
    if _string(
        _field(raw_state, "draw_pile_top", "state"),
        "state.draw_pile_top",
    ) != "back":
        raise SimulatorAdapterError("Unsupported simulator draw-pile ordering")
    scenario_id = _string(
        _field(raw_state, "scenario_id", "state"),
        "state.scenario_id",
    )
    if scenario_id not in D1_SCENARIOS:
        raise SimulatorAdapterError(f"Unsupported scenario_id: {scenario_id}")
    deck_preset = "starter"
    if "deck_preset" in raw_state:
        deck_preset = _string(raw_state["deck_preset"], "state.deck_preset")
    if deck_preset not in (*SUPPORTED_DECK_PRESETS, *COMBAT_SNAPSHOT_PRESETS):
        raise SimulatorAdapterError(f"Unsupported deck_preset: {deck_preset}")
    loadout_id: str | None = None
    recipe_tier: str | None = None
    deck_hash: str | None = None
    if "loadout" in raw_state:
        raw_loadout = _object(raw_state["loadout"], "state.loadout")
        if deck_preset in COMBAT_SNAPSHOT_PRESETS:
            if _string(
                _field(raw_loadout, "source", "state.loadout"),
                "state.loadout.source",
            ) != deck_preset:
                raise SimulatorAdapterError("state.loadout snapshot source disagrees")
            fingerprint = _string(
                _field(
                    raw_loadout,
                    "snapshot_fingerprint_fnv1a64",
                    "state.loadout",
                ),
                "state.loadout.snapshot_fingerprint_fnv1a64",
            )
            if len(fingerprint) != 16 or any(
                character not in "0123456789abcdef" for character in fingerprint
            ):
                raise SimulatorAdapterError("state.loadout snapshot fingerprint is invalid")
            loadout_id = f"{deck_preset}:{fingerprint}"
        else:
            loadout_id = _string(
                _field(raw_loadout, "loadout_id", "state.loadout"),
                "state.loadout.loadout_id",
            )
            recipe_tier = _string(
                _field(raw_loadout, "recipe_tier", "state.loadout"),
                "state.loadout.recipe_tier",
            )
            deck_hash = _string(
                _field(raw_loadout, "deck_hash", "state.loadout"),
                "state.loadout.deck_hash",
            )
            base_preset = _string(
                _field(raw_loadout, "base_preset", "state.loadout"),
                "state.loadout.base_preset",
            )
            if recipe_tier not in SUPPORTED_FEATURE_RECIPES:
                raise SimulatorAdapterError("state.loadout.recipe_tier is unsupported")
            if base_preset != deck_preset:
                raise SimulatorAdapterError("state.loadout.base_preset disagrees with deck_preset")
            if not loadout_id or len(deck_hash) != 64 or any(
                character not in "0123456789abcdef" for character in deck_hash
            ):
                raise SimulatorAdapterError("state.loadout identity/hash is invalid")
    terminal = _boolean(_field(raw_state, "terminal", "state"), "state.terminal")
    outcome = _string(_field(raw_state, "outcome", "state"), "state.outcome")
    expected_outcomes = (
        {"PLAYER_VICTORY", "PLAYER_LOSS"}
        if terminal
        else {"UNDECIDED"}
    )
    if outcome not in expected_outcomes:
        raise SimulatorAdapterError("state outcome/terminal fields disagree")
    input_state = _string(
        _field(raw_state, "input_state", "state"),
        "state.input_state",
    )
    if not terminal and input_state not in (
        ("PLAYER_NORMAL", "CARD_SELECT") if allow_card_selection else ("PLAYER_NORMAL",)
    ):
        raise SimulatorAdapterError("Non-terminal state is not a player decision")

    decision_id = _integer(
        _field(raw_state, "decision_id", "state"),
        "state.decision_id",
    )
    state = _parse_state(raw_state)
    if not allow_card_selection and any(
        isinstance(card, SelectionCardState)
        for pile in (state.combat.hand, state.combat.draw_pile,
                     state.combat.discard_pile, state.combat.exhaust_pile) for card in pile
    ):
        raise SimulatorAdapterError("Selection cards require explicit card-selection support")
    raw_monsters = _array(_field(raw_state, "monsters", "state"), "state.monsters")
    raw_hand = _array(_field(raw_state, "hand", "state"), "state.hand")
    for index, item in enumerate(raw_hand):
        raw_card = _object(item, f"state.hand[{index}]")
        if _integer(
            _field(raw_card, "hand_index", f"state.hand[{index}]"),
            f"state.hand[{index}].hand_index",
        ) != index:
            raise SimulatorAdapterError("state hand indices are not contiguous")
    if "public_draw_memory" in raw_state:
        memory = _object(raw_state["public_draw_memory"], "public_draw_memory")
        if memory.get("schema_version") != "known_draw_top_v1":
            raise SimulatorAdapterError("Unknown public draw memory schema")
        state = replace(state, known_draw_top=_parse_pile(
            memory.get("next_draw_first"), "known_draw_top", uuid_prefix="known-top",
        ))
    if "hand_upgrade_previews" in raw_state:
        previews = _array(raw_state["hand_upgrade_previews"], "hand_upgrade_previews")
        if len(previews) != len(state.combat.hand):
            raise SimulatorAdapterError("Upgrade previews must align with the current hand")
        parsed_previews = []
        for index, preview in enumerate(previews):
            card = state.combat.hand[index]
            parsed = None if preview is None else _parse_card(
                preview, f"hand_upgrade_previews[{index}]", uuid=card.uuid,
            )
            if parsed is not None and (parsed.card_id != card.card_id or parsed.upgrades != card.upgrades + 1):
                raise SimulatorAdapterError("Upgrade preview has a mismatched card identity or level")
            parsed_previews.append(parsed)
        state = replace(state, hand_upgrade_previews=tuple(parsed_previews))
    if not terminal and input_state == "CARD_SELECT":
        selection = _object(raw_state.get("card_selection"), "state.card_selection")
        if selection.get("schema_version") != "combat_card_selection_v1" or selection.get("task") not in TASKS:
            raise SimulatorAdapterError("Unsupported card selection schema/task")
        task = selection["task"]
        raw_actions = _array(response.get("legal_actions"), "legal_actions")
        indices = []
        for item in raw_actions:
            item = _object(item, "selection action")
            index = _integer(item.get("selection_index"), "selection_index")
            if item.get("kind") != "SELECT_CARD" or item.get("action_id") != f"SELECT:{index}":
                raise SimulatorAdapterError("Invalid native selection action")
            indices.append(index)
        state = CardSelectionState(**vars(state), selection_task=task,
                                   candidate_indices=tuple(indices),
                                   source_card=(_parse_card(selection["source_card"], "selection.source_card", uuid="selection-source")
                                                if "source_card" in selection else None),
                                   copies_created=_integer(selection.get("copies_created", 0), "copies_created"))
        actions = build_selection_actions(state)
        native_ids = tuple(f"SELECT:{index}" for index in indices)
    else:
        if "card_selection" in raw_state:
            raise SimulatorAdapterError("Card selection metadata outside selection phase")
        actions, native_ids = _parse_actions(
            _field(response, "legal_actions", "response"), state,
            raw_monsters, raw_hand, terminal=terminal,
        )
    reward = _number(_field(raw_state, "reward", "state"), "state.reward")
    expected_reward = 1.0 if outcome == "PLAYER_VICTORY" else -1.0 if outcome == "PLAYER_LOSS" else 0.0
    if reward != expected_reward:
        raise SimulatorAdapterError("state reward/outcome fields disagree")

    return SimulatorCanonicalDecision(
        state=state,
        actions=actions,
        native_action_ids=native_ids,
        decision_id=decision_id,
        scenario_id=scenario_id,
        deck_preset=deck_preset,
        loadout_id=loadout_id,
        recipe_tier=recipe_tier,
        deck_hash=deck_hash,
        terminal=terminal,
        outcome=outcome,
        reward=reward,
    )
