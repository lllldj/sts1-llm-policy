import re
from dataclasses import replace
from typing import Any

from .live_id_crosswalk import (
    canonicalize_monster_id,
    canonicalize_move_id,
    canonicalize_move_intent,
    canonicalize_power_id,
)
from .live_description_fallback import (
    LIVE_POWER_EFFECT_TEMPLATES,
    SourceDescribedPowerState,
    SourceDescribedRelicState,
    live_power_ids,
)
from .monster_behavior import build_monster_behavior
from .state_schema import (
    CanonicalState,
    CardState,
    CombatState,
    MonsterState,
    PlayerState,
    PowerState,
    RelicState,
)


_RELIC_ID_SEPARATOR = re.compile(r"[^A-Za-z0-9]+")


def _canonical_relic_id(native_id: str) -> str:
    """Map CommunicationMod display IDs onto the simulator catalog IDs."""

    without_possessive_marks = native_id.replace("'", "")
    return _RELIC_ID_SEPARATOR.sub("_", without_possessive_marks).strip("_").upper()


def _canonical_power_id(native_id: str) -> str:
    """Map CommunicationMod power IDs onto the lightspeed-derived catalog."""

    return canonicalize_power_id(
        native_id,
        live_power_ids(),
    )


def _require_dict(value: Any, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be a dict, "
            f"got {type(value).__name__}"
        )
    return value


def _require_list(value: Any, field_name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(
            f"{field_name} must be a list, "
            f"got {type(value).__name__}"
        )
    return value


def _parse_powers(
    raw_powers: list[object],
    field_name: str,
) -> tuple[PowerState, ...]:
    """
    Parse CommunicationMod power objects without depending on localized
    display names.

    Required fields:
    - id
    - name
    - amount

    CommunicationMod may also expose damage, card, misc, and
    just_applied for powers that need extra state.
    """

    result: list[PowerState] = []

    for index, raw_power_value in enumerate(raw_powers):
        field_prefix = f"{field_name}[{index}]"
        raw_power = _require_dict(
            raw_power_value,
            field_prefix,
        )

        power_id = _require_str_field(
            raw_power,
            "id",
            field_prefix,
        )
        name = _require_str_field(
            raw_power,
            "name",
            field_prefix,
        )
        amount = _require_int_field(
            raw_power,
            "amount",
            field_prefix,
        )
        damage = _optional_int_field(
            raw_power,
            "damage",
            field_prefix,
            default=0,
        )
        misc = _optional_int_field(
            raw_power,
            "misc",
            field_prefix,
            default=0,
        )
        just_applied = _optional_bool_field(
            raw_power,
            "just_applied",
            field_prefix,
            default=False,
        )

        raw_card = raw_power.get("card")
        card = None

        if raw_card is not None:
            card = _parse_card(
                _require_dict(
                    raw_card,
                    f"{field_prefix}.card",
                )
            )

        canonical_power_id = _canonical_power_id(power_id)
        result.append(
            SourceDescribedPowerState(
                power_id=canonical_power_id,
                name=name,
                amount=amount,
                damage=damage,
                misc=misc,
                just_applied=just_applied,
                card=card,
                source_description=LIVE_POWER_EFFECT_TEMPLATES.get(
                    canonical_power_id,
                    "",
                ),
            )
        )

    return tuple(result)


def _require_str_field(
    value: dict,
    key: str,
    field_prefix: str,
) -> str:
    result = value.get(key)

    if not isinstance(result, str):
        raise ValueError(
            f"{field_prefix}.{key} must be a str, "
            f"got {type(result).__name__}"
        )

    return result


def _require_int_field(
    value: dict,
    key: str,
    field_prefix: str,
) -> int:
    result = value.get(key)

    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(
            f"{field_prefix}.{key} must be an int, "
            f"got {type(result).__name__}"
        )

    return result


def _optional_int_field(
    value: dict,
    key: str,
    field_prefix: str,
    *,
    default: int,
) -> int:
    if key not in value:
        return default

    return _require_int_field(
        value,
        key,
        field_prefix,
    )


def _optional_bool_field(
    value: dict,
    key: str,
    field_prefix: str,
    *,
    default: bool,
) -> bool:
    if key not in value:
        return default

    result = value[key]

    if not isinstance(result, bool):
        raise ValueError(
            f"{field_prefix}.{key} must be a bool, "
            f"got {type(result).__name__}"
        )

    return result


def _parse_card(raw_card: dict) -> CardState:
    return CardState(
        card_id=raw_card["id"],
        name=raw_card["name"],
        card_type=raw_card["type"],
        cost=raw_card["cost"],
        upgrades=raw_card["upgrades"],
        description=raw_card["description"],
        exhausts=raw_card["exhausts"],
        ethereal=raw_card["ethereal"],
        has_target=raw_card["has_target"],
        is_playable=raw_card["is_playable"],
        uuid=raw_card["uuid"],
    )


def _parse_player(raw_player: dict) -> PlayerState:
    raw_powers = _require_list(
        raw_player.get("powers", []),
        "combat_state.player.powers",
    )

    return PlayerState(
        current_hp=raw_player["current_hp"],
        max_hp=raw_player["max_hp"],
        block=raw_player["block"],
        energy=raw_player["energy"],
        powers=_parse_powers(
            raw_powers,
            "combat_state.player.powers",
        ),
    )


def _parse_monster(raw_monster: dict) -> MonsterState:
    raw_powers = _require_list(
        raw_monster.get("powers", []),
        "combat_state.monsters[].powers",
    )

    native_monster_id = raw_monster["id"]
    monster_id = canonicalize_monster_id(native_monster_id)
    native_move_id = raw_monster["move_id"]

    return MonsterState(
        monster_id=monster_id,
        name=raw_monster["name"],
        current_hp=raw_monster["current_hp"],
        max_hp=raw_monster["max_hp"],
        block=raw_monster["block"],
        intent=canonicalize_move_intent(
            monster_id,
            native_move_id,
            raw_monster["intent"],
        ),
        move_id=canonicalize_move_id(
            monster_id,
            native_move_id,
        ),
        move_hits=raw_monster["move_hits"],
        move_base_damage=raw_monster["move_base_damage"],
        move_adjusted_damage=raw_monster[
            "move_adjusted_damage"
        ],
        powers=_parse_powers(
            raw_powers,
            "combat_state.monsters[].powers",
        ),
        is_gone=raw_monster["is_gone"],
        half_dead=raw_monster["half_dead"],
    )


def _parse_card_pile(
    raw_cards: list[dict],
) -> tuple[CardState, ...]:
    return tuple(
        _parse_card(raw_card)
        for raw_card in raw_cards
    )


def _previous_move_id(raw_monster: dict, field_prefix: str) -> int | None:
    previous_move_id = _optional_int_field(
        raw_monster,
        "last_move_id",
        field_prefix,
        default=-1,
    )
    if previous_move_id < 0:
        return None
    return canonicalize_move_id(
        canonicalize_monster_id(raw_monster["id"]),
        previous_move_id,
    )


def _has_used_entangle(raw_monster: dict, field_prefix: str) -> bool:
    if raw_monster.get("id") != "SlaverRed":
        return False

    last_move_id = _previous_move_id(raw_monster, field_prefix)
    second_last_move_id = _optional_int_field(
        raw_monster,
        "second_last_move_id",
        field_prefix,
        default=-1,
    )
    return last_move_id == 2 or second_last_move_id == 2


def from_communication_state(
    raw_state: dict,
) -> CanonicalState:
    """
    Convert a stable CommunicationMod combat decision state into
    our project-owned CanonicalState representation.

    This adapter currently supports stable combat decision states
    only.
    """

    if not raw_state.get("in_game", False):
        raise ValueError(
            "Cannot canonicalize state: not currently in game"
        )

    if not raw_state.get("ready_for_command", False):
        raise ValueError(
            "Cannot canonicalize unstable state: "
            "ready_for_command is false"
        )

    game_state = _require_dict(
        raw_state.get("game_state"),
        "game_state",
    )

    if game_state.get("room_phase") != "COMBAT":
        raise ValueError(
            "Only COMBAT room states are currently supported"
        )

    if game_state.get("action_phase") != "WAITING_ON_USER":
        raise ValueError(
            "Only WAITING_ON_USER combat states are "
            "currently supported"
        )

    raw_combat = _require_dict(
        game_state.get("combat_state"),
        "game_state.combat_state",
    )

    raw_player = _require_dict(
        raw_combat.get("player"),
        "game_state.combat_state.player",
    )

    raw_monsters = _require_list(
        raw_combat.get("monsters"),
        "game_state.combat_state.monsters",
    )

    raw_hand = _require_list(
        raw_combat.get("hand"),
        "game_state.combat_state.hand",
    )

    raw_draw_pile = _require_list(
        raw_combat.get("draw_pile"),
        "game_state.combat_state.draw_pile",
    )

    raw_discard_pile = _require_list(
        raw_combat.get("discard_pile"),
        "game_state.combat_state.discard_pile",
    )

    raw_exhaust_pile = _require_list(
        raw_combat.get("exhaust_pile"),
        "game_state.combat_state.exhaust_pile",
    )

    raw_relics = _require_list(
        game_state.get("relics", []),
        "game_state.relics",
    )
    relics: list[RelicState] = []
    for index, raw_relic_value in enumerate(raw_relics):
        field_prefix = f"game_state.relics[{index}]"
        raw_relic = _require_dict(raw_relic_value, field_prefix)
        relic_id = _require_str_field(raw_relic, "id", field_prefix)
        name_value = raw_relic.get("name", relic_id)
        if not isinstance(name_value, str):
            raise ValueError(f"{field_prefix}.name must be a string")
        description_value = raw_relic.get("description", "")
        if not isinstance(description_value, str):
            raise ValueError(f"{field_prefix}.description must be a string")
        counter_value = raw_relic.get("counter")
        if counter_value is not None and (
            isinstance(counter_value, bool) or not isinstance(counter_value, int)
        ):
            raise ValueError(f"{field_prefix}.counter must be an integer or null")
        relics.append(
            SourceDescribedRelicState(
                relic_id=_canonical_relic_id(relic_id),
                name=name_value,
                counter=counter_value,
                source_description=description_value,
            )
        )

    combat_turn = _require_int_field(
        raw_combat,
        "turn",
        "game_state.combat_state",
    )
    ascension_level = _require_int_field(
        game_state,
        "ascension_level",
        "game_state",
    )
    parsed_monsters = tuple(
        _parse_monster(raw_monster)
        for raw_monster in raw_monsters
    )
    living_monster_count = sum(
        not monster.is_gone for monster in parsed_monsters
    )
    monsters = tuple(
        replace(
            monster,
            behavior=build_monster_behavior(
                monster,
                previous_move_id=_previous_move_id(
                    raw_monster,
                    f"combat_state.monsters[{index}]",
                ),
                living_monster_count=living_monster_count,
                combat_turn=combat_turn,
                ascension=ascension_level,
                has_used_entangle=_has_used_entangle(
                    raw_monster,
                    f"combat_state.monsters[{index}]",
                ),
            ),
        )
        for index, (monster, raw_monster) in enumerate(
            zip(parsed_monsters, raw_monsters, strict=True)
        )
    )

    combat = CombatState(
        turn=combat_turn,
        player=_parse_player(raw_player),
        monsters=monsters,
        hand=_parse_card_pile(raw_hand),
        draw_pile=_parse_card_pile(raw_draw_pile),
        discard_pile=_parse_card_pile(
            raw_discard_pile
        ),
        exhaust_pile=_parse_card_pile(
            raw_exhaust_pile
        ),
    )

    return CanonicalState(
        seed=game_state["seed"],
        character=game_state["class"],
        ascension_level=ascension_level,
        act=game_state["act"],
        floor=game_state["floor"],
        relics=tuple(relics),
        combat=combat,
    )
