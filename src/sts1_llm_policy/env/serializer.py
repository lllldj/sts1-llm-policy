# src/sts1_llm_policy/env/serializer.py

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from .route_context import ROUTE_OBSERVATION_VERSION

from sts1_llm_policy.env.action_equivalence import (
    ModelActionClass,
    build_model_action_space,
    group_hand_cards,
)

from sts1_llm_policy.env.action_schema import (
    CanonicalAction,
    ActionType,
)
from sts1_llm_policy.env.mechanics_catalog import (
    SUPPORTED_KEYWORDS,
    describe_move,
    describe_move_by_id,
)
from sts1_llm_policy.env.live_description_fallback import (
    resolve_card_effect,
    resolve_power_effect,
    resolve_relic_effect,
)
from sts1_llm_policy.env.observation_v5_catalog import (
    MECHANIC_GLOSSARY as V5_MECHANIC_GLOSSARY,
    describe_move_by_id_v5,
    describe_move_v5,
    select_mechanics,
)
from sts1_llm_policy.env.state_schema import (
    CanonicalState,
    CardState,
    MonsterState,
    PowerState,
    RelicState,
)


OBSERVATION_SERIALIZER_VERSION = "observation_v2"
BEHAVIOR_OBSERVATION_SERIALIZER_VERSION = "observation_v3"
RELIC_OBSERVATION_SERIALIZER_VERSION = "observation_v4"
SEMANTIC_OBSERVATION_SERIALIZER_VERSION = "observation_v5"
SUPPORTED_OBSERVATION_SERIALIZER_VERSIONS = (
    ROUTE_OBSERVATION_VERSION,
    "observation_v6",
    "observation_v1",
    OBSERVATION_SERIALIZER_VERSION,
    BEHAVIOR_OBSERVATION_SERIALIZER_VERSION,
    RELIC_OBSERVATION_SERIALIZER_VERSION,
    SEMANTIC_OBSERVATION_SERIALIZER_VERSION,
)


def serialize_state(
    state: CanonicalState,
    *,
    version: str = OBSERVATION_SERIALIZER_VERSION,
    allow_source_descriptions: bool = False,
) -> str:
    """Serialize state with an explicitly versioned observation contract."""
    from .card_selection import CardSelectionState, SelectionCardState, TASKS
    extended = isinstance(state, CardSelectionState) or any(
        isinstance(card, SelectionCardState)
        for pile in (state.combat.hand, state.combat.draw_pile,
                     state.combat.discard_pile, state.combat.exhaust_pile)
        for card in pile
    )
    if state.route_context is not None and version != ROUTE_OBSERVATION_VERSION:
        raise ValueError("Route context requires observation_v7")
    if version == ROUTE_OBSERVATION_VERSION and state.route_context is None:
        raise ValueError("observation_v7 requires explicit route context")
    if extended and version not in {"observation_v6", ROUTE_OBSERVATION_VERSION}:
        raise ValueError("Combat card selection requires observation_v6 or observation_v7")
    if version in {"observation_v6", ROUTE_OBSERVATION_VERSION}:
        text = _serialize_state_v2(
            state, include_behavior=True, include_relics=True,
            include_semantic_closure=True,
            allow_source_descriptions=allow_source_descriptions,
        )
        if version == ROUTE_OBSERVATION_VERSION and state.known_draw_top is not None:
            known = " -> ".join(
                f"{c.name} | UPGRADE {c.upgrades} | COST {c.cost} | EFFECT {resolve_card_effect(c)}"
                for c in state.known_draw_top
            )
            text += "\n\nKNOWN_DRAW_TOP (next draw first): " + (known or "none")
            text += "\nOnly this observed prefix is ordered; the remaining draw pile is unknown."
        if isinstance(state, CardSelectionState):
            pile, effect = TASKS[state.selection_task]
            if state.selection_task == "DUAL_WIELD":
                effect += f" Create {state.copies_created} copies."
            if version == ROUTE_OBSERVATION_VERSION:
                if state.source_card is None:
                    raise ValueError("V7 card selection requires its public source card; rebuild the native bridge")
                source = state.source_card
                effect += f"\nSOURCE_CARD: {source.name} | UPGRADE {source.upgrades} | EFFECT {resolve_card_effect(source)}"
                if state.selection_task == "EXHAUST_ONE":
                    if source.card_id == "Burning Pact":
                        effect += f"\nPENDING: Exhaust the selected card, then attempt to draw {3 if source.upgrades else 2} cards (subject to No Draw and hand limit). Exhaust triggers also apply; future drawn identities are unknown."
                    elif source.card_id == "True Grit" and source.upgrades == 1:
                        effect += "\nPENDING: Exhaust the selected card. The source card's Block has already been applied; do not gain it again. Exhaust triggers also apply."
                    else:
                        raise ValueError("Unsupported source for V7 exhaust selection")
                elif state.selection_task == "WARCRY":
                    effect += "\nThe source card's draw has already resolved; do not draw again."
                elif state.selection_task == "HEADBUTT":
                    effect += "\nThe source card's attack has already resolved; do not deal its damage again."
                elif state.selection_task == "ARMAMENTS":
                    effect += "\nThe source card's Block has already been applied. Each action below shows the resulting upgraded card for this combat only."
                elif state.selection_task == "DUAL_WIELD":
                    effect += "\nCopies preserve the selected card's current upgrade and instance state; excess copies go to discard if the hand is full. They last only this combat."
            return text + f"\n\nCARD_SELECTION: {state.selection_task} | source={pile} | choose=1\n{effect}\nOnly the listed selection actions are legal; playing cards and ending the turn are unavailable."
        return text
    if version == "observation_v1":
        return _serialize_state_v1(state)
    if version == OBSERVATION_SERIALIZER_VERSION:
        return _serialize_state_v2(
            state,
            allow_source_descriptions=allow_source_descriptions,
        )
    if version == BEHAVIOR_OBSERVATION_SERIALIZER_VERSION:
        return _serialize_state_v2(
            state,
            include_behavior=True,
            allow_source_descriptions=allow_source_descriptions,
        )
    if version == RELIC_OBSERVATION_SERIALIZER_VERSION:
        return _serialize_state_v2(
            state,
            include_behavior=True,
            include_relics=True,
            allow_source_descriptions=allow_source_descriptions,
        )
    if version == SEMANTIC_OBSERVATION_SERIALIZER_VERSION:
        return _serialize_state_v2(
            state,
            include_behavior=True,
            include_relics=True,
            include_semantic_closure=True,
            allow_source_descriptions=allow_source_descriptions,
        )
    raise ValueError(f"Unsupported observation serializer version: {version}")


def _serialize_state_v2(
    state: CanonicalState,
    *,
    include_behavior: bool = False,
    include_relics: bool = False,
    include_semantic_closure: bool = False,
    allow_source_descriptions: bool = False,
) -> str:
    """
    Serialize a CanonicalState into a deterministic, model-facing text format.

    Design goals:
    - deterministic
    - compact and human-readable
    - independent of CommunicationMod raw JSON
    - do not expose internal identifiers such as card UUID
    - keep state serialization separate from legal-action serialization
    """
    combat = state.combat

    lines: list[str] = []

    if include_behavior:
        if state.route_context is not None:
            lines.extend((
                "ROUTE_OBJECTIVE:",
                "GOAL: Maximize the probability of completing the remaining route and defeating its final Boss.",
                "TRADEOFF: Balance the risk of dying in this combat against HP and persistent resources carried into later combats; a small increase in current combat risk can be worthwhile if it improves overall route survival.",
                "HORIZON: Evaluate actions over the remaining route, including known healing and deck improvements. Do not optimize immediate damage or current-combat win probability in isolation.",
                "RESOURCES: Current/max HP, the permanent deck and persistent relic counters carry between combats. Block, energy, temporary powers and combat-only card changes do not carry.",
                "TERMINAL: Death ends the route. Final Boss victory completes the goal; leftover HP has no further survival value.",
                "", state.route_context.serialize(),
            ))
        else:
            lines.extend(
                (
                    "COMBAT_OBJECTIVE:",
                    "PRIMARY: Maximize the probability of winning this combat.",
                    (
                        "SECONDARY: Among lines with comparable win probability, "
                        "maximize HP remaining at combat end."
                    ),
                    (
                        "HORIZON: Judge actions by their expected result over the "
                        "rest of the combat, not only immediate HP change."
                    ),
                )
            )

        if combat.accounting is not None:
            accounting = combat.accounting
            lines.extend(
                (
                    "",
                    "COMBAT_ACCOUNTING:",
                    f"STARTING_HP: {accounting.starting_hp}",
                    f"CURRENT_HP: {combat.player.current_hp}",
                    f"TOTAL_HP_LOSS: {accounting.total_hp_loss}",
                    f"ENEMY_DAMAGE_TAKEN: {accounting.enemy_damage_taken}",
                    f"SELF_HP_LOSS: {accounting.self_hp_loss}",
                )
            )

        if include_semantic_closure:
            keywords = _relevant_v5_mechanics(
                state,
                allow_source_descriptions=allow_source_descriptions,
            )
            keyword_catalog = V5_MECHANIC_GLOSSARY
        else:
            keywords = _relevant_keywords(state)
            keyword_catalog = SUPPORTED_KEYWORDS
        if keywords:
            lines.extend(("", "KEYWORDS:"))
            lines.extend(
                f"{keyword}: {keyword_catalog[keyword]}"
                for keyword in keywords
            )

    # ------------------------------------------------------------------
    # Run metadata
    # ------------------------------------------------------------------
    lines.append(f"CHARACTER: {state.character}")
    lines.append(f"ASCENSION: {state.ascension_level}")
    lines.append(f"ACT: {state.act}")
    lines.append(f"FLOOR: {state.floor}")
    lines.append(f"TURN: {combat.turn}")

    # ------------------------------------------------------------------
    # Player
    # ------------------------------------------------------------------
    player = combat.player

    lines.append("")
    lines.append("PLAYER:")
    lines.append(
        f"HP: {player.current_hp}/{player.max_hp}"
    )
    lines.append(f"BLOCK: {player.block}")
    lines.append(f"ENERGY: {player.energy}")

    if player.powers:
        lines.append(
            "POWERS: "
            + ", ".join(
                _serialize_power(
                    power,
                    allow_source_descriptions=allow_source_descriptions,
                )
                for power in player.powers
            )
        )
    else:
        lines.append("POWERS: NONE")

    if include_relics:
        lines.extend(("", "RELICS:"))
        if state.relics:
            lines.extend(
                _serialize_relic(
                    relic,
                    allow_source_descriptions=allow_source_descriptions,
                )
                for relic in state.relics
            )
        else:
            lines.append("NONE")

    # ------------------------------------------------------------------
    # Enemies
    # ------------------------------------------------------------------
    lines.append("")
    lines.append("ENEMIES:")

    if combat.monsters:
        for index, monster in enumerate(combat.monsters):
            monster_line = (
                f"[{index}] {monster.name}"
                f" | HP {monster.current_hp}/{monster.max_hp}"
                f" | BLOCK {monster.block}"
                f" | {_serialize_intent(monster)}"
                f" | EFFECT "
                f"{describe_move_v5(monster) if include_semantic_closure else describe_move(monster)}"
            )

            if monster.powers:
                monster_line += (
                    " | POWERS "
                    + ", ".join(
                        _serialize_power(
                            power,
                            allow_source_descriptions=allow_source_descriptions,
                        )
                        for power in monster.powers
                    )
                )
            if monster.stasis_card is not None:
                card = monster.stasis_card
                monster_line += (
                    f" | STASIS {card.name}"
                    f"{'+' if card.upgrades else ''}"
                    f" [{card.card_type}; {card.description}]"
                )

            lines.append(monster_line)
            if include_behavior:
                lines.append(
                    f"    {_serialize_behavior(monster, semantic_v5=include_semantic_closure)}"
                )
    else:
        lines.append("NONE")

    # ------------------------------------------------------------------
    # Hand
    # ------------------------------------------------------------------
    lines.append("")
    lines.append("HAND:")

    if combat.hand:
        for card_class in group_hand_cards(
            combat.hand,
            allow_source_descriptions=allow_source_descriptions,
        ):
            card = card_class.card
            lines.append(
                f"{card.name} x{card_class.copies}"
                f" | TYPE {card.card_type}"
                f" | COST {card.cost}"
                f" | UPGRADE {card.upgrades}"
                f" | TARGET {'ENEMY' if card.has_target else 'NONE'}"
                f" | EFFECT {card.effect}"
            )
    else:
        lines.append("EMPTY")

    # ------------------------------------------------------------------
    # Piles
    # ------------------------------------------------------------------
    lines.append("")
    lines.append("PILES:")
    lines.append(
        f"DRAW: {_serialize_pile(combat.draw_pile, allow_source_descriptions=allow_source_descriptions)}"
    )
    lines.append(
        f"DISCARD: {_serialize_pile(combat.discard_pile, allow_source_descriptions=allow_source_descriptions)}"
    )
    lines.append(
        f"EXHAUST: {_serialize_pile(combat.exhaust_pile, allow_source_descriptions=allow_source_descriptions)}"
    )

    return "\n".join(lines)


def serialize_actions(
    actions: Sequence[CanonicalAction],
) -> str:
    """
    Serialize the legal action set in its existing deterministic order.

    The serializer does NOT rebuild or reorder legal actions.
    """
    lines = ["LEGAL_ACTIONS:"]

    for action in actions:
        lines.append(_serialize_action(action))

    return "\n".join(lines)


def serialize_model_actions(
    state: CanonicalState,
    actions: Sequence[CanonicalAction],
    *,
    allow_source_descriptions: bool = False,
    include_upgrade_results: bool = False,
) -> str:
    """Serialize decision-local action equivalence classes for the model."""

    action_space = build_model_action_space(
        state,
        actions,
        allow_source_descriptions=allow_source_descriptions,
    )
    lines = ["LEGAL_ACTIONS:"]
    from .card_selection import CardSelectionState
    for item in action_space.classes:
        line = _serialize_model_action(item)
        selection_upgrade = isinstance(state, CardSelectionState) and state.selection_task == "ARMAMENTS"
        play_upgrade = item.action_type == ActionType.PLAY_CARD and item.card.card_id == "Armaments"
        if include_upgrade_results and (selection_upgrade or play_upgrade):
            previews = state.hand_upgrade_previews
            if len(previews) != len(state.combat.hand):
                raise ValueError("V7 Armaments requires native upgrade previews; rebuild the native bridge")
            if selection_upgrade:
                index = item.representative.selection_index
                if previews[index] is None:
                    raise ValueError("Armaments candidate is missing its upgrade preview")
                line += " | AFTER_UPGRADE: " + _upgrade_card_text(previews[index])
            else:
                mode = "ALL" if item.card.upgrades else "CHOOSE_ONE (automatic if only one eligible card)"
                line += f"\n    UPGRADE_MODE: {mode}; remaining hand only; combat-only changes."
                eligible = [(card, preview) for i, (card, preview) in enumerate(zip(state.combat.hand, previews))
                            if i != item.representative.hand_index and preview is not None]
                for card, preview in sorted(eligible, key=lambda pair: (pair[0].name, pair[0].cost, pair[0].upgrades)):
                    line += f"\n    UPGRADE_RESULT: {card.name} | COST {card.cost} | UPGRADE {card.upgrades} -> {_upgrade_card_text(preview)}"
                if not eligible:
                    line += "\n    UPGRADE_RESULT: NONE (no eligible remaining card)."
        lines.append(line)
    return "\n".join(lines)


def _upgrade_card_text(card: CardState) -> str:
    return (f"{card.name} | TYPE {card.card_type} | COST {card.cost} | UPGRADE {card.upgrades}"
            f" | TARGET {'ENEMY' if card.has_target else 'NONE'}"
            f" | EXHAUST {str(card.exhausts).upper()} | ETHEREAL {str(card.ethereal).upper()}"
            f" | EFFECT {resolve_card_effect(card)}")


def _relevant_keywords(state: CanonicalState) -> tuple[str, ...]:
    """Return only glossary entries reachable from the current combat state."""

    combat = state.combat
    cards = (
        combat.hand
        + combat.draw_pile
        + combat.discard_pile
        + combat.exhaust_pile
    )
    keywords: set[str] = set()

    for card in cards:
        if card.card_id in {"Bash", "Dropkick", "Thunderclap", "Uppercut", "Shockwave"}:
            keywords.add("VULNERABLE")
        if card.card_id in {"Clothesline", "Intimidate", "Uppercut", "Shockwave"}:
            keywords.add("WEAK")
        if card.card_id == "Battle Trance":
            keywords.add("NO_DRAW")
        if card.card_id == "Rage":
            keywords.add("RAGE")
        if card.card_id in {"Havoc", "Sentinel", "Sever Soul"}:
            keywords.add("EXHAUST")
        if card.card_id == "Reckless Charge":
            keywords.update(("ETHEREAL", "EXHAUST", "UNPLAYABLE"))
        if card.card_id in {"AscendersBane", "Dazed", "Wound"}:
            keywords.add("UNPLAYABLE")
        if card.ethereal:
            keywords.update(("ETHEREAL", "EXHAUST"))
        elif card.exhausts:
            keywords.add("EXHAUST")

    powers = combat.player.powers + tuple(
        power
        for monster in combat.monsters
        for power in monster.powers
    )
    for power in powers:
        if power.power_id == "Vulnerable":
            keywords.add("VULNERABLE")
        elif power.power_id == "Weak":
            keywords.add("WEAK")
        elif power.power_id == "No Draw":
            keywords.add("NO_DRAW")
        elif power.power_id == "Rage":
            keywords.add("RAGE")

    return tuple(sorted(keywords))


def _relevant_v5_mechanics(
    state: CanonicalState,
    *,
    allow_source_descriptions: bool = False,
) -> tuple[str, ...]:
    """Close the glossary over every rules text visible in this observation."""

    combat = state.combat
    texts = [
        resolve_card_effect(
            card,
            allow_source_description=allow_source_descriptions,
        )
        for card in (
            combat.hand
            + combat.draw_pile
            + combat.discard_pile
            + combat.exhaust_pile
        )
    ]
    texts.extend(
        resolve_relic_effect(
            relic,
            allow_source_description=allow_source_descriptions,
        )
        for relic in state.relics
    )
    if state.route_context is not None:
        texts.extend(resolve_card_effect(card) for card in state.hand_upgrade_previews if card is not None)
        source = getattr(state, "source_card", None)
        if source is not None:
            texts.append(resolve_card_effect(source))
    texts.extend(
        resolve_power_effect(
            power,
            allow_source_description=allow_source_descriptions,
        )
        for power in combat.player.powers
    )

    for monster in combat.monsters:
        texts.append(describe_move_v5(monster))
        texts.extend(
            resolve_power_effect(
                power,
                allow_source_description=allow_source_descriptions,
            )
            for power in monster.powers
        )
        if monster.behavior is not None:
            texts.append(monster.behavior.rule)
            texts.extend(
                describe_move_by_id_v5(monster.monster_id, move_id)[2]
                for move_id in monster.behavior.possible_next_move_ids
            )

    return select_mechanics(texts)


def serialize_observation(
    state: CanonicalState,
    actions: Sequence[CanonicalAction],
    *,
    version: str = OBSERVATION_SERIALIZER_VERSION,
    allow_source_descriptions: bool = False,
) -> str:
    """
    Serialize the complete policy observation:
        canonical state + current legal actions.
    """
    serialized_state = serialize_state(
        state,
        version=version,
        allow_source_descriptions=allow_source_descriptions,
    )
    if version == "observation_v1":
        serialized_actions = serialize_actions(actions)
    else:
        serialized_actions = serialize_model_actions(
            state,
            actions,
            allow_source_descriptions=allow_source_descriptions,
            include_upgrade_results=version == ROUTE_OBSERVATION_VERSION,
        )
    return f"{serialized_state}\n\n{serialized_actions}"


def _serialize_action(action: CanonicalAction) -> str:
    """
    Convert one CanonicalAction to model-facing text.

    action_id is the only identifier that the model will eventually
    need to output.
    """
    prefix = f"{action.action_id}:"

    if action.action_type == ActionType.END_TURN:
        return f"{prefix} END_TURN"

    if action.action_type == ActionType.PLAY_CARD:
        card_name = action.card_name or "UNKNOWN_CARD"

        if action.target_index is None:
            return f"{prefix} PLAY {card_name}"

        return (
            f"{prefix} PLAY {card_name}"
            f" -> TARGET_{action.target_index}"
        )

    raise ValueError(
        f"Unsupported action type: {action.action_type}"
    )


def _serialize_model_action(action: ModelActionClass) -> str:
    prefix = f"{action.model_action_id}:"
    if action.action_type == ActionType.END_TURN:
        return f"{prefix} END_TURN"
    if action.action_type == ActionType.SELECT_CARD and action.card is not None:
        return (f"{prefix} SELECT {action.card.name} | COST {action.card.cost}"
                f" | UPGRADE {action.card.upgrades} | EFFECT {action.card.effect}")
    if action.action_type != ActionType.PLAY_CARD or action.card is None:
        raise ValueError(f"Unsupported model action type: {action.action_type}")

    result = (
        f"{prefix} PLAY {action.card.name}"
        f" | COST {action.card.cost}"
        f" | UPGRADE {action.card.upgrades}"
    )
    if action.target_index is not None:
        result += f" -> TARGET_{action.target_index}"
    return f"{result} | COPIES {action.copies}"


def _serialize_power(
    power: PowerState,
    *,
    allow_source_descriptions: bool = False,
) -> str:
    """
    Serialize a power while omitting neutral optional values.

    The associated card's UUID remains internal and is never exposed.
    """
    result = f"{power.name}({power.amount})"
    details: list[str] = []

    if power.damage != 0:
        details.append(f"DAMAGE={power.damage}")

    if power.misc != 0:
        details.append(f"MISC={power.misc}")

    if power.just_applied:
        details.append("JUST_APPLIED")

    if power.card is not None:
        details.append(f"CARD={power.card.name}")

    if details:
        result += "[" + ", ".join(details) + "]"

    return (
        f"{result}: "
        + resolve_power_effect(
            power,
            allow_source_description=allow_source_descriptions,
        )
    )


def _serialize_relic(
    relic: RelicState,
    *,
    allow_source_descriptions: bool = False,
) -> str:
    counter = "NONE" if relic.counter is None else str(relic.counter)
    return (
        f"{relic.name} | COUNTER {counter}"
        " | EFFECT "
        + resolve_relic_effect(
            relic,
            allow_source_description=allow_source_descriptions,
        )
    )


def _serialize_intent(monster: MonsterState) -> str:
    """Serialize player-visible intent with current adjusted attack damage."""
    result = f"INTENT {monster.intent}"

    if "ATTACK" not in monster.intent:
        return result

    if monster.move_adjusted_damage < 0:
        raise ValueError(
            f"Attack intent has negative adjusted damage: {monster.name}"
        )
    if monster.move_hits <= 0:
        raise ValueError(
            f"Attack intent has non-positive hit count: {monster.name}"
        )

    total_damage = monster.move_adjusted_damage * monster.move_hits
    return (
        f"{result} | DAMAGE {monster.move_adjusted_damage} x "
        f"{monster.move_hits} | TOTAL {total_damage}"
    )


def _serialize_behavior(
    monster: MonsterState,
    *,
    semantic_v5: bool = False,
) -> str:
    """Serialize rules knowledge without exposing a realized future RNG draw."""

    behavior = monster.behavior
    if behavior is None:
        raise ValueError(
            f"observation_v3 requires behavior state for {monster.name}"
        )

    if behavior.previous_move_id is None:
        previous = "NONE"
    else:
        previous, _intent, _effect = describe_move_by_id(
            monster.monster_id,
            behavior.previous_move_id,
        )

    following: list[str] = []
    for move_id in behavior.possible_next_move_ids:
        describe = describe_move_by_id_v5 if semantic_v5 else describe_move_by_id
        name, intent, effect = describe(monster.monster_id, move_id)
        following.append(f"{name} [{intent}; {effect}]")

    return (
        f"BEHAVIOR: PHASE {behavior.phase}"
        f" | PREVIOUS {previous}"
        f" | FOLLOWING_AFTER_CURRENT "
        f"{'; '.join(following) if following else 'NONE'}"
        f" | SELECTION {behavior.selection.upper()}"
        f" | RULE {behavior.rule}"
    )


def _serialize_pile(
    cards: Sequence[CardState],
    *,
    allow_source_descriptions: bool = False,
) -> str:
    """Serialize visible composition while deliberately hiding pile order."""
    counts: Counter[tuple[str, str, int]] = Counter()

    for card in cards:
        # Validate that every exposed pile card remains inside the frozen P0
        # mechanics boundary, even though effects are shown only for the hand.
        resolve_card_effect(
            card,
            allow_source_description=allow_source_descriptions,
        )
        counts[(card.name, card.card_id, card.upgrades)] += 1

    if not counts:
        return "0 | EMPTY"

    entries: list[str] = []
    for (name, _card_id, upgrades), count in sorted(
        counts.items(),
        key=lambda item: (
            item[0][0].casefold(),
            item[0][1],
            item[0][2],
        ),
    ):
        upgrade_suffix = f"+{upgrades}" if upgrades else ""
        entries.append(f"{name}{upgrade_suffix} x{count}")

    return f"{len(cards)} | " + ", ".join(entries)


def _serialize_state_v1(state: CanonicalState) -> str:
    """Preserve the exact compact format used by the frozen v1 audit."""
    combat = state.combat
    player = combat.player
    lines = [
        f"CHARACTER: {state.character}",
        f"ASCENSION: {state.ascension_level}",
        f"ACT: {state.act}",
        f"FLOOR: {state.floor}",
        f"TURN: {combat.turn}",
        "",
        "PLAYER:",
        f"HP: {player.current_hp}/{player.max_hp}",
        f"BLOCK: {player.block}",
        f"ENERGY: {player.energy}",
    ]
    if player.powers:
        lines.append(
            "POWERS: "
            + ", ".join(_serialize_power_v1(power) for power in player.powers)
        )
    else:
        lines.append("POWERS: NONE")

    lines.extend(("", "ENEMIES:"))
    if combat.monsters:
        for index, monster in enumerate(combat.monsters):
            monster_line = (
                f"[{index}] {monster.name}"
                f" | HP {monster.current_hp}/{monster.max_hp}"
                f" | BLOCK {monster.block}"
                f" | INTENT {monster.intent}"
            )
            if monster.powers:
                monster_line += (
                    " | POWERS "
                    + ", ".join(
                        _serialize_power_v1(power) for power in monster.powers
                    )
                )
            if monster.stasis_card is not None:
                card = monster.stasis_card
                monster_line += f" | STASIS {card.name}{'+' if card.upgrades else ''}"
            lines.append(monster_line)
    else:
        lines.append("NONE")

    lines.extend(("", "HAND:"))
    if combat.hand:
        for index, card in enumerate(combat.hand):
            lines.append(f"[{index}] {card.name} | COST {card.cost}")
    else:
        lines.append("EMPTY")

    lines.extend(
        (
            "",
            "PILES:",
            f"DRAW: {len(combat.draw_pile)}",
            f"DISCARD: {len(combat.discard_pile)}",
            f"EXHAUST: {len(combat.exhaust_pile)}",
        )
    )
    return "\n".join(lines)


def _serialize_power_v1(power: PowerState) -> str:
    result = f"{power.name}({power.amount})"
    details: list[str] = []
    if power.damage != 0:
        details.append(f"DAMAGE={power.damage}")
    if power.misc != 0:
        details.append(f"MISC={power.misc}")
    if power.just_applied:
        details.append("JUST_APPLIED")
    if power.card is not None:
        details.append(f"CARD={power.card.name}")
    if details:
        result += "[" + ", ".join(details) + "]"
    return result
