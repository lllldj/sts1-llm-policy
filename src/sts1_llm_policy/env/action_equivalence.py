from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .action_schema import ActionType, CanonicalAction
from .live_description_fallback import resolve_card_effect
from .state_schema import CanonicalState, CardState


ACTION_EQUIVALENCE_VERSION = "action_equivalence_v1"

# Changing these fields changes the model action space and therefore requires
# a new action-equivalence version.
@dataclass(frozen=True)
class CardSemantics:
    """Complete P0 card semantics used for model-facing equivalence."""

    card_id: str
    name: str
    card_type: str
    cost: int
    upgrades: int
    effect: str
    exhausts: bool
    ethereal: bool
    has_target: bool


@dataclass(frozen=True)
class HandCardClass:
    """An unordered group of model-indistinguishable cards in hand."""

    card: CardSemantics
    copies: int


@dataclass(frozen=True)
class ModelActionClass:
    """One model action backed by one or more exact environment actions."""

    model_action_id: str
    action_type: ActionType
    card: CardSemantics | None
    target_index: int | None
    members: tuple[CanonicalAction, ...]

    @property
    def representative(self) -> CanonicalAction:
        return self.members[0]

    @property
    def copies(self) -> int:
        return len(self.members)

    @property
    def internal_action_ids(self) -> tuple[str, ...]:
        return tuple(member.action_id for member in self.members)


@dataclass(frozen=True)
class ModelActionSpace:
    """Decision-local equivalence classes presented to a policy."""

    classes: tuple[ModelActionClass, ...]
    version: str = ACTION_EQUIVALENCE_VERSION

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(item.model_action_id for item in self.classes)

    def resolve(self, action_id: str | None) -> ModelActionClass | None:
        if action_id is None:
            return None
        return next(
            (
                item
                for item in self.classes
                if item.model_action_id == action_id
            ),
            None,
        )

    def model_id_for_internal_action(
        self,
        internal_action_id: str,
    ) -> str | None:
        """Map an exact teacher/execution action back to its class label."""

        for item in self.classes:
            if internal_action_id in item.internal_action_ids:
                return item.model_action_id
        return None


def card_semantics(
    card: CardState,
    *,
    allow_source_descriptions: bool = False,
) -> CardSemantics:
    """Project a card instance onto the frozen public P0 semantics."""

    return CardSemantics(
        card_id=card.card_id,
        name=card.name,
        card_type=card.card_type,
        cost=card.cost,
        upgrades=card.upgrades,
        effect=resolve_card_effect(
            card,
            allow_source_description=allow_source_descriptions,
        ),
        exhausts=card.exhausts,
        ethereal=card.ethereal,
        has_target=card.has_target,
    )


def group_hand_cards(
    cards: Sequence[CardState],
    *,
    allow_source_descriptions: bool = False,
) -> tuple[HandCardClass, ...]:
    """Count hand cards by public semantics without exposing hand position."""

    counts: dict[CardSemantics, int] = {}
    for card in cards:
        semantics = card_semantics(
            card,
            allow_source_descriptions=allow_source_descriptions,
        )
        counts[semantics] = counts.get(semantics, 0) + 1

    return tuple(
        HandCardClass(card=semantics, copies=count)
        for semantics, count in sorted(
            counts.items(),
            key=lambda item: _card_sort_key(item[0]),
        )
    )


def build_model_action_space(
    state: CanonicalState,
    legal_actions: Sequence[CanonicalAction],
    *,
    allow_source_descriptions: bool = False,
) -> ModelActionSpace:
    """Collapse exact actions that have identical P0-visible consequences.

    Target index is part of the equivalence key. Exact card UUIDs and hand
    positions remain in the member actions for execution and audit only.
    """

    actions = tuple(legal_actions)
    if not actions:
        raise ValueError("legal_actions must not be empty")

    from .card_selection import CardSelectionState, selection_card
    if isinstance(state, CardSelectionState):
        # Keep exact choices distinct, including identical-looking instances.
        # Native pile position can matter to later effects and is not an
        # established equivalence relation in the historical PLAY contract.
        if len({a.action_id for a in actions}) != len(actions):
            raise ValueError("Duplicate selection action IDs")
        choices = [(a, card_semantics(selection_card(state, a),
                    allow_source_descriptions=allow_source_descriptions)) for a in actions]
        if {a.selection_index for a, _ in choices} != set(state.candidate_indices) or len(choices) != len(state.candidate_indices):
            raise ValueError("Selection actions do not cover exactly the candidate set")
        choices.sort(key=lambda item: _card_sort_key(item[1]))
        return ModelActionSpace(classes=tuple(
            ModelActionClass(f"ACTION_{i}", ActionType.SELECT_CARD, card, None, (action,))
            for i, (action, card) in enumerate(choices)
        ), version="combat_card_selection_v1")

    positions: dict[str, int] = {}
    grouped: dict[
        tuple[ActionType, CardSemantics | None, int | None],
        list[CanonicalAction],
    ] = {}
    end_turn_count = 0

    for position, action in enumerate(actions):
        if action.action_id in positions:
            raise ValueError(f"Duplicate legal action ID: {action.action_id}")
        positions[action.action_id] = position

        if action.action_type == ActionType.END_TURN:
            _validate_end_turn(action)
            key = (ActionType.END_TURN, None, None)
            end_turn_count += 1
        elif action.action_type == ActionType.PLAY_CARD:
            card = _card_for_action(state, action)
            semantics = card_semantics(
                card,
                allow_source_descriptions=allow_source_descriptions,
            )
            _validate_target(state, action, card)
            key = (ActionType.PLAY_CARD, semantics, action.target_index)
        else:
            raise ValueError(f"Unsupported action type: {action.action_type}")

        grouped.setdefault(key, []).append(action)

    if end_turn_count != 1:
        raise ValueError("legal_actions must contain exactly one END_TURN")

    ordered_keys = sorted(grouped, key=_action_class_sort_key)
    classes: list[ModelActionClass] = []

    for index, key in enumerate(ordered_keys):
        action_type, semantics, target_index = key
        members = tuple(
            sorted(
                grouped[key],
                key=lambda action: (
                    action.hand_index
                    if action.hand_index is not None
                    else len(state.combat.hand),
                    positions[action.action_id],
                ),
            )
        )
        classes.append(
            ModelActionClass(
                model_action_id=f"ACTION_{index}",
                action_type=action_type,
                card=semantics,
                target_index=target_index,
                members=members,
            )
        )

    return ModelActionSpace(classes=tuple(classes))


def model_action_space_records(
    action_space: ModelActionSpace,
) -> tuple[dict[str, object], ...]:
    """Return an explicit audit mapping without exposing it to the model."""

    records: list[dict[str, object]] = []
    for item in action_space.classes:
        records.append(
            {
                "model_action_id": item.model_action_id,
                "action_type": item.action_type.value,
                "card_name": item.card.name if item.card is not None else None,
                "target_index": item.target_index,
                "copies": item.copies,
                "equivalent_internal_action_ids": item.internal_action_ids,
                "representative_internal_action_id": (
                    item.representative.action_id
                ),
            }
        )
    return tuple(records)


def _card_for_action(
    state: CanonicalState,
    action: CanonicalAction,
) -> CardState:
    hand_index = action.hand_index
    if isinstance(hand_index, bool) or not isinstance(hand_index, int):
        raise ValueError("PLAY_CARD action requires an integer hand_index")
    if hand_index < 0 or hand_index >= len(state.combat.hand):
        raise ValueError(f"PLAY_CARD hand_index is out of range: {hand_index}")

    card = state.combat.hand[hand_index]
    if action.card_uuid != card.uuid:
        raise ValueError("PLAY_CARD card_uuid disagrees with current hand")
    if action.card_name != card.name:
        raise ValueError("PLAY_CARD card_name disagrees with current hand")
    if not card.is_playable:
        raise ValueError("PLAY_CARD references a currently unplayable card")
    return card


def _validate_target(
    state: CanonicalState,
    action: CanonicalAction,
    card: CardState,
) -> None:
    target_index = action.target_index
    if not card.has_target:
        if target_index is not None:
            raise ValueError("Non-targeted card action has a target_index")
        return

    if isinstance(target_index, bool) or not isinstance(target_index, int):
        raise ValueError("Targeted card action requires an integer target_index")
    if target_index < 0 or target_index >= len(state.combat.monsters):
        raise ValueError(f"Target index is out of range: {target_index}")

    monster = state.combat.monsters[target_index]
    if monster.is_gone or monster.half_dead or monster.current_hp <= 0:
        raise ValueError("Targeted card action references an invalid monster")


def _validate_end_turn(action: CanonicalAction) -> None:
    if any(
        value is not None
        for value in (
            action.hand_index,
            action.card_uuid,
            action.card_name,
            action.target_index,
        )
    ):
        raise ValueError("END_TURN action contains PLAY_CARD fields")


def _card_sort_key(card: CardSemantics) -> tuple[object, ...]:
    return (
        card.name.casefold(),
        card.card_id,
        card.upgrades,
        card.cost,
        card.card_type,
        card.effect,
        card.exhausts,
        card.ethereal,
        card.has_target,
    )


def _action_class_sort_key(
    key: tuple[ActionType, CardSemantics | None, int | None],
) -> tuple[object, ...]:
    action_type, card, target_index = key
    if action_type == ActionType.END_TURN:
        return (1,)
    if card is None:
        raise ValueError("PLAY_CARD equivalence key is missing card semantics")
    return (
        0,
        *_card_sort_key(card),
        -1 if target_index is None else target_index,
    )
