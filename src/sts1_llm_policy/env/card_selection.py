"""Explicit combat card-choice extension; legacy state schemas stay unchanged."""
from __future__ import annotations

from dataclasses import dataclass

from .action_schema import ActionType, CanonicalAction
from .state_schema import CanonicalState, CardState

CARD_SELECTION_VERSION = "combat_card_selection_v1"
SELECTION_OBSERVATION_VERSION = "observation_v6"
CARD_EFFECTS = {
    ("Burning Pact", 0): "Choose a card in your hand to Exhaust. Draw 2 cards.",
    ("Burning Pact", 1): "Choose a card in your hand to Exhaust. Draw 3 cards.",
    ("True Grit", 0): "Gain 7 Block. Exhaust a random card in your hand.",
    ("True Grit", 1): "Gain 9 Block. Choose a card in your hand to Exhaust.",
    ("Warcry", 0): "Draw 1 card. Choose a card in your hand to put on top of your draw pile. Exhaust.",
    ("Warcry", 1): "Draw 2 cards. Choose a card in your hand to put on top of your draw pile. Exhaust.",
    ("Dual Wield", 0): "Choose an Attack or Power in your hand. Add 1 copy of it to your hand.",
    ("Dual Wield", 1): "Choose an Attack or Power in your hand. Add 2 copies of it to your hand.",
    ("Armaments", 0): "Gain 5 Block. Choose a card in your hand to Upgrade for the rest of combat.",
    ("Armaments", 1): "Gain 5 Block. Upgrade all cards in your hand for the rest of combat.",
    ("Headbutt", 0): "Deal 9 damage. Choose a card in your discard pile to put on top of your draw pile.",
    ("Headbutt", 1): "Deal 12 damage. Choose a card in your discard pile to put on top of your draw pile.",
    ("Exhume", 0): "Choose a card other than Exhume in your exhaust pile and put it into your hand. Exhaust.",
    ("Exhume", 1): "Choose a card other than Exhume in your exhaust pile and put it into your hand. Exhaust.",
}
TASKS = {
    "EXHAUST_ONE": ("hand", "Exhaust one card; then continue resolving the played card."),
    "WARCRY": ("hand", "Put one card from hand on top of the draw pile."),
    "DUAL_WIELD": ("hand", "Copy one eligible Attack or Power."),
    "ARMAMENTS": ("hand", "Upgrade one eligible card for the rest of combat."),
    "HEADBUTT": ("discard_pile", "Put one card on top of the draw pile."),
    "EXHUME": ("exhaust_pile", "Return one eligible card to hand; Exhume cannot select itself."),
}


def selection_effect(card_id, upgrades):
    if card_id == "Searing Blow" and upgrades > 1:
        damage = 12 + upgrades * (upgrades + 7) // 2
        return f"Deal {damage} base damage. Can be upgraded any number of times."
    return CARD_EFFECTS.get((card_id, upgrades))


@dataclass(frozen=True)
class SelectionCardState(CardState):
    """A card whose semantics belong to the explicit selection extension."""


@dataclass(frozen=True)
class CardSelectionState(CanonicalState):
    selection_task: str = ""
    copies_created: int = 0
    # Exact source-pile indices; never serialized into the student prompt.
    candidate_indices: tuple[int, ...] = ()
    source_card: CardState | None = None


@dataclass(frozen=True)
class SelectCardAction(CanonicalAction):
    selection_index: int = -1


def selection_card(state: CardSelectionState, action: CanonicalAction) -> CardState:
    if not isinstance(action, SelectCardAction) or action.action_type != ActionType.SELECT_CARD:
        raise ValueError("Card selection permits SELECT_CARD actions only")
    if state.selection_task not in TASKS:
        raise ValueError("Unsupported combat card-selection task")
    index = action.selection_index
    if isinstance(index, bool) or not isinstance(index, int) or index not in state.candidate_indices:
        raise ValueError("Selection is outside the current candidate set")
    pile = getattr(state.combat, TASKS[state.selection_task][0])
    if not 0 <= index < len(pile):
        raise ValueError("Selection index is outside its source pile")
    card = pile[index]
    if action.card_uuid != card.uuid or action.card_name != card.name:
        raise ValueError("Selection card identity is stale")
    if action.hand_index is not None or action.target_index is not None:
        raise ValueError("Selection must not contain play-card fields")
    return card


def build_selection_actions(state: CardSelectionState) -> tuple[CanonicalAction, ...]:
    if state.selection_task not in TASKS or not state.candidate_indices:
        raise ValueError("Unsupported or empty card selection")
    if len(set(state.candidate_indices)) != len(state.candidate_indices):
        raise ValueError("Duplicate selection candidates")
    allowed_copies = (1, 2) if state.selection_task == "DUAL_WIELD" else (0,)
    if type(state.copies_created) is not int or state.copies_created not in allowed_copies:
        raise ValueError("Invalid selection copy count")
    pile = getattr(state.combat, TASKS[state.selection_task][0])
    actions = []
    for position, index in enumerate(state.candidate_indices):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(pile):
            raise ValueError("Invalid candidate index")
        card = pile[index]
        action = SelectCardAction(
            action_id=f"ACTION_{position}", action_type=ActionType.SELECT_CARD,
            card_uuid=card.uuid, card_name=card.name, selection_index=index,
        )
        selection_card(state, action)
        actions.append(action)
    return tuple(actions)
