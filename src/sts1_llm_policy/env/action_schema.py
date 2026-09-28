from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    PLAY_CARD = "play_card"
    END_TURN = "end_turn"
    SELECT_CARD = "select_card"


@dataclass(frozen=True)
class CanonicalAction:
    action_id: str
    action_type: ActionType

    # PLAY_CARD only.
    # Index in CanonicalState.combat.hand.
    hand_index: int | None = None

    # Internal unique identity of the card instance.
    card_uuid: str | None = None
    card_name: str | None = None

    # Index in CanonicalState.combat.monsters.
    # None for non-targeted cards and END_TURN.
    target_index: int | None = None
