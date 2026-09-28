from .action_schema import (
    ActionType,
    CanonicalAction,
)
from .state_schema import CanonicalState


def to_communication_command(
    action: CanonicalAction,
    state: CanonicalState,
) -> str:
    """
    Convert a CanonicalAction into a CommunicationMod command.

    Canonical conventions:
    - hand_index: 0-based
    - target_index: 0-based

    CommunicationMod conventions:
    - PLAY CardIndex: 1-based
    - TargetIndex: 0-based
    """

    if action.action_type == ActionType.END_TURN:
        return "end"

    if action.action_type != ActionType.PLAY_CARD:
        raise ValueError(
            f"Unsupported action type: {action.action_type}"
        )

    if action.hand_index is None:
        raise ValueError(
            "PLAY_CARD action requires hand_index"
        )

    if action.card_uuid is None:
        raise ValueError(
            "PLAY_CARD action requires card_uuid"
        )

    hand = state.combat.hand

    if not 0 <= action.hand_index < len(hand):
        raise ValueError(
            f"hand_index out of range: {action.hand_index}"
        )

    card = hand[action.hand_index]

    # Detect stale or mismatched actions.
    if card.uuid != action.card_uuid:
        raise ValueError(
            "Action no longer matches the card at "
            f"hand[{action.hand_index}]"
        )

    if not card.is_playable:
        raise ValueError(
            f"Card is not currently playable: {card.name}"
        )

    # CommunicationMod PLAY uses a 1-based card index.
    card_index = action.hand_index + 1

    if card.has_target:
        if action.target_index is None:
            raise ValueError(
                f"Targeted card requires target_index: {card.name}"
            )

        monsters = state.combat.monsters

        if not 0 <= action.target_index < len(monsters):
            raise ValueError(
                f"target_index out of range: "
                f"{action.target_index}"
            )

        target = monsters[action.target_index]

        if (
            target.is_gone
            or target.half_dead
            or target.current_hp <= 0
        ):
            raise ValueError(
                f"Invalid target: {target.name}"
            )

        return (
            f"play {card_index} "
            f"{action.target_index}"
        )

    if action.target_index is not None:
        raise ValueError(
            f"Non-targeted card must not have target_index: "
            f"{card.name}"
        )

    return f"play {card_index}"