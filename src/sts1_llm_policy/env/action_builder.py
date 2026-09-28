from .action_schema import (
    ActionType,
    CanonicalAction,
)
from .state_schema import CanonicalState


def _valid_target_indices(
    state: CanonicalState,
) -> tuple[int, ...]:
    """
    Return indices of monsters that are valid combat targets.

    v0 rule:
    - not gone
    - not half dead
    - HP > 0
    """

    result: list[int] = []

    for index, monster in enumerate(
        state.combat.monsters
    ):
        if monster.is_gone:
            continue

        if monster.half_dead:
            continue

        if monster.current_hp <= 0:
            continue

        result.append(index)

    return tuple(result)


def build_legal_actions(
    state: CanonicalState,
) -> tuple[CanonicalAction, ...]:
    """
    Build deterministic legal combat actions from CanonicalState.

    Ordering:
    1. Cards in hand order.
    2. For targeted cards, monsters in monster order.
    3. END_TURN last.

    ACTION_n is assigned sequentially in that order.
    """

    actions: list[CanonicalAction] = []

    target_indices = _valid_target_indices(state)

    for hand_index, card in enumerate(
        state.combat.hand
    ):
        # Important:
        # Only cards in hand are considered here.
        #
        # CommunicationMod may report is_playable=True for cards
        # outside the hand, so we must never scan draw/discard piles
        # when constructing legal PLAY_CARD actions.
        if not card.is_playable:
            continue

        if card.has_target:
            for target_index in target_indices:
                action_id = f"ACTION_{len(actions)}"

                actions.append(
                    CanonicalAction(
                        action_id=action_id,
                        action_type=ActionType.PLAY_CARD,
                        hand_index=hand_index,
                        card_uuid=card.uuid,
                        card_name=card.name,
                        target_index=target_index,
                    )
                )

        else:
            action_id = f"ACTION_{len(actions)}"

            actions.append(
                CanonicalAction(
                    action_id=action_id,
                    action_type=ActionType.PLAY_CARD,
                    hand_index=hand_index,
                    card_uuid=card.uuid,
                    card_name=card.name,
                    target_index=None,
                )
            )

    # A stable player combat decision state can always choose
    # to end the turn.
    actions.append(
        CanonicalAction(
            action_id=f"ACTION_{len(actions)}",
            action_type=ActionType.END_TURN,
        )
    )

    return tuple(actions)