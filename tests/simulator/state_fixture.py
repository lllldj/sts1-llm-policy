"""Synthetic simulator response payloads."""


def _card(
    enum_id: str,
    string_id: str,
    name: str,
    card_type: str,
    cost: int,
    *,
    has_target: bool,
    is_playable: bool = True,
    upgrades: int = 0,
    exhausts: bool = False,
    ethereal: bool = False,
) -> dict:
    return {
        "enum_id": enum_id,
        "string_id": string_id,
        "name": name,
        "type": card_type,
        "cost": cost,
        "cost_for_turn": cost,
        "upgrades": upgrades,
        "exhausts": exhausts,
        "ethereal": ethereal,
        "has_target": has_target,
        "is_playable": is_playable,
    }

def _response(
    *,
    decision_id: int = 0,
    terminal: bool = False,
    outcome: str = "UNDECIDED",
) -> dict:
    strike = _card("STRIKE_RED", "Strike_R", "Strike", "ATTACK", 1, has_target=True)
    defend = _card("DEFEND_RED", "Defend_R", "Defend", "SKILL", 1, has_target=False)
    is_gone = terminal
    state = {
        "schema_version": 1,
        "scenario_id": "cultist",
        "seed": 41,
        "ascension": 0,
        "act": 1,
        "floor": 1,
        "decision_id": decision_id,
        "turn": 0,
        "input_state": "PLAYER_NORMAL" if not terminal else "EXECUTING_ACTIONS",
        "outcome": outcome,
        "terminal": terminal,
        "reward": 1.0 if outcome == "PLAYER_VICTORY" else -1.0 if outcome == "PLAYER_LOSS" else 0.0,
        "draw_pile_top": "back",
        "player": {
            "current_hp": 80,
            "max_hp": 80,
            "block": 0,
            "energy": 3,
            "powers": [],
        },
        "monsters": [
            {
                "index": 0,
                "id": "CULTIST",
                "name": "CULTIST",
                "current_hp": 0 if terminal else 48,
                "max_hp": 48,
                "block": 0,
                "is_targetable": not terminal,
                "is_gone": is_gone,
                "half_dead": False,
                "move_id": 50,
                "move_name": "CULTIST_INCANTATION",
                "is_attacking": False,
                "base_damage": 0,
                "adjusted_damage": 0,
                "hits": 0,
                "powers": [
                    {
                        "power_index": 25,
                        "name": "Ritual",
                        "amount": 3,
                    }
                ],
            }
        ],
        "hand": [
            {**strike, "hand_index": 0},
            {**defend, "hand_index": 1},
        ],
        "draw_pile": [defend],
        "discard_pile": [],
        "exhaust_pile": [],
    }
    actions = [] if terminal else [
        {
            "action_id": "PLAY:0:0",
            "kind": "PLAY_CARD",
            "hand_index": 0,
            "target_index": 0,
            "card_id": "STRIKE_RED",
            "card_name": "Strike",
            "cost_for_turn": 1,
        },
        {
            "action_id": "PLAY:1:-1",
            "kind": "PLAY_CARD",
            "hand_index": 1,
            "target_index": None,
            "card_id": "DEFEND_RED",
            "card_name": "Defend",
            "cost_for_turn": 1,
        },
        {"action_id": "END", "kind": "END_TURN"},
    ]
    return {
        "v": 1,
        "id": "ignored-transport-id",
        "op": "reset" if decision_id == 0 else "step",
        "ok": True,
        "state": state,
        "legal_actions": actions,
    }
