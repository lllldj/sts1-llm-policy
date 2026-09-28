"""Synthetic game states and communication clients."""


def _raw_card() -> dict:
    return {
        "id": "Strike_R",
        "name": "Strike",
        "type": "ATTACK",
        "cost": 1,
        "upgrades": 0,
        "description": "Deal 6 damage.",
        "exhausts": False,
        "ethereal": False,
        "has_target": True,
        "is_playable": True,
        "uuid": "power-card-uuid",
    }

def _raw_state() -> dict:
    return {
        "in_game": True,
        "ready_for_command": True,
        "game_state": {
            "room_phase": "COMBAT",
            "action_phase": "WAITING_ON_USER",
            "seed": 1,
            "class": "IRONCLAD",
            "ascension_level": 0,
            "act": 1,
            "floor": 1,
            "combat_state": {
                "turn": 2,
                "player": {
                    "current_hp": 70,
                    "max_hp": 80,
                    "block": 0,
                    "energy": 3,
                    "powers": [
                        {
                            "id": "Strength",
                            "name": "Strength",
                            "amount": 2,
                        },
                        {
                            "id": "Nightmare",
                            "name": "Nightmare",
                            "amount": 1,
                            "damage": 4,
                            "misc": 3,
                            "just_applied": True,
                            "card": _raw_card(),
                        },
                    ],
                },
                "monsters": [
                    {
                        "id": "Cultist",
                        "name": "Cultist",
                        "current_hp": 42,
                        "max_hp": 48,
                        "block": 0,
                        "intent": "ATTACK",
                        "move_id": 1,
                        "move_hits": 1,
                        "move_base_damage": 6,
                        "move_adjusted_damage": 6,
                        "last_move_id": 3,
                        "powers": [
                            {
                                "id": "Ritual",
                                "name": "Ritual",
                                "amount": 3,
                                "just_applied": True,
                            }
                        ],
                        "is_gone": False,
                        "half_dead": False,
                    }
                ],
                "hand": [],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
            },
        },
    }


class RecordingClient:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def send_command(self, command: str) -> None:
        self.commands.append(command)

class SequenceClient(RecordingClient):
    def __init__(self, states: list[dict]) -> None:
        super().__init__()
        self.states = iter(states)
        self.handshakes = 0

    def receive_json(self) -> dict:
        return next(self.states)

    def handshake(self) -> None:
        self.handshakes += 1
