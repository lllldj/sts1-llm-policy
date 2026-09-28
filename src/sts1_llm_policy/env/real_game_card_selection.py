"""Opt-in combat selection transport for CommunicationMod HAND_SELECT/GRID."""
from copy import deepcopy
from dataclasses import replace

from .card_selection import (
    TASKS, CardSelectionState, SelectionCardState,
    build_selection_actions, selection_card, selection_effect,
)
from .action_schema import ActionType
from .real_game_adapter import from_communication_state
from .errors import CombatEndedError
from .real_game_env import RealGameEnv


ACTION_TASKS = {
    "ArmamentsAction": ("ARMAMENTS", "Armaments"),
    "HeadbuttAction": ("HEADBUTT", "Headbutt"),
    "ExhumeAction": ("EXHUME", "Exhume"),
    "ExhaustAction": ("EXHAUST_ONE", None),
    "WarcryAction": ("WARCRY", "Warcry"),
    "DualWieldAction": ("DUAL_WIELD", "Dual Wield"),
}


def is_selection_screen(raw):
    game = raw.get("game_state", {})
    return (raw.get("ready_for_command") is True and raw.get("in_game") is True
            and game.get("room_phase") == "COMBAT"
            and game.get("screen_type") in {"HAND_SELECT", "GRID"})


def promote_selection_cards(state):
    piles = {}
    for name in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        piles[name] = tuple(
            SelectionCardState(**vars(card)) if selection_effect(card.card_id, card.upgrades) is not None else card
            for card in getattr(state.combat, name)
        )
    return replace(state, combat=replace(state.combat, **piles))


def adapt_selection(raw, played_card=None):
    if not is_selection_screen(raw):
        raise ValueError("Not a stable combat selection screen")
    game = raw["game_state"]
    action = game.get("current_action")
    if action not in ACTION_TASKS:
        raise ValueError(f"Unaudited combat selection action: {action}")
    task, expected_card = ACTION_TASKS[action]
    in_play = game.get("combat_state", {}).get("card_in_play")
    source = in_play if isinstance(in_play, dict) else played_card
    if not isinstance(source, dict):
        raise ValueError("Selection is missing its originating card")
    source_id = source.get("id")
    upgrades = source.get("upgrades")
    if isinstance(upgrades, bool) or upgrades not in (0, 1):
        raise ValueError("Unsupported originating card upgrade")
    if expected_card is not None and source_id != expected_card:
        raise ValueError("Selection action disagrees with originating card")
    if task == "EXHAUST_ONE" and not (source_id == "Burning Pact" or source_id == "True Grit" and upgrades == 1):
        raise ValueError("Exhaust selection is not from a supported choosing card")
    if task == "ARMAMENTS" and upgrades != 0:
        raise ValueError("Upgraded Armaments must not request one-card selection")
    screen = game.get("screen_state", {})
    hand_screen = game["screen_type"] == "HAND_SELECT"
    if hand_screen != (TASKS[task][0] == "hand"):
        raise ValueError("Selection source disagrees with screen type")
    count = screen.get("max_cards" if hand_screen else "num_cards")
    if isinstance(count, bool) or count != 1:
        raise ValueError("Only mandatory single-card selection is supported")
    if screen.get("can_pick_zero" if hand_screen else "any_number") is not False:
        raise ValueError("Optional/multiple selection is outside this contract")
    selected = screen.get("selected" if hand_screen else "selected_cards")
    if selected != [] or screen.get("confirm_up", False):
        raise ValueError("Selection is already selected or awaiting confirmation")
    if not hand_screen and any(screen.get(k) is not False for k in ("for_upgrade", "for_transform", "for_purge")):
        raise ValueError("Non-combat grid operation is forbidden")
    candidates = screen.get("hand" if hand_screen else "cards")
    choices = game.get("choice_list")
    if "choose" not in raw.get("available_commands", []) or not isinstance(candidates, list) or not candidates:
        raise ValueError("Selection has no available choose command/candidates")
    if not isinstance(choices, list) or len(choices) != len(candidates):
        raise ValueError("choice_list does not match screen candidates")
    candidate_uuids = [c.get("uuid") for c in candidates]
    if any(not isinstance(u, str) or not u for u in candidate_uuids) or len(set(candidate_uuids)) != len(candidate_uuids):
        raise ValueError("Candidate UUIDs are missing or ambiguous")
    # The live action manager is EXECUTING_ACTIONS while awaiting this screen.
    # All selection guards above are checked before using the ordinary parser.
    projected = deepcopy(raw)
    projected["game_state"]["action_phase"] = "WAITING_ON_USER"
    state = promote_selection_cards(from_communication_state(projected))
    pile = getattr(state.combat, TASKS[task][0])
    indices = []
    for candidate in candidates:
        matches = [i for i, card in enumerate(pile) if card.uuid == candidate["uuid"]]
        if len(matches) != 1:
            raise ValueError("Screen card is absent or ambiguous in its source pile")
        card = pile[matches[0]]
        if candidate.get("id") != card.card_id or candidate.get("upgrades") != card.upgrades:
            raise ValueError("Screen card identity disagrees with source pile")
        if task == "EXHUME" and card.card_id == "Exhume":
            raise ValueError("Exhume cannot retrieve Exhume")
        if task == "DUAL_WIELD" and card.card_type not in {"ATTACK", "POWER"}:
            raise ValueError("Dual Wield candidate is not an Attack or Power")
        if task == "ARMAMENTS" and (card.card_type in {"STATUS", "CURSE"} or card.upgrades > 0 and card.card_id != "Searing Blow"):
            raise ValueError("Armaments candidate cannot upgrade")
        indices.append(matches[0])
    return CardSelectionState(**vars(state), selection_task=task,
                              candidate_indices=tuple(indices),
                              copies_created=1 + upgrades if task == "DUAL_WIELD" else 0)


class CardSelectionRealGameEnv(RealGameEnv):
    """Explicitly selected by a V6 session, leaving the V5 environment frozen."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._played_card = None
        self._selection_actions = ()

    @staticmethod
    def _is_combat_decision(raw_state):
        return is_selection_screen(raw_state) or RealGameEnv._is_combat_decision(raw_state)

    def _update_state(self, raw_state):
        state = (adapt_selection(raw_state, self._played_card) if is_selection_screen(raw_state)
                 else promote_selection_cards(from_communication_state(raw_state)))
        self._raw_state = deepcopy(raw_state)
        self._state = state
        self._selection_actions = build_selection_actions(state) if isinstance(state, CardSelectionState) else ()
        return state

    def legal_actions(self):
        state = self.get_state()
        if isinstance(state, CardSelectionState):
            return self._selection_actions
        return super().legal_actions()

    def step(self, action):
        state = self.get_state()
        if isinstance(state, CardSelectionState):
            card = selection_card(state, action)
            if not any(action is candidate for candidate in self.legal_actions()):
                raise ValueError("Selection action is not currently legal")
            # Screen order can differ from discard/exhaust pile order.
            choice = state.candidate_indices.index(action.selection_index)
            before = self.get_raw_state()
            self._client.send_command(f"choose {choice}")
            return self._wait_after_selection(action, card, before, f"choose {choice}")
        if action.action_type == ActionType.PLAY_CARD:
            self._played_card = deepcopy(self.get_raw_state()["game_state"]["combat_state"]["hand"][action.hand_index])
        else:
            self._played_card = None
        return super().step(action)

    def _wait_after_selection(self, action, card, before, command):
        confirmed = False
        commands = [command]
        while True:
            raw = self._client.receive_json()
            if not raw.get("ready_for_command"):
                continue
            if "error" in raw:
                raise RuntimeError(f"CommunicationMod error: {raw['error']}")
            if self._is_combat_terminal_state(raw):
                raw["card_selection_commands"] = commands
                self._remember_noncombat_state(raw)
                raise CombatEndedError(raw)
            if is_selection_screen(raw):
                game = raw["game_state"]
                old_game = before["game_state"]
                if game.get("current_action") != old_game.get("current_action"):
                    raise ValueError("Selection action changed before acknowledgement")
                screen = game.get("screen_state", {})
                selected = screen.get("selected" if game["screen_type"] == "HAND_SELECT" else "selected_cards", [])
                if selected:
                    if len(selected) != 1 or selected[0].get("uuid") != card.uuid:
                        raise ValueError("Selected card does not acknowledge the requested UUID")
                    if "confirm" in raw.get("available_commands", []) and not confirmed:
                        self._client.send_command("confirm")
                        commands.append("confirm")
                        confirmed = True
                    continue
                if screen == old_game.get("screen_state"):
                    continue  # queued pre-choice state
                # A queued repeated effect can open another selection directly.
                # Require a changed source pile before exposing it as a decision.
                source = TASKS[self.get_state().selection_task][0]
                if game["combat_state"].get(source) == old_game["combat_state"].get(source):
                    continue
                raw["card_selection_commands"] = commands
                return self._update_state(raw)
            if RealGameEnv._is_initial_combat_decision(raw):
                if self._selection_effect_observed(card, before, raw):
                    raw["card_selection_commands"] = commands
                    return self._update_state(raw)
                continue
            if "state" in raw.get("available_commands", []):
                self._request_state_after_backoff()

    def _selection_effect_observed(self, card, before, after):
        task = self.get_state().selection_task
        previous = before["game_state"]["combat_state"]
        current = after["game_state"]["combat_state"]
        pile = TASKS[task][0]
        if task == "ARMAMENTS":
            return any(c.get("uuid") == card.uuid and c.get("upgrades", -1) > card.upgrades
                       for c in current.get("hand", []))
        if task == "DUAL_WIELD":
            def count(combat):
                return sum(c.get("id") == card.card_id and c.get("upgrades") == card.upgrades
                           for name in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
                           for c in combat.get(name, []))
            return count(current) > count(previous)
        return all(c.get("uuid") != card.uuid for c in current.get(pile, []))
