from collections.abc import Callable
from copy import deepcopy
import math
import time

from .action_builder import build_legal_actions
from .action_mapper import to_communication_command
from .action_schema import ActionType, CanonicalAction
from .communication_client import CommunicationClient
from .errors import CombatEndedError
from .real_game_adapter import from_communication_state
from .state_schema import CanonicalState


class RealGameEnv:
    """
    Minimal real STS1 combat environment.

    v0 responsibilities:
    - wait for a stable combat decision state
    - expose CanonicalState
    - build legal actions
    - execute one CanonicalAction
    - refresh state after every action
    """

    def __init__(
        self,
        client: CommunicationClient,
        *,
        state_poll_interval_seconds: float = 0.25,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            isinstance(state_poll_interval_seconds, bool)
            or not isinstance(state_poll_interval_seconds, (int, float))
            or not math.isfinite(state_poll_interval_seconds)
            or state_poll_interval_seconds < 0
        ):
            raise ValueError(
                "state_poll_interval_seconds must be a finite "
                "non-negative number"
            )

        self._client = client
        self._state_poll_interval_seconds = float(
            state_poll_interval_seconds
        )
        self._sleep = sleeper

        self._raw_state: dict | None = None
        self._state: CanonicalState | None = None
        self._handshake_complete = False

    def handshake(self) -> None:
        """Signal readiness once, independently of slower policy loading."""

        if self._handshake_complete:
            return

        self._client.handshake()
        self._handshake_complete = True

    def _remember_noncombat_state(self, raw_state: dict) -> None:
        """Retain live evidence without exposing a stale combat state."""

        self._raw_state = deepcopy(raw_state)
        self._state = None

    def _request_state_after_backoff(self) -> None:
        """Throttle explicit state polling outside automatic game updates."""

        if self._state_poll_interval_seconds > 0:
            self._sleep(self._state_poll_interval_seconds)

        self._client.send_command("state")

    @staticmethod
    def _is_combat_decision(
        raw_state: dict,
    ) -> bool:
        if not raw_state.get(
            "ready_for_command",
            False,
        ):
            return False

        if not raw_state.get(
            "in_game",
            False,
        ):
            return False

        game_state = raw_state.get(
            "game_state"
        )

        if not isinstance(
            game_state,
            dict,
        ):
            return False

        combat_state = game_state.get(
            "combat_state"
        )

        if not isinstance(
            combat_state,
            dict,
        ):
            return False

        if (
            game_state.get("room_phase")
            != "COMBAT"
        ):
            return False

        if (
            game_state.get("action_phase")
            != "WAITING_ON_USER"
        ):
            return False

        available_commands = set(
            raw_state.get(
                "available_commands",
                [],
            )
        )

        return "end" in available_commands

    @staticmethod
    def _has_active_monster(raw_state: dict) -> bool:
        """Reject the brief post-kill COMBAT state with an empty encounter."""

        game_state = raw_state.get("game_state")

        if not isinstance(game_state, dict):
            return False

        combat_state = game_state.get("combat_state")

        if not isinstance(combat_state, dict):
            return False

        monsters = combat_state.get("monsters")

        if not isinstance(monsters, list):
            return False

        for monster in monsters:
            if not isinstance(monster, dict):
                continue

            current_hp = monster.get("current_hp")

            if (
                isinstance(current_hp, int)
                and not isinstance(current_hp, bool)
                and current_hp > 0
                and monster.get("is_gone") is not True
                and monster.get("half_dead") is not True
            ):
                return True

        return False

    @classmethod
    def _is_initial_combat_decision(
        cls,
        raw_state: dict,
    ) -> bool:
        """Accept a new episode only after a real encounter is populated."""

        return (
            cls._is_combat_decision(raw_state)
            and cls._has_active_monster(raw_state)
        )

    @staticmethod
    def _is_combat_terminal_state(
        raw_state: dict,
    ) -> bool:
        if not raw_state.get(
            "ready_for_command",
            False,
        ):
            return False

        # A return to the main menu after an action is still a terminal
        # episode event, although its outcome may only be classified as
        # aborted by the runner.
        if not raw_state.get("in_game", False):
            return True

        game_state = raw_state.get(
            "game_state"
        )

        if not isinstance(
            game_state,
            dict,
        ):
            return False

        screen_type = game_state.get("screen_type")

        if screen_type in {
            "CARD_REWARD",
            "COMBAT_REWARD",
            "BOSS_REWARD",
            "GAME_OVER",
            "COMPLETE",
        }:
            return True

        return game_state.get("room_phase") != "COMBAT"

    def _update_state(
        self,
        raw_state: dict,
    ) -> CanonicalState:
        state = from_communication_state(
            raw_state
        )

        self._raw_state = raw_state
        self._state = state

        return state

    @staticmethod
    def _combat_state(raw_state: dict) -> dict | None:
        game_state = raw_state.get("game_state")
        if not isinstance(game_state, dict):
            return None
        combat_state = game_state.get("combat_state")
        return combat_state if isinstance(combat_state, dict) else None

    @classmethod
    def _action_has_advanced(
        cls,
        action: CanonicalAction,
        state_before: dict,
        candidate: dict,
    ) -> bool:
        """Reject uncorrelated stale snapshots received after a command."""

        before_combat = cls._combat_state(state_before)
        after_combat = cls._combat_state(candidate)
        if before_combat is None or after_combat is None:
            return False

        if action.action_type == ActionType.PLAY_CARD:
            if not action.card_uuid:
                raise RuntimeError("Play action is missing its card UUID")
            hand = after_combat.get("hand")
            if not isinstance(hand, list):
                return False
            return all(
                not isinstance(card, dict)
                or card.get("uuid") != action.card_uuid
                for card in hand
            )

        if action.action_type == ActionType.END_TURN:
            before_turn = before_combat.get("turn")
            after_turn = after_combat.get("turn")
            return (
                isinstance(before_turn, int)
                and not isinstance(before_turn, bool)
                and isinstance(after_turn, int)
                and not isinstance(after_turn, bool)
                and after_turn > before_turn
            )

        raise RuntimeError(f"Unsupported action type: {action.action_type}")

    def _wait_for_initial_combat(
        self,
    ) -> CanonicalState:
        """
        Wait while the user manually enters a combat.
        """

        while True:
            raw_state = (
                self._client.receive_json()
            )

            if not raw_state.get(
                "ready_for_command",
                False,
            ):
                continue

            if "error" in raw_state:
                raise RuntimeError(
                    "CommunicationMod error: "
                    f"{raw_state['error']}"
                )

            if self._is_initial_combat_decision(
                raw_state
            ):
                return self._update_state(
                    raw_state
                )

            available_commands = set(
                raw_state.get(
                    "available_commands",
                    [],
                )
            )

            if (
                raw_state.get(
                    "in_game",
                    False,
                )
                and "wait"
                in available_commands
            ):
                self._client.send_command(
                    "wait 300"
                )

            elif (
                "state"
                in available_commands
            ):
                self._request_state_after_backoff()

            else:
                raise RuntimeError(
                    "No usable command while "
                    "waiting for combat"
                )

    @staticmethod
    def _is_run_terminal_state(raw_state: dict) -> bool:
        """Return whether a ready state proves that no later combat remains."""

        if not raw_state.get("ready_for_command", False):
            return False
        if not raw_state.get("in_game", False):
            return True

        game_state = raw_state.get("game_state")
        if not isinstance(game_state, dict):
            return False

        return game_state.get("screen_type") in {"GAME_OVER", "COMPLETE"}

    def _wait_for_session_combat(
        self,
        *,
        allow_initial_main_menu: bool,
    ) -> CanonicalState | None:
        """Wait for operator-driven navigation to reach the next combat.

        This method deliberately never issues ``choose``, ``proceed``, ``play``,
        ``end`` or ``wait``.  The only command it may emit is the read-only
        ``state`` request, leaving every non-combat decision to the user.
        ``None`` means that the active run ended or returned to the main menu.
        Before the first combat, the launch-time main menu is not an active-run
        terminal state and must remain under operator control.
        """

        run_observed = not allow_initial_main_menu

        while True:
            raw_state = self._client.receive_json()

            if not raw_state.get("ready_for_command", False):
                continue
            if "error" in raw_state:
                raise RuntimeError(
                    "CommunicationMod error: " f"{raw_state['error']}"
                )
            if raw_state.get("in_game", False):
                run_observed = True
            if self._is_initial_combat_decision(raw_state):
                return self._update_state(raw_state)

            self._remember_noncombat_state(raw_state)
            if run_observed and self._is_run_terminal_state(raw_state):
                return None

            available_commands = set(raw_state.get("available_commands", []))
            if "state" in available_commands:
                self._request_state_after_backoff()

    def wait_for_next_combat(self) -> CanonicalState | None:
        """Wait after a completed combat until the next combat or run end."""

        return self._wait_for_session_combat(allow_initial_main_menu=False)

    def _wait_after_action(
        self,
        action: CanonicalAction,
        state_before: dict,
    ) -> CanonicalState:
        """
        Wait for the next stable player decision state.

        If execution leaves combat, signal that explicitly
        instead of waiting forever for another combat state.
        """

        while True:
            raw_state = (
                self._client.receive_json()
            )

            if not raw_state.get(
                "ready_for_command",
                False,
            ):
                continue

            if "error" in raw_state:
                raise RuntimeError(
                    "CommunicationMod error: "
                    f"{raw_state['error']}"
                )

            if self._is_initial_combat_decision(
                raw_state
            ):
                if not self._action_has_advanced(
                    action,
                    state_before,
                    raw_state,
                ):
                    # CommunicationMod does not correlate state messages with
                    # commands. Explicit polling during operator-owned screens
                    # can leave duplicate ready snapshots queued at combat
                    # entry. Never expose one as a new policy decision.
                    continue
                return self._update_state(
                    raw_state
                )

            if self._is_combat_terminal_state(
                raw_state
            ):
                self._remember_noncombat_state(raw_state)
                raise CombatEndedError(
                    raw_state
                )

            available_commands = set(
                raw_state.get(
                    "available_commands",
                    [],
                )
            )

            if (
                raw_state.get(
                    "in_game",
                    False,
                )
                and "wait"
                in available_commands
            ):
                self._client.send_command(
                    "wait 300"
                )

            elif (
                "state"
                in available_commands
            ):
                self._request_state_after_backoff()

            else:
                raise RuntimeError(
                    "No usable command after action"
                )

    def connect(self) -> CanonicalState:
        """
        Perform CommunicationMod handshake and wait until
        the first stable combat decision state.
        """

        self.handshake()

        return self._wait_for_initial_combat()

    def connect_session(self) -> CanonicalState | None:
        """Handshake once and wait through the launch-time main menu."""

        self.handshake()
        return self._wait_for_session_combat(allow_initial_main_menu=True)

    def get_state(self) -> CanonicalState:
        if self._state is None:
            raise RuntimeError(
                "Environment is not connected yet"
            )

        return self._state

    def get_raw_state(self) -> dict:
        """Return an isolated snapshot of the latest CommunicationMod state."""

        if self._raw_state is None:
            raise RuntimeError(
                "Environment is not connected yet"
            )

        return deepcopy(self._raw_state)

    def legal_actions(
        self,
    ) -> tuple[CanonicalAction, ...]:
        state = self.get_state()

        return build_legal_actions(
            state
        )

    def step(
        self,
        action: CanonicalAction,
    ) -> CanonicalState:
        """
        Execute exactly one action.

        Important invariant:
        action is valid only for the current state.

        After execution, a fresh state is received and
        canonicalized before returning.
        """

        state_before = self.get_state()
        raw_state_before = self.get_raw_state()

        command = (
            to_communication_command(
                action,
                state_before,
            )
        )

        self._client.send_command(
            command
        )

        return self._wait_after_action(
            action,
            raw_state_before,
        )
