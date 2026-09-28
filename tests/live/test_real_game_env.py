import unittest

from sts1_llm_policy.env.action_schema import ActionType, CanonicalAction
from sts1_llm_policy.env.errors import CombatEndedError
from sts1_llm_policy.env.real_game_env import RealGameEnv

from tests.live.fixtures import RecordingClient, SequenceClient


class RealGameEnvTerminalTest(unittest.TestCase):
    def test_detects_reward_game_over_and_main_menu_terminal_states(self) -> None:
        reward = {
            "ready_for_command": True,
            "in_game": True,
            "game_state": {
                "room_phase": "COMPLETE",
                "screen_type": "COMBAT_REWARD",
            },
        }
        game_over = {
            "ready_for_command": True,
            "in_game": True,
            "game_state": {
                "room_phase": "COMPLETE",
                "screen_type": "GAME_OVER",
            },
        }
        main_menu = {
            "ready_for_command": True,
            "in_game": False,
        }

        self.assertTrue(RealGameEnv._is_combat_terminal_state(reward))
        self.assertTrue(RealGameEnv._is_combat_terminal_state(game_over))
        self.assertTrue(RealGameEnv._is_combat_terminal_state(main_menu))

    def test_does_not_end_at_active_or_unready_combat_state(self) -> None:
        combat = {
            "ready_for_command": True,
            "in_game": True,
            "game_state": {
                "room_phase": "COMBAT",
                "screen_type": "NONE",
            },
        }
        unready = {**combat, "ready_for_command": False}

        self.assertFalse(RealGameEnv._is_combat_terminal_state(combat))
        self.assertFalse(RealGameEnv._is_combat_terminal_state(unready))

    def test_raw_state_snapshot_cannot_mutate_environment(self) -> None:
        env = RealGameEnv(client=None)  # type: ignore[arg-type]
        env._raw_state = {"nested": {"value": 1}}

        snapshot = env.get_raw_state()
        snapshot["nested"]["value"] = 9

        self.assertEqual(env.get_raw_state()["nested"]["value"], 1)


class RealGameEnvInitialDecisionTest(unittest.TestCase):
    @staticmethod
    def _combat_state(monsters: list[dict]) -> dict:
        return {
            "ready_for_command": True,
            "in_game": True,
            "available_commands": ["play", "end", "state"],
            "game_state": {
                "room_phase": "COMBAT",
                "action_phase": "WAITING_ON_USER",
                "screen_type": "NONE",
                "combat_state": {"monsters": monsters},
            },
        }

    def test_accepts_initial_state_with_active_monster(self) -> None:
        raw_state = self._combat_state(
            [
                {
                    "name": "Cultist",
                    "current_hp": 48,
                    "is_gone": False,
                    "half_dead": False,
                }
            ]
        )

        self.assertTrue(RealGameEnv._is_initial_combat_decision(raw_state))

    def test_rejects_empty_post_kill_combat_transition(self) -> None:
        raw_state = self._combat_state([])

        self.assertTrue(RealGameEnv._is_combat_decision(raw_state))
        self.assertFalse(RealGameEnv._is_initial_combat_decision(raw_state))

    def test_rejects_only_gone_dead_or_half_dead_monsters(self) -> None:
        raw_state = self._combat_state(
            [
                {"current_hp": 0},
                {"current_hp": 10, "is_gone": True},
                {"current_hp": 10, "half_dead": True},
            ]
        )

        self.assertFalse(RealGameEnv._is_initial_combat_decision(raw_state))


class RealGameEnvPollingTest(unittest.TestCase):
    def test_state_request_waits_before_sending_command(self) -> None:
        client = RecordingClient()
        delays: list[float] = []
        env = RealGameEnv(
            client=client,  # type: ignore[arg-type]
            state_poll_interval_seconds=0.4,
            sleeper=delays.append,
        )

        env._request_state_after_backoff()

        self.assertEqual(delays, [0.4])
        self.assertEqual(client.commands, ["state"])

    def test_zero_interval_disables_sleep_but_still_requests_state(self) -> None:
        client = RecordingClient()
        delays: list[float] = []
        env = RealGameEnv(
            client=client,  # type: ignore[arg-type]
            state_poll_interval_seconds=0,
            sleeper=delays.append,
        )

        env._request_state_after_backoff()

        self.assertEqual(delays, [])
        self.assertEqual(client.commands, ["state"])

    def test_rejects_invalid_poll_intervals(self) -> None:
        for interval in (-0.1, float("inf"), float("nan"), True):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, "finite non-negative"):
                    RealGameEnv(
                        client=None,  # type: ignore[arg-type]
                        state_poll_interval_seconds=interval,
                    )


class RealGameEnvSessionTest(unittest.TestCase):
    @staticmethod
    def _combat_state() -> dict:
        return {
            "ready_for_command": True,
            "in_game": True,
            "available_commands": ["play", "end", "state"],
            "game_state": {
                "room_phase": "COMBAT",
                "action_phase": "WAITING_ON_USER",
                "screen_type": "NONE",
                "combat_state": {
                    "monsters": [
                        {"current_hp": 10, "is_gone": False, "half_dead": False}
                    ]
                },
            },
        }

    def test_operator_wait_only_issues_read_only_state_request(self) -> None:
        reward = {
            "ready_for_command": True,
            "in_game": True,
            "available_commands": ["choose", "proceed", "state"],
            "game_state": {"room_phase": "COMPLETE", "screen_type": "COMBAT_REWARD"},
        }
        client = SequenceClient([reward, self._combat_state()])
        env = RealGameEnv(client, state_poll_interval_seconds=0, sleeper=lambda _: None)
        marker = object()
        env._update_state = lambda raw: marker  # type: ignore[method-assign]

        result = env.wait_for_next_combat()

        self.assertIs(result, marker)
        self.assertEqual(client.commands, ["state"])
        self.assertNotIn("choose", client.commands)
        self.assertNotIn("proceed", client.commands)
        self.assertNotIn("wait 300", client.commands)

    def test_operator_wait_can_receive_manual_transition_without_command(self) -> None:
        reward = {
            "ready_for_command": True,
            "in_game": True,
            "available_commands": ["choose"],
            "game_state": {"room_phase": "COMPLETE", "screen_type": "CARD_REWARD"},
        }
        client = SequenceClient([reward, self._combat_state()])
        env = RealGameEnv(client)
        marker = object()
        env._update_state = lambda raw: marker  # type: ignore[method-assign]

        self.assertIs(env.wait_for_next_combat(), marker)
        self.assertEqual(client.commands, [])

    def test_session_connect_handshakes_once_and_stops_at_game_over(self) -> None:
        game_over = {
            "ready_for_command": True,
            "in_game": True,
            "available_commands": ["state"],
            "game_state": {"room_phase": "COMPLETE", "screen_type": "GAME_OVER"},
        }
        client = SequenceClient([game_over])
        env = RealGameEnv(client)

        self.assertIsNone(env.connect_session())
        self.assertEqual(client.handshakes, 1)
        self.assertEqual(client.commands, [])
        self.assertEqual(env.get_raw_state(), game_over)
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            env.get_state()

    def test_session_connect_waits_from_main_menu_for_first_combat(self) -> None:
        main_menu = {
            "ready_for_command": True,
            "in_game": False,
            "available_commands": ["start", "state"],
        }
        client = SequenceClient([main_menu, self._combat_state()])
        env = RealGameEnv(client, state_poll_interval_seconds=0)
        marker = object()
        env._update_state = lambda raw: marker  # type: ignore[method-assign]

        self.assertIs(env.connect_session(), marker)
        self.assertEqual(client.handshakes, 1)
        self.assertEqual(client.commands, ["state"])
        self.assertNotIn("start", client.commands)

    def test_early_handshake_is_not_repeated_by_session_connect(self) -> None:
        game_over = {
            "ready_for_command": True,
            "in_game": True,
            "available_commands": ["state"],
            "game_state": {
                "room_phase": "COMPLETE",
                "screen_type": "GAME_OVER",
            },
        }
        client = SequenceClient([game_over])
        env = RealGameEnv(client)

        env.handshake()

        self.assertIsNone(env.connect_session())
        self.assertEqual(client.handshakes, 1)

    def test_post_combat_wait_still_stops_at_main_menu(self) -> None:
        main_menu = {
            "ready_for_command": True,
            "in_game": False,
            "available_commands": ["start", "state"],
        }
        client = SequenceClient([main_menu])
        env = RealGameEnv(client, state_poll_interval_seconds=0)

        self.assertIsNone(env.wait_for_next_combat())
        self.assertEqual(client.commands, [])
        self.assertEqual(env.get_raw_state(), main_menu)

    def test_session_connect_stops_if_run_returns_to_menu_before_combat(self) -> None:
        map_screen = {
            "ready_for_command": True,
            "in_game": True,
            "available_commands": ["choose"],
            "game_state": {"room_phase": "INCOMPLETE", "screen_type": "MAP"},
        }
        main_menu = {
            "ready_for_command": True,
            "in_game": False,
            "available_commands": ["start", "state"],
        }
        client = SequenceClient([map_screen, main_menu])
        env = RealGameEnv(client, state_poll_interval_seconds=0)

        self.assertIsNone(env.connect_session())
        self.assertEqual(client.handshakes, 1)
        self.assertEqual(client.commands, [])
        self.assertEqual(env.get_raw_state(), main_menu)

    def test_combat_terminal_is_retained_before_exception(self) -> None:
        terminal = {
            "ready_for_command": True,
            "in_game": True,
            "available_commands": ["state"],
            "game_state": {"room_phase": "COMPLETE", "screen_type": "COMBAT_REWARD"},
        }
        client = SequenceClient([terminal])
        env = RealGameEnv(client)
        env._state = object()  # type: ignore[assignment]

        with self.assertRaises(CombatEndedError):
            env._wait_after_action(
                CanonicalAction("ACTION_0", ActionType.END_TURN),
                self._combat_state(),
            )

        self.assertEqual(env.get_raw_state(), terminal)
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            env.get_state()

    def test_post_kill_empty_combat_is_advanced_without_policy_decision(self) -> None:
        empty_combat = self._combat_state()
        empty_combat["available_commands"] = ["wait", "state"]
        empty_combat["game_state"]["combat_state"]["monsters"] = []
        terminal = {
            "ready_for_command": True,
            "in_game": True,
            "available_commands": ["state"],
            "game_state": {"room_phase": "COMPLETE", "screen_type": "COMBAT_REWARD"},
        }
        client = SequenceClient([empty_combat, terminal])
        env = RealGameEnv(client, state_poll_interval_seconds=0)
        env._state = object()  # type: ignore[assignment]

        with self.assertRaises(CombatEndedError):
            env._wait_after_action(
                CanonicalAction("ACTION_0", ActionType.END_TURN),
                self._combat_state(),
            )

        self.assertEqual(client.commands, ["wait 300"])

    def test_play_ignores_stale_ready_state_until_card_leaves_hand(self) -> None:
        before = self._combat_state()
        before["game_state"]["combat_state"].update(
            turn=1,
            hand=[{"uuid": "chosen-card"}, {"uuid": "other-card"}],
        )
        stale = {
            **before,
            "game_state": {
                **before["game_state"],
                "combat_state": {
                    **before["game_state"]["combat_state"],
                    "hand": [{"uuid": "chosen-card"}, {"uuid": "other-card"}],
                },
            },
        }
        advanced = {
            **before,
            "game_state": {
                **before["game_state"],
                "combat_state": {
                    **before["game_state"]["combat_state"],
                    "hand": [{"uuid": "other-card"}],
                },
            },
        }
        client = SequenceClient([stale, stale, advanced])
        env = RealGameEnv(client)
        marker = object()
        env._update_state = lambda raw: marker  # type: ignore[method-assign]

        result = env._wait_after_action(
            CanonicalAction(
                "ACTION_1",
                ActionType.PLAY_CARD,
                hand_index=0,
                card_uuid="chosen-card",
                card_name="Strike",
                target_index=0,
            ),
            before,
        )

        self.assertIs(result, marker)
        self.assertEqual(client.commands, [])

    def test_end_turn_ignores_stale_ready_state_until_turn_advances(self) -> None:
        before = self._combat_state()
        before["game_state"]["combat_state"]["turn"] = 3
        stale = {
            **before,
            "game_state": {
                **before["game_state"],
                "combat_state": {
                    **before["game_state"]["combat_state"],
                    "turn": 3,
                },
            },
        }
        advanced = {
            **before,
            "game_state": {
                **before["game_state"],
                "combat_state": {
                    **before["game_state"]["combat_state"],
                    "turn": 4,
                },
            },
        }
        client = SequenceClient([stale, advanced])
        env = RealGameEnv(client)
        marker = object()
        env._update_state = lambda raw: marker  # type: ignore[method-assign]

        result = env._wait_after_action(
            CanonicalAction("ACTION_0", ActionType.END_TURN),
            before,
        )

        self.assertIs(result, marker)
        self.assertEqual(client.commands, [])


if __name__ == "__main__":
    unittest.main()
