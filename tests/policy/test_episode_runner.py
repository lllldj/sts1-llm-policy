import tempfile
import unittest
from pathlib import Path

from sts1_llm_policy.data.trajectory import TrajectoryLogger, iter_trajectory_records
from sts1_llm_policy.env.action_builder import build_legal_actions
from sts1_llm_policy.env.action_schema import CanonicalAction
from sts1_llm_policy.env.errors import CombatEndedError
from sts1_llm_policy.env.state_schema import (
    CanonicalState,
    CardState,
    CombatState,
    MonsterState,
    PlayerState,
)
from sts1_llm_policy.data.trajectory import classify_terminal_outcome
from sts1_llm_policy.eval.episode_runner import extract_terminal_score, run_combat_episode
from sts1_llm_policy.policy.random_policy import RandomLegalPolicy


def _state(*, turn: int = 1) -> CanonicalState:
    card = CardState(
        card_id="Strike_R",
        name="Strike",
        card_type="ATTACK",
        cost=1,
        upgrades=0,
        description="Deal 6 damage.",
        exhausts=False,
        ethereal=False,
        has_target=True,
        is_playable=True,
        uuid=f"strike-{turn}",
    )

    return CanonicalState(
        seed=41,
        character="IRONCLAD",
        ascension_level=0,
        act=1,
        floor=1,
        combat=CombatState(
            turn=turn,
            player=PlayerState(80, 80, 0, 3),
            monsters=(
                MonsterState(
                    monster_id="Cultist",
                    name="Cultist",
                    current_hp=48,
                    max_hp=48,
                    block=0,
                    intent="BUFF",
                    move_id=3,
                    move_hits=1,
                    move_base_damage=0,
                    move_adjusted_damage=0,
                ),
            ),
            hand=(card,),
            draw_pile=(),
            discard_pile=(),
            exhaust_pile=(),
        ),
    )


def _raw_combat(turn: int) -> dict:
    return {
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "room_phase": "COMBAT",
            "screen_type": "NONE",
            "turn_fixture": turn,
        },
    }


def _game_over(*, victory: bool, score: int) -> dict:
    return {
        "ready_for_command": True,
        "in_game": True,
        "game_state": {
            "room_phase": "COMPLETE",
            "screen_type": "GAME_OVER",
            "screen_state": {"victory": victory, "score": score},
            "current_hp": 0 if not victory else 7,
        },
    }


class FakeCombatEnv:
    def __init__(
        self,
        states: list[CanonicalState],
        terminal_raw_state: dict | None = None,
    ) -> None:
        self.states = states
        self.terminal_raw_state = terminal_raw_state
        self.index = 0
        self.executed: list[CanonicalAction] = []

    def get_state(self) -> CanonicalState:
        return self.states[self.index]

    def get_raw_state(self) -> dict:
        return _raw_combat(self.states[self.index].combat.turn)

    def legal_actions(self) -> tuple[CanonicalAction, ...]:
        return build_legal_actions(self.get_state())

    def step(self, action: CanonicalAction) -> CanonicalState:
        self.executed.append(action)

        if self.index + 1 == len(self.states):
            if self.terminal_raw_state is None:
                raise AssertionError("Fake environment has no next state")

            raise CombatEndedError(self.terminal_raw_state)

        self.index += 1
        return self.get_state()


class TerminalClassificationTest(unittest.TestCase):
    def test_classifies_explicit_game_over_fields(self) -> None:
        victory = _game_over(victory=True, score=123)
        defeat = _game_over(victory=False, score=17)

        self.assertEqual(classify_terminal_outcome(victory).value, "victory")
        self.assertEqual(classify_terminal_outcome(defeat).value, "defeat")
        self.assertEqual(extract_terminal_score(victory), 123.0)

    def test_classifies_combat_reward_and_conservative_abort(self) -> None:
        reward = {
            "game_state": {
                "room_phase": "COMPLETE",
                "screen_type": "COMBAT_REWARD",
                "screen_state": {},
                "current_hp": 5,
            }
        }
        unknown = {"in_game": False}

        self.assertEqual(classify_terminal_outcome(reward).value, "victory")
        self.assertEqual(classify_terminal_outcome(unknown).value, "aborted")
        self.assertIsNone(extract_terminal_score(reward))

    def test_hp_zero_is_defeat_without_game_over_payload(self) -> None:
        raw_state = {
            "game_state": {
                "room_phase": "INCOMPLETE",
                "screen_type": "NONE",
                "current_hp": 0,
            }
        }

        self.assertEqual(classify_terminal_outcome(raw_state).value, "defeat")


class EpisodeRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "run.jsonl"

    def _logger(self) -> TrajectoryLogger:
        return TrajectoryLogger(
            self.path,
            episode_id="episode-001",
            game_seed=41,
            policy_seed=7,
            policy_name="random_legal_v1",
            encounter="Cultist",
        )

    def test_runs_until_environment_terminal_and_logs_final_action(self) -> None:
        env = FakeCombatEnv(
            [_state(turn=1), _state(turn=2)],
            terminal_raw_state=_game_over(victory=False, score=21),
        )
        callbacks: list[tuple[int, str]] = []

        summary = run_combat_episode(
            env,
            RandomLegalPolicy(policy_seed=7),
            self._logger(),
            on_step=lambda index, result: callbacks.append(
                (index, result.action.action_id)
            ),
        )

        records = list(iter_trajectory_records(self.path))
        self.assertEqual(summary.steps, 2)
        self.assertEqual(summary.outcome.value, "defeat")
        self.assertEqual(summary.score, 21.0)
        self.assertEqual(summary.termination_reason, "environment_terminal")
        self.assertEqual(len(env.executed), 2)
        self.assertEqual(len(records), 2)
        self.assertFalse(records[0]["done"])
        self.assertTrue(records[1]["done"])
        self.assertEqual(records[1]["terminal_outcome"], "defeat")
        self.assertEqual(records[1]["score"], 21.0)
        self.assertEqual(len(callbacks), 2)

    def test_step_limit_closes_episode_as_aborted(self) -> None:
        env = FakeCombatEnv([_state(turn=1), _state(turn=2)])

        summary = run_combat_episode(
            env,
            RandomLegalPolicy(policy_seed=7),
            self._logger(),
            max_steps=1,
        )

        record = list(iter_trajectory_records(self.path))[0]
        self.assertEqual(summary.steps, 1)
        self.assertEqual(summary.outcome.value, "aborted")
        self.assertEqual(summary.termination_reason, "max_steps")
        self.assertTrue(record["done"])
        self.assertEqual(record["terminal_outcome"], "aborted")
        self.assertIsNotNone(record["next_canonical_state"])

    def test_rejects_non_positive_step_limit_before_execution(self) -> None:
        env = FakeCombatEnv([_state()])

        with self.assertRaisesRegex(ValueError, "must be positive"):
            run_combat_episode(
                env,
                RandomLegalPolicy(policy_seed=7),
                self._logger(),
                max_steps=0,
            )

        self.assertEqual(env.executed, [])
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
