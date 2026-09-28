import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sts1_llm_policy.data.trajectory import (
    TerminalOutcome,
    TrajectoryFormatError,
    TrajectoryLogger,
    iter_trajectory_records,
)
from sts1_llm_policy.env.action_builder import build_legal_actions
from sts1_llm_policy.env.serializer import BEHAVIOR_OBSERVATION_SERIALIZER_VERSION
from sts1_llm_policy.env.state_schema import (
    CanonicalState,
    CardState,
    CombatState,
    MonsterBehaviorState,
    MonsterState,
    PlayerState,
)
from sts1_llm_policy.policy.random_policy import RandomLegalPolicy


def _state(*, turn: int = 1, seed: int = 41) -> CanonicalState:
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
        seed=seed,
        character="IRONCLAD",
        ascension_level=0,
        act=1,
        floor=1,
        combat=CombatState(
            turn=turn,
            player=PlayerState(
                current_hp=80,
                max_hp=80,
                block=0,
                energy=3,
            ),
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


class TrajectoryLoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "nested" / "run.jsonl"

    def _decision(self, state: CanonicalState):
        actions = build_legal_actions(state)
        result = RandomLegalPolicy(policy_seed=7).select_action(
            state,
            actions,
        )
        return actions, result

    def _logger(self, *, resume: bool = False) -> TrajectoryLogger:
        return TrajectoryLogger(
            self.path,
            episode_id="episode-001",
            game_seed=41,
            policy_seed=7,
            policy_name="random_legal_v1",
            encounter="Cultist",
            resume=resume,
        )

    def test_writes_self_contained_jsonl_transition(self) -> None:
        state = _state(turn=1)
        next_state = _state(turn=2)
        actions, result = self._decision(state)
        logger = self._logger()

        returned = logger.log_transition(
            state=state,
            raw_state={"ready_for_command": True},
            legal_actions=actions,
            policy_result=result,
            next_state=next_state,
            next_raw_state={"ready_for_command": True, "step": 2},
            reward=0.25,
            score=3.5,
        )

        records = list(iter_trajectory_records(self.path))

        self.assertEqual(records, [returned])
        record = records[0]
        self.assertEqual(record["schema_version"], "trajectory_v1")
        self.assertEqual(record["record_type"], "transition")
        self.assertEqual(record["episode_id"], "episode-001")
        self.assertEqual(record["step_index"], 0)
        self.assertEqual(record["game_seed"], 41)
        self.assertEqual(record["policy_seed"], 7)
        self.assertEqual(
            record["observation_serializer_version"],
            "observation_v2",
        )
        self.assertEqual(
            record["action_equivalence_version"],
            "action_equivalence_v1",
        )
        self.assertIsNone(record["evidence_class"])
        self.assertIsNone(record["scenario_id"])
        self.assertIsNone(record["preflight"])
        self.assertEqual(record["turn"], 1)
        self.assertEqual(record["raw_state"], {"ready_for_command": True})
        self.assertEqual(record["canonical_state"]["seed"], 41)
        self.assertEqual(record["legal_actions"][0]["action_type"], "play_card")
        self.assertEqual(
            record["model_legal_actions"][0]["model_action_id"],
            "ACTION_0",
        )
        self.assertEqual(
            record["policy_result"]["selected_model_action_id"],
            result.selected_model_action_id,
        )
        self.assertEqual(
            record["policy_result"]["equivalent_internal_action_ids"],
            list(result.equivalent_internal_action_ids),
        )
        self.assertEqual(record["action"], record["policy_result"]["action"])
        self.assertEqual(record["policy_result"]["decision_source"], "random")
        self.assertIsNone(record["policy_result"]["parse_success"])
        self.assertIn("LEGAL_ACTIONS", record["serialized_state"])
        self.assertFalse(record["done"])
        self.assertIsNone(record["terminal_outcome"])
        self.assertEqual(logger.next_step_index, 1)
        self.assertFalse(logger.finished)

        raw_lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(raw_lines), 1)
        self.assertIsInstance(json.loads(raw_lines[0]), dict)

    def test_terminal_transition_finishes_episode(self) -> None:
        state = _state()
        actions, result = self._decision(state)
        logger = self._logger()

        logger.log_transition(
            state=state,
            legal_actions=actions,
            policy_result=result,
            next_state=None,
            reward=1.0,
            done=True,
            terminal_outcome=TerminalOutcome.VICTORY,
        )

        record = list(iter_trajectory_records(self.path))[0]
        self.assertTrue(record["done"])
        self.assertEqual(record["terminal_outcome"], "victory")
        self.assertIsNone(record["next_canonical_state"])
        self.assertTrue(logger.finished)

        with self.assertRaisesRegex(RuntimeError, "already finished"):
            logger.log_transition(
                state=state,
                legal_actions=actions,
                policy_result=result,
                next_state=None,
                done=True,
                terminal_outcome=TerminalOutcome.VICTORY,
            )

    def test_explicit_behavior_version_is_recorded_and_serialized(self) -> None:
        state = _state()
        monster = replace(
            state.combat.monsters[0],
            behavior=MonsterBehaviorState(
                phase="opening_buff",
                previous_move_id=None,
                possible_next_move_ids=(1,),
                selection="deterministic",
                rule="Opens with Incantation, then repeats Dark Strike.",
            ),
        )
        state = replace(
            state,
            combat=replace(state.combat, monsters=(monster,)),
        )
        actions, result = self._decision(state)
        logger = TrajectoryLogger(
            self.path,
            episode_id="episode-v3",
            game_seed=41,
            policy_seed=7,
            policy_name="random_legal_v1",
            observation_serializer_version=(
                BEHAVIOR_OBSERVATION_SERIALIZER_VERSION
            ),
        )

        record = logger.log_transition(
            state=state,
            legal_actions=actions,
            policy_result=result,
            next_state=None,
            done=True,
            terminal_outcome=TerminalOutcome.VICTORY,
        )

        self.assertEqual(
            record["observation_serializer_version"],
            "observation_v3",
        )
        self.assertIn("BEHAVIOR:", record["serialized_state"])

    def test_records_duplicate_action_class_and_exact_members(self) -> None:
        state = _state()
        duplicate = replace(
            state.combat.hand[0],
            uuid="strike-duplicate",
        )
        state = replace(
            state,
            combat=replace(
                state.combat,
                hand=(state.combat.hand[0], duplicate),
            ),
        )
        actions, result = self._decision(state)

        record = self._logger().log_transition(
            state=state,
            legal_actions=actions,
            policy_result=result,
            next_state=None,
            done=True,
            terminal_outcome=TerminalOutcome.VICTORY,
        )

        play_class = record["model_legal_actions"][0]
        self.assertEqual(play_class["copies"], 2)
        self.assertEqual(
            play_class["equivalent_internal_action_ids"],
            ["ACTION_0", "ACTION_1"],
        )
        self.assertEqual(
            play_class["representative_internal_action_id"],
            "ACTION_0",
        )

    def test_validates_terminal_and_current_action_invariants(self) -> None:
        state = _state()
        actions, result = self._decision(state)
        logger = self._logger()

        with self.assertRaisesRegex(ValueError, "requires terminal_outcome"):
            logger.log_transition(
                state=state,
                legal_actions=actions,
                policy_result=result,
                next_state=None,
                done=True,
            )

        with self.assertRaisesRegex(ValueError, "requires next_state"):
            logger.log_transition(
                state=state,
                legal_actions=actions,
                policy_result=result,
                next_state=None,
            )

        actions_without_selected = tuple(
            action for action in actions if action != result.action
        )

        with self.assertRaisesRegex(ValueError, "current legal action set"):
            logger.log_transition(
                state=state,
                legal_actions=actions_without_selected,
                policy_result=result,
                next_state=_state(turn=2),
            )

        self.assertFalse(self.path.exists())

    def test_resume_continues_step_index_and_rejects_finished_episode(self) -> None:
        state = _state(turn=1)
        next_state = _state(turn=2)
        actions, result = self._decision(state)
        self._logger().log_transition(
            state=state,
            legal_actions=actions,
            policy_result=result,
            next_state=next_state,
        )

        with self.assertRaisesRegex(ValueError, "already exists"):
            self._logger()

        resumed = self._logger(resume=True)
        self.assertEqual(resumed.next_step_index, 1)
        next_actions, next_result = self._decision(next_state)
        resumed.log_transition(
            state=next_state,
            legal_actions=next_actions,
            policy_result=next_result,
            next_state=None,
            done=True,
            terminal_outcome=TerminalOutcome.DEFEAT,
        )

        records = list(iter_trajectory_records(self.path))
        self.assertEqual([record["step_index"] for record in records], [0, 1])

        with self.assertRaisesRegex(ValueError, "Cannot resume finished"):
            self._logger(resume=True)

    def test_reader_reports_malformed_jsonl(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.write_text("not json\n", encoding="utf-8")

        with self.assertRaisesRegex(TrajectoryFormatError, "line 1"):
            list(iter_trajectory_records(self.path))

    def test_rejects_seed_mismatch_and_non_finite_scores(self) -> None:
        state = _state(seed=99)
        actions, result = self._decision(state)
        logger = self._logger()

        with self.assertRaisesRegex(ValueError, "does not match game_seed"):
            logger.log_transition(
                state=state,
                legal_actions=actions,
                policy_result=result,
                next_state=None,
                done=True,
                terminal_outcome=TerminalOutcome.ABORTED,
            )

        valid_state = _state()
        valid_actions, valid_result = self._decision(valid_state)

        with self.assertRaisesRegex(ValueError, "score must be finite"):
            logger.log_transition(
                state=valid_state,
                legal_actions=valid_actions,
                policy_result=valid_result,
                next_state=_state(turn=2),
                score=float("nan"),
            )

    def test_parity_clean_requires_and_records_accepted_preflight(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires scenario_id"):
            TrajectoryLogger(
                self.path,
                episode_id="clean-001",
                game_seed=41,
                policy_seed=7,
                policy_name="random_legal_v1",
                evidence_class="parity_clean",
            )

        with self.assertRaisesRegex(ValueError, "accepted preflight"):
            TrajectoryLogger(
                self.path,
                episode_id="clean-001",
                game_seed=41,
                policy_seed=7,
                policy_name="random_legal_v1",
                evidence_class="parity_clean",
                scenario_id="two_louse",
                preflight={"accepted": False},
            )

        state = _state()
        actions, result = self._decision(state)
        logger = TrajectoryLogger(
            self.path,
            episode_id="clean-001",
            game_seed=41,
            policy_seed=7,
            policy_name="random_legal_v1",
            encounter="Cultist",
            evidence_class="parity_clean",
            scenario_id="cultist",
            preflight={
                "schema_version": "real_game_capture_preflight_v1",
                "accepted": True,
            },
        )
        logger.log_transition(
            state=state,
            legal_actions=actions,
            policy_result=result,
            next_state=None,
            done=True,
            terminal_outcome=TerminalOutcome.VICTORY,
        )

        record = list(iter_trajectory_records(self.path))[0]
        self.assertEqual(record["evidence_class"], "parity_clean")
        self.assertEqual(record["scenario_id"], "cultist")
        self.assertTrue(record["preflight"]["accepted"])


if __name__ == "__main__":
    unittest.main()
