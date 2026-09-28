from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sts1_llm_policy.env.combat_snapshot import with_snapshot_relics
from sts1_llm_policy.data.trajectory import TrajectoryLogger, iter_trajectory_records
from sts1_llm_policy.env.action_equivalence import build_model_action_space
from sts1_llm_policy.env.simulator_execution import SELECTION, resolve_simulator
from sts1_llm_policy.env.simulator_env import StsLightspeedEnv
from sts1_llm_policy.env.route_context import CombatRouteContext, RouteOperation
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.eval.episode_runner import run_combat_episode
from sts1_llm_policy.data.trajectory_replay import replay_combat
from sts1_llm_policy.policy.random_policy import RandomLegalPolicy
from tests.simulator.native_support import NativeSelectionTestCase


class DrawAndSelectPolicy(RandomLegalPolicy):
    def select_action(self, state, actions):
        result = super().select_action(state, actions)
        space = build_model_action_space(state, actions)
        priorities = {"Warcry": 0, "True Grit": 1, "Rage": 2, "Strike": 3}
        selected = min(space.classes, key=lambda c: priorities.get(
            c.card.name.rstrip("+") if c.card else "", 4,
        ))
        return replace(result, action=selected.representative,
                       selected_model_action_id=selected.model_action_id,
                       equivalent_internal_action_ids=selected.internal_action_ids)


class NativeReplayTests(NativeSelectionTestCase):
    def test_rebuild_public_memory_preserves_historical_evidence_and_transitions(self):
        with resolve_simulator(required_capabilities={SELECTION}).create_client() as client, TemporaryDirectory() as temporary:
            client.reward_reset(712, ascension=0, act=1, relics=[])
            snapshot = client.export_combat_snapshot()["combat_snapshot"]
            snapshot["deck"] = [{"string_id": n, "upgraded": False}
                                for n in ("Warcry", "True Grit", "Rage", "Strike_R", "Defend_R")]
            snapshot = with_snapshot_relics(snapshot, [])
            context = CombatRouteContext(1, 2, "normal_weak", "the_guardian", (RouteOperation("combat", "boss"),))
            env = StsLightspeedEnv(client, allow_card_selection=True)
            env.reset("jaw_worm", 712, combat_snapshot=snapshot, route_context=context)
            path = Path(temporary) / "combat.jsonl"
            logger = TrajectoryLogger(path, episode_id="rebuild", game_seed=712, policy_seed=101,
                                      policy_name="fixture", observation_serializer_version="observation_v7")
            run_combat_episode(env, DrawAndSelectPolicy(101), logger, max_steps=12)
            records = list(iter_trajectory_records(path))
            original = deepcopy(records)
            observations = []
            def consume(index, state, actions):
                self.assertEqual(state.route_context, context)
                self.assertIsNotNone(state.known_draw_top)
                observations.append(serialize_observation(state, actions, version="observation_v7"))
            result = replay_combat(env, scenario_id="jaw_worm", combat_seed=712, snapshot=snapshot,
                                   records=records, route_context=context, public_observation_consumer=consume)
            self.assertEqual(records, original)
            self.assertEqual(result["verified_transitions"], len(observations))
            self.assertTrue(any("KNOWN_DRAW_TOP" in text for text in observations))
            corrupted = deepcopy(records)
            corrupted[0]["next_raw_state"]["state"]["player"]["current_hp"] -= 1
            with self.assertRaisesRegex(ValueError, "transition mismatch"):
                replay_combat(env, scenario_id="jaw_worm", combat_seed=712, snapshot=snapshot,
                              records=corrupted, route_context=context, public_observation_consumer=consume)

    def test_draw_selection_prefix_and_tampered_evidence(self):
        with resolve_simulator(required_capabilities={SELECTION}).create_client() as client, TemporaryDirectory() as temporary:
            client.reward_reset(712, ascension=0, act=1, relics=[])
            snapshot = client.export_combat_snapshot()["combat_snapshot"]
            snapshot["deck"] = [{"string_id": n, "upgraded": n == "True Grit"}
                                for n in ("Warcry", "True Grit", "Rage", "Strike_R", "Defend_R")]
            snapshot = with_snapshot_relics(snapshot, [])
            env = StsLightspeedEnv(client, allow_card_selection=True)
            env.reset("jaw_worm", 712, combat_snapshot=snapshot)
            path = Path(temporary) / "combat.jsonl"
            logger = TrajectoryLogger(path, episode_id="replay", game_seed=712,
                                      policy_seed=101, policy_name="fixture",
                                      observation_serializer_version="observation_v6")
            run_combat_episode(env, DrawAndSelectPolicy(101), logger, max_steps=12)
            records = list(iter_trajectory_records(path))
            result = replay_combat(env, scenario_id="jaw_worm", combat_seed=712,
                                   snapshot=snapshot, records=records)
            self.assertEqual(result["verified_transitions"], len(records))
            self.assertGreater(result["secondary_selection_decisions"], 0)
            selection_index = next(i for i, r in enumerate(records)
                                   if "selection_task" in r["canonical_state"])
            replay_combat(env, scenario_id="jaw_worm", combat_seed=712, snapshot=snapshot,
                          records=records, stop_before=selection_index)
            # A restored intermediate choice is usable by the actual Teacher.
            search = env.search_model_actions(simulations=64, search_seed=101)
            self.assertIn(search.suggested_model_action_id,
                          build_model_action_space(env.get_state(), env.legal_actions()).action_ids)
            corrupted = deepcopy(records)
            corrupted[selection_index]["serialized_state"] += "tampered"
            with self.assertRaisesRegex(ValueError, "observation mismatch"):
                replay_combat(env, scenario_id="jaw_worm", combat_seed=712, snapshot=snapshot,
                              records=corrupted, stop_before=selection_index)
            corrupted = deepcopy(records)
            corrupted[0]["next_raw_state"]["state"]["player"]["current_hp"] -= 1
            with self.assertRaisesRegex(ValueError, "transition mismatch"):
                replay_combat(env, scenario_id="jaw_worm", combat_seed=712,
                              snapshot=snapshot, records=corrupted)


if __name__ == "__main__":
    unittest.main()
