import unittest

from sts1_llm_policy.env.combat_snapshot import with_snapshot_relics
from sts1_llm_policy.env.simulator_execution import SELECTION, resolve_simulator
from sts1_llm_policy.env.simulator_env import StsLightspeedEnv
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.env.route_context import CombatRouteContext, RouteOperation
from sts1_llm_policy.env.action_equivalence import build_model_action_space
from tests.simulator.native_support import NativeSelectionTestCase


class PublicDrawMemoryTests(NativeSelectionTestCase):
    def setUp(self):
        self.client = resolve_simulator(required_capabilities={SELECTION}).create_client()
        self.addCleanup(self.client.close)
        self.env = StsLightspeedEnv(self.client, allow_card_selection=True)

    def reset(self, names):
        self.client.reward_reset(712, ascension=0, act=1, relics=[])
        snapshot = self.client.export_combat_snapshot()["combat_snapshot"]
        snapshot["deck"] = [{"string_id": n, "upgraded": False} for n in names]
        self.context = CombatRouteContext(1, 2, "normal_weak", "the_guardian", (RouteOperation("combat", "boss"),))
        self.env.reset("jaw_worm", 712, combat_snapshot=with_snapshot_relics(snapshot, []), route_context=self.context)
        self.env.public_state()
        self.assertEqual(self.env.get_state().route_context, self.context)

    def play(self, card):
        space = build_model_action_space(self.env.get_state(), self.env.legal_actions())
        action = next(a for a in space.classes if a.card and a.card.card_id == card)
        self.env.step(action.representative)

    def choose(self, card):
        space = build_model_action_space(self.env.get_state(), self.env.legal_actions())
        action = next(a for a in space.classes if a.card and a.card.card_id == card)
        self.env.step(action.representative)

    def test_headbutt_warcry_known_card_survives_sampling_and_is_consumed(self):
        self.reset(["Strike_R", "Headbutt", "Warcry", "Defend_R", "Battle Trance"])
        self.play("Strike_R")
        self.play("Headbutt")
        # A singleton discard pile is resolved automatically by the simulator.
        self.assertEqual([c.card_id for c in self.env.get_state().known_draw_top], ["Strike_R"])
        self.env.public_state(seed=55)
        self.assertEqual(self.env.get_state().route_context, self.context)
        self.assertEqual(self.env.get_raw_state()["state"]["draw_pile"][-1]["string_id"], "Strike_R")
        self.play("Warcry")
        self.assertEqual(self.env.get_state().known_draw_top, ())
        self.choose("Defend_R")
        self.assertEqual(self.env.get_state().route_context, self.context)
        self.assertEqual([c.card_id for c in self.env.get_state().known_draw_top], ["Defend_R"])
        self.play("Battle Trance")
        self.assertEqual(self.env.get_state().known_draw_top, ())

    def test_inner_search_does_not_depend_on_outer_hidden_draw_order(self):
        self.reset(["Strike_R", "Defend_R", "Bash", "Rage", "Anger", "Shrug It Off", "Power Through", "Second Wind", "Body Slam"])
        self.env.public_state(seed=1)
        before = serialize_observation(self.env.get_state(), self.env.legal_actions(), version="observation_v7")
        first = self.env.search_model_actions(simulations=128, search_seed=7, public_state_seed=999, minimum_root_action_visits=8)
        self.env.public_state(seed=2)
        after = serialize_observation(self.env.get_state(), self.env.legal_actions(), version="observation_v7")
        second = self.env.search_model_actions(simulations=128, search_seed=7, public_state_seed=999, minimum_root_action_visits=8)
        self.assertEqual(before, after)
        self.assertEqual(first.model_actions, second.model_actions)
        self.assertEqual(first.native_result.hidden_order_fingerprint, second.native_result.hidden_order_fingerprint)


if __name__ == "__main__":
    unittest.main()
