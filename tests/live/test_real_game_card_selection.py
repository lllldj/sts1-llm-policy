from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from sts1_llm_policy.env.card_selection import build_selection_actions
from sts1_llm_policy.env.real_game_card_selection import (
    adapt_selection, CardSelectionRealGameEnv,
)
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.policy.llm_policy import LLMPolicy
from sts1_llm_policy.data.trajectory import TrajectoryLogger, iter_trajectory_records

from .fixtures import _raw_state, _raw_card, SequenceClient as FakeClient


def selection_fixture(action="ArmamentsAction", source="Armaments", upgraded=0):
    raw = _raw_state()
    game = raw["game_state"]
    game["combat_state"]["player"]["powers"] = []
    game["action_phase"] = "EXECUTING_ACTIONS"
    game["current_action"] = action
    grid = action in {"HeadbuttAction", "ExhumeAction"}
    game["screen_type"] = "GRID" if grid else "HAND_SELECT"
    cards = [{**_raw_card(), "uuid": f"choice-{i}", "is_playable": False} for i in range(2)]
    game["combat_state"]["card_in_play"] = {"id": source, "upgrades": upgraded}
    pile = "discard_pile" if action == "HeadbuttAction" else "exhaust_pile" if grid else "hand"
    game["combat_state"][pile] = deepcopy(cards)
    if grid:
        cards.reverse()  # UI order deliberately differs from source pile.
        game["screen_state"] = {"cards": cards, "selected_cards": [], "num_cards": 1,
                                "any_number": False, "for_upgrade": False,
                                "for_transform": False, "for_purge": False, "confirm_up": False}
    else:
        game["screen_state"] = {"hand": cards, "selected": [], "max_cards": 1, "can_pick_zero": False}
    game["choice_list"] = ["strike", "strike"]
    raw["available_commands"] = ["choose", "state"]
    return raw


class RealGameCardSelectionTests(unittest.TestCase):
    def test_seven_sources_use_selection_only_actions(self):
        cases = [("ArmamentsAction", "Armaments", 0), ("HeadbuttAction", "Headbutt", 0),
                 ("ExhumeAction", "Exhume", 1), ("ExhaustAction", "Burning Pact", 0),
                 ("ExhaustAction", "True Grit", 1), ("WarcryAction", "Warcry", 1),
                 ("DualWieldAction", "Dual Wield", 1)]
        for action, source, upgrades in cases:
            with self.subTest(source=source):
                state = adapt_selection(selection_fixture(action, source, upgrades))
                actions = build_selection_actions(state)
                self.assertEqual(len(actions), 2)
                text = serialize_observation(state, actions, version="observation_v6")
                self.assertIn("ACTION_0: SELECT Strike", text)
                self.assertNotIn("choice-0", text)
                self.assertNotIn("ACTION_2: END_TURN", text)
                if source == "Dual Wield":
                    self.assertIn("Create 2 copies", text)

    def test_duplicate_snapshots_and_confirm_are_acknowledged_once(self):
        raw = selection_fixture()
        selected = deepcopy(raw)
        card = selected["game_state"]["screen_state"]["hand"].pop(0)
        selected["game_state"]["screen_state"]["selected"] = [card]
        selected["available_commands"] = ["confirm", "state"]
        after = _raw_state()
        after["game_state"]["combat_state"]["hand"] = [{**card, "upgrades": 1}]
        after["available_commands"] = ["play", "end", "state"]
        client = FakeClient([raw, raw, selected, selected, after])
        env = CardSelectionRealGameEnv(client, sleeper=lambda _: None)
        env._update_state(raw)
        env.step(env.legal_actions()[0])
        self.assertEqual(client.commands, ["choose 0", "confirm"])
        self.assertEqual(env.get_raw_state()["card_selection_commands"], client.commands)

    def test_grid_uses_ui_index_instead_of_pile_index(self):
        raw = selection_fixture("HeadbuttAction", "Headbutt")
        after = _raw_state()
        after["available_commands"] = ["end"]
        client = FakeClient([after])
        env = CardSelectionRealGameEnv(client)
        env._update_state(raw)
        action = env.legal_actions()[0]
        self.assertEqual(action.selection_index, 1)
        env.step(action)
        self.assertEqual(client.commands, ["choose 0"])

    def test_unknown_optional_noncombat_and_stale_choices_rejected(self):
        raw = selection_fixture()
        variants = []
        for path, value in [("current_action", "UnknownAction"), ("room_phase", "EVENT")]:
            wrong = deepcopy(raw); wrong["game_state"][path] = value; variants.append(wrong)
        for key, value in [("max_cards", 2), ("max_cards", True), ("can_pick_zero", True)]:
            wrong = deepcopy(raw); wrong["game_state"]["screen_state"][key] = value; variants.append(wrong)
        wrong = deepcopy(raw); wrong["game_state"]["screen_state"]["hand"][0]["uuid"] = "missing"; variants.append(wrong)
        for wrong in variants:
            with self.assertRaises(ValueError):
                adapt_selection(wrong)
        env = CardSelectionRealGameEnv(FakeClient([])); env._update_state(raw)
        with self.assertRaises(ValueError):
            env.step(replace(env.legal_actions()[0], card_uuid="stale"))
        self.assertEqual(env._client.commands, [])
        with self.assertRaises(ValueError):
            env.step(replace(env.legal_actions()[0]))

    def test_selection_uses_shared_retry_and_fallback(self):
        class Backend:
            def generate(self, request):
                return "invalid"
        state = adapt_selection(selection_fixture())
        result = LLMPolicy(Backend(), 1, observation_version="observation_v6").select_action(
            state, build_selection_actions(state))
        self.assertTrue(result.fallback_used)
        self.assertIn(result.action, build_selection_actions(state))

    def test_v6_trajectory_preserves_choice_identity_and_can_resume(self):
        class Backend:
            def generate(self, request):
                return "ACTION_0"
        raw = selection_fixture()
        state = adapt_selection(raw)
        actions = build_selection_actions(state)
        result = LLMPolicy(Backend(), 1, observation_version="observation_v6").select_action(state, actions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "choice.jsonl"
            options = dict(episode_id="choice", game_seed=state.seed, policy_seed=1,
                           policy_name="test", observation_serializer_version="observation_v6",
                           evidence_class="coverage_collection")
            logger = TrajectoryLogger(path, **options)
            logger.log_transition(state=state, legal_actions=actions, policy_result=result,
                                  next_state=state, raw_state=raw)
            rows = list(iter_trajectory_records(path))
            self.assertEqual(rows[0]["action_equivalence_version"], "combat_card_selection_v1")
            self.assertIn("selection_index", rows[0]["action"])
            self.assertIn("selection_task", rows[0]["canonical_state"])
            TrajectoryLogger(path, resume=True, **options)

    def test_play_reaches_selection_and_old_play_snapshot_is_ignored(self):
        initial = _raw_state()
        initial["available_commands"] = ["play", "end", "state"]
        initial["game_state"]["combat_state"]["hand"] = [
            {**_raw_card(), "id": "Armaments", "name": "Armaments", "type": "SKILL", "has_target": False}
        ]
        selection = selection_fixture()
        client = FakeClient([initial, selection])
        env = CardSelectionRealGameEnv(client)
        env._update_state(initial)
        env.step(env.legal_actions()[0])
        self.assertEqual(client.commands, ["play 1"])
        self.assertEqual(env.get_state().selection_task, "ARMAMENTS")
