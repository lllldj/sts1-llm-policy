"""Short native integration checks; never load a model or retained datasets."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from sts1_llm_policy.env.combat_snapshot import with_snapshot_relics
from sts1_llm_policy.env.simulator_client import SimulatorRemoteError
from sts1_llm_policy.env.simulator_search import parse_battle_scum_search_result
from sts1_llm_policy.env.simulator_adapter import from_sts_lightspeed_response
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.env.card_selection import CardSelectionState
from sts1_llm_policy.env.simulator_execution import SELECTION, resolve_simulator
from sts1_llm_policy.env.simulator_env import StsLightspeedEnv
from tests.simulator.native_support import NativeSelectionTestCase
from sts1_llm_policy.env.action_equivalence import build_model_action_space
from sts1_llm_policy.env.action_schema import ActionType
from sts1_llm_policy.policy.llm_policy import LLMPolicy
from sts1_llm_policy.env.route_context import CombatRouteContext, RouteOperation


def v7_text(decision):
    context = CombatRouteContext(1, 2, "normal_weak", "the_guardian", (RouteOperation("combat", "boss"),))
    return serialize_observation(replace(decision.state, route_context=context), decision.actions, version="observation_v7")


class NativeCardSelectionTests(NativeSelectionTestCase):
    def setUp(self):
        self.client = resolve_simulator(required_capabilities={SELECTION}).create_client()
        self.addCleanup(self.client.close)
        self.exhaust_resolution_checks = []

    def snapshot(self, cards):
        self.client.reward_reset(712, ascension=0, act=1, relics=[])
        snapshot = self.client.export_combat_snapshot()["combat_snapshot"]
        snapshot["deck"] = [{"string_id": name, "upgraded": upgraded,
                             **({"upgrade_count": int(upgraded)} if name == "Searing Blow" else {})}
                            for name, upgraded in cards]
        return with_snapshot_relics(snapshot, [])

    def reset(self, cards):
        return self.client.reset("jaw_worm", 712, combat_snapshot=self.snapshot(cards))

    def test_draw_and_secondary_selection_expose_fresh_legal_decisions(self):
        env = StsLightspeedEnv(self.client, allow_card_selection=True)
        for card in ("Battle Trance", "Armaments", "Warcry"):
            with self.subTest(card=card):
                snapshot = self.snapshot([(name, False) for name in (
                    card, "Strike_R", "Defend_R", "Bash", "Defend_R",
                    "Strike_R", "Defend_R", "Strike_R")])
                for seed in range(712, 728):
                    env.reset("jaw_worm", seed, combat_snapshot=snapshot)
                    action = next((a for a in env.legal_actions() if a.card_name == card), None)
                    if action is not None:
                        break
                else:
                    self.fail("No opening hand for information-boundary fixture")
                before = env.get_raw_state()
                state = env.step(action)
                after = env.get_raw_state()
                text = serialize_observation(state, env.legal_actions(), version="observation_v6")
                if card == "Battle Trance":
                    self.assertGreater(len(after["state"]["hand"]), len(before["state"]["hand"]))
                    self.assertEqual(after["state"]["input_state"], "PLAYER_NORMAL")
                else:
                    self.assertIsInstance(state, CardSelectionState)
                    self.assertIn("CARD_SELECTION:", text)
                self.assertNotEqual(after["state"]["decision_id"], before["state"]["decision_id"])
                result = env.search_model_actions(simulations=64, search_seed=101)
                legal = build_model_action_space(state, env.legal_actions()).action_ids
                self.assertIn(result.suggested_model_action_id, legal)
                self.assertEqual(env.get_raw_state(), after)

    def test_armaments_plus_skips_curse_and_already_upgraded_cards(self):
        response = self.reset([
            ("Armaments", True), ("AscendersBane", False),
            ("Defend_R", False), ("Strike_R", True), ("Bash", False),
        ])
        response = self.play(response, "Armaments")
        decision = from_sts_lightspeed_response(response, allow_card_selection=True)
        versions = {card.card_id: card.upgrades for card in decision.state.combat.hand}
        self.assertEqual(versions, {"AscendersBane": 0, "Defend_R": 1, "Strike_R": 1, "Bash": 1})

    def test_armaments_plus_skips_generated_status_cards(self):
        for scenario, status_id in (("slime_boss", "Slimed"), ("three_sentries", "Dazed")):
            with self.subTest(scenario=scenario):
                response = self.client.reset(scenario, 712, combat_snapshot=self.snapshot([
                    ("Armaments", True), ("Armaments", True),
                    ("Defend_R", False), ("Defend_R", False), ("Defend_R", False),
                ]))
                checked = False
                for _ in range(12):
                    cards = response["state"]["hand"]
                    if any(card["string_id"] == status_id for card in cards):
                        response = self.play(response, "Armaments")
                        decision = from_sts_lightspeed_response(response, allow_card_selection=True)
                        statuses = [card for card in decision.state.combat.hand if card.card_id == status_id]
                        self.assertTrue(statuses)
                        self.assertTrue(all(card.upgrades == 0 for card in statuses))
                        checked = True
                        break
                    for _ in range(3):
                        if not any(a.get("card_name", "").rstrip("+") == "Defend" for a in response["legal_actions"]):
                            break
                        response = self.play(response, "Defend")
                    response = self.client.step(response["state"]["decision_id"], "END")
                self.assertTrue(checked, "Expected generated status and Armaments in the same hand")

    def play(self, response, name):
        action = next(a for a in response["legal_actions"] if a.get("card_name", "").rstrip("+") == name)
        return self.client.step(response["state"]["decision_id"], action["action_id"])

    def select(self, response, task):
        decision = from_sts_lightspeed_response(response, allow_card_selection=True)
        self.assertIsInstance(decision.state, CardSelectionState)
        self.assertEqual(decision.state.selection_task, task)
        self.assertTrue(all(a["kind"] == "SELECT_CARD" for a in response["legal_actions"]))
        text = serialize_observation(decision.state, decision.actions, version="observation_v6")
        self.assertIn("CARD_SELECTION:", text)
        self.assertIn(" SELECT ", text)
        current_text = v7_text(decision)
        self.assertIn("SOURCE_CARD:", current_text)
        if task == "ARMAMENTS":
            self.assertIn("AFTER_UPGRADE:", current_text)
        if task == "EXHAUST_ONE":
            self.assertIn("PENDING:", current_text)
            if decision.state.source_card.card_id == "Burning Pact":
                count = 3 if decision.state.source_card.upgrades else 2
                self.assertIn(f"then attempt to draw {count} cards", current_text)
        with self.assertRaises(ValueError):
            serialize_observation(decision.state, decision.actions, version="observation_v5")
        with self.assertRaises(SimulatorRemoteError):
            self.client.step(decision.decision_id, "END")
        after = self.client.step(decision.decision_id, response["legal_actions"][0]["action_id"])
        with self.assertRaises(SimulatorRemoteError):
            self.client.step(decision.decision_id, response["legal_actions"][0]["action_id"])
        self.assertEqual(after["state"]["input_state"], "PLAYER_NORMAL")
        return after

    def test_all_seven_cards_and_upgrades(self):
        for name in ("Armaments", "Headbutt", "Exhume", "Burning Pact", "True Grit", "Warcry", "Dual Wield"):
            for upgraded in (False, True):
                with self.subTest(card=name, upgraded=upgraded):
                    support = ["Strike_R", "Defend_R", "Bash", "Defend_R"]
                    if name == "Exhume":
                        support = ["Disarm", "Intimidate", "Strike_R", "Defend_R"]
                    response = self.reset([(name, upgraded)] + [(x, False) for x in support])
                    if name == "Headbutt":
                        response = self.play(response, "Defend")
                        response = self.play(response, "Strike")
                    if name == "Exhume":
                        response = self.play(response, "Disarm")
                        response = self.play(response, "Intimidate")
                    response = self.play(response, name)
                    if name == "Armaments" and upgraded:
                        self.assertTrue(all(c["upgrades"] == 1 for c in response["state"]["hand"]))
                        self.assertEqual(response["state"]["player"]["block"], 5)
                        continue
                    if name == "True Grit" and not upgraded:
                        self.assertEqual(len(response["state"]["exhaust_pile"]), 1)
                        self.assertEqual(response["state"]["input_state"], "PLAYER_NORMAL")
                        continue
                    task = {"Burning Pact": "EXHAUST_ONE", "True Grit": "EXHAUST_ONE",
                            "Dual Wield": "DUAL_WIELD"}.get(name, name.upper())
                    source = {"HEADBUTT": "discard_pile", "EXHUME": "exhaust_pile"}.get(task, "hand")
                    index = response["legal_actions"][0]["selection_index"]
                    selected = deepcopy(response["state"][source][index])
                    after = self.select(response, task)
                    state = after["state"]
                    if task == "ARMAMENTS":
                        self.assertEqual(sum(c["upgrades"] for c in state["hand"]), 1)
                    elif task in ("HEADBUTT", "WARCRY"):
                        self.assertEqual(state["draw_pile"][-1]["string_id"], selected["string_id"])
                    elif task == "EXHUME":
                        self.assertTrue(any(c["string_id"] == selected["string_id"] for c in state["hand"]))
                    elif task == "EXHAUST_ONE":
                        self.assertTrue(any(c["string_id"] == selected["string_id"] for c in state["exhaust_pile"]))
                    elif task == "DUAL_WIELD":
                        self.assertEqual(len(state["hand"]), len(response["state"]["hand"]) + 1 + int(upgraded))

    def test_empty_and_single_candidate_auto_resolution(self):
        for name in ("Armaments", "Headbutt", "Exhume", "Burning Pact", "True Grit", "Warcry", "Dual Wield"):
            for support in ([], [("Strike_R", False)]):
                with self.subTest(card=name, support=support):
                    response = self.reset([(name, name == "True Grit")] + support)
                    response = self.play(response, name)
                    # Some effects can create a one-choice screen; that is a
                    # legitimate decision, not a reason to issue END_TURN.
                    if response["state"]["input_state"] == "CARD_SELECT":
                        self.select(response, response["state"]["card_selection"]["task"])
                    else:
                        if response["state"]["terminal"]:
                            self.assertEqual(response["legal_actions"], [])
                        else:
                            self.assertEqual(response["state"]["input_state"], "PLAYER_NORMAL")

    def test_teacher_can_search_selection_without_mutating_it(self):
        response = self.reset([("Armaments", False), ("Bash", False), ("Strike_R", False)])
        response = self.play(response, "Armaments")
        before = self.client.legal_actions()
        result = self.client.search(response["state"]["decision_id"], simulations=16, search_seed=42)
        parse_battle_scum_search_result(
            result, current_decision_id=response["state"]["decision_id"],
            current_native_action_ids=[a["action_id"] for a in response["legal_actions"]],
        )
        self.assertIn(result["suggested_action_id"], [a["action_id"] for a in response["legal_actions"]])
        after = self.client.legal_actions()
        self.assertEqual(before["decision_id"], after["decision_id"])
        self.assertEqual(before["legal_actions"], after["legal_actions"])

    def test_armaments_repeated_searing_blow_upgrade(self):
        response = self.reset([("Armaments", False), ("Armaments", False),
                               ("Searing Blow", False), ("Bash", False)])
        for _ in range(2):
            response = self.play(response, "Armaments")
            index = next(i for i, c in enumerate(response["state"]["hand"]) if c["string_id"] == "Searing Blow")
            preview = response["state"]["hand_upgrade_previews"][index]
            self.assertIn("AFTER_UPGRADE:", v7_text(from_sts_lightspeed_response(response, allow_card_selection=True)))
            response = self.client.step(response["state"]["decision_id"], f"SELECT:{index}")
            actual = next(c for c in response["state"]["hand"] if c["string_id"] == "Searing Blow")
            self.assertEqual(actual["special_data"], preview["special_data"])
            self.assertEqual(actual["upgrades"], preview["upgrades"])

        decision = from_sts_lightspeed_response(response, allow_card_selection=True)
        self.assertIn("Deal 21 base damage", serialize_observation(decision.state, decision.actions, version="observation_v6"))

    def test_armaments_previews_match_execution_across_supported_reward_cards(self):
        from sts1_llm_policy.workflows.continuous_plan import (
            load_scope,
        )
        root = Path(__file__).resolve().parents[2]
        scope = load_scope(root / "configs/env/d_expansion_card_selection_v1_scope.json", project_root=root)
        from sts1_llm_policy.env.simulator_adapter import _CARD_ENUM_TO_ID
        for enum_id in scope["reward_cards"]["included_enum_ids"]:
            card_id = _CARD_ENUM_TO_ID[enum_id]
            with self.subTest(card=card_id):
                response = self.reset([("Armaments", False), (card_id, False), ("Defend_R", False)])
                initial = from_sts_lightspeed_response(response, allow_card_selection=True)
                self.assertIn("UPGRADE_RESULT:", v7_text(initial))
                self.assertNotIn("UPGRADE_RESULT:", serialize_observation(initial.state, initial.actions, version="observation_v6"))
                response = self.play(response, "Armaments")
                index = next(i for i, c in enumerate(response["state"]["hand"]) if c["string_id"] == card_id)
                preview = response["state"]["hand_upgrade_previews"][index]
                self.assertIsNotNone(preview)
                decision = from_sts_lightspeed_response(response, allow_card_selection=True)
                self.assertIn("AFTER_UPGRADE:", v7_text(decision))
                with self.assertRaisesRegex(ValueError, "native upgrade previews"):
                    v7_text(replace(decision, state=replace(decision.state, hand_upgrade_previews=())))
                after = self.client.step(response["state"]["decision_id"], f"SELECT:{index}")
                actual = next(c for c in after["state"]["hand"] if c["string_id"] == card_id and c["upgrades"] == 1)
                for key in ("enum_id", "string_id", "name", "type", "cost", "cost_for_turn", "upgrades", "special_data", "exhausts", "ethereal", "has_target"):
                    self.assertEqual(actual[key], preview[key], key)

    def test_upgraded_armaments_displays_and_executes_all_remaining_upgrades(self):
        response = self.reset([("Armaments", True), ("Corruption", False), ("True Grit", False), ("Sentinel", False)])
        decision = from_sts_lightspeed_response(response, allow_card_selection=True)
        text = v7_text(decision)
        self.assertIn("UPGRADE_MODE: ALL", text)
        self.assertIn("Corruption | TYPE POWER | COST 2 | UPGRADE 1", text)
        self.assertIn("Choose a card in your hand to Exhaust", text)
        self.assertIn("gain 3 Energy", text)
        after = self.play(response, "Armaments")
        self.assertTrue(all(c["upgrades"] == 1 for c in after["state"]["hand"]))

    def test_exhume_excludes_other_exhumes(self):
        response = self.reset([("Exhume", False), ("Exhume", False), ("Disarm", False),
                               ("Intimidate", False), ("Strike_R", False)])
        for name in ("Exhume", "Disarm", "Intimidate", "Exhume"):
            response = self.play(response, name)
        self.assertEqual(response["state"]["input_state"], "CARD_SELECT")
        for action in response["legal_actions"]:
            card = response["state"]["exhaust_pile"][action["selection_index"]]
            self.assertNotEqual(card["string_id"], "Exhume")

    def test_lethal_headbutt_does_not_request_a_choice(self):
        response = self.reset([("Headbutt", False), ("Bludgeon", True), ("Seeing Red", True),
                               ("Defend_R", False), ("Strike_R", False)])
        for name in ("Seeing Red", "Bludgeon", "Headbutt"):
            response = self.play(response, name)
        self.assertTrue(response["state"]["terminal"])
        self.assertEqual(response["state"]["outcome"], "PLAYER_VICTORY")
        self.assertEqual(response["legal_actions"], [])

    def test_exhaust_zero_single_and_multiple_candidates_preserve_effects(self):
        for name in ("Burning Pact", "True Grit"):
            for upgraded in (False, True):
                for candidates in (0, 1, 2):
                    for deck_size in (5, 10):
                        with self.subTest(card=name, upgraded=upgraded, candidates=candidates, deck_size=deck_size):
                            # At seed 712 both decks draw the tested card and
                            # four Angers. Playing Anger costs no Energy and
                            # supplies a nonempty discard pile for reshuffling.
                            response = self.reset([(name, upgraded)] + [("Anger", False)] * (deck_size - 1))
                            for _ in range(4 - candidates):
                                response = self.play(response, "Anger")
                            before = deepcopy(response["state"])
                            self.assertEqual(len(before["hand"]), 1 + candidates)
                            self.assertEqual(before["exhaust_pile"], [])
                            self.assertEqual(len(before["draw_pile"]), deck_size - 5)
                            response = self.play(response, name)
                            choose_required = candidates > 1 and (name == "Burning Pact" or upgraded)
                            if choose_required:
                                self.assertEqual(response["state"]["input_state"], "CARD_SELECT")
                                self.assertEqual(len(response["legal_actions"]), candidates)
                                self.assertEqual(response["state"]["exhaust_pile"], [])
                                # Burning Pact's draw must wait until the
                                # required choice is resolved, not precede it.
                                self.assertEqual(len(response["state"]["hand"]), candidates)
                                self.assertEqual(len(response["state"]["draw_pile"]), len(before["draw_pile"]))
                                response = self.select(response, "EXHAUST_ONE")
                            else:
                                # Do not tolerate a needless one-choice screen.
                                self.assertEqual(response["state"]["input_state"], "PLAYER_NORMAL")
                                self.assertNotIn("card_selection", response["state"])
                                self.assertTrue(all(a["kind"] != "SELECT_CARD" for a in response["legal_actions"]))
                            after = response["state"]
                            burned = int(candidates > 0)
                            drawn = 2 + int(upgraded) if name == "Burning Pact" else 0
                            block = (9 if upgraded else 7) if name == "True Grit" else 0
                            self.assertFalse(after["terminal"])
                            self.assertEqual(after["turn"], before["turn"])
                            self.assertEqual(after["decision_id"] - before["decision_id"], 1 + int(choose_required))
                            self.assertEqual(len(after["exhaust_pile"]), burned)
                            self.assertEqual(len(after["hand"]), candidates - burned + drawn)
                            self.assertEqual(after["player"]["energy"], before["player"]["energy"] - 1)
                            self.assertEqual(after["player"]["block"], before["player"]["block"] + block)
                            self.assertEqual(after["player"]["current_hp"], before["player"]["current_hp"])
                            if deck_size == 10:
                                self.assertEqual(len(after["draw_pile"]), len(before["draw_pile"]) - drawn)
                            decision = from_sts_lightspeed_response(response, allow_card_selection=True)
                            serialize_observation(decision.state, decision.actions, version="observation_v6")
                            self.exhaust_resolution_checks.append({
                                "card": name, "upgraded": upgraded, "candidate_count": candidates,
                                "initial_draw_pile": deck_size - 5, "choice_required": choose_required,
                                "cards_exhausted": burned, "cards_drawn": drawn, "block_gained": block,
                                "energy_spent": 1, "final_hand_count": len(after["hand"]),
                                "final_input_state": after["input_state"],
                            })

    def test_no_exhaust_does_not_trigger_feel_no_pain(self):
        for candidates in (0, 1):
            with self.subTest(candidates=candidates):
                response = self.reset([("Burning Pact", False), ("Feel No Pain", False)] + [("Anger", False)] * 3)
                response = self.play(response, "Feel No Pain")
                for _ in range(3 - candidates):
                    response = self.play(response, "Anger")
                response = self.play(response, "Burning Pact")
                after = response["state"]
                self.assertEqual(after["input_state"], "PLAYER_NORMAL")
                self.assertEqual(len(after["hand"]), 2)
                self.assertEqual(len(after["exhaust_pile"]), candidates)
                self.assertEqual(after["player"]["block"], 3 * candidates)
                self.exhaust_resolution_checks.append({
                    "card": "Burning Pact", "candidate_count": candidates,
                    "power": "Feel No Pain", "cards_exhausted": candidates,
                    "cards_drawn": 2, "block_gained": 3 * candidates,
                })

    def test_automatic_single_exhaust_triggers_sentinel_energy(self):
        response = self.reset([("Burning Pact", False), ("Sentinel", False)] + [("Anger", False)] * 3)
        for _ in range(3):
            response = self.play(response, "Anger")
        response = self.play(response, "Burning Pact")
        after = response["state"]
        self.assertEqual(after["input_state"], "PLAYER_NORMAL")
        self.assertEqual([c["string_id"] for c in after["exhaust_pile"]], ["Sentinel"])
        self.assertEqual(after["player"]["energy"], 4)  # 3 - Pact's 1 + Sentinel's 2.
        self.assertEqual(len(after["hand"]), 2)
        self.exhaust_resolution_checks.append({
            "card": "Burning Pact", "candidate_count": 1, "selected_card": "Sentinel",
            "cards_exhausted": 1, "cards_drawn": 2, "final_energy": 4,
        })

    def test_canonical_environment_needs_no_secondary_policy_call_for_zero_or_one(self):
        class Backend:
            calls = 0
            output = ""

            def generate(self, request):
                self.calls += 1
                return self.output

        for candidates in (0, 1):
            with self.subTest(candidates=candidates):
                env = StsLightspeedEnv(self.client, allow_card_selection=True)
                env.reset("jaw_worm", 712, combat_snapshot=self.snapshot(
                    [("Burning Pact", False)] + [("Anger", False)] * 4
                ))
                for _ in range(4 - candidates):
                    env.step(next(a for a in env.legal_actions() if a.card_name == "Anger"))
                state = env.get_state()
                actions = env.legal_actions()
                action_space = build_model_action_space(state, actions)
                backend = Backend()
                backend.output = next(c.model_action_id for c in action_space.classes
                                      if c.card is not None and c.card.card_id == "Burning Pact")
                result = LLMPolicy(backend, 7, observation_version="observation_v6").select_action(state, actions)
                after = env.step(result.action)
                self.assertEqual(backend.calls, 1)
                self.assertNotIsInstance(after, CardSelectionState)
                self.assertEqual(len(after.combat.hand), 2)
                self.assertEqual(len(after.combat.exhaust_pile), candidates)
                self.assertTrue(all(a.action_type != ActionType.SELECT_CARD for a in env.legal_actions()))
                self.exhaust_resolution_checks.append({
                    "path": "LLMPolicy_stub -> StsLightspeedEnv -> native bridge -> canonical state",
                    "candidate_count": candidates, "policy_calls": backend.calls,
                    "cards_exhausted": candidates, "cards_drawn": 2, "secondary_decision": False,
                })
