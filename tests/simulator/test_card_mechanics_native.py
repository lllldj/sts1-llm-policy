"""Short card-rule regressions; reference: the installed game's card classes.

Regression expectations come from independently reproduced audit findings.
These checks do not require the proprietary game JAR at runtime.
"""
from dataclasses import replace

from sts1_llm_policy.env.combat_snapshot import resign_snapshot, with_snapshot_relics
from sts1_llm_policy.env.simulator_execution import SELECTION, resolve_simulator
from sts1_llm_policy.env.simulator_client import SimulatorRemoteError
from sts1_llm_policy.env.simulator_adapter import from_sts_lightspeed_response
from sts1_llm_policy.env.live_description_fallback import resolve_card_effect
from sts1_llm_policy.env.serializer import serialize_observation
from sts1_llm_policy.eval.reward_support import reward_state_snapshot
from tests.simulator.native_support import NativeSelectionTestCase


# Normal/upgraded base costs, checked against red-card constructors and upgrade()
# in the game JAR. Whirlwind uses the native X-cost sentinel.
BASE_COST_GROUPS = {
    -1: "Whirlwind",
    0: "Anger|Battle Trance|Berserk|Bloodletting|Brutality|Clash|Flex|Intimidate|Offering|Rage|Reckless Charge|Warcry",
    1: "Armaments|Body Slam|Burning Pact|Cleave|Combust|Defend_R|Disarm|Double Tap|Dropkick|Dual Wield|Evolve|Exhume|Feed|Feel No Pain|Fire Breathing|Ghostly Armor|Havoc|Headbutt|Hemokinesis|Inflame|Iron Wave|Limit Break|Metallicize|Pommel Strike|Power Through|Pummel|Rampage|Rupture|Second Wind|Seeing Red|Sentinel|Shrug It Off|Spot Weakness|Strike_R|Sword Boomerang|Thunderclap|True Grit|Twin Strike|Wild Strike",
    2: "Bash|Carnage|Clothesline|Dark Embrace|Entrench|Fiend Fire|Flame Barrier|Heavy Blade|Immolate|Impervious|Juggernaut|Perfected Strike|Reaper|Searing Blow|Sever Soul|Shockwave|Uppercut",
    3: "Barricade|Bludgeon|Corruption|Demon Form",
    4: "Blood for Blood",
}
UPGRADE_COSTS = {
    "Body Slam": 0, "Exhume": 0, "Havoc": 0, "Seeing Red": 0,
    "Dark Embrace": 1, "Entrench": 1, "Barricade": 2, "Corruption": 2,
    "Blood for Blood": 3,
}


class CardFixture:
    def setUp(self):
        self.client = resolve_simulator(required_capabilities={SELECTION}).create_client()
        self.addCleanup(self.client.close)

    def reset(self, cards, relics=(), scenario="jaw_worm"):
        self.client.reward_reset(712, ascension=0, act=1, relics=[])
        snapshot = self.client.export_combat_snapshot()["combat_snapshot"]
        snapshot["deck"] = [{"string_id": name, "upgraded": up,
                             **({"upgrade_count": int(up)} if name == "Searing Blow" else {})}
                            for name, up in cards]
        return self.client.reset(scenario, 712, combat_snapshot=with_snapshot_relics(snapshot, list(relics)))

    def play(self, response, name):
        action = next(a for a in response["legal_actions"] if a.get("card_name", "").rstrip("+") == name)
        return self.client.step(response["state"]["decision_id"], action["action_id"])

    @staticmethod
    def card(response, name):
        return next(c for c in response["state"]["hand"] if c["string_id"] == name)


class NativeCardMechanicsTests(CardFixture, NativeSelectionTestCase):
    def test_snapshot_signing_matches_native_export_and_relic_reset(self):
        self.client.reward_reset(712, ascension=0, act=1,
                                 relics=[{"id": "NUNCHAKU", "counter": 6}])
        snapshot = self.client.export_combat_snapshot()["combat_snapshot"]
        self.assertEqual(resign_snapshot(snapshot), snapshot)
        replacement = with_snapshot_relics(snapshot, [{"id": "NUNCHAKU", "counter": 9}])
        response = self.client.reset("jaw_worm", 712, combat_snapshot=replacement)
        counter = next(r["counter"] for r in response["state"]["relics"] if r["id"] == "NUNCHAKU")
        self.assertEqual(counter, 9)
        response = self.client.reset("jaw_worm", 712, combat_snapshot=with_snapshot_relics(snapshot, []))
        self.assertEqual(response["state"]["relics"], [])

    def test_supported_card_base_costs_and_upgraded_costs(self):
        costs = {name: cost for cost, names in BASE_COST_GROUPS.items() for name in names.split("|")}
        for up in (False, True):
            response = self.reset([(name, up) for name in costs])
            cards = [c for pile in ("hand", "draw_pile") for c in response["state"][pile]]
            self.assertCountEqual([c["string_id"] for c in cards], costs)
            for c in cards:
                name = c["string_id"]
                with self.subTest(card=name, upgraded=up):
                    expected = UPGRADE_COSTS.get(name, costs[name]) if up else costs[name]
                    self.assertEqual(c["cost"], expected)
                    self.assertEqual(c["cost_for_turn"], expected)
                    self.assertEqual(c["upgrades"], int(up))
        # Also exercise in-combat upgrade, independently of snapshot restoration.
        response = self.reset([("Searing Blow", False), ("Armaments", True)])
        response = self.play(response, "Armaments")
        card = self.card(response, "Searing Blow")
        self.assertEqual((card["upgrades"], card["cost"], card["cost_for_turn"]), (1, 2, 2))

    def test_rage_cost_block_order_and_expiry(self):
        for up, block in ((False, 3), (True, 5)):
            for rage_first in (True, False):
                with self.subTest(upgraded=up, rage_first=rage_first):
                    response = self.reset([("Rage", up)] + [("Strike_R", False)] * 4)
                    self.assertEqual(self.card(response, "Rage")["cost_for_turn"], 0)
                    if not rage_first:
                        response = self.play(response, "Strike")
                    before = response["state"]["player"]["energy"]
                    response = self.play(response, "Rage")
                    self.assertEqual(response["state"]["player"]["energy"], before)
                    for _ in range(before):
                        response = self.play(response, "Strike")
                    self.assertEqual(response["state"]["player"]["block"], before * block)
                    response = self.client.step(response["state"]["decision_id"], "END")
                    response = self.play(response, "Strike")
                    self.assertEqual(response["state"]["player"]["block"], 0)

    def test_rage_is_playable_at_zero_energy_and_not_exhausted(self):
        for up in (False, True):
            with self.subTest(upgraded=up):
                response = self.reset([("Rage", up)] + [("Defend_R", False)] * 4)
                for _ in range(3):
                    response = self.play(response, "Defend")
                self.assertEqual(response["state"]["player"]["energy"], 0)
                response = self.play(response, "Rage")
                self.assertEqual(response["state"]["player"]["energy"], 0)
                self.assertTrue(any(c["string_id"] == "Rage" for c in response["state"]["discard_pile"]))
                self.assertEqual(response["state"]["exhaust_pile"], [])

    def test_rage_stacks_and_triggers_per_card_not_per_hit(self):
        response = self.reset([("Rage", False), ("Rage", True), ("Twin Strike", False), ("Defend_R", False)])
        response = self.play(self.play(response, "Rage"), "Rage")
        response = self.play(response, "Twin Strike")
        self.assertEqual(response["state"]["player"]["block"], 8)

    def test_upgrading_rage_in_hand_keeps_zero_cost(self):
        response = self.reset([("Armaments", True), ("Rage", False), ("Strike_R", False)])
        response = self.play(response, "Armaments")
        self.assertEqual(self.card(response, "Rage")["upgrades"], 1)
        self.assertEqual(self.card(response, "Rage")["cost_for_turn"], 0)
        response = self.play(self.play(response, "Rage"), "Strike")
        self.assertEqual(response["state"]["player"]["block"], 10)

    def test_direct_upgraded_disarm_applies_exactly_three_strength(self):
        response = self.play(self.reset([("Disarm", True)]), "Disarm")
        strength = next(p["amount"] for p in response["state"]["monsters"][0]["powers"] if p["name"] == "Strength")
        self.assertEqual(strength, -3)


class FixedCardMechanicsRegressions(CardFixture, NativeSelectionTestCase):
    def test_iron_wave_applies_dexterity_once(self):
        for up in (False, True):
            with self.subTest(upgraded=up):
                response = self.reset([("Iron Wave", up)], [{"id": "ODDLY_SMOOTH_STONE", "counter": -1}])
                response = self.play(response, "Iron Wave")
                self.assertEqual(response["state"]["player"]["block"], 8 if up else 6)

    def test_double_tap_should_discard_after_play(self):
        for up in (False, True):
            with self.subTest(upgraded=up):
                response = self.reset([("Double Tap", up), ("Strike_R", False), ("Strike_R", False)])
                decision = from_sts_lightspeed_response(response, allow_card_selection=True)
                card = next(c for c in decision.state.combat.hand if c.card_id == "Double Tap")
                self.assertNotIn("EXHAUST", resolve_card_effect(card))
                self.assertIn("EXHAUST", resolve_card_effect(replace(card, exhausts=True)))
                response = self.play(response, "Double Tap")
                self.assertEqual([c["string_id"] for c in response["state"]["discard_pile"]], ["Double Tap"])
                hp = response["state"]["monsters"][0]["current_hp"]
                response = self.play(self.play(response, "Strike"), "Strike")
                self.assertEqual(hp - response["state"]["monsters"][0]["current_hp"], 24 if up else 18)

    def test_double_tap_still_exhausts_under_corruption(self):
        response = self.reset([("Corruption", True), ("Double Tap", False), ("Strike_R", False)])
        response = self.play(self.play(response, "Corruption"), "Double Tap")
        self.assertEqual([c["string_id"] for c in response["state"]["exhaust_pile"]], ["Double Tap"])

    def test_blood_for_blood_upgrade_keeps_damage_discount(self):
        for losses in (0, 1, 2, 3):
            with self.subTest(losses=losses):
                response = self.reset([("Blood for Blood", False), ("Armaments", True)] + [("Bloodletting", False)] * 3)
                for _ in range(losses):
                    response = self.play(response, "Bloodletting")
                self.assertEqual(self.card(response, "Blood for Blood")["cost"], 4 - losses)
                response = self.play(response, "Armaments")
                self.assertEqual(self.card(response, "Blood for Blood")["cost"], 3 - losses)
                if losses:
                    before = response["state"]["player"]["energy"]
                    response = self.play(response, "Blood for Blood")
                    self.assertEqual(response["state"]["player"]["energy"], before - (3 - losses))

    def test_snapshot_preserves_searing_blow_upgrade(self):
        # V2 restores the persistent upgrade count into Card::misc.
        response = self.reset([("Searing Blow", True)])
        self.assertEqual(self.card(response, "Searing Blow")["upgrades"], 1)

    def test_havoc_played_disarm_plus_reduces_three_strength(self):
        # The bridge correction only recognizes Disarm as the root hand action.
        response = self.reset([
            ("Havoc", False), ("Defend_R", False), ("Defend_R", False),
            ("Disarm", True), ("Defend_R", False), ("Defend_R", False),
        ])
        self.assertEqual(response["state"]["draw_pile"][0]["string_id"], "Disarm")
        response = self.play(response, "Havoc")
        strength = next(p["amount"] for p in response["state"]["monsters"][0]["powers"] if p["name"] == "Strength")
        self.assertEqual(strength, -3)

    def test_disarm_plus_respects_artifact(self):
        response = self.reset([("Disarm", True)], scenario="three_sentries")
        response = self.play(response, "Disarm")
        powers = response["state"]["monsters"][0]["powers"]
        self.assertFalse(any(p["name"] in {"Artifact", "Strength"} for p in powers))

    def test_reward_upgrade_snapshot_roundtrip_and_counterfactuals(self):
        self.client.reward_reset(32, ascension=0, act=1, relics=[])
        reward = self.client.sample_card_reward(room="MONSTER", floor=5)
        choice = next(c for c in reward["choices"] if c["string_id"] == "Searing Blow")
        candidate = reward_state_snapshot(reward, choice["choice"])
        self.assertEqual(candidate["schema_version"], "combat_snapshot_v2")
        self.assertEqual(candidate["deck"][-1]["upgrade_count"], 0)
        result = self.client.apply_card_reward_choice(reward["reward_id"], choice["choice"])
        index = next(i for i, c in enumerate(result["reward_state"]["deck"]) if c["string_id"] == "Searing Blow")
        for count, damage in ((1, 16), (2, 21), (3, 27)):
            with self.subTest(upgrade_count=count):
                result = self.client.upgrade_reward_card(index)
                self.assertEqual(result["upgraded_card_after"]["upgrade_count"], count)
                snapshot = self.client.export_combat_snapshot()["combat_snapshot"]
                self.assertEqual(snapshot["deck"][index]["upgrade_count"], count)
                candidate = reward_state_snapshot({"reward_state": result["reward_state"], "choices": [], "reward_id": count}, "SKIP")
                self.assertEqual(candidate["schema_version"], "combat_snapshot_v2")
                self.assertEqual(candidate["deck"][index]["upgrade_count"], count)
                response = self.client.reset("jaw_worm", 712, combat_snapshot=snapshot)
                cards = response["state"]["hand"] + response["state"]["draw_pile"]
                self.assertEqual(next(c["upgrades"] for c in cards if c["string_id"] == "Searing Blow"), count)
                snapshot["deck"] = [snapshot["deck"][index]]
                response = self.client.reset("jaw_worm", 712, combat_snapshot=with_snapshot_relics(snapshot, []))
                decision = from_sts_lightspeed_response(response, allow_card_selection=True)
                self.assertTrue(decision.loadout_id.startswith("combat_snapshot_v2:"))
                text = serialize_observation(decision.state, decision.actions, version="observation_v6")
                self.assertIn(f"Deal {damage} base damage.", text)
                hp = response["state"]["monsters"][0]["current_hp"]
                response = self.play(response, "Searing Blow")
                self.assertEqual(hp - response["state"]["monsters"][0]["current_hp"], damage)

    def test_v2_rejects_missing_invalid_or_inconsistent_counts(self):
        self.client.reward_reset(712, ascension=0, act=1, relics=[])
        snapshot = self.client.export_combat_snapshot()["combat_snapshot"]
        for count in (None, True, -1, 0, 1.5, 32768):
            with self.subTest(count=count):
                snapshot["deck"] = [{"string_id": "Searing Blow", "upgraded": True}]
                if count is not None:
                    snapshot["deck"][0]["upgrade_count"] = count
                with self.assertRaises(SimulatorRemoteError):
                    self.client.reset("jaw_worm", 712, combat_snapshot=with_snapshot_relics(snapshot, []))

    def test_v1_snapshot_keeps_historical_identity_and_count_semantics(self):
        self.client.reward_reset(712, ascension=0, act=1, relics=[])
        snapshot = self.client.export_combat_snapshot()["combat_snapshot"]
        snapshot["schema_version"] = "combat_snapshot_v1"
        snapshot["deck"] = [{"string_id": "Searing Blow", "upgraded": True}]
        response = self.client.reset("jaw_worm", 712, combat_snapshot=with_snapshot_relics(snapshot, []))
        self.assertEqual(self.card(response, "Searing Blow")["upgrades"], 0)
        self.assertEqual(response["state"]["loadout"]["source"], "combat_snapshot_v1")
