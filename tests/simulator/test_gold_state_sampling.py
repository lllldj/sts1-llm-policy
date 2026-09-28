import random
import unittest

from sts1_llm_policy.data.gold.state_sampling import (
    action_events,
    decision_features,
    sample_decisions,
    transition_events,
)


def card(name, cost=1, kind="SKILL", **extra):
    return {"card_id": name, "card_type": kind, "cost": cost, "upgrades": 0, "exhausts": False, **extra}


def record(hand, *, turn=1, energy=3, powers=(), selected=0):
    actions = [{"action_type": "play_card", "hand_index": i} for i in range(len(hand))]
    actions.append({"action_type": "end_turn"})
    return {"turn": turn, "canonical_state": {"combat": {
        "hand": hand, "player": {"energy": energy, "powers": list(powers)},
        "monsters": [], "draw_pile": [], "discard_pile": [], "exhaust_pile": []}},
        "raw_state": {"state": {"hand": [{"cost_for_turn": c["cost"]} for c in hand]}},
        "legal_actions": actions, "model_legal_actions": actions,
        "action": actions[selected]}


class GoldStateSamplingTests(unittest.TestCase):
    def test_opportunity_does_not_follow_teacher_choice_or_card_pairs(self):
        r = record([card("Defend_R"), card("Offering"), card("Rage")])
        f = decision_features([r])[0]
        self.assertEqual(f["opportunity_events"], ["energy_cost", "hand_access", "rule_change"])
        r["action"] = r["legal_actions"][-1]
        self.assertEqual(f, decision_features([r])[0])
        self.assertEqual(action_events(r["canonical_state"], r["legal_actions"][0]), set())

    def test_net_zero_energy_and_same_size_hand_still_detect_events(self):
        before = record([card("Dropkick", kind="ATTACK"), card("Defend_R")], energy=2)
        after = record([card("Strike_R", kind="ATTACK"), card("Defend_R")], energy=2)
        self.assertEqual(transition_events(before, after), {"energy_cost", "hand_access"})
        after["turn"] = 2
        self.assertEqual(transition_events(before, after), set())

    def test_dropkick_condition_and_no_draw(self):
        r = record([card("Dropkick", kind="ATTACK")])
        r["canonical_state"]["combat"]["monsters"] = [{"powers": []}]
        a = {**r["action"], "target_index": 0}
        self.assertEqual(action_events(r["canonical_state"], a), set())
        r["canonical_state"]["combat"]["monsters"][0]["powers"] = [{"power_id": "Vulnerable", "amount": 1}]
        r["canonical_state"]["combat"]["player"]["powers"] = [{"power_id": "NO_DRAW", "amount": 1}]
        self.assertEqual(action_events(r["canonical_state"], a), {"energy_cost"})

    def test_secondary_selection_and_self_exhaust_boundary(self):
        r = record([card("True Grit", upgrades=1), card("Defend_R")])
        after = record([])
        after["canonical_state"]["combat"]["exhaust_pile"] = [card("Defend_R")]
        self.assertIn("hand_exhaust", transition_events(r, after))
        own = record([card("Impervious", exhausts=True), card("Defend_R")])
        after = record([card("Defend_R")])
        after["canonical_state"]["combat"]["exhaust_pile"] = [card("Impervious")]
        self.assertNotIn("hand_exhaust", transition_events(own, after))
        r["canonical_state"]["selection_task"] = "EXHAUST_ONE"
        a = {"action_type": "select_card", "selection_index": 0}
        self.assertEqual(action_events(r["canonical_state"], a), {"secondary_selection", "hand_exhaust"})

    def test_upgrade_discount_and_rule_changes_ignore_identity_churn(self):
        before = record([card("Armaments"), card("Body Slam", uuid="old")])
        after = record([card("Body Slam", cost=0, upgrades=1, uuid="new")], energy=2)
        self.assertEqual(transition_events(before, after), {"upgrade_top", "energy_cost"})
        after["canonical_state"]["combat"]["player"]["powers"] = [{"power_id": "Rage", "amount": 3}]
        self.assertIn("rule_change", transition_events(before, after))

    def test_mixture_excludes_forced_and_weights_change_actual_selection(self):
        records = [record([card("Strike_R", kind="ATTACK")]),
                   record([card("Rage")]), record([]),
                   record([card("Defend_R")], turn=2)]
        a, coverage = sample_decisions(records, 9, random.Random(4))
        self.assertEqual(set(a), {0, 1, 3})
        self.assertEqual(coverage["excluded_single_action_decisions"], 1)
        self.assertTrue(all(0 < f["conditional_draw_probability"] <= 1 for f in a.values()))
        self.assertEqual((a, coverage), sample_decisions(records, 9, random.Random(4)))
        starts = events = 0
        for seed in range(100):
            s, _ = sample_decisions(records, 1, random.Random(seed), weights={"turn_start": .99, "event": 0, "random": .01})
            e, _ = sample_decisions(records, 1, random.Random(seed), weights={"turn_start": 0, "event": .99, "random": .01})
            starts += next(iter(s)) in {0, 3}
            events += next(iter(e)) == 1
        self.assertGreater(starts, 95)
        self.assertGreater(events, 95)

    def test_multi_tag_state_has_one_bucket_and_empty_stratum_falls_back(self):
        records = [record([card("Offering")]), record([card("Pommel Strike", kind="ATTACK")], turn=2)]
        chosen, _ = sample_decisions(records, 2, random.Random(5))
        self.assertEqual(chosen[0]["assigned_event_bucket"], "energy_cost")
        self.assertEqual(chosen[1]["assigned_event_bucket"], "hand_access")
        plain = [record([card("Defend_R")])]
        chosen, _ = sample_decisions(plain, 1, random.Random(0), weights={"turn_start": 0, "event": .9, "random": .1})
        self.assertEqual(chosen[0]["sampling_channel"], "random")
        self.assertEqual(chosen[0]["conditional_draw_probability"], 1)
        forced = record([])
        chosen, _ = sample_decisions([forced], 2, random.Random(0))
        self.assertFalse(chosen)


if __name__ == "__main__":
    unittest.main()
