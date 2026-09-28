import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sts1_llm_policy.data.gold.preparation import select_panel
from sts1_llm_policy.data.gold.candidate_statistics import collection_statistics
from sts1_llm_policy.artifacts import atomic_write_json as write_json


def action(identity, card, card_type, target=None, kind="play_card"):
    return {"id": identity, "card": card, "target": target, "action_type": kind,
            "card_semantics": {"card_id": card, "card_type": card_type, "upgrades": 0} if card_type else None}


class GoldCollectionTests(unittest.TestCase):
    def test_card_distribution_ignores_target_multiplicity_and_keeps_empty_states(self):
        attack = [action(f"A{i}", "Strike_R", "ATTACK", i) for i in range(3)]
        defend = action("D", "Defend_R", "SKILL")
        end = action("E", "end_turn", None, kind="end_turn")
        def state(roots, kept):
            return {"status": "completed", "encounter_family": "elite", "root_actions": roots,
                    "estimated_acceptable_actions": kept, "paired_comparisons": [],
                    "actions": [{"action": a["id"], "passes_win_floor_lcb": False} for a in roots]}
        states = [state(attack + [defend, end], ["A2", "D"]),
                  state([defend, end], []), state([action("S", "Defend_R", "SKILL", kind="select_card")], ["S"])]
        summary = collection_statistics(states)["overall"]
        self.assertEqual(summary["primary_metric"], .5)
        self.assertEqual(summary["primary_metric_opportunity_states"], 2)
        self.assertEqual(summary["legal_cards"]["category_state_equal_share"]["ATTACK"], .25)
        self.assertEqual(summary["candidate_cards"]["category_state_equal_share"]["ATTACK"], .5)
        self.assertEqual(summary["candidate_cards"]["states_without_items"], 2)
        self.assertEqual(summary["legal_actions"]["category_item_counts"]["ATTACK"], 3)
        self.assertEqual(summary["quality_counts"]["secondary_selection_states"], 1)
        self.assertEqual(summary["quality_counts"]["empty_candidate_states"], 1)

    def test_mechanic_defense_can_overlap_attack_and_basic_includes_bash(self):
        from sts1_llm_policy.data.gold.candidate_statistics import categories
        self.assertTrue({"ATTACK", "defense", "direct_block", "nonbasic"} <= categories(action("A", "Iron Wave", "ATTACK")))
        self.assertIn("conditional_or_retained_block", categories(action("R", "Rage", "SKILL")))
        self.assertIn("basic", categories(action("B", "Bash", "ATTACK")))
        self.assertNotIn("defense", categories(action("S", "Body Slam", "ATTACK")))

    def test_route_sampling_covers_reached_boss_and_changes_group_behavior(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "report.json", {"status": "completed",
                       "configuration": {"data_source": {"evidence_class": "teacher_candidate_pool"}},
                       "scope": {"routes": 3, "combat_seed_groups_per_route": 2}})
            for r in range(3):
                for g in range(2):
                    combats = []
                    for c, family in enumerate(("normal_weak", "boss"), 1):
                        path = f"r{r}-g{g}-c{c}.jsonl"
                        records = [{"schema_version": "trajectory_v1", "record_type": "transition",
                                    "step_index": step, "legal_actions": ["A"], "model_legal_actions": ["A"],
                                    "teacher_action": "anything", "outcome": "defeat"} for step in range(5)]
                        (root / path).write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
                        combats.append({"combat_index": c, "encounter_family": family, "scenario_id": family,
                                        "trajectory": path, "summary": {"decisions": 5}})
                    # An early death leaves one group without a Boss; never fabricate it.
                    if r == 0 and g == 0:
                        combats.pop()
                    write_json(root / f"teacher/routes/route-{r:03d}-combat-seed-group-{g:02d}.json", {"combats": combats})
            kwargs = {"route_count": 3, "seed": 17, "states_per_combat": {"normal_weak": 1, "boss": 4},
                      "sampling": "uniform_decisions_v1"}
            samples, selected = select_panel(root, Path("report.json"), groups_per_route=1, **kwargs)
            self.assertEqual((samples, selected), select_panel(root, Path("report.json"), groups_per_route=1, **kwargs))
            self.assertTrue(all(len(s["groups"]) == 1 for s in selected["route_groups"]))
            more, expanded = select_panel(root, Path("report.json"), groups_per_route=2, **kwargs)
            self.assertEqual(len(expanded["combats"]), 11)
            self.assertGreater(len(more), len(samples))
            self.assertEqual(len(more), len({s["id"] for s in more}))
            self.assertEqual(expanded["states_by_family"], {"boss": 20, "normal_weak": 6})
            self.assertTrue(all(c["selected_decisions"] == (4 if c["encounter_family"] == "boss" else 1) for c in expanded["combats"]))
            filtered, manifest = select_panel(root, Path("report.json"), groups_per_route=1,
                                               **{**kwargs, "route_count": 2}, excluded_routes=[0])
            self.assertEqual({s["route"] for s in filtered}, {1, 2})
            self.assertEqual(manifest["route_inclusion_probability"], 1.)
            self.assertEqual(manifest["excluded_routes"], [0])
            self.assertEqual(manifest["unselected_eligible_routes"], [])
            with self.assertRaisesRegex(ValueError, "Not enough routes"):
                select_panel(root, Path("report.json"), groups_per_route=1, **kwargs, excluded_routes=[0])


if __name__ == "__main__":
    unittest.main()
