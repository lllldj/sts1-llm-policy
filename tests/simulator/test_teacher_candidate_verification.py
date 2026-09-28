from copy import deepcopy
import unittest

from scripts.verify_teacher_candidate_pool import _verify_arm_metrics


class CandidateMetricVerificationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"route": {"route_index": i, "combat_seed_group_index": 0},
             "status": "completed" if i == 0 else "defeated",
             "death_combat_index": None if i == 0 else 5,
             "combats": [{"encounter_family": "boss" if i == 0 else "elite",
                          "summary": {"outcome": "victory" if i == 0 else "defeat",
                                      "decisions": 12,
                                      "protocol": {"retry_count": 0, "fallback_count": 0}}}]}
            for i in range(2)
        ]
        self.reported = {
            "route_count": 2, "route_execution_count": 2,
            "combat_seed_groups_per_route": 1, "boss_victories": 1,
            "truncated_routes": 0, "boss_arrivals": 1,
            "combat_count": 2, "combat_victories": 1,
            "death_combat_index_counts": {"5": 1}, "death_combat_index_mean": 5.0,
            "decisions": 24, "retry_count": 0, "fallback_count": 0,
        }

    def test_death_histogram_matches_json_string_keys(self):
        _verify_arm_metrics(self.rows, self.reported)

    def test_real_count_and_position_mismatches_are_still_rejected(self):
        for changes in ({"death_combat_index_counts": {"5": 2}},
                        {"death_combat_index_counts": {"6": 1}},
                        {"boss_victories": 2}, {"decisions": 25}):
            with self.subTest(changes=changes):
                reported = {**deepcopy(self.reported), **changes}
                with self.assertRaisesRegex(ValueError, "Aggregate metrics"):
                    _verify_arm_metrics(self.rows, reported)


if __name__ == "__main__":
    unittest.main()
