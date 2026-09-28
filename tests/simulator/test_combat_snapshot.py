"""Snapshot signing must preserve content and isolate caller-owned state."""
from copy import deepcopy
import unittest

from sts1_llm_policy.env.combat_snapshot import resign_snapshot, with_snapshot_relics


class CombatSnapshotTests(unittest.TestCase):
    def test_signing_preserves_schema_and_nested_content_without_mutation(self):
        for schema in ("combat_snapshot_v1", "combat_snapshot_v2"):
            with self.subTest(schema=schema):
                snapshot = {"schema_version": schema, "current_hp": 30,
                            "deck": [{"string_id": "Searing Blow", "upgraded": True,
                                      "upgrade_count": 3}],
                            "relics": [{"id": "NUNCHAKU", "counter": 6}],
                            "fingerprint_fnv1a64": "old"}
                before = deepcopy(snapshot)
                signed = resign_snapshot(snapshot)
                self.assertEqual(snapshot, before)
                self.assertEqual({**signed, "fingerprint_fnv1a64": "old"}, before)
                self.assertNotEqual(signed["fingerprint_fnv1a64"], "old")
                self.assertEqual(resign_snapshot(dict(reversed(list(signed.items())))), signed)
                changed = resign_snapshot({**signed, "current_hp": 29})
                self.assertNotEqual(changed["fingerprint_fnv1a64"], signed["fingerprint_fnv1a64"])
                signed["deck"][0]["upgrade_count"] = 4
                signed["relics"][0]["counter"] = 0
                self.assertEqual(snapshot, before)

    def test_relic_replacement_copies_both_inputs_and_resigns(self):
        snapshot = resign_snapshot({"current_hp": 30, "deck": [{"string_id": "Strike_R"}],
                                    "relics": [{"id": "BURNING_BLOOD", "counter": 0}]})
        before = deepcopy(snapshot)
        relics = [{"id": "NUNCHAKU", "counter": 6}]
        changed = with_snapshot_relics(snapshot, relics)
        self.assertEqual(changed["relics"], relics)
        self.assertEqual(changed, resign_snapshot(changed))
        self.assertNotEqual(changed["fingerprint_fnv1a64"], snapshot["fingerprint_fnv1a64"])
        changed["relics"][0]["counter"] = 9
        changed["deck"][0]["string_id"] = "Defend_R"
        self.assertEqual(snapshot, before)
        self.assertEqual(relics, [{"id": "NUNCHAKU", "counter": 6}])
