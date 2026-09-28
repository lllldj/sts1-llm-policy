"""Synthetic GOLD source records and export parameters."""
from hashlib import sha256
import itertools
from sts1_llm_policy.data.gold.export_source import candidate_ids
from sts1_llm_policy.artifacts import canonical_json_bytes


PARAMS = dict(dataset_ids={a: f"test-{a}" for a in "abc"}, hp_tolerance=1., minimum_paired_trials=64,
              se_multiplier=2., minimum_adjusted_hp_gap=1., coarse_hp_gap=3., weight_hp_cap=5.)

def sources():
    states = []
    for index, hps in enumerate(((50, 47, 44), (50, 48, 46), (50, 49, 48), (50, 46))):
        roots = [{"id": f"ACTION_{i}", "card": "end_turn" if i == len(hps)-1 else "Strike_R",
                  "target": None, "action_type": "end_turn" if i == len(hps)-1 else "play_card",
                  "card_semantics": None if i == len(hps)-1 else {"card_id": "Strike_R", "card_type": "ATTACK", "upgrades": 0}}
                 for i in range(len(hps))]
        state = {"sample": {"id": f"state-{index}", "route": 4, "group": 1, "combat": 1, "step": index},
                 "status": "completed", "encounter_family": "normal_weak", "outer_trials_completed": 64,
                 "minimum_win_rate": .99, "win_floor_rounding": "floor", "minimum_wins": 63,
                 "root_actions": roots, "executions": 64 * len(roots), "executed_decisions": 100,
                 "observation": "PUBLIC\nLEGAL_ACTIONS:\n" + "\n".join(
                     f"{a['id']}: {'END_TURN' if a['action_type']=='end_turn' else 'PLAY Strike'}" for a in roots),
                 "actions": [{"action": a["id"], "trials": 64, "wins": 64, "expected_carried_hp": hp}
                             for a, hp in zip(roots, hps)],
                 "paired_comparisons": [{"a": f"ACTION_{i}", "b": f"ACTION_{j}", "paired_trials": 64,
                                         "standard_error": 0., "mean_carried_hp_a_minus_b": hps[i]-hps[j],
                                         "unpriced_resource_conflict_trials": 0}
                                        for i, j in itertools.combinations(range(len(hps)), 2)]}
        state["estimated_acceptable_actions"] = sorted(candidate_ids(state, 1.))
        states.append(state)
    report = {"status": "completed", "states": states, "configuration": {
        "observation_version": "observation_v7", "expected_hp_tolerance": 1.,
        "minimum_win_rate": {"normal_weak": .99}, "samples": [s["sample"] for s in states],
        "source_report": "pool/report.json", "source_partition": {
            "role": "train_candidate", "excluded_routes": [1], "reserved_routes": [5]}}}
    verification = {"status": "completed", "verified_executions": sum(s["executions"] for s in states),
                    "verified_decisions": 400,
                    "report_canonical_sha256": sha256(canonical_json_bytes(report)).hexdigest()}
    return report, verification
