# Frozen 192-combat Boss comparison

Evaluation component of [single](single.md). Training and data provenance belong
to that stage; this page specifies the frozen testing panel and its results.

## Scope and results

The table records the historical frozen development experiment. Current single
configs reuse its input panel under corrected mechanics as a new evaluation;
the historical scores below have not been remeasured under those rules.
Ironclad, Act 1, Ascension 0; 24 route-derived loadouts × 8 combat seeds = 192
independent Boss combats per policy. This does not execute the preceding route
or carry damage between these combats. Each Boss has 64 combats.

| Policy | Victories / 192 | Retry | Fallback |
| --- | --- | --- | --- |
| Teacher | 184 | 0 | 0 |
| 1.5B Base | 106 | 1135 | 78 |
| 1.5B Gold SFT | 162 | 1247 | 1 |
| 1.5B Silver DPO | 164 | 851 | 13 |
| 7B Base | 123 | 0 | 0 |
| 7B Gold SFT | 172 | 54 | 0 |
| 7B Silver DPO | 171 | 447 | 0 |

Primary metric: Boss victories on shared combat identities. HP loss and output
legality are supporting metrics; defeats contribute terminal loss of 80 on this
panel. Cross-size timing is not comparable across historical runtime/hardware.
No sealed/test data is used. [Detailed comparisons](../stageresult.md#7b-sft-and-dpo-on-the-frozen-boss-panel)
retain the paired statistics and original evidence links.

## Inputs and configuration

- [Input generator configuration](../../../configs/generation/single_boss_inputs.json):
  `combat_panel_generation_v2`, panel ID `counterfactual_v2_act1_a0_boss_observation_v5_corrected_v1`.
  It selects `corrected_v1` and generates a new panel; it does not reconstruct
  or overwrite the retained historical input manifest.
- [Panel manifest](../../../assets/eval/qwen2_5_7b_base_act1_a0_boss_inputs_v1.json):
  shared by all three 7B evaluations. Frozen panel fingerprint:
  `fed8c5b2d3330308dbcc66cd530e95057c4c8f222db03e9a89003e4f6e0b7d56`.
- [Scope](../../../configs/env/d_expansion_v1_scope.json) and the configured
  `assets/picker/card_pick_metrics_v1.sqlite3` supply mechanics and picker
  statistics. The generator config records its required database identity,
  deck-stage rules, card/relic pools and all source-generation parameters.
- Evaluation configs: [7B Base](../../../configs/runs/evaluation/single_qwen2_5_7b_base_boss.json),
  [7B SFT](../../../configs/runs/evaluation/single_qwen2_5_7b_sft_boss.json),
  [7B DPO](../../../configs/runs/evaluation/single_qwen2_5_7b_dpo_boss.json).
  Each uses `frozen_policy_panel_evaluation_v1`, explicit `simulator_mechanics: corrected_v1`,
  V5 and a 1000-decision limit. Platform selection does not change this requirement;
  enabling the corrected engine does not enable secondary selection.
  Omitted checkpoint selects Base; the other two select the corresponding
  `assets/adapters/single/qwen2_5_7b_{gold_sft,silver_dpo}_v1` adapters.

### Observation, models and algorithms

[V5 semantics](../policy_observation.md#maintained-observation-versions) provide public
combat state, canonical legal actions and a transitive mechanics glossary.
Each decision returns one `ACTION_n`; serializer, projection, parser, retry and
seeded fallback are shared by learned policies. Hidden draw order, realized
future RNG and Teacher search evidence are excluded; there is no chat history.
This is the historical single-action interface, without V6 secondary selection.

Base means unadapted **Instruct** weights: Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, or Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`. See [1.5B runtime](../../../configs/runtime/base_model_runtime_1_5b_v2.json)
and [7B runtime](../../../configs/runtime/base_model_runtime_7b_v2.json).
7B uses BF16/SDPA, no quantization, greedy batch size 1, at most 8 generated
tokens, no constrained decoding, and independent decisions. Its execution profile
requires [CUDA and native BF16](../../../configs/runtime/training_cuda_bf16.json).

Teacher uses `BattleScumSearcher2`, 8192 simulations per decision and search seed
101. This is the historical combat-search baseline, not public-information Gold
certification. Historical configuration lineage was recorded at Git `33c85dc`:
`configs/eval/counterfactual_v2_act1_a0_expanded_v5_evaluation.json` →
`counterfactual_v2_silver_dpo_v5_evaluation.json`; Teacher settings originate in
`configs/eval/expanded_teacher_simulator_v1.json`. Those historical paths are not
current run configs; their broader source-panel counts are not this 192-Boss subset.

Input generation uses `teacher_counterfactual` reward selection,
`starter_alternating` removal, `value_priority` upgrades and `act_topology` routes.
Reward search uses budget 8192, search seed 101, hidden-order seeds 701/809/907,
opening damage multiplier 1.5, required pair wins versus skip 2, and conflict seed
offsets 100000/200000. Weights are search/intrinsic/optionality .5/.35/.15,
current/future target .7/.3, duplicate penalty .08, premium/strong override .15/.08.
Picker temperature/exploration/epsilon are .85/.15/.02. These are **loadout
generation** parameters, not model decoding or Gold certification thresholds.

### Training provenance

[7B SFT run](../../../configs/runs/training/single_qwen2_5_7b_gold_sft.json) consumes
`assets/datasets/single/observation-v5/gold_sft.manifest.json`, seed
2026090302. [7B DPO run](../../../configs/runs/training/single_qwen2_5_7b_silver_dpo.json)
consumes the sibling `silver_preferences.manifest.json`, seed 2026090303; it starts
from SFT, which also supplies the frozen reference. These are the same V5 dataset
sources used by the historical 1.5B experiment.

[SFT recipe](../../../configs/profiles/training/sft_lora_v1.json) and
[DPO recipe](../../../configs/profiles/training/dpo_lora_v1.json): q_proj/v_proj
LoRA rank 8, alpha 16, dropout 0; one epoch, microbatch 1, accumulation 8,
response-only supervision including assistant terminator, maximum 3072 tokens
without truncation. AdamW learning rates are 1e-4 / 2e-5, weight decay 0, gradient
norm cap 1; DPO beta .1 and label smoothing 0. 7B completed 8221 SFT units/1028
steps and 471 DPO units/59 steps, with finite losses and exact fresh-Base reload.
Historical 1.5B provenance remains in its [SFT report](../../../report/training/qwen2_5_1_5b_teacher_v2_gold_sft_observation_v5_v1.json)
and [DPO report](../../../report/training/qwen2_5_1_5b_teacher_v2_silver_dpo_observation_v5_v1.json);
current recipes do not rewrite those report bindings.

### Seeds

| Domain | Recorded base/settings |
| --- | --- |
| Source matrix reward / combat | 202614010000 / 202614020000 |
| Source matrix policy / picker / relic | 4100000 / 4200000 / 4300000 |
| Route panel reward / combat / picker | 202617010000 / 202617020000 / 7200000 |
| Original evaluation combat / policy | 202620020000 / 8200000; 2 seeds per route |
| Additional evaluation combat / policy | 202640000000 / 8400000; expanded to 8 per route |
| Additional episode index start | 192 |

The generator preserves the frozen shared-RNG call order. Exact per-combat
identities are in the panel manifest, not inferred from table ordering.

## Outputs and reproduction boundary

[Historical 1.5B/Teacher report](../../../report/evaluation/counterfactual_v2_act1_a0_expanded_v5_evaluation.json)
contains the completed comparison. The [7B evidence table](../stageresult.md#7b-sft-and-dpo-on-the-frozen-boss-panel)
identifies each training/evaluation report, episode export and recorded hash.
Historical 7B evaluation outputs are `outputs/eval/qwen2-5-7b-{base,sft,dpo}-act1-a0-boss-v1/formal/`:
`report.json`, `episodes.jsonl`, resumable combat records and step trajectories.
Raw evaluation trajectories are retained locally and are not distributed. These
are logical run-relative locations; reports do not replace the original trajectories.
Current configs write to `outputs/eval/qwen2-5-7b-{base,sft,dpo}-act1-a0-boss-corrected-v1/formal/`
with mechanics in their resume bindings. They reject historical episode bindings.

[Input reproduction report](../../../report/data/counterfactual_v2_act1_a0_boss_inputs_v2.json)
records 24 routes and 192 inputs matching the frozen fingerprint on Windows at
Git `4c46b83` (2026-09-06), in 131.923 seconds.
That evidence applies to the historical Windows mechanism and snapshot format.
The current corrected generator retains the seeds and algorithms but can produce
different loadouts and snapshot content; it has no historical digest target.
The input entry is [generate_combat_panel.py](../../../scripts/generate_combat_panel.py);
the 7B evaluation entry is [run_frozen_policy_panel_evaluation.py](../../../scripts/run_frozen_policy_panel_evaluation.py).
Input reproduction is distinct from policy execution evidence. Native upstream
revision is `7476a81954020087da31d41d16fddf475746ec2d`; asset/binary and candidate
bindings remain in original reports. The retired source was recorded at Git
`33c85dc`; that revision and its external assets are not included in a release
snapshot. No rerun is implied.
