# Stage results

The [README](../../README.md#current-offline-results) summarizes current results.
This page retains comparisons, statistical methods and supporting evidence.
Panels differ, so their win rates are not directly comparable. They are development
experiments, not held-out tests or evidence of full-game performance.

Original execution reports remain the source evidence. Raw reconstruction and
replay collections are local and undistributed; `outputs/` paths below are evidence
locations, not download links. Earlier source revisions identify development history
outside this repository. [Marked path redactions](../../report/README.md) change
public report bytes, not experimental results.

## 7B SFT and DPO on the frozen Boss panel

Completed comparison: 2026-09-07T12:28:39.3635392Z. Primary metric: victories on the shared frozen Act-1/A0 Boss panel. Scope: 24 routes, 192 combats per object, seven objects, `observation_v5`. Panel SHA-256: `fed8c5b2d3330308dbcc66cd530e95057c4c8f222db03e9a89003e4f6e0b7d56`.

### Policy outcomes

| Metric | Teacher | 1.5B Base | 1.5B SFT | 1.5B DPO | 7B Base | 7B SFT | 7B DPO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| victories | 184 | 106 | 162 | 164 | 123 | 172 | 171 |
| defeats | 8 | 86 | 30 | 28 | 69 | 20 | 21 |
| win_rate | 0.958333 | 0.552083 | 0.84375 | 0.854167 | 0.640625 | 0.895833 | 0.890625 |
| win_rate_wilson_95 | `[0.91995, 0.97874]` | `[0.48141, 0.62072]` | `[0.78571, 0.8883]` | `[0.7973, 0.89714]` | `[0.57061, 0.70512]` | `[0.84459, 0.93155]` | `[0.83858, 0.92734]` |
| total_hp_loss_mean | 29.5 | 65.661 | 46.661 | 48.062 | 58.417 | 42.74 | 43.932 |
| total_hp_loss_median | 29.0 | 75.5 | 47.0 | 49.0 | 61.5 | 45.0 | 46.0 |
| victory_hp_loss_mean | 27.304 | 54.028 | 40.488 | 42.61 | 46.309 | 38.407 | 39.503 |
| decisions | 4980 | 6993 | 5923 | 5960 | 7450 | 5722 | 5858 |
| decisions_mean | 25.938 | 36.422 | 30.849 | 31.042 | 38.802 | 29.802 | 30.51 |
| retry_count | 0 | 1135 | 1247 | 851 | 0 | 54 | 447 |
| fallback_count | 0 | 78 | 1 | 13 | 0 | 0 | 0 |
| first_pass_legal_rate | 1.0 | 0.837695 | 0.789465 | 0.857215 | 1.0 | 0.990563 | 0.923694 |
| hexaghost victories | 64 | 46 | 59 | 62 | 47 | 61 | 61 |
| slime_boss victories | 56 | 39 | 52 | 50 | 32 | 53 | 55 |
| the_guardian victories | 64 | 21 | 51 | 52 | 44 | 58 | 55 |

Rates are fractions. `win_rate_wilson_95` lists the reported Wilson interval endpoints. All-combat HP-loss means include defeats, which contribute terminal loss (80 on this panel); victory-only means include victories only. Each Boss has 64 combats.

### Training completion

| Metric | 7B SFT | 7B DPO |
| --- | --- | --- |
| status | trained_pending_development_evaluation | trained_pending_development_evaluation |
| processed_units | 8221 | 471 |
| optimizer_steps | 1028 | 59 |
| seconds | 1976.65 | 234.084 |
| minimum_loss | 5.395070184022188e-05 | 0.36341550946235657 |
| maximum_loss | 2.0852115154266357 | 1.021500825881958 |
| peak_memory_reserved_bytes | 32677822464 | 32686211072 |
| checkpoint_weights_sha256 | 49f968c0420333382908e9065a821b57c6ef9e8813093423a82c79b10368ece0 | 47ad268838bf9ee2116fda978e9bbe5b23345f33b20794e617478ae694ec1aea |
| reference_groups_computed | — | 471 |

The formal 7B experiment's tracked sources correspond to Git `88ecc52`.

Status strings describe the original training stage before development evaluation. Subsequent evaluation completed for both candidates. Training completed the full sample/step counts with finite loss and exact fresh-Base reload. Times are seconds; memory is bytes.

### Contrasts

| Comparison | Statistic | Value |
| --- | --- | --- |
| 7B SFT minus 7B Base | victory_delta | 49 |
| 7B SFT minus 7B Base | win_rate_delta | 0.255208 |
| 7B SFT minus 7B Base | total_hp_loss_mean_delta | -15.677 |
| 7B SFT minus 7B Base | victory_hp_loss_mean_delta | -7.902 |
| 7B DPO minus 7B Base | victory_delta | 48 |
| 7B DPO minus 7B Base | win_rate_delta | 0.25 |
| 7B DPO minus 7B Base | total_hp_loss_mean_delta | -14.485 |
| 7B DPO minus 7B Base | victory_hp_loss_mean_delta | -6.806 |
| 7B DPO minus 7B SFT (paired) | sft_loss_to_dpo_win | 4 |
| 7B DPO minus 7B SFT (paired) | sft_win_to_dpo_loss | 5 |
| 7B DPO minus 7B SFT (paired) | net_victory_delta | -1 |
| 7B DPO minus 7B SFT (paired) | win_rate_delta | -0.005208 |
| 7B DPO minus 7B SFT (paired) | exact_mcnemar_two_sided_p | 1.0 |
| 7B DPO minus 7B SFT (paired) | dpo_minus_sft_hp_loss_mean | 1.193 |
| 7B DPO minus 7B SFT (paired) | dpo_minus_sft_victory_hp_loss_mean | 1.096 |
| 7B DPO minus 7B SFT (paired) | dpo_hp_loss_lower_same_higher | `[22, 130, 40]` |
| 7B minus 1.5B (descriptive) | base.victory_delta | 17 |
| 7B minus 1.5B (descriptive) | base.win_rate_delta | 0.088542 |
| 7B minus 1.5B (descriptive) | base.total_hp_loss_mean_delta | -7.244 |
| 7B minus 1.5B (descriptive) | base.victory_hp_loss_mean_delta | -7.719 |
| 7B minus 1.5B (descriptive) | sft.victory_delta | 10 |
| 7B minus 1.5B (descriptive) | sft.win_rate_delta | 0.052083 |
| 7B minus 1.5B (descriptive) | sft.total_hp_loss_mean_delta | -3.921 |
| 7B minus 1.5B (descriptive) | sft.victory_hp_loss_mean_delta | -2.081 |
| 7B minus 1.5B (descriptive) | dpo.victory_delta | 7 |
| 7B minus 1.5B (descriptive) | dpo.win_rate_delta | 0.036458 |
| 7B minus 1.5B (descriptive) | dpo.total_hp_loss_mean_delta | -4.13 |
| 7B minus 1.5B (descriptive) | dpo.victory_hp_loss_mean_delta | -3.107 |
| Teacher minus 7B SFT | victory_delta | 12 |
| Teacher minus 7B SFT | win_rate_delta | 0.0625 |
| Teacher minus 7B SFT | total_hp_loss_mean_delta | -13.24 |
| Teacher minus 7B SFT | victory_hp_loss_mean_delta | -11.103 |

Win-rate deltas are fractions; 0.01 is one percentage point. The HP-loss vector
is lower / same / higher for DPO relative to SFT. Cross-size contrasts are descriptive.

### Interpretation

7B SFT leads learned candidates at 172/192; DPO is one win lower. Only nine paired
outcomes differ (four gains, five losses; exact McNemar p=1.0). DPO also adds 393
retries and lowers first-pass legality by 6.687 percentage points. This does not
establish improvement over SFT. Cross-size timing is not comparable because
runtime/hardware differ. The live default remains unchanged.

### Recorded verification and original evidence

Recorded checks confirmed complete finite-loss training and exact fresh-Base
reload, completed SFT/DPO evaluation with 192 unique shared combat identities,
consistent combat accounting and frozen-panel bindings. The historical 1.5B report
declares the same 192 identities. No test/sealed data was read. These are historical
checks, not new verification performed by reading this page.

| Evidence field | Value |
| --- | --- |
| historical_1_5b_and_teacher_report.path | [report/evaluation/counterfactual_v2_act1_a0_expanded_v5_evaluation.json](../../report/evaluation/counterfactual_v2_act1_a0_expanded_v5_evaluation.json) |
| historical_1_5b_and_teacher_report.sha256 | f79fbf2c0037ab0ad35b645e94a627cb46b315f4d0e3d3cc153640208d9ac691 |
| qwen2_5_7b_base_report.path | `outputs/eval/qwen2-5-7b-base-act1-a0-boss-v1/formal/report.json` |
| qwen2_5_7b_base_report.sha256 | 0aef029c9c47527ba236204e82f22743a9330ede901778b06a9feb0acb77ecc2 |
| qwen2_5_7b_sft_training_report.path | `outputs/training/qwen2_5_7b_gold_sft_v1/report.json` |
| qwen2_5_7b_sft_training_report.sha256 | 5dad843be3213a59156e5e3b2247965cfd0bb333b4dd8d4ac4f18bdceb7c93d8 |
| qwen2_5_7b_dpo_training_report.path | `outputs/training/qwen2_5_7b_silver_dpo_v1/report.json` |
| qwen2_5_7b_dpo_training_report.sha256 | e11038cee27a6aa05f8a3a094224c42be27255ab12938cc90f3cf13920a80a05 |
| qwen2_5_7b_sft_evaluation.report_path | `outputs/eval/qwen2-5-7b-sft-act1-a0-boss-v1/formal/report.json` |
| qwen2_5_7b_sft_evaluation.report_sha256 | eadc0a39851255cb0882dae3deea5b939cdf384723f434e871b9592ab93fb6da |
| qwen2_5_7b_sft_evaluation.episodes_path | `outputs/eval/qwen2-5-7b-sft-act1-a0-boss-v1/formal/episodes.jsonl` |
| qwen2_5_7b_sft_evaluation.episodes_sha256 | 94e6689124aff81b77a0665d712f26e027260d06e2030d89d8320ddf0c2108fc |
| qwen2_5_7b_dpo_evaluation.report_path | `outputs/eval/qwen2-5-7b-dpo-act1-a0-boss-v1/formal/report.json` |
| qwen2_5_7b_dpo_evaluation.report_sha256 | b8a50b9013c036f7857c8c18abb60de7bb1e713d7d39c32df3ca14f7db848b99 |
| qwen2_5_7b_dpo_evaluation.episodes_path | `outputs/eval/qwen2-5-7b-dpo-act1-a0-boss-v1/formal/episodes.jsonl` |
| qwen2_5_7b_dpo_evaluation.episodes_sha256 | 2caa72131fee87fdf5961d6ee873ff5f99eec0699cf203e7a973a82dc8ec17f3 |

## 7B Base and Teacher on continuous routes

Completed comparison: 2026-09-07T18:04:34.5149829Z. Source revision: `9653109729e921265954e86c0f4fe998f4037c22`. Primary metric: Boss victories on paired continuous route executions. Scope: 20 route topologies, four combat/policy seed groups per topology, 80 executions per arm, up to eight combats per route, `observation_v6` / `combat_card_selection_v1`. Four seed groups are nested within each topology; uncertainty is summarized using 20 topology clusters.

### Route and policy outcomes

| Metric | 7B Base | teacher |
| --- | --- | --- |
| boss_victories | 1 | 68 |
| boss_win_rate_all_routes | 0.0125 | 0.85 |
| boss_reaches | 12 | 76 |
| boss_win_rate_given_reach | 0.083333 | 0.894737 |
| combats_started | 364 | 628 |
| combat_victories | 285 | 616 |
| combat_win_rate_given_reach | 0.782967 | 0.980892 |
| mean_combats_won_per_route | 3.5625 | 7.7 |
| death_combat_index_counts | `{"1": 10, "2": 4, "3": 3, "4": 23, "5": 23, "6": 1, "7": 4, "8": 11}` | `{"5": 4, "8": 8}` |
| decisions | 6295 | 8251 |
| first_pass_legal_rate | 1.0 | 1.0 |
| retry_count | 0 | 0 |
| fallback_count | 0 | 0 |
| upgrade_events | 131 | 312 |
| mean_upgrade_candidates | 6.022901 | 6.330128 |
| run_seconds | 1044.969 | 1843.983 |
| model_load_seconds | 21.046 | 0.0 |

Rates are fractions; times are seconds. Combat win rate is conditional on reaching a combat, so the two arms do not play identical sets of reached encounters. Boss victories over all 80 routes are the primary outcome. Boss win rate given reach uses each arm's survivor population. Death-count keys are one-based combat positions.

### Progression

| Combat position | Base attempts | Base victories | Teacher attempts | Teacher victories |
| --- | --- | --- | --- | --- |
| 1 | 80 | 70 | 80 | 80 |
| 2 | 70 | 66 | 80 | 80 |
| 3 | 66 | 63 | 80 | 80 |
| 4 | 63 | 40 | 80 | 80 |
| 5 | 40 | 17 | 80 | 76 |
| 6 | 17 | 16 | 76 | 76 |
| 7 | 16 | 12 | 76 | 76 |
| 8 | 12 | 1 | 76 | 68 |

### Paired contrasts

| Statistic | Value |
| --- | --- |
| teacher_minus_base_boss_victories | 67 |
| teacher_minus_base_boss_win_rate | 0.8375 |
| route_execution_outcomes.both_win | 0 |
| route_execution_outcomes.teacher_only_win | 68 |
| route_execution_outcomes.base_only_win | 1 |
| route_execution_outcomes.both_lose | 11 |
| execution_level_exact_mcnemar_two_sided_p_descriptive | 2.371692252312041e-19 |
| teacher_more_combats_won | 77 |
| equal_combats_won | 2 |
| base_more_combats_won | 1 |
| mean_teacher_minus_base_combats_won | 4.1375 |
| median_teacher_minus_base_combats_won | 4.0 |
| topology_cluster_comparison.teacher_better_topologies | 20 |
| topology_cluster_comparison.ties | 0 |
| topology_cluster_comparison.base_better_topologies | 0 |
| topology_cluster_comparison.two_sided_exact_sign_p | 1.9073486328125e-06 |

The topology-cluster exact sign test is the primary paired uncertainty summary. The execution-level McNemar p-value is descriptive only because the 80 executions are not 80 independent route topologies.

### Interpretation

Teacher's advantage spans all 20 topologies. Base loses 23 routes at combat four
and another 23 at the first elite, reaching the Boss only 12 times; Teacher loses
four at the first elite and eight at the Boss. Conditional Boss and combat win
rates compare different survivor populations. The topology sign test is the
primary paired uncertainty summary; the 80-execution McNemar value is descriptive.
Both arms make all 14,546 decisions legally on the first pass; all 443 upgrade
events retain 12-cell evidence per candidate. This supports a gap on this configured
development generator, not native full-game performance.

### Recorded verification and original evidence

Recorded checks matched combined and per-arm reports/inputs, with 20 unique
topologies and four groups sharing each schedule. Encounter bags exhausted before
repeats; all 640 combat seeds and all 640 policy seeds were unique. Upgrade
comparisons retained 12 cells per candidate, all policy outputs were legal on the
first pass, and no test/sealed data was read.

| Evidence field | Value |
| --- | --- |
| combined_report.path | `outputs/generation/qwen2-5-7b-base-teacher-act1-continuous-v1/formal/report.json` |
| combined_report.sha256 | c83b576d7a85dd9572f3543b314399eb00286e74c9ae3fcee3dd12a467b00316 |
| combined_inputs.path | `outputs/generation/qwen2-5-7b-base-teacher-act1-continuous-v1/formal/inputs.json` |
| combined_inputs.sha256 | c81a4732032903a83521c63b84db56e2ac3bf9f4c2420f8b8149e851d47f6907 |
| teacher_report_sha256 | 8cd4c8a2730206c9b82e60a9d4678e454bd3e2de5818e60d684ad101bfebd32e |
| teacher_inputs_sha256 | 404d050c84ef755429f93cf7f07e8ab54b5654a5abaf557d0527f602c0455181 |
| qwen2_5_7b_base_report_sha256 | c2df1d8d23ccad313430e0a811917067afd5a136a8b45b4d32d3cf29ded8c3e8 |
| qwen2_5_7b_base_inputs_sha256 | d080f96096a08caa3ca3448ae3665ebfe55b1bed0d88a84faae89bac6a1dd87d |

## 7B Base with observation V7 on continuous routes

Primary metric: final Boss victories on the paired Act-1/A0 development routes. Decision-limit truncations remain in the denominator as non-victories and are reported separately from deaths.

| Policy | Boss victories | Truncated routes | Boss arrivals | Combat victories | Decisions | Retry | Fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V7 Base | 4/80 (5.00%) | 1 | 18 | 341 | 10509 | 0 | 0 |
| V6 Base | 1/80 (1.25%) | 0 | 12 | 285 | 6295 | 0 | 0 |
| V6 Teacher (local search) | 68/80 (85.00%) | 0 | 76 | 616 | 8251 | 0 | 0 |

Against V6 Base: V7-only wins=4, baseline-only wins=1, both win=0, both lose=75.
By route topology: V7 better=3, baseline better=1, ties=16; two-sided exact sign p=0.625.

Against V6 Teacher (local search): V7-only wins=0, baseline-only wins=64, both win=4, both lose=12.
By route topology: V7 better=0, baseline better=20, ties=0; two-sided exact sign p=1.9073486e-06.

Four formal seed groups share each route topology; execution-level pairs are not independent topology samples. Changes cover the V7 route objective/context and explicit card-selection/upgrade information together; this experiment does not isolate their separate effects. Teacher remains a local-combat baseline. No sealed data or training is involved.

The native bridge includes public preview metadata and a correction to UpgradeAllCardsInHand: only cards passing canUpgrade are upgraded. Retained V6 Base/Teacher results predate this engine correction, so this comparison cannot isolate observation V7 effects. V7 raw trajectories and combat/route counts were reviewed locally; previously reviewed V6 results are reused.

Original evidence:

- `outputs/generation/qwen2-5-7b-base-teacher-act1-continuous-v7-upgrade-fix/formal/report-qwen2_5_7b_base.json`
- `outputs/generation/qwen2-5-7b-base-teacher-act1-continuous-v7-upgrade-fix/formal/inputs-qwen2_5_7b_base.json`
- `outputs/generation/qwen2-5-7b-base-teacher-act1-continuous-v7-upgrade-fix/revision.txt`
- `outputs/generation/qwen2-5-7b-base-teacher-act1-continuous-v1/formal/report.json`
- `outputs/generation/qwen2-5-7b-base-teacher-act1-continuous-v1/formal/inputs.json`

## 7B mixed GOLD/Teacher SFT on matched V7 routes

Completed: 2026-09-17T01:28:02Z, evaluation revision `0e77c53995ae28899e8916e9ba46d814f2d99d4b`. Mixed SFT wins 26/80 routes, compared with 13/80 for GOLD-only SFT and 4/80 for Base. All 80 mixed-policy executions survive the first three weak combats. The result supports improved performance on this development panel, with remaining encounter-specific weaknesses and no default-policy change.

### Scope, training and evidence

The primary metric is final Boss victories per route execution. All three policies use the same 20 Act-1/A0 route topologies, four seed groups per topology, V7 observation, action protocol and non-combat strategies. The mixed evaluation contains one arm; the retained Base/GOLD-only evaluation supplies the controls. Configuration fields excluding run labels, output paths and arms match, as do all 80 generated schedules, native bridge content, simulator revision, picker database, model runtime and execution profile. First-combat snapshots are identical across policies. Later HP, decks and route decisions can diverge as a consequence of earlier play.

Both SFT runs start from fresh Qwen2.5-7B-Instruct with the same q/v LoRA rank 8, alpha 16, learning rate 1e-4 and one epoch. GOLD-only processes 5,843 states in 731 optimizer steps; mixed SFT processes 17,893 states in 2,237 steps. Mixed loss mass is 70% GOLD, 25% Teacher strategic demonstrations and 5% sole-legal END formatting. Teacher demonstrations cover the first three complete combats from the existing 480 training routes, with GOLD retained at overlaps. The mixed run therefore changes both data coverage/weighting and optimization budget; this comparison cannot isolate their contributions or establish that the chosen ratio is optimal.

The mixed training processed all states without truncation, with finite losses and an exact adapter reload on fresh Base. Evaluation took 2,265.962 seconds (37.8 minutes). Local checks reconstructed the declared 80-route schedule, matched the evaluated checkpoint metadata to the retained adapter, checked all 80 new route records and all 531 trajectory hashes, and reproduced terminal combat accounting and aggregate metrics. Protocol statistics below were recomputed from all three policies' trajectories. No inference or simulator replay was rerun for this analysis.

### Route outcomes

| Metric | Base | GOLD-only SFT | Mixed SFT |
| --- | ---: | ---: | ---: |
| Boss victories / 80 | 4 (5.00%) | 13 (16.25%) | 26 (32.50%) |
| Survived first three combats / 80 | 73 | 66 | 80 |
| Reached first elite / 80 | 53 | 48 | 71 |
| Survived first elite / 80 | 20 | 25 | 49 |
| Boss arrivals / 80 | 18 | 23 | 44 |
| Combat victories / combats started | 341/417 | 348/415 | 477/531 |
| Decision-limit truncations | 1 | 0 | 0 |
| Decisions | 10,509 | 11,746 | 9,880 |
| Retry / fallback | 0 / 0 | 834 / 0 | 0 / 0 |

Truncations remain non-victories in the denominator. GOLD-only and mixed complete all route executions without truncation; a route ending in defeat is still a completed evaluation execution.

The mixed-minus-GOLD Boss difference is **+16.25 percentage points**: mixed alone wins 15 paired executions, GOLD alone wins 2, both win 11, and neither wins 52. Grouping the four seeds by topology gives 8 improved topologies, 0 worsened and 12 tied. A paired topology bootstrap gives a descriptive 95% interval of **[+7.50, +27.50] percentage points**; the two-sided exact topology sign test gives p=0.0078125. Against Base, the difference is +27.50 points, with 23 mixed-only wins, 1 Base-only win, 3 shared wins and 53 shared non-wins; topology counts are 10 improved, 0 worsened and 10 tied, with bootstrap interval [+13.75, +42.50] points.

Bootstrap method: average the four paired binary victory differences within each of 20 topologies, resample 20 topology means with replacement 20,000 times using NumPy `default_rng(20260917)`, and take the 2.5th/97.5th percentiles. The sign test excludes tied topologies. These are nominal development-panel summaries, not evidence from 80 independent topologies or an untouched held-out test. This panel also informed the mixed-data design.

### Early combat behavior

All first combats begin from the same 80-HP starter deck, so this slice avoids the survivor and loadout differences of later stages. Damage is gross combat HP loss, before distinguishing victory healing; it is not simply entry HP minus exit HP. Means include defeats.

| First encounter | Executions | Mean HP loss: Base / GOLD / Mixed | Mean turns: Base / GOLD / Mixed |
| --- | ---: | --- | --- |
| Cultist | 28 | 9.96 / 23.07 / 6.39 | 4.14 / 7.50 / 3.18 |
| Jaw Worm | 16 | 19.19 / 38.81 / 13.69 | 6.75 / 13.00 / 2.88 |
| Small Slimes | 12 | 17.33 / 7.00 / 8.00 | 6.92 / 7.50 / 3.25 |
| Two Louse | 24 | 17.21 / 5.29 / 10.42 | 6.83 / 6.25 / 3.00 |
| All first combats | 80 | 15.09 / 18.48 / 9.30 | 5.89 / 8.23 / 3.08 |

The large GOLD-only regressions on Cultist and Jaw Worm are reversed on this panel. This is not a uniform reduction in damage: mixed loses 1.00 more HP against Small Slimes and 5.13 more against Two Louse than GOLD-only, while still taking less damage than Base on both.

The action traces show a shift away from prolonged Defend-heavy play. Across the 28 first Cultist combats, GOLD-only plays Defend 348 times and Bash 36 times; mixed plays Defend 14 times and Bash 47 times. Across 16 first Jaw Worm combats, those counts change from 394/38 to 8/25. These are descriptive totals over trajectories of different lengths, not a controlled action-value estimate. Neither SFT policy voluntarily ends a turn in any first combat while another model action is legal. The shorter combats accompany changed card choices.

After combat three, mean HP among survivors is 58.07 for Base, 55.47 for GOLD-only and 69.78 for mixed. Counting earlier deaths as zero instead gives 52.99, 45.76 and 69.78 over all 80 executions. On the 66 executions where GOLD reaches combat four, mixed enters with mean 69.67 HP versus GOLD's 55.47, but loses 25.33 versus 21.83 HP in that combat. Higher early resource retention therefore does not imply uniformly better play in the later strong encounter.

### Formatting and later bottlenecks

| First-attempt legal output | Base | GOLD-only SFT | Mixed SFT |
| --- | ---: | ---: | ---: |
| Sole-legal END states | 2,541/2,541 (100%) | 1,821/2,655 (68.59%) | 2,193/2,193 (100%) |
| Multiple legal model actions | 7,968/7,968 (100%) | 9,091/9,091 (100%) | 7,687/7,687 (100%) |

All 834 GOLD-only retries occur in sole-legal END states; the mixed trajectories need none. These are each policy's visited states, not a fixed common formatting test. Since the old retries already recovered a legal action with zero fallback, correcting this output defect alone does not explain the route victory increase.

Mixed reaches the Boss in 44 executions versus GOLD's 23. Victory conditional on reaching the Boss is 26/44 (59.09%) versus 13/23 (56.52%); these cohorts differ. Among the 22 executions where both reach the Boss, mixed wins 16 and GOLD wins 13, but mixed also enters with more HP (52.73 versus 46.45 on average), and loadouts may differ. The evidence is consistent with a substantial contribution from improved survival/resource retention before the Boss; it does not isolate a change in Boss tactical skill.

The largest remaining death location is the first elite: 22 of mixed's 54 deaths, including 16 Lagavulin defeats out of 32 arrivals. It also loses 18 of 44 Boss arrivals. Two Louse damage and Lagavulin play are useful next diagnostic slices; the current evidence does not justify another broad data expansion or immediate attribution of all gains to Teacher labels. Any subsequent training comparison should explicitly account for the 2,237-versus-731-step budget difference.

### Original evidence

- [Mixed SFT training report](../../report/training/qwen2_5_7b_gold_teacher_mixed_sft_v7_v1.json)
- [Mixed SFT evaluation report](../../report/evaluation/qwen2_5_7b_gold_teacher_mixed_sft_act1_continuous_v7_v1.json)
- [Matched Base/GOLD-only evaluation report](../../report/evaluation/qwen2_5_7b_base_gold_sft_act1_continuous_v7_v1.json)
- `outputs/generation/qwen2-5-7b-gold-teacher-mixed-sft-act1-continuous-v7-v1/formal/inputs.json`
- `outputs/generation/qwen2-5-7b-base-gold-sft-act1-continuous-v7-v1/formal/inputs.json`

## Earlier single-stage evidence

Teacher search is privileged supervision infrastructure. The completed
public-information V2 process certified 8,221 Gold, 471 Silver, and 1,574 Abstain
states from the historical train source. Those numbers and all Teacher search
parameters are results, recorded in
`report/data/public_information_teacher_train_collection_v1.json`; they are no
longer maintained as runnable configuration in the current checkout.

The earlier Expanded SFT evaluation recorded 376/420 victories, versus Base
315/420 and privileged Teacher 408/420. The original reports are
[Expanded SFT](../../report/evaluation/expanded_sft_simulator_v1.json) and
[topology DPO](../../report/evaluation/topology_dpo_simulator_v1.json).

On the separate V5 192-combat panel, Base/Gold SFT/Silver DPO/Teacher won
32/83/86/141. Its original [SFT](../../report/evaluation/counterfactual_v2_gold_sft_v5_evaluation.json)
and [DPO](../../report/evaluation/counterfactual_v2_silver_dpo_v5_evaluation.json)
reports retain their scope. The larger focused Act-1/A0 comparison did not
establish a material DPO advantage; Gold SFT remained the default. These
panels are distinct from the focused 7B comparison above.

## GOLD supporting diagnostics

These historical panels explain tested choices and their limits. They do not
define the current collection/export contract. Original `outputs/` evidence
is retained separately from the source snapshot; the current algorithms are
specified in the [GOLD method](data_training.md#teacher-and-gold).

### Targeted order diagnostic

The completed order diagnostic (`outputs/inspection/teacher-order-probe-v1/report.json`)
compared historical first-stage and primary Gold predicates on corrected-mechanics
states. Its dedicated runner, config and tests are retired; Git `e1045b0` retains
the implementation and config recorded by the original report. The report embeds
the source-pool selection settings and reference contracts; the classifier source
revision is `d74ad724cbab405db09164b9d2b671211c36117b`. These references document
provenance, not a maintained diagnostic reproduction environment. The Silver stress
pass was not run, so primary Silver results remain provisional.

Real states are selected by public defense opportunities, independently of the
collected Teacher action. Each selected decision is restored and checked against
its recorded trajectory. Fixed branch families compare defense permutations,
omitted defense, an extra-attack tradeoff where legal, and Rage placement using
actual native transitions. The experimental conditional prefix sets retain the
best immediate HP within matching pre-END enemy states, prioritizing survival.
This restricted one-enemy-phase comparison is not a production certification
rule: it does not establish full-combat value, future draw-order equivalence, or
optimality over unenumerated branches. Raw transitions preserve those differences.
Draw and secondary-selection fixtures expose fresh observations and fresh legal
searches; the deterministic branch probe refuses to cross those information
boundaries. Reports contain no certified SFT labels or population coverage claims.

### Executed-continuation panel

The historical `teacher_gold_probe_v3` diagnostic consumed an explicit panel of
restored decisions from the fresh candidate pool. Its dedicated config is retired;
the shared GOLD executor remains maintained, and its tuning routes are retained
in the fixed source-exclusion input.
Each legal model-action class is forced once in each paired execution world;
subsequent decisions run the configured Teacher search again until combat ends.
The panel includes combination opportunities and ordinary multi-target, draw,
secondary-selection and low-HP decisions. Selection uses public opportunities,
not the recorded Teacher action or final outcome. Related source-route seed groups
remain one lineage; this targeted panel does not estimate population coverage.

### Continuation-policy comparison

The historical continuation-policy comparison retains its original
B (`outputs/inspection/teacher-continuation-policy-v1/arm-b/formal/report.json`)
and C (`outputs/inspection/teacher-continuation-policy-v1/arm-c/formal/report.json`)
reports. Its dedicated panel preparer and run configs are retired; development
commit `4ba3057` identifies their historical source, which is outside the public
Git history. No diagnostic source download or maintained reproduction environment
is provided.
The original preparer required a completed, replay-verified baseline with the
same native runtime. It sampled six states per encounter family on distinct
source routes and separately included legal Warcry opportunities. Random sampling
excludes diagnostic routes and uses no outcome or candidate-label filter. The
source collection is already an event-enriched panel; this subset is not a uniform
sample of the original route pool. The panel's `selection.json` declares the
baseline, random and diagnostic IDs, pairing, configs and primary metric.

The recorded configurations retain baseline sample IDs, execution seed, outer trials,
hidden samples, search budget and root allocation. Arm B uses internal search
thresholds 0.95 for non-Boss encounters and 0.85 for Boss; arm C uses 0.90 and 0.80.
The external family win-rate thresholds remain unchanged. Reports use a 1 HP
tolerance; the saved trials also support 5 HP rescoring. Baseline A results were
reused and require the same HP tolerance when comparing labels.
The primary comparison averages the paired carried-HP difference over all forced
roots within each state, then equally over the random states. Warcry diagnostics
are separate. Executed win rates, label changes and state-equal card distributions
are secondary; internal threshold activation and action changes can be recovered
from the saved per-step search evidence with `continuation_selection`. This changes
only how the search results select the next executed action, not the native tree
search objective. The same seeds pair starting worlds; after policies diverge,
their subsequent public states and random-event consumption can differ.

### Sampling-ladder calibration

The historical sampling-ladder diagnostic retains its
original report (`outputs/inspection/teacher-gold-ladder-v1/formal/report.json`).
Its dedicated preparer and run config are retired at the same source boundary.
Preparation consumed completed and verified collection/continuation reports and
their selection manifest. The panel retained the 24 random states and four Warcry
diagnostics from that comparison and added
four old empty-candidate Boss states from different unused routes. These three groups
remain separate in the report; outcome-selected diagnostics do not estimate population
rates. It uses the declared continuation policy, all legal roots, paired trial IDs,
and an independent output directory. Existing execution artifacts are not modified or
imported: the full 128 trials are executed for every state in this calibration run.

### Multi-target budget diagnostic

The historical multi-target budget diagnostic retains its
selection record (`outputs/inspection/teacher-gold-multitarget-v1/selection.json`)
and partial standard-budget report (`outputs/inspection/teacher-gold-multitarget-v1/budget-2048/formal/report.json`).
Its preparer and both budget variants' configs are retired. Their excluded tuning
routes are preserved in the source-bound
[GOLD exclusion input](../../assets/datasets/continuous/gold-exclusions.json);
formal training-source selection does not open the retired configs.
The diagnostic shuffled declared route/seed-group/combat slots, rejected unreached
combats or those without a public multi-target action opportunity, and sampled one eligible decision uniformly
within each accepted combat. Each source route appears once. This samples combat
slots, not decisions uniformly over the whole pool; selection never uses the
recorded Teacher action or final outcome. The initial panel has eight states.
A separate crowded selection (`outputs/inspection/teacher-gold-multitarget-crowded-v1/selection.json`)
used four states, seed 2026090905, at least three living enemies, and excluded
the initial panel's routes. The strata remain
separate when interpreting results.

Each panel declared two V3 probe configurations, using 2,048 versus 8,192
simulations per inner world, with identical states, outer/inner/search seed domains,
four inner worlds, and minimum 32 visits per native root edge. An action with
several equivalent card instances can consume several native root allocations.
The primary comparison is the state-equal mean paired carried-HP difference
(8,192 minus 2,048) over identical forced root actions; candidate-set changes,
target choices, root visits and runtime explain that comparison. More visits
do not automatically imply better decisions, and post-divergence paths no longer
share identical public states. Per-search budget, native root count and elapsed
time are included in new traces and checked during replay where applicable.

`minimum_win_rate` accepts a scalar or an explicit encounter-family map; missing
families fail. These panels initialize floors from the fresh pool's reached-battle
Teacher win rates minus one percentage point, pooling ordinary battles, and use
a five-HP tolerance. This historical sampling Teacher is an initial calibration
reference for the continuation Teacher, not a same-state baseline. The 32 outer
trials remain diagnostic estimates; no production certification is implied.


## Native adaptation check

The first local regression of the shared native resolver recorded 330 passing
tests, including native fixtures, with no skips. That historical result is not
a current test count or acceptance of a different native build. Fresh compilation
was covered through dispatch fixtures; builders retain their separate acceptance.

## Reconstruction and source evidence

The historical Boss panel contains **24 route-derived loadouts × eight seeds**,
64 combats per Boss, with 80 entry HP. Each fight is independent; preceding route
damage is not carried in. The [frozen input manifest](../../assets/eval/qwen2_5_7b_base_act1_a0_boss_inputs_v1.json)
contains exact combat identities. Historical Teacher search used 8,192 simulations
and search seed 101. The [input reconstruction report](../../report/data/counterfactual_v2_act1_a0_boss_inputs_v2.json)
matched the original panel in 131.923 seconds on Windows at development revision
`4c46b83`. Current [generation](../../configs/generation/single_boss_inputs.json)
and single evaluation use corrected mechanics; new loadouts and scores need not
match that historical engine. Generating new inputs never implicitly replaces
the frozen panel.

The early continuous V6 Base/Teacher comparison predates the hand-upgrade
correction. Its V7 Base successor also predates known draw-prefix memory.
Observation and mechanics changed together; the V6 Teacher is not a matched
current V7 control. Routes contain up to eight combats with rewards, upgrades,
removals, relics and healing, not native map/shop/event play. Settings and seed
streams are recorded in inputs; the [current panel](../../configs/panels/continuous_act1_development.json)
declares the maintained route family.

The historical Teacher candidate pool completed **800 routes × four seed groups**:
25,541 combats, 25,166 combat wins, 3,176 Boss arrivals, 2,825 Boss wins and 337,769
decisions, with no truncations/retries/fallbacks. All reached decisions, including
losing routes, were retained and raw-transition replay completed. These counts
establish collection, not GOLD label quality or route-optimal supervision.
The original report is `outputs/generation/teacher-act1-candidate-pool-v1/formal/report.json`;
source revisions are `c4d3b44d194d4938e917ecd4585b6e9d935988fe` and
`f718ab57cba8a2a462deae4318227d98267c690f`. Its V7 text predates draw memory.
The [generation config](../../configs/generation/continuous_teacher_pool.json)
and [development exclusions](../../assets/datasets/continuous/development-exclusions.json)
remain available, but the raw pool is not distributed. Current V2 route execution
cannot resume this historical V1 pool in place.

Local reconstruction on 2026-09-23 matched V4/V5 Gold (8,221 records each), Silver
(471 each) and V5 development (2,406), including both audit hashes for the restored
4,680-episode summaries. The full check took 108.3 seconds. Original certification
exports and episode sources are required and are not distributed; maintained
reconstruction does not rerun the retired certification pipeline.

The same local reconstruction matched all continuous GOLD SFT, mixed SFT and
DPO A/B/C records. Mixed export replayed all 1,440 selected Teacher combats.
GOLD replay covered **240 of 7,134 complete states**: 216 stratified random plus
24 targeted, totaling 89,664 continuations and 901,211 decisions. Other states
were not replayed. Full export equality is not full continuation verification;
the raw inputs, imported V3 trials and new report-bound receipt are undistributed.
