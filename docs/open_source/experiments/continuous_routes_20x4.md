# Continuous Act-1 routes: 20 × 4

Historical evaluation and configuration-change evidence within
[continuous](continuous.md); this page does not define a separate experiment stage.

## Scope, status and results

This record covers the early Base/Teacher experiments. Later V7 GOLD-only SFT,
mixed SFT and DPO A/B/C results and their original reports are indexed in
[current offline results](../../../README.md#current-offline-results).

Ironclad, Act 1/A0, 20 source routes with four formal combat/policy seed groups
per route. Each arm executes 80 routes, at most eight combats each. Both Base
versions are frozen and no longer used in ongoing experiments. The completed V6
Teacher remains a historical comparison baseline; it is not a corrected V7 rerun.

| Recorded arm | Boss wins / 80 | Boss arrivals | Combat wins | Decisions | Truncations |
| --- | --- | --- | --- | --- | --- |
| V6 7B Base | 1 | 12 | 285 | 6295 | 0 |
| V6 Teacher | 68 | 76 | 616 | 8251 | 0 |
| Corrected V7 7B Base | 4 | 18 | 341 | 10509 | 1 |

All three recorded zero retry/fallback. Primary metric is final Boss completion;
truncations remain non-victories in the denominator but are distinct from deaths.
Four groups share a route schedule, so topology-level comparisons have 20 source
clusters, not 80 independent routes. [V6 results](../stageresult.md#7b-base-and-teacher-on-continuous-routes)
and [V7 results](../stageresult.md#7b-base-with-observation-v7-on-continuous-routes)
identify the original reports, inputs and paired analysis.

## Inputs and execution

| Version | Configuration | Recorded source revision |
| --- | --- | --- |
| V6 | `continuous_v1` (retired) | `9653109729e921265954e86c0f4fe998f4037c22` |
| V7 upgrade fix | `continuous_v7` (retired) | `9425277df1cce52e4db23b59d2d72ce7d4ee30fb` |

Both historically used `continuous_combat_panel_generation_v1` through
[run_continuous.py](../../../scripts/run_continuous.py). The original configs
declared Base and Teacher arms; the recorded V7 execution is Base only. Its
Teacher comparison reuses V6 evidence. The mechanics/reward scope is
[d_expansion_card_selection_v1_scope.json](../../../configs/env/d_expansion_card_selection_v1_scope.json);
picker statistics come from `assets/picker/card_pick_metrics_v1.sqlite3`.
Resolved scope, database, runtime and native identities are recorded in the run.
The current checkout retains their explicit route/seed exclusions in
[development-exclusions.json](../../../assets/datasets/continuous/development-exclusions.json),
without requiring these old execution configs.

Base is Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`,
without adapter. [Runtime](../../../configs/runtime/base_model_runtime_7b_v2.json):
BF16/SDPA, no quantization, greedy batch 1, maximum 8 new tokens, no constrained
decoding or chat history. [Execution profile](../../../configs/runtime/training_cuda_bf16.json)
requires CUDA and native BF16. Teacher requires the native simulator but no model weights.

## Observation and action contract

V6 uses `observation_v6` / `combat_card_selection_v1`, adding Armaments, Headbutt,
Exhume, Burning Pact, True Grit, Warcry and Dual Wield secondary choices to the
public combat interface. Every model decision, including a secondary choice,
returns one legal-list `ACTION_n` through the shared parser/retry/fallback path.

V7 requires route context: current position/type, remaining ordered operations,
public Boss identity and healing amounts. It provides upgrade previews and
secondary-choice source/effect details. Future ordinary monster identities,
random rewards/relics, seed values and hidden draw order are not model input.
The model objective is final Boss completion, allowing current combat risk versus
future HP/resources. The native Teacher does not consume that text: it remains
local `BattleScumSearcher2` combat search rather than a route-objective optimizer.

The V7 run also corrected `UpgradeAllCardsInHand` to upgrade only eligible cards.
V6 results predate this correction, so observed differences do not isolate V7
prompt effects. These historical V7 trajectories also predate the later known
draw-prefix memory addition; current V7 text is not their original serializer.
See the [observation contract](../policy_observation.md#continuous-route-observation-v7).

## Route and algorithms

Initial relic: Burning Blood. Combat floors: 1, 3, 5, 7, 9, 11, 13, 15.
The exact operation sequence is:

```text
weak → pick → weak → pick → weak → pick → upgrade → remove → relic
→ strong → pick → upgrade → elite → relic → pick → strong → pick
→ remove → heal 24 → upgrade → elite → relic → pick → upgrade → Boss
```

This is an abstract configured route, not a native map/shop/event run. Victory
carries current/max HP and native relic counters forward; healing caps at max HP.
Death stops the route. The decision limit is 1000 per combat. Category bags for
weak/strong/elite encounters prevent repeats until their respective pools exhaust.
Card and relic pools are enumerated by the linked scope and run configuration.

| Consumer | Algorithm and exact settings |
| --- | --- |
| Actual Teacher combat | `teacher_search`, `BattleScumSearcher2`, 8192 simulations, search seed 101 |
| Reward selection | `teacher_counterfactual`; picker epsilon .02; search 8192, search seed 101, hidden-order seeds 701/809/907 |
| Reward comparison | Opening-damage multiplier 1.5; pair wins vs skip 2; conflict combat offsets 100000/200000 |
| Reward strategic weights | search/intrinsic/optionality .5/.35/.15; current/future .7/.3; duplicate .08; premium/strong override .15/.08 |
| Upgrade selection | `teacher_combat_priority`; all three Act-1 elites plus route-visible Boss, 3 combat seeds each, 12 target cells per candidate |
| Upgrade search | 2048 simulations, search seed 101, hidden-order seed 1103; rank by mean win rate, victorious ending HP, then evaluation |
| Removal / route | `starter_alternating` / `configured_act1_survival` |

Upgrade candidates share identical evaluation cells. The upgrade panel excludes
the actual next monster and formal combat seed. These upgrade seeds are separate
from actual combat seeds. Reward search is a separate consumer: its evaluator is
initialized from the configured final Boss combat seed, with the declared conflict
offsets. The two search protocols must not be conflated with label certification.

## Seeds and sharing

| Stream | Base (both versions) |
| --- | --- |
| route | 202609070000 |
| encounter | 202609070100 |
| relic | 202609070200 |
| reward | 202609071000 |
| combat | 202609072000 |
| policy | 202609073000 |
| picker | 202609074000 |
| upgrade | 202609075000 |
| remove | 202609076000 |
| upgrade_combat | 202609077000 |

Route-level RNGs use base + zero-based route index. For route `r`, group `g`
(0–3), combat `c` (1–8), formal combat/policy offset is `(r * 4 + g) * 8 + c - 1`.
Upgrade evaluation offsets allocate 4 upgrades × 12 cells per route, then the
upgrade position, target and replicate. Exact schedules are stored in inputs.
Arms/groups share source scheduling, but survival and accumulated state can
change the operations reached and their resulting choices.

## Outputs, recovery and limits

Output roots are `outputs/generation/qwen2-5-7b-base-teacher-act1-continuous-v1/`
and `outputs/generation/qwen2-5-7b-base-teacher-act1-continuous-v7-upgrade-fix/`.
Formal combined runs write `formal/report.json` and `formal/inputs.json`; selected
arms write `report-ID.json` and `inputs-ID.json`. Route/group summaries, starting
snapshots, reward/upgrade decisions and per-combat JSONL are retained below each
arm. Raw evidence is retained locally and is not distributed; its logical run-relative
locations and original identities are recorded in the result sections above.

Completed route/group caches are resumable; an unfinished route is regenerated.
Combining completed arms validates caches without rerunning them. Local review
completed for the reported results. The retained V6 Teacher is
usable only with its historical interface/engine boundary explicitly stated.
Neither development trajectories nor Teacher actions are certified training labels.
