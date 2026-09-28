# Teacher candidate pool: 800 × 4

Source-data component of [continuous](continuous.md). GOLD certification and
student training are downstream steps documented on the stage page.
The original pool is an external asset; its
[download status](continuous.md#downloads-and-starting-points) is separate from
the recorded completion status below.

## Purpose, status and results

Completed fresh candidate collection: 800 source routes × 4 formal seed groups
= 3200 route executions, not 3200 independent combats. This is the current source
for candidate sampling and certification experiments. All decisions are retained,
including losing routes; no old Gold filter or victory-only selection is applied.

| Metric | Recorded result |
| --- | --- |
| Route executions | 3200 |
| Actual combats / maximum | 25541 / 25600 |
| Combat victories | 25166 |
| Boss arrivals | 3176 |
| Boss victories | 2825 / 3200 |
| Decisions | 337769 |
| Truncations / retry / fallback | 0 / 0 / 0 |

The original report (`outputs/generation/teacher-act1-candidate-pool-v1/formal/report.json`)
records `status=completed`; collection and raw-transition replay verification are
complete. Completion, quantity and replay integrity are the collection criteria;
Boss wins describe the generated trajectories, not Gold label quality. Collection
does not establish certification, a training dataset, training completion or
route-optimal supervision. The collection report does not establish public
availability of the raw assets.

## Inputs and configuration

[Run configuration](../../../configs/generation/continuous_teacher_pool.json):
`continuous_combat_panel_generation_v1`, run ID `teacher_act1_candidate_pool_v1`.
Entry: [run_continuous.py](../../../scripts/run_continuous.py).
Mechanics/reward scope: [card-selection scope](../../../configs/env/d_expansion_card_selection_v1_scope.json).
Picker asset: `assets/picker/card_pick_metrics_v1.sqlite3`. The original
report embeds the run configuration, resolved configuration/asset identity,
execution target and source-isolation evidence. No model, checkpoint or GPU is used.

revision.txt (`outputs/generation/teacher-act1-candidate-pool-v1/revision.txt`)
records `c4d3b44d194d4938e917ecd4585b6e9d935988fe` and
`f718ab57cba8a2a462deae4318227d98267c690f`: original collection and continuation
with worker support. The later aggregate-key correction at `3241efc` did not
rewrite the original evidence. Windows execution uses the corrected card-selection
extension over native revision `7476a81954020087da31d41d16fddf475746ec2d`;
the exact resolved binary/asset identity in the report distinguishes it from
earlier builds sharing that upstream revision.

## Route, observation and algorithms

Ironclad, Act 1/A0; Burning Blood initially. Combat floors 1/3/5/7/9/11/13/15:

```text
weak → pick → weak → pick → weak → pick → upgrade → remove → relic
→ strong → pick → upgrade → elite → relic → pick → strong → pick
→ remove → heal 24 → upgrade → elite → relic → pick → upgrade → Boss
```

Victory carries current/max HP and native relic counters; death stops the route.
Healing caps at max HP. Maximum 1000 decisions per combat; truncation is retained
as an abort, not a gameplay defeat. Category encounter bags exhaust before repeat.
The full relic pool is in the run config, card/encounter scope in the linked scope.

The recorded interface is `observation_v7` / `combat_card_selection_v1`, including
route context, secondary choices and corrected upgrade previews. It predates the
later V7 known draw-prefix memory change. Do not reinterpret saved observations
using the latest serializer simply because their version string matches.
Student-visible text excludes hidden order, future random outcomes and seeds.
Teacher consumes native search state, not the V7 text; the observations' route
objective does not make its local search route-aware.

| Consumer | Algorithm/settings |
| --- | --- |
| Combat policy | Teacher only; `BattleScumSearcher2`, 8192 simulations per decision, search seed 101 |
| Card pick | `teacher_counterfactual`, picker epsilon .02; 8192 search, seed 101, hidden-order seeds 701/809/907 |
| Pick comparison | Opening multiplier 1.5; 2 pair wins vs skip; conflict seed offsets 100000/200000 |
| Pick weights | search/intrinsic/optionality .5/.35/.15; current/future .7/.3; duplicate .08; premium/strong override .15/.08 |
| Upgrade | `teacher_combat_priority`; all Act-1 elites + public route Boss; 3 seeds per target, 12 cells per candidate |
| Upgrade search | 2048 simulations, seed 101, hidden-order seed 1103; mean win rate, victorious ending HP, then evaluation |
| Remove / route | `starter_alternating` / `configured_act1_survival` |

The upgrade panel uses independent seeds and does not inspect the actual next
monster. Reward evaluation separately uses the configured final Boss combat seed
and conflict offsets. These generation algorithms match the configured continuous
route family; they are not the later forced-action external-execution Gold probe.

## Seeds and source isolation

| Stream | Base |
| --- | --- |
| route | 202610000000 |
| encounter | 202611000000 |
| relic | 202612000000 |
| reward | 202613000000 |
| combat | 202614000000 |
| policy | 202615000000 |
| picker | 202616000000 |
| upgrade | 202617000000 |
| remove | 202618000000 |
| upgrade_combat | 202619000000 |

Route RNGs use base + route index. Formal combat/policy seed offsets are
`(route_index * 4 + group_index) * 8 + combat_index - 1`, with zero-based routes
and groups, one-based combats. Upgrade seeds allocate 48 cells per route
(4 upgrades × 4 targets × 3 replicates). Completed inputs retain exact schedules.
Worker count changes concurrency, not seeds or source identity; collection
continued with two local workers.

`data_source.evidence_class=teacher_candidate_pool` explicitly excludes the
V6 and V7 development sources through
[development-exclusions.json](../../../assets/datasets/continuous/development-exclusions.json).
Preflight reads the declared source-route identities and exact seed intervals,
including upgrade panels and reward conflict
offsets. It does not read their trajectories. Recurring opening hands do not
constitute source overlap. All groups/arms of one source route must share any
future train/dev assignment; source lineage is metadata, never model input.

## Outputs and verification boundary

Root: teacher-act1-candidate-pool-v1 (`outputs/generation/teacher-act1-candidate-pool-v1`).
`formal/report.json` summarizes the execution; `formal/inputs.json` stores the
declared inputs. `formal/teacher/` retains route/group completion records and
`route-work/route-NNN/combat-seed-group-NN/combat-NN.jsonl` trajectories.
Each combat retains the starting V2 snapshot, scenario and combat seed, action
representatives, raw transitions, public observations and route lineage.

Every new combat is replayed without search before finalization. Replay compares
raw states and public observations, excluding only session decision IDs from
state equality. [trajectory_replay.py](../../../src/sts1_llm_policy/data/trajectory_replay.py)
can restore an intermediate decision via `stop_before`, including secondary
choices. Existing full-pool verification need not be repeated to inspect samples.
Under the original implementation, completed route/groups resume and an interrupted
unfinished route is regenerated. Current V2 execution requires a new output
directory for this V1 pool; the original evidence remains readable by its consumers.
Original trajectories remain bound to their generation code and native build.

Search-score labels and certification judgments are not source-pool fields.
Later certification outputs belong to their own experiment and must not rewrite
this pool's identity or turn the 20-route development baseline into training data.
