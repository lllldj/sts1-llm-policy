# Teacher and GOLD contract

Executed continuation scoring, collection, replay verification and source sampling.
Student-visible information is defined in [policy and observation](policy_observation.md);
exported training records are defined in [data and training](data_training.md#dataset-exports).
Run order and inputs are listed in [continuous experiments](experiments/continuous.md).

The implementation is grouped in `data/gold/`: `preparation.py` selects inputs,
`collection.py` executes resumable collection, `continuation.py` owns shared
continuation and scoring behavior, `verification.py` replays evidence,
`storage.py` maintains compact traces and imports, and the export modules produce
SFT, DPO and mixed records. Both generation and evaluation use
`data/trajectory_replay.py`; replay does not depend on an evaluation runner.

## GOLD execution and replay

The shared `scripts/collect_gold.py` entry consumes an explicit restored
decision panel, forces every legal model-action class in paired execution worlds,
and runs fresh continuation search until combat ends. Formal collection is
selected by `configs/generation/continuous_gold_collection.json`; stage ordering and
source prerequisites are in [continuous](experiments/continuous.md). Historical
panel designs are retained under [GOLD supporting diagnostics](stageresult.md#gold-supporting-diagnostics).

### Continuation scoring

Outer execution worlds and inner search worlds use separate seed domains.
`fresh_future_rng_known_draw_top_v1` preserves the current battle state and its
observed top-of-draw-pile prefix, permutes only the unknown draw remainder, and
starts fresh independent AI, random-card, miscellaneous, monster-HP, potion and
shuffle RNG streams. It is an explicitly defined conditional-state experiment,
not posterior inference over original game seeds. The current enemy intention
remains fixed. Inner worlds are independently sampled from each new public
decision; the actual outer future is not used to choose an action. Native tree
search inside each sampled world remains privileged, and retains its finite-budget
search bias. Model-action win-rate estimates are averaged with equal weight per
inner world, then victory HP breaks ties and action ID makes exact ties deterministic.
This is the default `win_rate_then_hp` continuation policy. An explicit
`continuation_policy` with mode `win_floor_then_expected_hp` instead admits actions
whose mean internal search win rate reaches `minimum_search_win_rate`, then maximizes
the equally weighted mean of `win_rate * victory_ending_hp_mean` across inner worlds.
Equal expected HP is broken by mean search win rate and then action ID. When no
action reaches the threshold, it falls back to the default policy. The search
threshold may be a scalar or a mapping by encounter family; missing families are
errors. It is independent of the external `minimum_win_rate` certification gate.
These internal scores select the continuation; they are not the reported root value.

`minimum_win_rate` accepts a scalar or an explicit encounter-family map; missing
families fail. The report scores complete executed battles: victory contributes settled ending
HP, defeat contributes zero. Estimated acceptable actions first meet the configured
win-rate floor and then fall within the configured expected-HP tolerance of the
best eligible candidate. Each state's win-rate lower bounds use one-sided exact
binomial intervals with Bonferroni adjustment across its root actions. Paired HP
differences and standard errors are descriptive; resource differences in max HP
or surviving relic counters are recorded without an HP exchange rate. Collection
does not certify HP non-inferiority, export Gold/Silver labels, or train.
An unfinished execution is `aborted`, makes the state partial, and is never counted
as a defeat or silently replaced with another seed.

### Execution, verification and receipts

Use `--mode preflight` for restoration, observation-preservation and root-budget checks;
`--mode smoke` uses one paired trial for the config's declared smoke subset in a separate
directory. The ordinary run resumes completed state/trial/action records. Ctrl+C
preserves completed records; repeat the same command to rerun unfinished executions.
Input semantics or native-runtime changes require a different output directory.
Before collecting or reusing records, the runner binds the source pool report,
the selected route reports and their consumed combat trajectories in
`teacher_gold_source_identity_v1`. JSON inputs use canonical content hashes;
trajectories use file-content hashes, checked against declared hashes when present.
Each shared source file is checked once per invocation. Changing source content at
the same path rejects resume, and explicit full-trace imports require the same source
binding. A nonempty output without an identity, or an older identity without this
binding, cannot be adopted for continuation. Use a fresh output directory for new runs.
New collection reports carry this binding and replay verifies it too. Historical
reports remain readable and replayable under their recorded checks; they are not
rewritten or upgraded into resumable runs by adding a digest.
Each execution writes a gzip trajectory plus terminal result before becoming
resumable. `--verify` independently replays final traces without search, checks
observations, chosen actions against saved search evidence, terminal scores, counts
and reported statistics, applying the recorded continuation policy and encounter
threshold. A successful CLI verification writes `verification.json` alongside the
execution report. Combine `--mode smoke --verify` to verify the short run. Existing
receipts are never overwritten: use `--verification-output` with a new path
relative to the project root when repeating verification. The destination is
checked before replay; a failed replay writes no receipt.

The replay receipt includes `report_canonical_sha256`, binding it to the report
actually replayed. It is SHA-256 of the parsed report serialized as UTF-8 JSON with
sorted object keys, compact separators, unescaped Unicode and no non-finite numbers.
Formatting, line endings and object-key order do not affect this digest; all report
fields and array order do. Verification also rejects a report that changes during
replay. This binds report content, not its filesystem location.

SFT and DPO exports reject missing or mismatched bindings, even if all counts agree.
Historical receipts without this field remain historical evidence; do not add a
digest to them after the fact. A new export requires actual replay verification
into a new receipt, then a new export config selecting that receipt and an unused
dataset output directory. Archived configs retain their original receipt paths.
Existing reports, dataset manifests and training inputs are not migrated.

Replay defaults to the complete collection. `--verification-selection <path>`
instead selects complete states from a JSON document with
`schema_version: "gold_replay_selection_v1"`, `report_canonical_sha256`, an integer
`seed`, and `groups`, each containing a unique `name` and a nonempty `state_ids`
list. IDs must be unique across groups and form a proper subset of the report.
Fix the selection before replay; group names can distinguish stratified random
samples from targeted mechanism checks. Every selected state's roots, trials,
terminal scores and ladder decisions are still checked in full. Collection-wide
card and ladder summaries are also recomputed from the report; this does not replay
unselected trajectories. A failed replay writes no final receipt; repeating an
interrupted replay starts the selected states again.

The receipt records `scope` (`full` or `sampled`), the selection, `verified_states`,
`total_states`, actual replay counts and elapsed time. SFT/DPO export configs default
to `verification_scope: "full"`; consuming a sampled receipt requires explicitly
setting `verification_scope: "sampled"`. Export still validates the complete report
and derives all eligible records. Replay counts are checked against the selected
states, and the receipt is preserved in each new dataset's lineage. Full export
equality and sampled trajectory agreement are separate claims. A sample does not
certify unselected executions or establish a population error rate.

Verification rejects a different recorded observation version before replay;
historical artifacts require the implementation matching their recorded contract.
New reports capture Git revision and worktree status at run start, marked by
`git_provenance_at: run_start`. Earlier reports captured these fields at completion,
so they alone do not identify code already loaded by a running process.

### Paired sampling ladder

`sampling_ladder` supports `mode: shadow` and `mode: adaptive`, stages `[32, 64, 128]`,
and floor-rounded external gates. It checkpoints each complete paired stage and
records whether the proposed scheduler would expand. At 32, proper candidate subsets
advance to 64. Empty sets stop if every root is at least two wins below the gate;
all-pass sets stop only outside the HP/win boundary rules. At 64, a candidate-set
change or boundary triggers 128. The initial HP boundary is within 0.25 HP of the
configured tolerance. Win-boundary roots have exactly the required wins or one fewer,
and must have expected HP at least best eligible HP minus tolerance minus the boundary
band; when no root is eligible, all roots are relevant. Thus full wins do not themselves
trigger expansion at the configured high non-Boss gates. Boundary decisions use the
current empirical scores and may miss high-variance roots.

A deterministic hash of the audit seed, sample ID and stage advances 10% of otherwise
stopped states by one stage; all their roots advance together. Shadow execution always
continues to 128, allowing inspection of every proposed early stop, including those
outside the audit subset. The primary metric is the fraction of hypothetical early
stops whose candidate set differs at 128, separately per panel group. Actual recorded
search-call counts estimate savings relative to the complete 128 reference. Stage
reports also retain empirical labels, boundary/flip flags and reference sensitivity;
these flags do not alter labels or constitute certified SFT/DPO exports. The 128 result
is a higher-budget empirical reference, not ground truth. `--verify` replays executions
and recomputes all stage decisions and group statistics from the trial records.
The regular one-trial `--mode smoke` checks the execution/replay path; it disables the ladder.
The configuration embedded in the historical shadow report retains its original
trace format and fixed HP band.

### Adaptive collection and compact storage

The formal training-candidate collection uses
`configs/generation/continuous_gold_collection.json`: 480 source routes, one seed group per
route, six workers, continuation policy B, and the event-mixture quotas described
under [bounded route sampling](#bounded-route-sampling-for-candidate-distributions). Its
`source_partition` excludes every route declared by the earlier collection, mechanism
and multi-target diagnostic configs. Remaining unselected routes are reserved; this
collection does not draw validation states or add extra mechanism/end-turn quotas.
The preparer accepts source-bound `--exclusions`, `--training-candidates`,
`--adaptive`, `--hp-tolerance`, `--trace-observation`, `--compact-storage`, and `--route-progress` arguments;
these control the generated execution config and recorded selection. Exclusion
probabilities are conditional on the eligible route population. Route lineage,
including all alternative seed groups, stays within one partition.

In adaptive mode the runner stops only after all roots finish a paired stage. At 64,
`paired_se_multiplier_at_64: 2` additionally expands when a win-eligible root's HP gap
to the deterministic best eligible reference is within
`max(hp_boundary_band, 2 * paired_standard_error)` of the HP tolerance. This preserves
the 32-trial rules and does not turn descriptive uncertainty into a confidence claim.
Reports record each state's actual trial count, stage decisions, terminal outcomes,
and actual stop distribution. They do not report unobserved full-128 disagreement or
search savings. Replaying the records must reproduce the stopping decision as well
as the scores. A stopped 32/64 state cannot be expanded by editing its config in place;
resume requires the same declared execution and scheduling semantics.

`trace_observation: reconstruct` omits intermediate serialized observation text.
Raw state (including known draw-order memory), selected actions, all continuation
search evidence, terminal scores and trial seeds remain stored. Source-prefix replay
restores route context and public memory, and the root V7 text is stored once per
selected state and checked during verification. Intermediate raw-state equality,
action legality, search-rule selection and terminal equality remain checked; no
comparison against a nonexistent saved intermediate text is claimed. Historical
configs default to `trace_observation: full`. This removes derived text, not all
reconstructible simulator state or search records, and does not change Teacher inputs.

The V4 collection adds `continuation_storage` with `mode: compact_v1`,
`full_trace_probability: 0.01`, `keep_first_trial: true`, and
`keep_boundary_trials: true`. A finished state first undergoes full trace replay,
including Teacher selection and all terminal scores. It then atomically writes one
`continuations.json.gz` containing every root/trial terminal row and executed model
action sequence, plus the pre-compaction verification counts. Rows retain wins and
losses, current/max HP, relic counters, decision/search counts and trial identity;
the archive includes rejected roots and all-pass states. Root observation and action
metadata remain in the state report. Per-trial JSON files and unselected full traces
are removed only after the archive roundtrip equals the verified content.

Full traces are retained for trial zero of every root, a deterministic 1% sample of
root/trial executions, and the last trial at a stage for roots marked HP/win-boundary
or whose candidate membership changed since the preceding stage. These categories
overlap; 1% is not the total retained fraction. The boundary rule provides diagnostic
examples, not all executions that contributed to uncertainty. An unfinished state
keeps its full per-trial artifacts, bounding temporary storage by active states.
Interrupts cancel queued states and finish stopping active work at execution/replay
boundaries; complete trials remain resumable.

Compact `--verify` replays **every** action sequence within its selected states
(all states by default) and checks terminal scores,
source-root metadata, summaries and ladder decisions. It additionally checks raw
states and recorded Teacher search selection for retained full traces. Its
`full_search_trace_executions` count distinguishes that scope from
`verified_executions`; deleted search evidence cannot be independently audited
again from actions alone. HP tolerance, external win gates, resource utility and
export weights can be recalculated from terminal rows. Such rescoring does not
provide unexecuted higher-sample trials or the outcome of another continuation policy.

An explicit `resume_from` imports a stopped full-trace run into a separate output.
The importer requires identical execution inputs, panel and native binary, excluding
only worker count, output and storage/import settings. It records the source identity,
original run provenance, source report checksum and transferred file checksums.
Completed states are verified and compacted without Teacher search; incomplete states
reuse completed trials and search only the missing ones. Source files remain unchanged.
After all available source states have import receipts, resuming the new run no longer
requires the old GOLD directory. The original candidate pool is still required for
root restoration: V7 observation alone is not a simulator snapshot. SFT/DPO exports
are derived products and do not replace these source/terminal/replay assets.

With `route_progress: true`, selection, preflight, collection and verification print
explicit route percentages. Collection/verification count a route complete only
after every selected state on that route has finished; state percentages are printed
alongside it. Collection emits updates every eight paired trials and at each stage.
Completed state reports are reused after the run identity check; partial states resume
their completed trial/action files. Verification reads every execution, including
records reused on resume, with the evidence scope described above. These are candidate-collection reports;
removing all-pass states and assigning equal total SFT weight per state remain export
operations and do not delete raw candidate evidence.

### Bounded route sampling for candidate distributions

Preparation and execution share configuration validation and seed derivation in
`data/gold/configuration.py`. Preparing a panel does not import the GOLD executor,
simulator runtime or model libraries. Continuation-policy, sampling-ladder and
storage settings are checked by the same rules before execution.

`scripts/prepare_gold_collection.py` reuses the executed-continuation runner with
an explicit random panel from the fresh candidate pool. It samples distinct routes
uniformly, then the requested number of seed groups uniformly within each route.
Every reached combat contributes up to one decision for a weak ordinary combat,
two for a strong ordinary or elite combat, and four for a Boss. The default
`event_mixture_v1` excludes states with only one legal **model** action, including
forced end turn and equivalent-copy-only selections. Voluntary end turn remains
among the roots whenever alternatives exist. Unreached combats and combats with
no eligible decisions are not replaced; shortages are recorded.

Each slot independently chooses a sampling channel: 40% turn starts, 30% event
states, 30% uniform over all remaining eligible states. Sampling is without
replacement within the combat. Empty channels transfer their mass to the uniform
channel. These are channel probabilities, not hard quotas or exclusive state
categories. A uniformly drawn state can also be a turn start or event state.

Six mechanism tags cover hand access, energy/cost changes, secondary selection,
upgrades/known-top control, exhausting other hand cards, and subsequent-play rule
changes. Legal-action opportunities are tagged even if the source Teacher chose
another action. Result states use same-turn public transitions; ordinary turn-start
draw/energy is not a second event. Card mechanisms and conservative multiset
changes are implemented in `src/sts1_llm_policy/data/gold/state_sampling.py`;
position-based simulator UUIDs are not treated as persistent card identities.
Potential mechanisms are sampling hints, not a claim that an effect will resolve
or yield a useful label. Outcome, hidden pile order and Teacher scores do not
participate in sampling.

Each event state is assigned once to its least frequent event tag within the
eligible states of that combat (lexical tie break). The event channel samples
nonempty assigned buckets uniformly, then states uniformly. This limits domination
by frequent draw events without multiplying the weight of multi-tag states.
`selection.json` records source positions, channel and event tags, draw order,
eligible/omitted counts and the total conditional probability of each draw given
previous draws. This is **not** a marginal inclusion probability for the final
panel. Historical `uniform_decisions_v1` remains explicit and records its true
per-combat inclusion probability; it does not apply the new eligibility filter.

The preparer requires explicit `--template-config`, `--config-output`, `--output`
and `--groups-per-route`, and defaults to 80 routes. Business logic is in
`src/sts1_llm_policy/data/gold/preparation.py`; it does not import a diagnostic script.
`--route-count`, `--weak-states`, `--strong-states`, `--elite-states`, `--boss-states`,
`--seed`, `--sampling`, `--turn-start-weight`, `--event-weight`, `--random-weight`,
`--workers`, `--template-config`, `--config-output` and `--output` control actual
sampling or execution inputs. Weights must sum to one with a positive uniform
fraction. Both generated documents are validated and checked for destination
conflicts before either is written; existing identical inputs are reusable.
The template declares the source pool and continuation search settings. Generated configs use
the source Teacher's family win rates minus one percentage point, a five-HP
tolerance, `win_floor_rounding: floor` and `card_statistics: state_equal_cards_v1`.
The earlier 80-route collection config is retired; its excluded tuning routes
remain in `assets/datasets/continuous/gold-exclusions.json`.
The formal entry is `configs/generation/continuous_gold_collection.json`; its recorded
panel can be executed or resumed without preparing a new panel. Its `selection`
path references the ordered samples and source partition; loading expands them
before existing validation, report snapshots and resume comparisons. Inline
samples remain valid, but cannot override a referenced selection. Mixed SFT
reads that same fixed input directly. The extraction preserves the historical
expanded configuration, including provenance strings; retired config names in
`exclude_configs` are never opened. New exclusions bind a source report by path
and SHA-256, so route numbers cannot be silently reused for another pool. The shared
`scripts/collect_gold.py` requires explicit `--config`; `--verify` replays
the final traces without search and `--verification-output` selects a new receipt.

For a new panel, preparation reads the template execution settings without opening
its old selection file. It discards old samples, smoke membership, panel groups, source
partition, resume source, ladder, compact-storage policy and progress switch.
It rebuilds those fields only from the supplied preparation options; explicit
`--trace-observation full` also overrides a template containing `reconstruct`.
The source and continuation policy remain inherited. Training-candidate preparation
requires an explicit `--exclusions` document and recomputes selected/excluded/reserved
route membership. It does not reuse a previous panel's partition or import trials.
Use separate config/output destinations for a new panel; the frozen collection's
existing import source and completed outputs retain their original identities.

With `win_floor_rounding: floor`, the minimum number of wins is
`floor(outer_trials * minimum_win_rate)`. At a 98.9% target this accepts 31/32 or
126/128; the report includes the actual integer requirement and effective rate.
An incomplete state produces no candidates. Exact binomial lower bounds remain
descriptive and do not gate acceptance. Omitting the option preserves historical
exact point-rate thresholds and report statistics. A one-trial smoke tests the
execution pipeline, not selection quality; the same rounding gives a zero-win
minimum there. Normal collection uses the configured outer trial count.

The completed execution report includes distributions before and after screening,
both overall and by encounter family. Each state has equal weight; each nonempty
state divides that weight uniformly among its distinct public card semantics.
The card view merges targets and indistinguishable copies, retaining different
upgrades or other differing public semantics. The action view keeps every legal
model action and target. Empty candidate states are reported separately; they do
not contribute a fictitious card to the candidate composition denominator.

Card types `ATTACK`, `SKILL` and `POWER` are distinct from defense mechanic tags.
Direct block includes Defend, Iron Wave, Shrug It Off, Flame Barrier, Ghostly Armor,
Power Through, Sentinel, Impervious, True Grit and Armaments. Mitigation includes
Clothesline, Uppercut, Disarm, Intimidate and Shockwave. Conditional, delayed or
retained block includes Rage, Second Wind, Entrench, Feel No Pain, Metallicize and
Barricade. Their union is the defense category; these tags describe intrinsic card
mechanics, not realized defensive benefit in the current state. Attack and defense
therefore overlap. Strike, Defend and Bash, including upgrades, are basic cards;
Strike plus Defend is also reported separately. Status/curse plays, end turn and
secondary selections remain in the action view but outside playable-card shares.

The primary metric is the fraction of states with a legal defense-card opportunity
whose candidate set retains at least one defense card. Category-specific opportunity
counts, retention fractions, card composition, empty/all-action candidate sets and
unpriced persistent-resource conflicts accompany it. These are distributions of
the stratified sampled panel, not unbiased estimates of all pool decisions.
No SFT export or student training is performed. `--verify` checks the restored card
metadata and recomputes these distributions in addition to replaying every trial.
