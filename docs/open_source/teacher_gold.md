# Teacher and GOLD

The Teacher searches the simulator for useful actions. GOLD evaluates those
choices through repeated completed combats, then supplies action sets and
preferences for [SFT and DPO](data_training.md). It estimates local combat quality;
it does not prove optimal play or optimize the full route objective.

## GOLD execution and replay

The [collection config](../../configs/generation/continuous_gold_collection.json)
selects source states, search budget, continuation policy and sampling schedule.
At each selected state, collection forces every legal model-action class, uses
paired execution seeds, and plays each continuation until combat ends. The
original candidate pool is needed to restore these states; a V7 prompt alone is
not a simulator snapshot.

### Continuation scoring

Outer execution worlds and inner search worlds use separate random streams.
Known top-of-draw-pile cards are preserved; the unknown remainder and future RNG
are resampled under the declared public-state contract. The current enemy intent
stays fixed. This is a conditional-state experiment, not reconstruction of the
original game's hidden seed. Native search sees each sampled world's internal
state, but never uses the actual execution world's future to select an action.

At each continuation decision, search estimates are averaged across inner worlds.
The default policy ranks win rate, then victory HP. The formal collection instead
uses **win floor, then expected HP**: consider actions reaching the configured
internal win threshold, then maximize expected carried HP. If none qualify, fall
back to win-rate priority. This internal search rule is separate from the external
acceptance gate computed from completed battles.

A victory contributes settled ending HP; defeat contributes zero. Acceptable
actions meet the external win floor and lie within the HP tolerance of the best
eligible action. The formal collection uses a 1-HP tolerance. With floor rounding,
the win requirement is `floor(trials × target rate)`: a 98.9% target permits 31/32
or 126/128 wins. Incomplete continuations produce no candidates and are not counted
as defeats. Max-HP and relic-resource differences are recorded without an invented
HP exchange rate.

Reported binomial bounds and paired standard errors describe uncertainty; they
do not certify HP non-inferiority. The collection does not itself export labels.
[Export rules](data_training.md#dataset-exports) decide which states and actions
enter training.

### Bounded route sampling for candidate distributions

The source pool contains independent training-source routes. Preparation samples
routes and seed groups, then reached combat states. Formal collection selects
480 routes with one seed group each. All groups belonging to a source route stay
in one partition; tuning and development routes are excluded explicitly.

The default per-combat limits are one weak-combat state, two strong/elite states,
and four Boss states. States with only one legal model action are excluded;
voluntary end turn remains a candidate when alternatives exist. Unreached or
ineligible combats are not replaced.

Each draw selects a channel: **40% turn start, 30% event, 30% uniform**. Draws are
without replacement, and empty channels transfer their mass to uniform sampling.
These are probabilities, not fixed quotas. Event tags cover hand access,
energy/cost, secondary choices, upgrades/known-top control, exhaust and effects on
later plays. Opportunities can count even if the source Teacher chose another
move. Outcome, hidden order and Teacher scores do not guide sampling.

Multi-tag states are assigned once to their least frequent local tag; event
buckets are sampled evenly. Recorded probabilities are conditional on earlier
draws, not final marginal inclusion probabilities. This enriched panel therefore
does not estimate the distribution of all pool decisions without bias.

`prepare_gold_collection.py` reads an explicit template/source and writes a new
selection and execution config. Training selections require a source-bound
exclusion file; route numbers from another pool cannot be reused. Old selections
and trial imports are cleared when preparing a new panel. Settings and destinations
are explicit; use `--help` and [source isolation](experiments/continuous.md#selection-and-source-isolation)
for the entry points. The fixed public selection does not contain the raw pool.

### Paired sampling ladder

Collection uses **32 → 64 → 128** paired trials to spend more effort on uncertain
states. All actions in a state finish a stage together. At 32, proper candidate
subsets continue; clearly empty or all-pass sets may stop. At 64, changed candidates
or a nearby win/HP boundary trigger 128. A deterministic 10% audit advances other
stopped states by one stage. The full rules live in
[the scheduler](../../src/sts1_llm_policy/data/gold/sampling_ladder.py) and collection config.

The initial HP boundary band is 0.25 HP. At 64 it expands to the larger of that band
and twice the paired standard error. Win boundaries concern actions at the required
win count or one short and close enough in HP to matter; when none qualify, all
roots matter. Empty sets stop early only when every root is at least two wins below
the gate. These are heuristic stopping rules and can miss high-variance actions.

Shadow mode always runs to 128 and compares hypothetical early stops with that
reference; adaptive mode reports only the trials actually executed. Even 128 trials
are an empirical reference, not ground truth. One-trial smoke disables the ladder
and checks execution only: floor rounding can give it a zero-win gate.

### Adaptive collection and compact storage

Compact storage keeps every action sequence, trial identity and terminal outcome,
including rejected actions and all-pass states. A completed state is fully replayed
before compaction. Full traces are retained for trial zero, a deterministic 1%
sample, and selected boundary/changed-candidate trials; those categories overlap.
The 1% setting is not the total retained fraction.

Replay checks every stored action sequence and terminal score in its selected
states. Detailed raw-state and Teacher-choice checks apply only to retained full
traces; action tapes cannot recover deleted search evidence. With reconstructed
observations, intermediate text is regenerated rather than compared against a
saved text field. Root observations, route context and known draw memory remain
checked. Storage savings do not increase label confidence.

Terminal rows permit rescoring HP tolerances or export weights, not inventing
unexecuted trials or a different continuation policy. Explicit imports require
matching execution inputs and native content, preserve the source evidence, and
record receipts. Historical collection imports are additional reconstruction
inputs; see [availability](experiments/continuous.md#downloads-and-starting-points).

### Execution, verification and receipts

Use `collect_gold.py --config ...` to collect and `--verify` to replay without
new search. Preflight checks restoration and budgets; smoke uses a separate small
output. Ctrl+C preserves complete trials; rerun unchanged inputs to continue.
Changing sources, native runtime or stopping rules requires new output, not editing
an old result into compatibility.

Verification writes a new receipt bound to the actual report content. Repeated
verification needs a new `--verification-output`; failed replay writes no receipt.
Exports reject missing or mismatched bindings. Old receipts without that binding
remain historical evidence and cannot be repaired by adding a digest manually.

`--verification-selection` can restrict replay to an explicit subset of complete
states, fixed before replay. Its receipt records sampled coverage, and export must
explicitly opt into `verification_scope: sampled`. Export still processes all
records; matching a full exported dataset does not establish replay agreement for
unselected continuations. The local reconstruction used 240 of 7,134 complete
states; that is not full replay or a population error-rate estimate.

Collection summaries distinguish card composition from action/target counts.
States have equal weight; attack and defense tags can overlap, and empty sets add
no fictitious card. Opportunity retention describes the sampled panel, not all
possible gameplay. Original diagnostic comparisons and their limits remain in
[stage results](stageresult.md#gold-supporting-diagnostics).
