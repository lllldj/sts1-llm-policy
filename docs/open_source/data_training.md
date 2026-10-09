# Methods and training

The [README](../../README.md#experiment-design) gives the overview. This page
keeps the main algorithm choices and limits; exact settings live in the linked
configs. Commands are in the [running guide](README.md).

## Model inputs and routes

Each decision reads public state and legal actions, without conversation history,
and returns one `ACTION_n`. Base, SFT and DPO share serialization, parsing, one
retry and a logged, seeded legal fallback. Equivalent playable card copies share
an action per target; secondary choices preserve individual candidates. Unknown
outcome-relevant mechanics are rejected. Hidden draw order, future RNG, execution
IDs and Teacher scores are excluded from student text.

| Interface | Use |
| --- | --- |
| V5 | Single combat and the default live policy: state, legal actions and mechanics glossary. |
| V6 | Opt-in live secondary card selection, using the existing V5-trained adapter. |
| V7 | Continuous combat: selection context, upgrade previews, remaining route and known draw-prefix memory. |

V7 exposes the public Boss and remaining operation types, but not future ordinary
monsters, random offers or seeds. Headbutt/Warcry can establish known draw order;
draws consume it, and shuffle or unknown insertion clears it. Upgrade previews
come from copies without advancing combat. Missing context causes an error rather
than a guessed prompt. Earlier V7 records predate this memory correction; the
version name alone does not make their observations equivalent.

The [continuous panel](../../configs/panels/continuous_act1_development.json)
controls rewards, upgrades, removals, relics and healing. HP and persistent relic
counters carry after victory; temporary combat effects do not. Burning Blood
heals once; route healing occurs only at its declared node. Death ends a route,
while decision-limit truncation is a separate non-victory in the denominator.
Later-stage HP comparisons are conditional on reaching that stage.

The [reward picker](../../src/sts1_llm_policy/eval/counterfactual_reward_picker_v2.py)
compares each offered card with skipping, using search, pick-rate priors and deck
synergy. Its reward evaluation uses the configured final Boss seed and conflict
offsets. Upgrades instead compare all Act 1 elites and the public Boss with separate
seeds; they do not inspect the actual next monster or formal combat seed. These
are different search uses from GOLD certification. Shared schedules do not force
policies to reach the same later states or construct identical decks.

## Teacher and GOLD

The source [Teacher pool](../../configs/generation/continuous_teacher_pool.json)
retains reached decisions, including losing routes, and replays each new combat
before finalization. GOLD restores selected states, forces every legal model-action
class and plays each continuation to combat end with paired execution seeds.
A prompt alone cannot restore a simulator state.

Outer execution and inner search use independent random streams. Known draw-prefix
cards and current enemy intent stay fixed; unknown order and future RNG are
resampled. Search sees each sampled world's internal state, but not the actual
execution world's future. This is a conditional-state experiment, not recovery
of the original game's hidden seed. The Teacher optimizes local combat, not the
student's full-route objective.

At each continuation decision, search estimates are averaged across inner worlds.
The default ranks win rate then victory HP; formal collection instead chooses
maximum expected carried HP among actions meeting an internal win threshold,
falling back to win-rate priority if none qualify. The external GOLD gate uses
completed battles, not these internal search estimates.

Victory contributes settled ending HP; defeat contributes zero. Acceptable
choices meet the external win floor and fall within **1 HP** of the best eligible
action. Required wins are `floor(trials × target rate)`: a 98.9% target permits
31/32 or 126/128. Incomplete continuations yield no candidates, not defeats.
Max-HP and relic-resource differences have no invented HP exchange rate. Binomial
bounds and paired standard errors describe uncertainty; they do not certify
HP non-inferiority or route-optimal actions.

### Sampling and replay

[Formal collection](../../configs/generation/continuous_gold_collection.json)
uses 480 training routes, one seed group each, excluding development and tuning
routes. All groups of a source route stay in the same partition. Repeated opening
hands are not leakage; sharing a source route is. Single-action states are omitted.
Reached combats contribute at most 1 weak, 2 strong/elite or 4 Boss states.
Sampling mixes 40% turn start, 30% event and 30% uniform draws without replacement;
empty channels transfer mass to uniform. Event buckets balance public opportunities,
not outcomes or Teacher choices. These are probabilities, not quotas, and the
result is an enriched sample rather than a population-frequency estimate.

Paired trials follow **32 → 64 → 128**. Proper candidate subsets continue at 32;
changed sets or near-boundary results advance at 64, with a deterministic 10%
audit of other stops. The HP boundary is 0.25 initially, then `max(0.25, 2 × paired
SE)` at 64. Relevant win counts at the threshold or one short trigger scrutiny;
empty sets stop early only when every root is at least two wins below the gate.
See the [scheduler](../../src/sts1_llm_policy/data/gold/sampling_ladder.py) for exact
rules. Early stopping can miss high-variance actions; even 128 trials are an
empirical reference. One-trial smoke checks execution only, with a possible
zero-win gate from floor rounding.

Compact storage retains all action tapes, trial identities and terminal outcomes,
including rejected/all-pass cases, after full state replay. Detailed traces remain
for trial zero, a deterministic 1% sample and selected boundary trials (overlapping
categories). Replay checks action tapes and terminal scores for selected states;
raw-state/search checks cover only retained detailed traces. Regenerated intermediate
text is not a comparison against deleted text. Storage savings do not raise
label confidence or recover deleted search evidence.

Exports require a receipt bound to the actual report. `collect_gold.py --verify`
writes a new receipt via `--verification-output`; sampled replay also needs an
explicit selection and export's `verification_scope: sampled`. Old receipts cannot
be fixed by adding a digest manually. Terminal records allow rescoring tolerances,
not inventing trials or changing the continuation policy. Imports require matching
execution/native inputs and retain their original evidence.

## SFT and preference targets

**GOLD-only SFT** omits empty and all-pass action sets. Each state contributes one
weighted sum of response cross-entropies, including the terminator. States have
equal total weight; nonbasic Attack/Skill/Power plays have coefficient 1.5, other
actions 1, normalized within the state. More acceptable actions do not increase
state weight. See [export settings](../../configs/data/continuous_gold_sft.json).

**Mixed SFT** adds victorious complete Teacher combats from the selected routes'
first three fights; it does not require later Boss success, but rejects truncated
routes. GOLD wins at overlaps. Other Teacher choices are imitation labels, not
GOLD-certified. Loss mass is **70% GOLD / 25% Teacher strategy / 5% sole-legal END**.
GOLD/formatting states are equal within source; strategy combats divide equal mass
among their states. Mean state coefficient is one without minibatch renormalization.
Mixed starts from fresh Base and has more steps than GOLD-only, so the comparison
does not isolate mixture quality at equal cost. See [mixed export](../../configs/data/continuous_mixed_sft.json).

**DPO** pairs a 1-HP GOLD action with a rejected action. Each pair needs at least
64 paired continuations, no lower chosen win count, no unpriced persistent-resource
conflict, and `mean HP gap - 2 × paired SE > 1`. All variants share ordered states,
each with at least one eligible pair whose mean gap is strictly greater than 3.

| Variant | Selection and weighting |
| --- | --- |
| A | All eligible pairs, sharing equal total weight per state. |
| B | Only pairs with mean HP gap > 3, sharing equal total weight per state. |
| C | A's pairs multiplied by `min(1, HP gap / 5)`; global normalization keeps mean state weight at one. |

C retains downweighting across states; per-state or minibatch renormalization must
not erase it. All arms start independently from mixed SFT, also their frozen
reference, with the same seed and state-update budget. Pair counts and compute can
differ. These are empirical filters on adaptive samples, not confidence certificates.
See [DPO export](../../configs/data/continuous_dpo.json) and the
[export report](../../report/data/gold_dpo_v7_v1.json).

## Configuration and artifact reuse direction

Configs choose the actual model, dataset, recipe, checkpoint and panel; manifests
identify assets and outputs record execution. Changing a display name is not a
model switch. The single runner remains V5-specific; a new observation name does
not make an unsupported protocol work. Continuous panels are complete inputs,
not partial/nested overrides. An omitted checkpoint selects Base; an invalid
checkpoint or unknown field fails instead of silently selecting Base.

Training validates legal labels, source isolation, finite values and input length
without truncation. Development and sealed/test sources are excluded from training.
Exports use declared sources and separate destinations, rejecting conflicting
files. Writes are atomic per file, not across an export; after interruption retry
unchanged inputs or choose a new destination. Historical lineage paths are
provenance, not implicit runtime dependencies.

### Adapters and resume

`project_lora_v1` adapters contain `adapter_config.json` and
`adapter_model.safetensors`, but are **not directly loadable PEFT adapters**.
The project loader requires matching Base revision, shapes and weight content.
Final adapters support inference or new-run initialization; continuing an optimizer
needs the run's checkpoint, optimizer and Python/PyTorch/CUDA RNG state, including
for nonzero dropout. Missing/corrupt state fails. Completed training is reused
only when the final adapter still matches; interrupted saves resume from the last
valid checkpoint.

Reuse depends on model/tokenizer, data, recipe, observation, seeds and native
mechanics. Path relocation must preserve content and referenced artifacts; comments
and source-file renaming alone are not behavior changes. Candidate-pool run IDs
remain bound to trajectory lineage. Scope identity includes selected reward cards
and ordered encounter pools, not descriptive text. Hashes detect mismatches, not
a malicious publisher who can replace both bytes and expected hashes.

Changed behavior requires updated compatibility contracts and separate outputs;
old/unidentified resume state is rejected. Readiness must apply to the actual
machine and run. Optimizer changes invalidate resume even if optimizer-free
backward evidence remains reusable. Reports preserve their original identities.
Current DPO uses FP32 response log probabilities and rejects older reference caches,
backward receipts and resume state. Published DPO experiments were not rerun under
that correction; their reports and inference adapters remain historical results.

Complete training means finite loss, all declared steps and exact fresh-Base
adapter reload; policy improvement needs matched evaluation. Keep whole output
trees for resume and analysis: reports alone cannot replace trajectories. Earlier
development commits, retired executors and original raw reconstruction evidence
are not distributed. Do not alter historical digests to make them appear current.
