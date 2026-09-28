# Data and training contract

Dataset inputs, exports, training, compatibility and evidence boundaries.
See [policy and observation](policy_observation.md) for student inputs and
[Teacher and GOLD](teacher_gold.md) for collection and replay.
Stage workflows are in [single](experiments/single.md) and
[continuous](experiments/continuous.md); setup is in
[runtime and simulator](runtime_and_simulator.md).
Original execution results live under `report/` or `outputs/`;
retained comparisons are in [stage results](stageresult.md).

## Dataset and training interface

### Maintained training inputs

Training consumes a validated `dataset_manifest_v1`. The manifest owns task
type, observation version, stable identity fields, split artifacts, lineage,
target/preference semantics, record counts, byte sizes, and content hashes.
The `single` 7B SFT and DPO runs consume `gold_sft.manifest.json` and
`silver_preferences.manifest.json` respectively under
`assets/datasets/single/observation-v5/`, with their declared train artifacts.
The manifest and its content are the maintained training input. Reconstruction
from retained Teacher V2 exports uses `prepare_dataset.py` and
`migrate_observation_v5.py` with the explicit single-stage data configs; see
[the reconstruction workflow](experiments/single.md#rebuilding-v4-and-v5-datasets).
The earlier collection/certification execution surface remains retired.

`scripts/run_training.py` is the shared SFT/preference entry. An algorithm
adapter owns record validation, tokenization, loss semantics, prepared-input
identity, generation-probe prompt, and reference requirements. The runner
owns model and LoRA loading, deterministic ordering, optimization, resumable
checkpoint state, fresh-Base reload, and reports. Dataset counts and optimizer
steps are always derived from the verified manifest.

The completed 1.5B Gold SFT remains the development and live policy. Its V1
training config remains because `real_game_gold_sft_v5_session.json` and
`real_game_card_selection_session.json` bind it, its report, and its checkpoint.
This config is an execution dependency of the live profiles.

Preference support uses the shared adapter and `dpo_lora_v1` recipe. The
completed 1.5B Silver DPO result did not replace Gold SFT. Its experiment-specific
config, topology pipeline, evaluation runners, and snapshot tests have been
removed from the current checkout; the measured result remains in reports.

### Adapter checkpoint format

Reference adapters under `assets/adapters/` and newly saved final adapters use
the project's `project_lora_v1` format (`schema_version: 1`). Each directory contains:

| File | Contract |
| --- | --- |
| `adapter_config.json` | Format identifiers, exact `base_model_id` and `base_revision`, LoRA `spec` (target module suffixes, rank, alpha, dropout), ordered `replaced_modules`, and the weights filename and SHA-256. |
| `adapter_model.safetensors` | Adapter tensors only, named `<module>.lora_a.weight` and `<module>.lora_b.weight`; Base weights are obtained separately. |

These filenames are also used by [PEFT](https://huggingface.co/docs/peft/en/developer_guides/checkpoint),
but the metadata and tensor names differ. Direct loading with `PeftModel.from_pretrained`
or Transformers' PEFT integration is unsupported. The project loader accepts only
`project_lora_v1`; no PEFT import/export or automatic conversion is provided.

Use the matching evaluation config's `checkpoint` field (or `checkpoint` in a
continuous arm), or `initial_checkpoint` in a training config. These paths reach
[`train.lora.load_lora_checkpoint`](../../src/sts1_llm_policy/train/lora.py)
through the existing model backend or training runtime. The loader applies the
adapter to a separately loaded Base with the declared model ID and revision,
checks the weight hash, and requires matching module names and tensor shapes.
Unsupported format/schema errors are distinct from Base identity mismatches.
An adapter for one Base size or revision cannot be substituted for another by
editing its metadata.

A final adapter supports inference and explicit initialization of a new training
run. It does not resume optimizer progress. V2 training continuation additionally
requires the matching run's `latest.json`, checkpoint state, optimizer and RNG
files under the [current recovery contract](#configuration-and-artifact-reuse-direction).
The included reference adapters contain no such continuation state.

## Dataset exports

### Executed-GOLD V7 SFT groups

`uv run python scripts/export_gold_sft.py --config configs/data/continuous_gold_sft.json`
is the SFT export entry point; the retained config describes the original export
of the completed V4 candidate report and its final verification into
`outputs/datasets/gold-sft-v7-v1/train.jsonl.gz` and `manifest.json`. The exporter
checks the verification's report-content binding, declared training partition,
complete state IDs and execution counts, scope-appropriate replay counts, and reproduces the original
candidate sets before applying the export HP tolerance.
The export config explicitly selects both source files, the destination, HP
tolerance and nonbasic-card multiplier. Conflicting existing exports are refused.
Re-exporting historical data requires the new receipt and destination described above.

SFT and DPO share collection validation, candidate selection and public decision
metadata in `data/gold/export_source.py`; each exporter constructs its own record
schema and weights. DPO does not construct intermediate SFT training records.
All three exporters build their dataset artifacts and manifests in memory and
complete their export checks before writing. DPO also completes the A/B/C checks and
builds its combined report first. Destination, temporary-file and existing-content
conflicts are checked across the entire export before any file is written.
Identical existing files can be reused. Writes remain atomic per file; an I/O
failure or interruption can leave a partial export. Retrying with the same inputs
reuses identical files and refuses conflicting files or leftover temporary files.
This is not a multi-file transaction.

`decision_sft_group_v1` stores one V7 public observation, its legal action metadata,
source lineage, and explicit candidate action IDs and positive normalized weights.
Empty and all-pass states are omitted. The initial export retains the collection's
floor-rounded encounter win gates and a 1 HP tolerance. Nonbasic Attack/Skill/Power
play actions receive coefficient 1.5; Strike, Defend and Bash (including upgrades),
end turn, playable status cards and secondary selections receive coefficient 1.
Normalize these coefficients within each state. Targets remain distinct actions;
the rule does not collapse them or multiply the total weight of a state. This is
an empirical training target, not a new statistical certification of the labels.

The SFT adapter accepts the group schema alongside historical single-label records.
It checks prompt/action membership, observation integrity, unique candidates and
normalized weights. Every candidate supervises the existing response-only token
cross entropy, including the assistant terminator. The state loss is the weighted
sum of candidate losses; batches and epochs count states. Training backpropagates
each candidate before computing the next to bound activation memory. Recovery
identities include every candidate's tokens and weight. Generation probes consume
only the shared prompt. Historical single-label records retain their behavior;
secondary-selection labels require the new group schema.

`configs/runs/training/continuous_qwen2_5_7b_gold_sft.json` selects the fresh 7B Base,
the exported manifest and existing `sft_lora_v1` recipe: one epoch, eight states
per optimizer step, learning rate 1e-4, q/v LoRA rank 8 and alpha 16. Tokenization
must fit the 3,072-token cap without truncation. GPU recovery and longest-state
backward, followed by optimizer/save/reload smoke, apply to this new run.

Training consumes the train-only manifest and its declared dataset artifact.
Source reports and continuation archives are export provenance, not training
inputs. The loader verifies the artifact, record count, sample semantics and
weights, and rejects development/test splits before reading their assets.
Model/tokenizer assets and any initial checkpoint must be available at the
configured paths. Dataset validity and execution readiness are separate checks.
The existing 20-topology/four-seed continuous development panel supports subsequent
closed-loop evaluation; observation and simulator comparability with historical
Base results must be checked. A separate GOLD-labeled offline validation collection
is optional and is not a prerequisite for this first run.

### Matched V7 DPO preference datasets

`uv run python scripts/export_gold_dpo.py --config configs/data/continuous_dpo.json`
is the DPO export entry point. The retained config describes the original three
train-only datasets from the completed GOLD collection and an explicitly scoped replay verification
in `outputs/datasets/gold-dpo-v7-v1/{a,b,c}/`,
each containing `train.jsonl.gz` and `manifest.json`. The original export result is
[gold_dpo_v7_v1.json](../../report/data/gold_dpo_v7_v1.json). Export takes a few
seconds locally and does not run new continuations or load a model.
It shares the GOLD source report-content binding check and historical-receipt requirements;
the archived config and existing datasets retain their original identities.

Chosen actions come from the current 1-HP GOLD set; rejected actions are outside
that set. Every retained pair has at least 64 paired continuations, no unpriced
persistent-resource conflict, no lower observed chosen win count, and
`mean_carried_hp_gap - 2 * paired_standard_error > 1`. Defeat contributes zero
carried HP. These adaptive-sample estimates are heuristic preference labels,
not statistically certified or route-optimal values. No within-GOLD pairs,
sole-action states, or development trajectories are added.

All arms use the same ordered states, restricted to states with at least one
eligible pair whose raw mean HP gap is **strictly greater than 3**:

- A retains every eligible pair, with equal state mass and equal pair mass within
  each state.
- B retains only pairs with raw gap `> 3`, then divides each state's mass equally
  among its remaining pairs. Exactly 3 is excluded.
- C retains exactly A's pairs and multiplies each base pair weight by
  `f = min(1, mean_carried_hp_gap / 5)`. Let `f_bar_s` be the mean factor within
  state s and `Z` the mean of `f_bar_s` over all states. The serialized edge weight
  is `f / sum_state(f)` and state `loss_weight` is `f_bar_s / Z`. Their product is
  `f / (state_pair_count * Z)`: small-gap states remain downweighted, while the
  dataset mean state coefficient stays one. Weights are not normalized again
  within minibatches. Factors are fixed from the recorded outcomes, not scheduled
  by training step.

The three run configs are
`configs/runs/training/continuous_qwen2_5_7b_dpo_{a,b,c}.json`. They use the existing
`scripts/run_training.py`, the same `dpo_lora_v1` recipe and seed, and the completed
mixed SFT checkpoint as both policy initialization and frozen reference. Each
runs one epoch with eight states per optimizer update, learning rate `2e-5` and
beta `0.1`. Each run has separate reference-cache, checkpoint and report outputs.

`decision_preference_group_v1` accepts optional positive state `loss_weight` only
with explicit manifest semantics and complete mean-one weights. State weights
participate in training/recovery identity. Each pair is backpropagated before the
next pair's graphs are allocated, preserving the weighted group gradient without
holding all pair graphs at once. V7 secondary selections use the same action-only
response boundary as SFT. Historical groups without state weights retain weight one.

Local export and CPU gradient checks do not establish 7B CUDA readiness. Before
formal training, each configured dataset needs current tokenizer/length checks,
longest-sample backward evidence and optimizer/save/reload smoke. Training
consumes the exported manifest and artifact, the declared Base assets and mixed
SFT checkpoint; it does not reconstruct data from the Teacher collection.

The corresponding evaluation configs are
`configs/runs/evaluation/continuous_qwen2_5_7b_dpo_{a,b,c}.json`.
Each loads its own final DPO checkpoint and writes to a separate generation output
directory. Route construction, all RNG streams, fixed noncombat strategies,
V7 protocol and the 20-topology/four-seed development panel match the mixed SFT
evaluation. The primary comparison is final Boss completion; original route
records and combat trajectories remain available for local analysis.

### Mixed GOLD and Teacher V7 SFT

`scripts/export_gold_sft.py --config configs/data/continuous_mixed_sft.json`
uses the existing GOLD manifest and the explicitly selected Teacher pool. It
selects the first three combats of each training route, with the same seed group
as the GOLD collection. The excluded tuning routes, reserved routes and development
sources remain outside training. `--mode smoke` replays only the configured representative
combats and writes a separate smoke dataset; the training run uses the formal dataset.

The GOLD input must be a train-only V7 SFT manifest; its type and split boundary
are checked before any split artifact is read. Mixture weights, dataset identity,
encounter-family declarations and destination paths are checked before native
replay. Source integrity, route isolation, transition replay and training-loader
validation remain separate checks at their respective input boundaries.

Pool location is separate from provenance. Optional `source_locations` in a mixed
export config maps a historical pool-report path to an object with `path` and
`sha256`. Set `path` to the relocated report and `sha256` to its previously verified
content digest. References in the GOLD manifest and selection are resolved through
this mapping without rewriting those files. The current `source_report` may retain
its historical reference or point directly to the new report. Every referenced
report must have identical content; mapped files must also match their declared
digest. Missing references and mismatches fail before simulator startup. A digest
chosen alongside a replacement file does not authenticate its historical origin.

Move the pool as a directory, keeping `teacher/routes/` and each trajectory's
report-relative path intact. This mapping applies only to the declared pool reports,
not arbitrary dataset files or a global asset search. Route identity, excluded and
reserved routes, trajectory hashes, and GOLD overlap checks still apply. New mixed
manifests record the resolved pool path/content digest and the selection digest.

Public route context is constructed in `env/route_context.py`, shared by generation,
GOLD replay and mixed export. Counterfactual picker lookup and Teacher evaluation
live in `eval/counterfactual_reward_picker_v2.py`; both generators consume that module.

Every selected combat is replayed from its recorded input snapshot and action tape,
including secondary choices. Original observations, states, transitions and outcomes
are checked. Public draw memory is exposed without sampling a new RNG world, and
current V7 observations are rebuilt. Old source files retain their original text
and identity. Hidden order, RNG and search evidence are not student inputs.
Reconstruction mismatches fail export instead of silently changing labels.

Teacher demonstrations must come from complete victorious combats. They are
single-action imitation targets, not certified GOLD or route-optimal labels.
The containing route may subsequently end in defeat; later Boss success is not
a filter on early demonstrations. Truncated route executions remain rejected.
At source states already represented in GOLD, only the original GOLD candidate
set is retained; rebuilt observations and action mappings must agree. Other
demonstrations preserve every decision, with sole-legal END states separately
marked as formatting supervision. Active END and secondary choices remain
strategy demonstrations. Winning trajectories are not a proof of optimal play.

The initial loss mixture is 70% GOLD, 25% Teacher strategy and 5% sole-legal END.
GOLD states remain equally weighted within their source and retain the original
candidate coefficients. Strategy combats contribute equally; each combat divides
its mass among its retained strategy states. Formatting states are equally weighted.
For N total records, a state's `loss_weight` is N times its normalized source mass:
`N * source_mass / source_states` for GOLD and formatting, or
`N * source_mass / (strategy_combats * retained_states_in_combat)` for strategy.
The dataset mean coefficient is one. Candidate weights still sum to one within
each state. The existing SFT adapter multiplies each candidate CE by both weights;
it does not renormalize source weights within individual minibatches. Manifests
declare and loaders verify the source masses; state weights participate in recovery
identity. Historical groups without explicit state weights retain weight one.

`configs/runs/training/continuous_qwen2_5_7b_mixed_sft.json` uses the existing
`run_training.py` and `sft_lora_v1` recipe, fresh 7B Base and one epoch over all
mixed records. Eight states form an optimizer step. More records imply more
optimizer steps than GOLD-only training; this comparison therefore includes a
training-budget difference. Changed dataset inputs require current longest-sample
backward and optimizer/save/reload smoke before formal training. Source manifests,
trajectories and native binaries are export inputs; training needs only
the resulting manifest and its declared compressed training artifact.

## Configuration and artifact reuse direction

Configuration locations follow their consumers: `configs/runtime/` owns model
and execution settings; `configs/runs/training/` and `configs/runs/evaluation/`
own offline runs; `configs/generation/` owns Teacher/GOLD and frozen input
production; `configs/live/` owns real-game sessions. Shared training recipes,
panels, data-export settings and experiment member lists remain in
`configs/profiles/`, `configs/panels/`, `configs/data/` and `configs/experiments/`.
Historical report paths describe the original executions and are not rewritten.

`artifacts.py` owns shared file/path operations and `configuration.py` owns
configuration documents. `data/manifest.py` validates dataset manifests.
`workflows/` composes experiment preparation and continuous-route execution;
its `continuous_config.py` expands the shared panel without introducing generic
configuration inheritance. Data, training and evaluation import the foundational
modules directly rather than through workflow orchestration.

Shared model configuration and asset checks live in `model_runtime.py`;
dependency and hardware checks live in `execution_environment.py`. Training and
policy inference consume these modules directly. Data, policy, training and
evaluation interfaces are imported from their defining modules, without
package-wide eager imports. For example, use `data.trajectory.TrajectoryLogger`,
`policy.llm_policy.LLMPolicy` and `train.lora.load_lora_checkpoint`.

Within training, `identity.py` owns run bindings, `runtime.py` owns Base and
initial-adapter loading, and `readiness.py` validates recovery evidence.
SFT tokenization, collation and batch transfer live in `sft_data.py`.
`recovery.py` executes recovery probes through the training runner; the runner
consumes readiness checks without importing the recovery workflow. Module moves
preserve portable compatibility contracts. Legacy byte-bound runs still reject
changed source bindings; their original reports are not rewritten.

Reward construction uses `eval/card_profiles.py` for card tags and coverage,
`eval/reward_support.py` for shared scores and counterfactual snapshots, and
`eval/counterfactual_reward_picker_v2.py` for the maintained selection algorithm.
Live and simulator environments share `env.errors.CombatEndedError`;
replay verification uses `data.trajectory_replay.comparable_execution_state`
to compare raw states without bridge-session decision IDs.

| Layer | Responsibility |
|---|---|
| Engine (`src/`) | Reusable validation, tokenization, loss, resume, and runtime behavior. CLI scripts only adapt arguments. |
| Profile/run (`configs/`) | Select model/runtime semantics, recipe, dataset manifest, seed, allowed modes, and optional destinations. |
| Manifest/assets | Bind dataset, model, tokenizer, adapter, and checkpoint content whose exact bytes matter. |
| Result (`outputs/`, published `report/`) | Store resolved configuration, fingerprints, execution evidence, and measurements. |

Execution outcomes such as `training_started` and `stable_claim` belong in
reports, not continuous or frozen evaluation run inputs.

Generation follows the same split. A continuous-route run config selects its
strategies, operation schedule, observation/interaction protocol, legal reward
scope, policy arms, target panels, and independent RNG streams. Resolved asset and
runtime identities are generated into results for resume/audit; they are not
hand-authored expected bindings in the run config. In particular, a Teacher
upgrade target panel must be distinct from the actual next encounter and formal
combat seeds so its decision cannot consume hidden route information.

### Prepare independent training and evaluation configs

The recorded configs select supplied reference adapters for evaluation.
Use `prepare_experiment.py` to connect new training outputs and evaluations without
overwriting those inputs. From the repository root (typically under five seconds;
configuration files only, no model or simulator execution):

```text
uv run --locked python scripts/prepare_experiment.py --config configs/experiments/single_7b.json --config configs/experiments/continuous_7b.json --output outputs/reproduction
```

Select either member list on its own to prepare one stage. For another experiment,
create an explicit `experiment_preparation_v1` list with `training` and `evaluation` arrays;
list upstream training before its dependents. Optional `reference_checkpoints` maps
each supplied adapter directory to its upstream training config. Preparation replaces
both reference-adapter inputs and original training-output references with the new
checkpoint locations. Every mapped training config must be a selected member;
config filenames and adapter directory names are not interpreted.
Missing upstream members, duplicate members/output identities, unknown schemas and
conflicting destinations fail before any prepared configs are written. The command
prints all generated paths on success and exits nonzero on a conflict. It does not
start or schedule training/evaluation.

Continuous evaluation panels are expanded into the generated config snapshots.
Those generated copies preserve the selected conditions even if a shared panel is
later edited; the maintained source configs reference the single shared definition.
The existing runners perform full runtime and asset validation when invoked.

Use the generated paths, prefixed with `outputs/reproduction/`, for both training
and evaluation of these new models. Dataset, Base, recipe and environment
inputs remain explicit references to the original configs/assets. Single DPO uses
the new single SFT checkpoint; continuous DPO A/B/C each use the new mixed SFT
checkpoint. The DPO recipe freezes that same initial checkpoint as its reference.
Evaluation configs select the corresponding new checkpoints, while Base arms remain
unadapted. DPO readiness and adapted-policy evaluation require their upstream
training to have completed.

Repeating preparation with unchanged inputs preserves existing files. To start
another independent experiment, choose another `--output` and use that prefix throughout;
to resume, retain the existing configs and output directories. An input/config
conflict stops preparation without overwriting an earlier run. Evaluate the
included reference adapters with the original evaluation configs instead.

### Asset validation and compatibility

Git records source and ordinary configuration revisions. Model weights, datasets,
adapters, and native binaries have content identities in their manifests or
metadata. Loaders validate the declared assets and reuse validated metadata
within an operation. Training and evaluation use semantic bindings to determine
whether saved results or checkpoint state can be reused.

These checks cover input/action validity, split isolation, complete counts,
finite values, and exact adapter reload onto a fresh Base model. Deduplication
and deterministic ordering also use semantic fingerprints as part of their
algorithms. The binding rules and compatibility limits are described below.

### Current configuration and identity implementation

Continuous generation and evaluation use
`continuous_combat_configuration_identity_v2`. This binds resolved scope, route and
seed rules, reward/Teacher strategies, observation and interaction contracts,
decision limits, model/tokenizer/decoding semantics, dependency versions, adapter
content, picker database bytes, and native binary/revision. Candidate-source
exclusions bind explicitly declared source-route identities and exact RNG seed
sets rather than the locations of retired configuration files. Exclusion inputs
retain original run/output identities and configuration labels as provenance;
those configuration labels are never opened.

The scope fingerprint binds the resolved allowed reward-card set, the selected
act's encounter pools (including their order), and the scope interaction contract.
Documentation links, explanatory text, descriptive counts and unused acts are
excluded. Reward-card ordering does not affect eligibility or this fingerprint.
Earlier whole-scope fingerprints are not automatically accepted for resume;
historical outputs retain their recorded identities.

JSON formatting, object-key order, configuration paths, model-cache and adapter
directory locations and output destinations do not change this identity.
Evaluation run labels are excluded; a Teacher candidate pool's `run_id` remains
bound because exports require it to match the original trajectory lineage.
A relocated output must retain its route records and referenced
trajectories; resume still verifies snapshot and trajectory content. Hardware
requirements remain separate and are enforced when the backend loads a model;
reading completed routes does not load the model or certify the current GPU.
The existing supported dtype and decoding protocol remain enforced by the
runtime parser.

Before execution writes output, it checks existing route and summary identities
across all arms. A generated `configuration_identity.json` also identifies a
run interrupted before its first completed route. V1 or unidentified partial
output directories are rejected; new executions must select a new directory.
Historical V1 reports and trajectories retain their original identities and
remain readable for analysis, validation and supported data exports. They are
not migrated or re-signed for V2 resume. Continuing an interrupted V1 run requires
its original implementation; a current release snapshot does not provide it.

`training_run_v2` requires only `schema_version`, `run_id`, `allowed_modes`,
`training_recipe`, `model_runtime`, `dataset_manifest`, `execution_profile`, and
`seed`. It derives `outputs/training/<run_id>/report.json` and separate
`backward/` and `smoke/` destinations. Explicit destination overrides and
`required_simulator_capabilities` remain available. Optional `initial_checkpoint`
names a project-relative adapter directory; DPO requires it even at preflight,
and SFT may use it to continue from an explicit adapter. The loader validates
the adapter's declared Base identity and weight content. Unknown fields, including
legacy hand-maintained `expected_*` digests and `implementation_components`,
are rejected. The loader enforces the train-only and no-sealed-data boundary.

Both single and grouped SFT/DPO inputs bind the observation text to its content
digest. Legal action IDs must be unique and contiguous from `ACTION_0`, match the
record's action metadata, and contain every supervised or preferred action. These
checks apply to V5 as well as V7 before tokenization reaches model execution.

`training_recipe_v1` validates required and unknown fields at the root and in
each section before opening model/runtime or dataset references. Numeric fields
require JSON numbers of the declared kind, excluding booleans and non-finite
values. Errors name the field or section, such as `optimizer.learning_rate` or
`dpo.beta`. SFT recipes omit the DPO section; DPO requires positive beta, zero
label smoothing, and the frozen initial-checkpoint reference. Existing numerical
and tokenization limits remain enforced for both algorithms.

`train/identity.py` owns checkpoint and recovery identities. It generates
fingerprints from resolved model/tokenizer semantics, the recipe values relevant
to the requested mode, dataset semantics and content, and actual ordered input
IDs, attention masks, labels, and prompt boundaries. DPO additionally binds all
ordered chosen/rejected branches, edge and state weights, beta, and the initial checkpoint
content used as the frozen reference. Implementation compatibility uses an
engine-owned backward contract, an algorithm-owned loss contract and, for smoke
or training, an update/resume contract. These are implementation constants, not
config overrides. Changes to batching, loss computation, or optimizer/update order
must update the affected contract when previous evidence or continuation is no
longer valid. Git records source provenance; source-file bytes are not a portable
compatibility gate.

JSON formatting, object-key order, source comments or module relocation, profile paths,
cache paths, output destinations, and observed hardware do not change training
identity. Token, label, relevant recipe, model, dataset content or compatibility
contract changes do. Micro-batch and gradient-checkpointing settings participate
in backward compatibility. Optimizer/schedule changes invalidate continuation but not the
optimizer-free backward probe; checkpoint save frequency is also excluded.
Initial-checkpoint relocation or JSON formatting preserves identity; changing its
weights does not. `portable_training_identity_v3` rejects earlier portable
training/recovery identities; no source-hash fields are silently discarded to
accept an old checkpoint or receipt. Existing reports and checkpoints retain their
original bindings and require their matching implementation for continuation.
A new run needs current recovery evidence; published final adapters remain usable
as inference or explicitly declared initialization assets. Frozen `training_run_v1`
keeps its original byte-bound contract.

The update/resume contract is `adamw_ordered_accumulation_rng_resume_v2`.
Each resume checkpoint includes a content-bound `rng.pt` with Python, PyTorch CPU
and the selected CUDA device's random state. Recovery checks this state; continuation
restores it after model and optimizer construction so nonzero LoRA dropout resumes
the same random stream. Earlier update contracts or missing/corrupt RNG state are
rejected, without rewriting historical checkpoints. The backward contract is unchanged;
matching optimizer-free readiness evidence remains reusable under its existing rules.

Final adapters are written to a temporary sibling directory and renamed only after
both weights and metadata have been saved. An interrupted final save can be retried
from the last resume checkpoint. Reusing a completed training report first checks
the final adapter's Base identity, metadata and weight content against that report;
missing or changed artifacts fail instead of returning the previous success status.
This check does not load another Base model or rerun training.

DPO computes reference log probabilities from the declared initial checkpoint
before optimization, persists bound per-group values, then releases its reference
model before loading the trainable model. Recovery reuses its already loaded
initial model under inference mode for reference values and then backward; it
neither loads a second Base simultaneously nor freezes the trainable adapter.
Response log probabilities normalize logits and sum response-token scores in FP32
in both reference inference and training. The loss contract
`frozen_reference_weighted_dpo_fp32_v2` invalidates earlier DPO reference caches,
backward receipts and resumable training state. Use fresh output directories and
current readiness evidence; historical reports and final inference adapters retain
their original identities. This numerical correction has not been evaluated by
rerunning the published DPO experiments.

Existing V2 reports and checkpoint state carry a generated `configuration`
snapshot and versioned `binding`; no hand-authored lock file or migration asset
inventory is required. Execution-profile compatibility and the observed machine
environment are checked separately before backward evidence is reused. A
different machine requires current recovery evidence and does not imply bitwise
equivalent continuation.

SHA-256 here detects a mismatch relative to chosen inputs. It does not
authenticate a publisher who can replace both an artifact and its expected
digest. Source provenance comes from the chosen Git revision or release snapshot;
asset hashes establish content identity relative to the declared inputs.

## Evidence and tests

Training and policy-evaluation preflight share their execution's existing
configuration/input preparation paths.
Preflight returns after those checks; execution continues with the prepared inputs.
In portable training, preparation checks model metadata, tokenizer assets, dataset
content and any initial adapter. Recovery or training then checks the current
environment and Base weights once before model execution. Reference, trainable
and final-reload Base models reuse that same execution's validation. A separate
invocation checks its own environment and assets; no CI result or process-global
cache bypasses these checks. Runtime legality, finite values, checkpoint integrity
and final-output checks remain at their consumption or production boundaries.

A formal training run requires finite loss, complete sample/step accounting,
resumable checkpoints, and exact adapter reload onto a fresh Base model. Before
a long GPU run, recovery must load verified assets, execute a bounded generation,
and pass longest-sample backward without creating an optimizer.

Default test discovery covers the reusable `policy`, `simulator`, `live`, and
`training` packages under `tests/`; each package owns one stable behavior area.
Tests protect behavior that can change experimental results or corrupt artifacts:
public observations and legal actions, data isolation and label/loss computation,
checkpoint loading, and resumable execution. Historical configuration snapshots,
fixed experiment counts, duplicated source hashes and completed one-off rebuild
checks are not a separate maintained suite. Historical results
are read from reports or, when the required assets remain available, reproduced
at their recorded Git revision; they are not a third test suite. The directory
map is kept in `tests/README.md`.

Recorded comparisons and their original report references are in
[stage results](stageresult.md); the evidence directory map is in
[report/README.md](../../report/README.md). Test success does not replace
experimental results or establish that historical assets are available.

## Retention and retirement

The current checkout keeps only files required by a maintained runtime,
protocol, input manifest, or active configuration. Retained artifacts include
current inputs, result reports, final adapters, source evidence, and unresolved
failures. Retired experiment scripts, run configs, profile snapshots, and dedicated
tests have no consumers among the maintained entries.

After a completed run, redundant shards, smoke artifacts, optimizer/resume state
and their latest pointers may be retired when the retained results meet its
remaining uses. Final adapters support inference, not continuation of the old
optimizer state. Retained per-combat results support recorded outcome comparisons;
deleting step trajectories ends step-by-step replay and action-level reanalysis.

Retired experiment execution files were recorded at Git commit `9278d51` before
decoupling. That historical implementation is not included in a release snapshot.
Historical reproduction requires the original source and the assets named by its
report, which this release does not distribute. Git does not restore ignored artifacts, and retired runs are
not guaranteed to be exactly reproducible. The current checkout does not make a
historical byte-bound config runnable by replacing its old digest.
