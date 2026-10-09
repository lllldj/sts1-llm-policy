# Data and training

The experiments compare learning from acceptable actions with learning from
preferences between actions. [Teacher and GOLD](teacher_gold.md) explains how
those targets are obtained; this page covers how they become training data.
The README summarizes [LoRA and training settings](../../README.md#lora-and-training-settings).

## Dataset and training interface

Training reads a dataset manifest and its declared compressed records. Inputs
include the public observation, legal actions, targets and source lineage. Labels
must be legal; observation text must match its recorded content; training and
development sources stay separate. Tokenization never truncates a sample to fit.
The shared runner supports SFT and DPO, while configs select model, data and recipe.

The maintained single stage uses V5 Gold SFT and Silver preference datasets.
Its [reconstruction entries](experiments/single.md#rebuilding-v4-and-v5-datasets)
start from completed historical certification exports; they do not restore the
retired certification pipeline. The retained 1.5B training config is also a live
profile dependency, not a promise to reproduce its original training today.

### Adapter checkpoint format

Adapters use the project's `project_lora_v1` format: `adapter_config.json` and
`adapter_model.safetensors`. Despite the familiar filenames, they are not directly
loadable PEFT adapters. The project loader requires the declared Base revision,
module shapes and weight content. Select an adapter through the evaluation
`checkpoint` or training `initial_checkpoint` field; changing its name does not
make it compatible with another Base.

Final adapters support inference or initialization of a new run. Continuing an
interrupted optimizer requires that run's resume checkpoint, optimizer and RNG
state. The supplied reference adapters do not include these.

## Dataset exports

Exports read explicitly declared source reports and evidence, then write separate
dataset destinations. Conflicting files are rejected; identical exports may be
reused. Writes are atomic per file, not across the whole export. After an I/O
interruption, retry with the same inputs and inspect any leftover temporary files.
Training with the supplied datasets does not require rebuilding their raw sources.

### Executed-GOLD V7 SFT groups

GOLD SFT trains on a set of acceptable actions at each state. The current export
uses the collection's win gate and a 1-HP tolerance; empty sets and states where
all actions pass are omitted. A state contributes one weighted sum of action
cross-entropies, including the response terminator.

States have equal total weight. Within a state, nonbasic Attack/Skill/Power plays
receive coefficient 1.5; basic cards, end turn, status plays and secondary choices
receive 1. Normalize those coefficients within the state. Targets remain distinct
and a state with more acceptable actions does not gain more total weight.

[The export config](../../configs/data/continuous_gold_sft.json) selects sources and
weights. The [training config](../../configs/runs/training/continuous_qwen2_5_7b_gold_sft.json)
connects the dataset, Base model and SFT recipe.

### Matched V7 DPO preference datasets

Chosen actions come from the 1-HP GOLD set; rejected actions are outside it.
Every pair needs at least 64 paired continuations, no lower observed chosen win
count, no unpriced persistent-resource conflict, and
`mean HP gap - 2 × paired standard error > 1`. Defeat contributes zero carried HP.
These are empirical filters on adaptively sampled outcomes, not confidence
certificates or route-optimal values.

All variants use the same ordered states, each containing at least one eligible
pair whose mean HP gap is strictly greater than 3:

| Variant | Selection and weighting |
| --- | --- |
| A | All eligible pairs; equal state weight, divided equally among its pairs. |
| B | Only pairs with mean HP gap strictly greater than 3; equal state weight, divided over the remaining pairs. |
| C | A's pairs, with each base weight multiplied by `min(1, HP gap / 5)`. Small-gap states also lose weight; a global normalization keeps the mean state weight at one. |

C's weighting is not renormalized separately per state or minibatch in a way that
would erase that downweighting. All three start independently from mixed SFT,
which is also the frozen reference, with the same seed, state-update budget and
DPO recipe. Pair counts and compute cost can still differ.
See [export settings](../../configs/data/continuous_dpo.json) and the
[original export report](../../report/data/gold_dpo_v7_v1.json).

### Mixed GOLD and Teacher V7 SFT

Mixed SFT adds complete victorious Teacher combats from the first three fights
of the selected training routes and seed groups. Later Boss success is not a
filter, but truncated route executions are rejected. Source states already in
GOLD keep only their GOLD targets. Other Teacher choices are imitation labels,
not certified GOLD. Sole-legal end-turn states are separated as formatting examples.

The loss mixture is **70% GOLD, 25% Teacher strategy, 5% forced end turn**.
Within each source, GOLD/formatting states are equally weighted; strategy combats
are equally weighted and divide their mass among their retained decisions.
Candidate weights within GOLD states are preserved. The dataset's mean state
coefficient is one, without per-minibatch renormalization.

[The export config](../../configs/data/continuous_mixed_sft.json) declares sources
and mixture weights. Export replays the recorded actions and rebuilds public V7
observations, including draw memory; hidden order and search evidence never enter
student text. Source mismatches fail export. Relocated pools need explicit
`source_locations` mapping with unchanged report content and internal layout.
Mixed SFT starts from fresh Base. More records mean more optimizer steps than
GOLD-only SFT, so the comparison does not isolate data mixture at equal cost.

## Configuration and artifact reuse direction

Configs choose behavior; manifests identify data and model assets; outputs record
what actually ran. Model, checkpoint, dataset and panel changes reach their actual
consumers. Changing a display name alone is not a model switch, and the V5 single
runner does not accept arbitrary observation versions.

### Prepare independent training and evaluation configs

The included evaluation configs select reference adapters. For newly trained
models, prepare connected copies with separate outputs:

```text
uv run --locked python scripts/prepare_experiment.py --config configs/experiments/single_7b.json --config configs/experiments/continuous_7b.json --output outputs/reproduction
```

Preparation takes a few seconds and starts no experiment. Either member list can
be used alone. It connects DPO initialization/reference and evaluation to the new
upstream checkpoints, and expands continuous panels. Use the generated configs
under `outputs/reproduction/` for the subsequent stage commands.

Member lists declare dependencies explicitly, with upstream training first;
filenames are not interpreted. Missing dependencies, duplicate identities and
conflicting destinations fail. Repeating unchanged preparation preserves files;
choose a new output root for a new experiment.

### Asset validation and compatibility

Code and ordinary configs are tracked by Git. Dataset, model, adapter and native
content is checked against its declared identity. A hash detects mismatched bytes;
it does not authenticate a publisher who can replace both data and expected hash.
Keep raw output trees for resume and analysis, not just their summary reports.

### Current configuration and identity implementation

Reuse depends on the inputs and behavior that affect execution: data/tokens,
model and tokenizer, recipe, observation, seed rules, adapters and native mechanics.
Formatting, comments and relocated paths alone generally do not change that
identity. Relocation must preserve content and referenced artifacts. Candidate
pool run IDs remain bound to trajectory lineage, and scope identity includes the
selected reward cards and ordered encounter pools, not descriptive text.

Current training and route execution reject older or unidentified resume state.
Historical reports are not rewritten to pass new checks. Final reference adapters
remain valid for inference or explicit initialization; that does not restore old
optimizer state. A new run needs readiness on its actual machine. Optimizer changes
invalidate continuation even when optimizer-free backward evidence can be reused.

Resume restores Python/PyTorch/CUDA RNG state, including for nonzero LoRA dropout;
missing or corrupt state fails. A completed report is reused only if its final
adapter still matches. Interrupted final saves can resume from the last valid
checkpoint. Changing implementation behavior requires updating its compatibility
contract; source-file renaming alone is not a semantic change.

DPO computes response log probabilities in FP32 for both policy and reference.
The current numerical contract rejects older reference caches, backward receipts
and resume state. The published DPO experiments have not been rerun under that
correction; their reports and inference adapters retain their original identity.

## Evidence and tests

Before a long run, readiness loads verified assets, generates a short response and
checks the longest-sample backward; optimizer smoke covers update/save behavior.
Training completion means finite loss, all declared steps and exact adapter reload
onto a fresh Base. Policy improvement requires matched evaluation evidence.

[Tests](../../tests/README.md) cover observations, actions, data isolation, loss and
recovery behavior. Passing tests does not reproduce historical experiments or make
missing raw assets available. [Stage results](stageresult.md) links the evidence.

## Retention and retirement

The snapshot includes maintained code/configs, datasets, reference adapters and
reports. Earlier development commits, retired executors and original raw
reconstruction/replay collections are not distributed. Historical paths and hashes
identify provenance, not downloadable inputs. Final adapters do not replace resume
state; per-combat summaries do not replace trajectories for action-level replay.
Do not make historical configurations appear current by replacing their digests.
