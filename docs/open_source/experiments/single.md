# single

SFT and DPO with public single-combat information. The stage includes the earlier
expanded/topology branches and the Gold SFT/Silver DPO comparisons on 1.5B and 7B
models. The frozen Boss panel is an evaluation component, not the training task.
Its primary metric is victories on matched combats; HP loss and output legality
provide supporting evidence.

The source snapshot includes the maintained training/evaluation entries, formal
configs, tracked Boss panel and original reports. Earlier branches retain their
results and explanations. Dataset reconstruction from retained Teacher V2 exports
is supported by the entries below; external assets are still required. The auxiliary
live interface retains its separate 1.5B V5 default.

## First short run

Complete [Python setup](../../../README.md#setup-and-verification), obtain the
[7B Base snapshot](../../../assets/README.md#external-assets), and satisfy the
[GPU requirements](../runtime_and_simulator.md#execution-requirements-and-recorded-hardware)
and [native build prerequisites](../runtime_and_simulator.md#native-build-prerequisites).
Use the included Gold SFT adapter and frozen Boss panel. Run these commands from
the repository root in PowerShell or Bash, continuing only after each succeeds:

```text
uv run --locked python scripts/simulator.py build
uv run --locked python scripts/run_frozen_policy_panel_evaluation.py --config configs/runs/evaluation/single_qwen2_5_7b_sft_boss.json --mode smoke --smoke-combats 2
```

The build selects the host's corrected simulator and includes its protocol smoke;
[build time and recovery](../runtime_and_simulator.md#callable-native-environment-selection)
depend on the existing installation. **Rough model-smoke budget: 2–10 minutes**
after asset installation, including model loading; GPU, storage and combat length
dominate. This estimate is not a measurement of the current corrected build.

The runner checks inputs before loading the policy and prints combat progress.
Open
`outputs/eval/qwen2-5-7b-sft-act1-a0-boss-corrected-v1/smoke/report.json`.
Success is `status: smoke_completed`, `scope.combat_identities: 2`, and passing
`checks`. Keep the whole `smoke/` directory, including `episodes.jsonl`,
per-combat records and `trajectories/`; the report alone is insufficient for resume.
Ctrl+C stops execution; rerunning validates completed combats and runs the missing
ones. Formal evaluation writes separately to `formal/` and does not reuse smoke
as a full-panel result.

This checks the execution chain, not policy quality or historical-score reproduction.
Next choose [full reference evaluation](#evaluate-reference-adapters) or
[new training](#running-the-7b-comparison). Dataset reconstruction is an optional
[historical-input workflow](#rebuilding-v4-and-v5-datasets).

## Evaluate reference adapters

With the [first-run prerequisites](#first-short-run) installed, run this block
from the repository root in Linux Bash or Windows Git Bash. It evaluates Base,
Gold SFT and Silver DPO on the included panel without training. This is the full
comparison; the short run above is sufficient for an initial execution check.

```bash
(
set -euo pipefail
uv run --locked python scripts/simulator.py build
for arm in base sft dpo; do
    evaluation="configs/runs/evaluation/single_qwen2_5_7b_${arm}_boss.json"
    uv run --locked python scripts/run_frozen_policy_panel_evaluation.py --config "$evaluation"
done
)
```

Each call validates its inputs before execution; a separate `--mode preflight`
call is optional for diagnosis. **Rough budget: 15–45 minutes per policy** plus
the [build time](../runtime_and_simulator.md#callable-native-environment-selection).
The historical Base panel took about 15 minutes including load; corrected
mechanics and policy-dependent combat length can change that timing.

Success is `status: completed` in each config's `output_dir/formal/report.json`,
with all 192 combats accounted for. Retain the entire `formal/` directory,
including `episodes.jsonl`, per-combat records and trajectories, for resume and
paired comparisons. Boss victories are the primary metric; HP and legality are
supporting metrics. Ctrl+C stops execution; rerunning validates and reuses
compatible completed combats. The block stops at the first failure.

## Running the 7B comparison

With the [first-run prerequisites](#first-short-run) installed and a clean Git
checkout, prepare independent configs from the repository root. Preparation takes
under five seconds; the build validates/reuses an existing compatible installation:

```text
uv run --locked python scripts/simulator.py build
uv run --locked python scripts/prepare_experiment.py --config configs/experiments/single_7b.json --output outputs/reproduction
```

[Configuration preparation](../data_training.md#prepare-independent-training-and-evaluation-configs) creates training and evaluation configs
under `outputs/reproduction/configs/`, using the included datasets and the declared
Base/native assets. The commands below use those generated configs. DPO cannot
pass full preflight until the new SFT checkpoint exists; its initialization and
frozen reference both use that checkpoint. The original evaluation configs above
remain available for evaluating the included reference weights.
Follow [current training readiness](../runtime_and_simulator.md#training-readiness-on-the-current-machine);
historical readiness does not certify a changed implementation or environment.
Training requires finite loss, complete derived steps, changed adapters and
exact fresh-Base reload. Evaluation requires all declared terminal summaries
and report checks. Compare matched Boss victories first, then HP and legality.

First establish a matching Base result for the corrected-mechanics comparison:

```text
uv run --locked python scripts/run_frozen_policy_panel_evaluation.py --config outputs/reproduction/configs/runs/evaluation/single_qwen2_5_7b_base_boss.json
```

This writes `outputs/reproduction/evaluation/qwen2_5_7b_base_act1_a0_boss_corrected_v1/formal/`.
Expect `completed` and 192 combats; keep the whole output tree. Allow roughly
15–45 minutes, with Ctrl+C and resume behavior as in [reference evaluation](#evaluate-reference-adapters).
If a complete Base result already exists under the same panel, seeds, mechanics,
model/decoding and execution bindings, reuse that result and its raw artifacts;
record its location alongside the new comparison. Historical legacy-mechanics
scores cannot substitute for this baseline. Repeating the command checks existing
compatible output in this destination before running missing combats.

Run the following in Linux Bash or Windows Git Bash on a machine matching the
execution profile. Each subshell stops at the first failure and uses a fresh
readiness report name. Recovery includes configuration/data/tokenizer checks and
longest-sample backward; optimizer smoke then checks the update/save path.
Standalone training `--mode preflight` remains available for diagnosis.

```bash
(
set -euo pipefail
sft=outputs/reproduction/configs/runs/training/single_qwen2_5_7b_gold_sft.json
sft_eval=outputs/reproduction/configs/runs/evaluation/single_qwen2_5_7b_sft_boss.json
sft_recovery="outputs/recovery/7b-sft-$(date -u +%Y%m%dT%H%M%S)-$$.json"
uv run --locked python scripts/restore_training.py --config "$sft" --report "$sft_recovery"
uv run --locked python scripts/run_training.py --config "$sft" --mode smoke --recovery-report "$sft_recovery"
uv run --locked python scripts/run_training.py --config "$sft" --mode run --recovery-report "$sft_recovery"
uv run --locked python scripts/run_frozen_policy_panel_evaluation.py --config "$sft_eval" --mode smoke --smoke-combats 2
uv run --locked python scripts/run_frozen_policy_panel_evaluation.py --config "$sft_eval"
)
```

After SFT succeeds, run the DPO stage:

```bash
(
set -euo pipefail
dpo=outputs/reproduction/configs/runs/training/single_qwen2_5_7b_silver_dpo.json
dpo_eval=outputs/reproduction/configs/runs/evaluation/single_qwen2_5_7b_dpo_boss.json
dpo_recovery="outputs/recovery/7b-dpo-$(date -u +%Y%m%dT%H%M%S)-$$.json"
uv run --locked python scripts/restore_training.py --config "$dpo" --report "$dpo_recovery"
uv run --locked python scripts/run_training.py --config "$dpo" --mode smoke --recovery-report "$dpo_recovery"
uv run --locked python scripts/run_training.py --config "$dpo" --mode run --recovery-report "$dpo_recovery"
uv run --locked python scripts/run_frozen_policy_panel_evaluation.py --config "$dpo_eval" --mode smoke --smoke-combats 2
uv run --locked python scripts/run_frozen_policy_panel_evaluation.py --config "$dpo_eval"
)
```

Budget estimates, not measured 7B training times: recovery plus smoke roughly
5–20 minutes per algorithm; reserve several hours for each formal training run.
Historical 1.5B reports recorded about 2.68 hours
SFT and 5.24 hours DPO on different hardware. Group count alone does not predict
DPO duration. Use the new smoke's measured step time to refine the training
budget (1,028 SFT / 59 DPO steps for these manifests), allowing extra reference
calculation and load/save time. Longest-biased smoke can overestimate step cost.
The recorded 7B Base panel took about 15 minutes including load; allow roughly
15–45 minutes per candidate evaluation, with policy-dependent combat length.

Success signals are `ready`, `smoke_passed`,
`trained_pending_development_evaluation`, and evaluation `completed`. Stop with
Ctrl+C. Keep code/config/data unchanged during a run. Repeating a stage block
performs current recovery and resumes from saved training steps/completed combats;
unsaved work since the last checkpoint is repeated. An intact completed training
report is reused. A failed smoke blocks formal training. After completion,
keep both training reports and each final adapter's two files, the recovery
reports and generated configs, and the complete Base/SFT/DPO `formal/` evaluation
trees, including trajectories. Retain training resume state while continuation
is needed. These commands do not change the default policy.

## Downloads and starting points

These stage inputs are included at the repository-relative paths below.
[Asset acquisition](../../../assets/README.md) owns external downloads, adapter
format and availability; the table connects each input to its use in this stage.

| Asset | Repository path | Availability | Generation / compatibility |
| --- | --- | --- | --- |
| V5 Gold SFT dataset | `assets/datasets/single/observation-v5/gold_sft.manifest.json` and its declared split artifact | Included | Retained training input; reconstruction from V2 exports is described below |
| V5 Silver preference dataset | `assets/datasets/single/observation-v5/silver_preferences.manifest.json` and its declared split artifact | Included | Retained DPO input; reconstruction from V2 exports is described below |
| V5 development SFT dataset | `assets/datasets/single/observation-v5/development_sft.manifest.json` and its declared split artifact | Included | Separate development split; excluded from training |
| 7B Gold SFT adapter | `assets/adapters/single/qwen2_5_7b_gold_sft_v1/` | Included | `run_training.py` with the 7B Gold SFT config below |
| 7B Silver DPO adapter | `assets/adapters/single/qwen2_5_7b_silver_dpo_v1/` | Included | `run_training.py` with the 7B Silver DPO config below, initialized from Gold SFT |
| 1.5B Gold SFT live adapter | `assets/adapters/single/qwen2-5-1-5b-teacher-v2-gold-sft-observation-v5-v1/` | Included | Frozen V5 live checkpoint; its original training has additional historical bindings |

Use the [new-training sequence](#running-the-7b-comparison) with these manifests,
or [evaluate reference weights](#evaluate-reference-adapters) directly. Rebuilding
the Boss input panel additionally uses the included picker database. Detailed
replay of a historical run needs its original trajectories.

The V5 datasets are included; their original Teacher V2 source assets are retained
locally and are not distributed. Reconstruction starts from completed certification exports; it does not rerun
the retired Teacher certification pipeline. The 1.5B historical
adapter and reports also do not establish that current code can reproduce its
original training. These limits are separate from the maintained 7B entries.

## Training and checkpoints

| Model / arm | Configuration or retained evidence | Checkpoint lineage |
| --- | --- | --- |
| 1.5B Gold SFT | [Training config](../../../configs/runs/training/single_qwen2_5_1_5b_gold_sft.json), [report](../../../report/training/qwen2_5_1_5b_teacher_v2_gold_sft_observation_v5_v1.json) | Fresh Base; the completed adapter remains the live default |
| 1.5B Silver DPO | [Original training report](../../../report/training/qwen2_5_1_5b_teacher_v2_silver_dpo_observation_v5_v1.json) | Historical execution; its dedicated config is retired |
| 7B Gold SFT | [Training config](../../../configs/runs/training/single_qwen2_5_7b_gold_sft.json) | Fresh Base; output `outputs/training/qwen2_5_7b_gold_sft_v1/checkpoint` |
| 7B Silver DPO | [Training config](../../../configs/runs/training/single_qwen2_5_7b_silver_dpo.json) | Initializes from 7B Gold SFT, also used as the frozen reference |

The shared training entry is [run_training.py](../../../scripts/run_training.py).
Runtime, recipe, dataset and seed are selected by each config. The 1.5B config is a
frozen adapter with stricter historical bindings; its presence does not imply that
the latest implementation reproduces the original training. The
[7B evidence table](../stageresult.md#recorded-verification-and-original-evidence)
links recovered training reports and evaluation outputs.

## Evaluation and delivery

[Boss input configuration](../../../configs/generation/single_boss_inputs.json)
generates new inputs with corrected simulator mechanics. Evaluation uses
[Base](../../../configs/runs/evaluation/single_qwen2_5_7b_base_boss.json),
[SFT](../../../configs/runs/evaluation/single_qwen2_5_7b_sft_boss.json) and
[DPO](../../../configs/runs/evaluation/single_qwen2_5_7b_dpo_boss.json) configs through
[run_frozen_policy_panel_evaluation.py](../../../scripts/run_frozen_policy_panel_evaluation.py).
The [Boss record](frozen_boss_192.md) retains panel identities, results, seed rules,
raw-output locations and reproduction limits. Other historical panels remain in
their original reports; the Boss subset does not stand in for all evaluations.

Current single configs explicitly require `corrected_v1` mechanics on Windows
and Linux, while retaining V5 single-action observations and the included frozen
panel. They write new `*-boss-corrected-v1` outputs. Historical training data,
reference adapters and recorded results keep their original identities; a new
evaluation under corrected mechanics does not reproduce their original scores.
Replacing the panel is explicit: use the generated `inputs.json`, its fingerprint
and counts in each evaluation config, with separate output directories.

The [evaluation commands](#evaluate-reference-adapters) define the completion
checks and retained output tree. [Training readiness](../runtime_and_simulator.md#training-readiness-on-the-current-machine)
uses the intended training config so its input and seed identity matches training.

## Data and earlier branches

The retained lineage includes the [expanded source collection](../../../report/data/expanded_sft_full_v3_4680.json),
[expanded training data](../../../report/data/expanded_sft_training_data_v1.json),
[topology preferences](../../../report/data/topology_preference_collection_v1.json),
[public-information Teacher training data](../../../report/data/public_information_teacher_v2_training_data_v1.json)
and [observation migration](../../../report/data/observation_v5_migration_v1.json).
These are different construction steps and training branches, not interchangeable
datasets. Their original reports define counts, algorithms and source identities.

Earlier [Expanded SFT](../../../report/training/expanded_sft_training_v1.json) and
[topology DPO](../../../report/training/topology_dpo_training_v1.json) retain their
own [SFT evaluation](../../../report/evaluation/expanded_sft_simulator_v1.json) and
[DPO evaluation](../../../report/evaluation/topology_dpo_simulator_v1.json).
Their retired execution surfaces require the separately supplied historical
source identified in the [retention contract](../data_training.md#retention-and-retirement).

The maintained Gold/Silver training inputs are the explicit dataset manifests at
`assets/datasets/single/observation-v5/{gold_sft,silver_preferences}.manifest.json`.
They select the artifacts, splits and observation contract. The two maintained
data entries below reconstruct records from retained certification exports and
the declared historical source. Earlier collection and certification executors
remain retired.

### Rebuilding V4 and V5 datasets

Run from the project root, after supplying the declared source assets:

```powershell
uv run python scripts/prepare_dataset.py --config configs/data/single_teacher_v2.json
uv run python scripts/migrate_observation_v5.py --config configs/data/single_observation_v5.json
```

The first command reads the completed V2 certification report, its Gold/Silver
artifacts and the historical eligible source JSONL. If that JSONL is absent,
first use [the existing summary rebuild](#rebuilding-expanded-source-summaries).
It checks source identity, counts, accepted training membership, public state,
legal labels and disjoint Gold/Silver identities. Output is V4 SFT/preference
data under `outputs/datasets/single-rebuilt-v4/`.

The second command consumes those manifests and writes V5 records under
`outputs/datasets/single-rebuilt-v5/`. It reuses the current V4-to-V5 text
conversion, preserves actions, labels and preference weights, and records each
old observation hash. The formal config also reconstructs the separate
development SFT split from the declared historical source. No Teacher search,
model loading or training runs. Each output directory contains manifests,
compressed records and an original execution `report.json`.

Both configs select input and output paths; dataset IDs are explicit and counts
come from the inputs. Migration supports one train or development split per V4
SFT/preference manifest; test splits and other observation versions are rejected.
The optional `development` input may be omitted for training-only reconstruction.
Train/development source episodes cannot overlap. Source artifacts retain content
checks; ordinary source/config files are tracked by Git rather than pinned by byte hash.

Existing files are never overwritten with different content. Repeating an
identical export is safe; use new output paths for changed inputs. Rebuilt
manifests record their actual source paths and identities, so they are not
byte-identical historical manifests. Record-by-record JSON equality is the
reproduction criterion; gzip encoding and provenance metadata can differ.
Point a new training config at the rebuilt manifest to train from it. Historical
training configs and datasets retain their original bindings.

The entries accept relocated inputs and reject source leakage, invalid labels
and conflicting outputs. Local full reconstruction on 2026-09-23 matched all V4/V5
Gold (8,221 each), Silver (471 each), and V5 development (2,406) records. Restoring
the 4,680-episode source summaries also matched both original audit hashes.
Source restoration, export, migration and comparison together took 108.3 seconds
on the collection machine; allow roughly 1–5 minutes depending on storage and CPU.
These checks used retained local evidence; that evidence is not distributed.

## Historical artifact retention

The earlier single-stage cleanup retains the full 4,680-combat source, formal V4/V5 datasets,
Teacher V2 certification artifacts and source manifest, four final adapters,
formal per-combat evaluation results, and necessary runtime/live evidence.
Superseded medium/pilot runs, debug/smoke artifacts, old training intermediates,
certification shards, and evaluation step trajectories have been retired.
Retained JSON reports preserve experimental results; public path redactions are
[explicitly marked](../../../report/README.md). Historical hashes describe the
original evidence, not a promise that every attachment still exists.

### Rebuilding expanded-source summaries

The historical `expanded-sft-data-v3-8192-full-4680` collection retains its
`episodes/`, `episodes.jsonl`, `audit.json` and `progress.json`. Its two large
JSONL views, `source_records.jsonl` and `eligible_source_records.jsonl`, are
reproducible derivatives; they are not shipped in the repository. Given the local
episode files and audit, they can be restored with:

```powershell
uv run --no-sync --cache-dir .cache/uv python scripts/rebuild_expanded_sft_summaries.py
```

The default source and destination are
`outputs/expanded_sft/expanded-sft-data-v3-8192-full-4680/`. Restoration takes
roughly 1–3 minutes on the local collection machine and writes about 4.39 GiB;
actual time depends on concurrent work and disk throughput. `--source-dir` and
`--output-dir` select explicit alternate directories. `--verify-only` instead
reconstructs the hashes in memory without writing the views. Existing destination
files are not overwritten; both reconstructed hashes must match the original audit
before publication.

This standalone standard-library script preserves the export rule from commit
`7b7a76e`: sort episodes by index, flatten their original record arrays, and
exclude audited quarantined episode IDs and hidden-order-invalid observation hashes
from the eligible view. It also verifies the quarantine against train/dev deck
overlap. JSON uses sorted keys, compact separators, UTF-8 and LF. Existing observation
strings, labels and the audit remain unchanged; no model, Teacher search, current
serializer or retired collection pipeline is loaded. The two restored views are
historical provenance/export inputs, not the maintained training manifest artifacts.
