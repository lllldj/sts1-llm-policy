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

Allow **roughly 5–20 minutes per algorithm** for recovery plus smoke. The
[historical 7B training results](../stageresult.md#training-completion) recorded
about 33 minutes for SFT and 4 minutes for DPO, with 1,028 and 59 optimizer steps
respectively. These timings describe the original implementation and machine;
they are not a current-runtime guarantee. Use the new smoke's measured step time
to estimate the run, allowing extra reference calculation and load/save time.
Group count alone does not predict DPO duration, and longest-biased smoke can
overestimate step cost.
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

V5 Gold SFT, Silver preference and separate development datasets are included in
`assets/datasets/single/observation-v5/`. The 7B SFT/DPO and 1.5B live reference
adapters are under `assets/adapters/single/`. The [asset index](../../../assets/README.md)
lists their files and external Base downloads.

Training and reference evaluation use these included assets directly. Historical
reconstruction instead needs local Teacher V2 certification exports and original
source records, which are not distributed. It does not rerun the retired
certification pipeline. The retained 1.5B adapter supports the live profile;
its presence does not promise reproduction of the original training.

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

The included frozen panel is evaluated with corrected simulator mechanics on
Windows and Linux. Current configs write separate `*-boss-corrected-v1` outputs;
they need not reproduce the [historical Boss scores](frozen_boss_192.md).
The reference adapters and historical reports retain their original identities.

[Boss input generation](../../../configs/generation/single_boss_inputs.json)
can build a new panel. Select its `inputs.json`, fingerprint and counts explicitly
in evaluation configs and use separate outputs. [Evaluation commands](#evaluate-reference-adapters)
above describe completion and resume.

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
Their retired execution surfaces require the original historical source and
assets identified in the [retention contract](../data_training.md#retention-and-retirement).
That source is not included in the public Git history or offered as a separate download.

The maintained Gold/Silver training inputs are the explicit dataset manifests at
`assets/datasets/single/observation-v5/{gold_sft,silver_preferences}.manifest.json`.
They select the artifacts, splits and observation contract. The two maintained
data entries below reconstruct records from retained certification exports and
the declared historical source. Earlier collection and certification executors
remain retired.

### Rebuilding V4 and V5 datasets

With the declared historical source assets available, run from the project root:

```powershell
uv run python scripts/prepare_dataset.py --config configs/data/single_teacher_v2.json
uv run python scripts/migrate_observation_v5.py --config configs/data/single_observation_v5.json
```

The first command reads completed V2 certification exports and eligible source
JSONL, writing V4 data to `outputs/datasets/single-rebuilt-v4/`. If the source
JSONL is absent, [rebuild its summaries](#rebuilding-expanded-source-summaries) first.
The second writes V5 data to `outputs/datasets/single-rebuilt-v5/`, adding the
glossary while preserving actions, labels and weights. The formal config also
rebuilds the separate development split. Neither command runs a model or Teacher.

Configs select inputs and destinations. Migration accepts one train or development
split per manifest; test data and overlapping source episodes are rejected.
Different existing content is never overwritten. Each output contains manifests,
compressed records and `report.json`; compare records as JSON, since gzip bytes
and provenance paths can differ. Point new training configs at the rebuilt manifests.

Local reconstruction on 2026-09-23 matched V4/V5 Gold (8,221 records each), Silver
(471 each), and V5 development (2,406). Restoring the 4,680-episode summaries also
matched both original audit hashes. The full local check took 108.3 seconds;
allow roughly 1–5 minutes depending on storage and CPU. These raw inputs are not
distributed. Ctrl+C stops reconstruction; retry unchanged inputs to reuse
identical outputs, or choose a new destination for changed inputs.

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

The script preserves original records and labels, excludes audited quarantined
sources from the eligible view, and checks train/development deck isolation.
It uses no model, Teacher search or current serializer. See the
[implementation](../../../scripts/rebuild_expanded_sft_summaries.py) for the exact
serialization rules. The restored views are historical export inputs, not the
maintained training manifests.
