# continuous

SFT and DPO with public continuous-route context, evaluated by final Boss
completion on configured Act 1/A0 routes. The student sees route position,
remaining public operations and known draw-prefix memory under the current V7
contract. Teacher search and GOLD certification still optimize/evaluate one
combat at a time; they do not provide route-optimal labels.

The source snapshot includes preparation, GOLD collection/replay verification,
export, training and evaluation code, formal configs and original reports.
The maintained datasets, reference adapters and picker database are included in
Git. Base weights and native builds are obtained separately. Original raw evidence
remains outside Git; historical GOLD reconstruction has the additional input and
receipt requirements below.

## First short run

Complete [Python setup](../../../README.md#setup-and-verification), obtain the
[7B Base snapshot](../../../assets/README.md#external-assets), and satisfy the
[GPU requirements](../runtime_and_simulator.md#execution-requirements-and-recorded-hardware)
and [native build prerequisites](../runtime_and_simulator.md#native-build-prerequisites).
The mixed SFT reference adapter and picker database are included. Run these
commands from the repository root in PowerShell or Bash, continuing only after
each succeeds:

```text
uv run --locked python scripts/simulator.py build --require combat_card_selection_v1
uv run --locked python scripts/run_continuous.py --config configs/runs/evaluation/continuous_qwen2_5_7b_mixed_sft.json --mode smoke --smoke-routes 1
```

The build includes its native protocol smoke; see [build time and recovery](../runtime_and_simulator.md#callable-native-environment-selection).
`--smoke-routes 1` selects **one topology and all four configured seed groups**:
four route executions for this single policy. Each route continues until defeat,
the decision limit, or final Boss completion. **Rough model-smoke budget:
3–15 minutes** after installation, including model loading and reward/upgrade
searches. This is an estimate from full-panel timings, not a measured short run;
a first-topology subset need not have the panel's average combat length.

The runner checks inputs before loading the policy and prints route progress.
Open
`outputs/generation/qwen2-5-7b-gold-teacher-mixed-sft-act1-continuous-v7-v1/smoke/report.json`.
Success is `status: smoke_completed`, with `scope.routes: 1`,
`scope.route_executions: 4` and `scope.arms: 1`. A completed execution can include
defeats or truncated routes; inspect the policy's `metrics` for actual outcomes.
Keep the whole `smoke/` tree, including `inputs.json`, route records and combat
trajectories. Ctrl+C stops execution; rerunning validates completed routes and
restarts an interrupted route. Formal evaluation writes separately to `formal/`.

This checks execution and artifacts, not a reliable Boss-win estimate.
Next choose [full reference evaluation](#evaluate-reference-adapters) or
[new training](#train-and-evaluate-new-models). [Data generation and historical
reconstruction](#data-and-labels) are separate starting points.

## Evaluate reference adapters

With the [first-run prerequisites](#first-short-run) installed, run this block
from the repository root in Linux Bash or Windows Git Bash to evaluate all six
reference policies. The first config contains both Base and GOLD-only SFT; the
other configs each contain one arm. Each invocation validates its own inputs.

```bash
(
set -euo pipefail
uv run --locked python scripts/simulator.py build --require combat_card_selection_v1
for arm in base_gold_sft mixed_sft dpo_a dpo_b dpo_c; do
    evaluation="configs/runs/evaluation/continuous_qwen2_5_7b_${arm}.json"
    uv run --locked python scripts/run_continuous.py --config "$evaluation"
done
)
```

The recorded DPO A evaluation took about 41 minutes for 80 routes; **rough budget:
30–90 minutes per policy**, plus [native build time](../runtime_and_simulator.md#callable-native-environment-selection).
GPU, CPU search cost and combat length affect timing. All six policies execute
480 routes in total. For one supplied policy, use only its [tabled config](#training-and-evaluation).
`--arm` can select Base or GOLD SFT separately; filtered runs write
`report-<arm-id>.json` and `inputs-<arm-id>.json`. After both arms complete, call
without `--arm` to validate their caches and write the combined report.
Standalone `--mode preflight` is optional for diagnosis before model execution.

Success is `status: completed` in each configured `output_dir/formal/report.json`,
with 80 route executions per arm. Retain the complete `formal/` directory:
reports, inputs, route records and combat trajectories. Compare final Boss wins
first, then deaths, legality and fallback metrics. Ctrl+C interrupts execution;
rerun the block to reuse completed route caches. Keep inputs and configs unchanged
for resume; historical V1 caches are not accepted as current V2 results.

## Train and evaluate new models

Use a clean Git checkout and the [first-run prerequisites](#first-short-run).
From the repository root in Linux Bash or Windows Git Bash, this block prepares
independent outputs and runs GOLD-only SFT, mixed SFT and DPO A/B/C in dependency
order. Every DPO arm uses the new mixed checkpoint
as both initialization and frozen reference. The first evaluation includes Base.

```bash
(
set -euo pipefail
uv run --locked python scripts/prepare_experiment.py --config configs/experiments/continuous_7b.json --output outputs/reproduction
uv run --locked python scripts/simulator.py build --require combat_card_selection_v1
for arm in gold_sft mixed_sft dpo_a dpo_b dpo_c; do
    training="outputs/reproduction/configs/runs/training/continuous_qwen2_5_7b_${arm}.json"
    evaluation_arm="$arm"
    if [ "$arm" = gold_sft ]; then evaluation_arm=base_gold_sft; fi
    evaluation="outputs/reproduction/configs/runs/evaluation/continuous_qwen2_5_7b_${evaluation_arm}.json"
    recovery="outputs/recovery/continuous-${arm}-$(date -u +%Y%m%dT%H%M%S)-$$.json"
    uv run --locked python scripts/restore_training.py --config "$training" --report "$recovery"
    uv run --locked python scripts/run_training.py --config "$training" --mode smoke --recovery-report "$recovery"
    uv run --locked python scripts/run_training.py --config "$training" --mode run --recovery-report "$recovery"
    uv run --locked python scripts/run_continuous.py --config "$evaluation"
done
)
```

Preparation takes seconds. Recovery includes configuration/data/tokenizer checks,
generation and longest-sample backward; optimizer smoke checks the update/save path.
Standalone `--mode preflight` is available for diagnosis.
**Rough readiness plus smoke budget: 5–20 minutes per run.** Recorded training took
approximately 1.3 hours for GOLD SFT, 2.2 hours for mixed SFT and 3.5–4.1 hours per
DPO arm; current hardware and token lengths determine
actual time. Evaluation adds the per-policy budget above. Use current smoke timing
to refine the estimate. These are multi-hour experiments, not installation checks.

The block stops at the first error. Expected signals are `ready`, `smoke_passed`,
`trained_pending_development_evaluation`, then evaluation `completed`.
Ctrl+C stops the current stage. Rerunning performs fresh readiness and reuses
compatible completed training/checkpoints and route caches; unsaved steps repeat.
Preserve generated configs, `outputs/reproduction/training/` and
`outputs/reproduction/evaluation/` with its route records and trajectories, plus
the recovery receipts. Retain training resume state while continuation is needed.
Training reports must show complete steps, finite losses and exact fresh-Base
reload; evaluation reports must account for all declared routes. Review the final reports and retain
the raw artifacts before drawing a policy comparison.

The [current results](../../../README.md#current-offline-results) report 80 routes
per arm, sharing 20 topologies and four seed groups. This is a development panel,
not independent generalization evidence. The [early route record](continuous_routes_20x4.md)
retains prior Base/Teacher executions with different observation/mechanics
conditions; those results are not additional matched arms of this comparison.

## Training and evaluation

| Arm | Training configuration | Evaluation configuration | Original training / evaluation reports |
| --- | --- | --- | --- |
| Base | Unadapted 7B Instruct | [Base / GOLD-only](../../../configs/runs/evaluation/continuous_qwen2_5_7b_base_gold_sft.json) | [Evaluation](../../../report/evaluation/qwen2_5_7b_base_gold_sft_act1_continuous_v7_v1.json) |
| GOLD-only SFT | [Gold SFT](../../../configs/runs/training/continuous_qwen2_5_7b_gold_sft.json) | [Base / GOLD-only](../../../configs/runs/evaluation/continuous_qwen2_5_7b_base_gold_sft.json) | [Training](../../../report/training/qwen2_5_7b_gold_sft_v7_v1.json) / [evaluation](../../../report/evaluation/qwen2_5_7b_base_gold_sft_act1_continuous_v7_v1.json) |
| Mixed SFT | [Mixed SFT](../../../configs/runs/training/continuous_qwen2_5_7b_mixed_sft.json) | [Mixed SFT](../../../configs/runs/evaluation/continuous_qwen2_5_7b_mixed_sft.json) | [Training](../../../report/training/qwen2_5_7b_gold_teacher_mixed_sft_v7_v1.json) / [evaluation](../../../report/evaluation/qwen2_5_7b_gold_teacher_mixed_sft_act1_continuous_v7_v1.json) |
| DPO A | [DPO A](../../../configs/runs/training/continuous_qwen2_5_7b_dpo_a.json) | [DPO A](../../../configs/runs/evaluation/continuous_qwen2_5_7b_dpo_a.json) | [Training](../../../report/training/qwen2_5_7b_gold_dpo_a_v7_v1.json) / [evaluation](../../../report/evaluation/qwen2_5_7b_gold_dpo_a_act1_continuous_v7_v1.json) |
| DPO B | [DPO B](../../../configs/runs/training/continuous_qwen2_5_7b_dpo_b.json) | [DPO B](../../../configs/runs/evaluation/continuous_qwen2_5_7b_dpo_b.json) | [Training](../../../report/training/qwen2_5_7b_gold_dpo_b_v7_v1.json) / [evaluation](../../../report/evaluation/qwen2_5_7b_gold_dpo_b_act1_continuous_v7_v1.json) |
| DPO C | [DPO C](../../../configs/runs/training/continuous_qwen2_5_7b_dpo_c.json) | [DPO C](../../../configs/runs/evaluation/continuous_qwen2_5_7b_dpo_c.json) | [Training](../../../report/training/qwen2_5_7b_gold_dpo_c_v7_v1.json) / [evaluation](../../../report/evaluation/qwen2_5_7b_gold_dpo_c_act1_continuous_v7_v1.json) |

Both SFT runs start from fresh Base; DPO arms independently initialize from
mixed SFT and use that checkpoint as the frozen reference. The table identifies
the supplied reference weights and their original evidence. The
[new-training sequence](#train-and-evaluate-new-models) prepares separate configs
whose evaluations and DPO inputs select the newly trained checkpoints.

## Shared evaluation panel

All six arms use [one development panel](../../../configs/panels/continuous_act1_development.json)
for routes, seeds, public inputs, simulator mechanics and non-combat strategies.
Run configs select models and outputs. To change conditions, select a complete
new panel; partial overrides and nested panels are rejected. Reports save the
expanded settings. [Experiment preparation](../data_training.md#prepare-independent-training-and-evaluation-configs)
connects new checkpoints without changing the reference configurations.

## Downloads and starting points

Datasets and reference adapters are included under `assets/datasets/continuous/`
and `assets/adapters/continuous/`; see the [asset index](../../../assets/README.md)
for all paths. Original Teacher pool trajectories, GOLD continuation evidence and
evaluation trajectories are local only. The public selection files do not contain
those raw inputs.

For new data, generate an independent Teacher pool, prepare its GOLD selection,
then collect, verify and export to separate destinations. Select rebuilt manifests
explicitly in mixed export and training configs. Regeneration does not promise
byte-identical historical data.

Historical GOLD reconstruction additionally requires the imported
`teacher-gold-collection-v3/formal` state results. Older replay receipts lack the
report binding required by current exports: with the raw inputs present, run
`collect_gold.py --verify --verification-output` to write a new receipt and use it
in a separate export config. Keep the original evidence. A sampled replay receipt
requires explicit `verification_scope: sampled`; see [replay coverage](../teacher_gold.md#execution-verification-and-receipts).

Local reconstruction on 2026-09-23 matched all GOLD SFT, DPO A/B/C and mixed SFT
records. Mixed export replayed all 1,440 selected Teacher combats. GOLD replay
covered 240 complete states—216 stratified random and 24 targeted—with 89,664
continuations and 901,211 decisions. The remaining states were not replayed;
full dataset equality is not full continuation verification. Raw inputs and the
new receipt are not distributed.

Use new output directories for current runs. Historical V1 route caches cannot
resume as current V2 executions; see [compatibility](../data_training.md#current-configuration-and-identity-implementation).

## Data and labels

| Step | Configuration | Output and evidence |
| --- | --- | --- |
| Teacher source pool | [continuous_teacher_pool](../../../configs/generation/continuous_teacher_pool.json) | [Collection record](teacher_candidate_pool_800x4.md), original trajectories and replay evidence |
| GOLD collection | [continuous_gold_collection](../../../configs/generation/continuous_gold_collection.json) | `outputs/generation/teacher-gold-collection-v4/formal/`; report, execution traces and replay receipt |
| GOLD-only SFT export | [continuous_gold_sft](../../../configs/data/continuous_gold_sft.json) | `outputs/datasets/gold-sft-v7-v1/manifest.json` |
| Mixed SFT export | [continuous_mixed_sft](../../../configs/data/continuous_mixed_sft.json) | `outputs/datasets/gold-teacher-mixed-sft-v7-v1/formal/manifest.json` |
| DPO A/B/C export | [continuous_dpo](../../../configs/data/continuous_dpo.json) | `outputs/datasets/gold-dpo-v7-v1/{a,b,c}/manifest.json`, [original export report](../../../report/data/gold_dpo_v7_v1.json) |

The source pool contains independent training-source routes, excluding the
development sources. The formal GOLD selection uses 480 routes; mixed SFT adds
Teacher demonstrations from the selected seed groups' first three combats, with
overlaps assigned to GOLD. A/B/C share states and differ in preference filtering
or weighting. Exact selection, weighting and certification settings belong to
the configs and [data contracts](../data_training.md#dataset-exports).

Generation uses [run_continuous.py](../../../scripts/run_continuous.py).
[prepare_gold_collection.py](../../../scripts/prepare_gold_collection.py) prepares
new panels from explicit templates, sources and source-bound exclusion lists;
[collect_gold.py](../../../scripts/collect_gold.py) executes the selected
config and performs replay verification. Exports use
[export_gold_sft.py](../../../scripts/export_gold_sft.py) and
[export_gold_dpo.py](../../../scripts/export_gold_dpo.py).
Existing dataset manifests remain usable. New GOLD exports require an actual
report-bound replay receipt; old receipts are not upgraded by renaming configs.

### Selection and source isolation

[The fixed selection](../../../assets/datasets/continuous/gold-selection.json)
identifies ordered states and their source pool; mixed export uses the same file.
For a new panel, `prepare_gold_collection.py` needs an explicit source and a
matching exclusion document. The [included exclusions](../../../assets/datasets/continuous/gold-exclusions.json)
apply only to their original pool. An empty list is valid only when the new pool
has no tuning routes to exclude.

Preparation clears inherited imports and old samples, then writes a new selection
and execution config. It does not need the old raw pool. Subsequent collection
still needs the newly declared source reports and trajectories. Use separate
outputs for collection, [replay verification](../teacher_gold.md#gold-execution-and-replay)
and export.

## Artifacts, checks and supporting experiments

Keep the reports and complete output trees specified above. Training completion,
evaluation outcomes and action-level replay establish different things.
[Supporting experiments](README.md#supporting-work) retain action-order,
multi-target, continuation-policy and sampling-ladder results; retired preparers
are not required to train from the included datasets.
