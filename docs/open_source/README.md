# Running the project

Use the included adapters for a first run; training and historical data
reconstruction are optional. Commands below run from the repository root.
See the [project overview](../../README.md), [methods](data_training.md) and
[result analysis](stageresult.md) for the experiment itself.

## Setup

Install Git and `uv`, then:

```text
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

Installation time depends on downloads. Tests use CPU fixtures and need no Base
weights; absent native builds are skipped, while invalid builds fail. See
[test coverage](../../tests/README.md). A passing test suite is not GPU readiness.

Model runs require one CUDA GPU with native BF16, SDPA and unquantized weights.
CPU inference, ROCm and multi-GPU sharding are unsupported. Historical 7B training
used an RTX 5090 D; this is not a minimum-memory guarantee. Download the exact
[Base snapshot](../../assets/README.md#external-assets), including tokenizer files,
to `model.snapshot_path` in the selected runtime config; `cuda:N` selects the GPU.
`uv.lock` records the environment. If deliberately using another compatible
PyTorch build, use `uv run --no-sync` and repeat readiness on that environment.

## Build the simulator

Both platforms need Git and C++17 `g++` with static GCC/C++ runtime libraries.
Windows needs native MinGW and PowerShell; Linux needs pthread support. The first
build downloads pinned upstream sources and submodules. No GPU is needed.

```text
uv run --locked python scripts/simulator.py build --require combat_card_selection_v1
```

Success reports `ready` and `smoke.status: passed`. Valid builds are reused.
Typical build time is 2–4 minutes on Linux, 10–30 seconds for a Windows extension
with an existing base, or several minutes for a fresh Windows base. Ctrl+C can
leave an incomplete build.

Invalid builds require explicit recovery: on Windows, run
`scripts/build_sts_lightspeed.ps1` followed by `scripts/build_card_selection_bridge.ps1`;
on Linux, run `uv run --locked python scripts/build_sts_lightspeed_linux.py`.
Then repeat the shared build command. Rebuild old manifests missing mechanics or
patch records; do not edit them to bypass checks. Both platforms check search and
exhaust patches. Preserve the checkout's declared source line endings.

`simulator.py verify` checks installed bindings without launching native code;
`resolve` reports host selection and `smoke` checks the protocol. Current runs use
`corrected_v1`. Explicit `legacy_v1` supports only the Windows base without
secondary selection. Corrected mechanics alone do not enable V6 actions in V5.

## First evaluation

Choose either short run. These commands work in PowerShell or Bash, after setup:

```text
uv run --locked python scripts/run_frozen_policy_panel_evaluation.py --config configs/runs/evaluation/single_qwen2_5_7b_sft_boss.json --mode smoke --smoke-combats 2
```

This runs two standalone Boss combats with the supplied SFT adapter. **Rough time:
2–10 minutes**, including model loading. Its report is
`outputs/eval/qwen2-5-7b-sft-act1-a0-boss-corrected-v1/smoke/report.json`.
Expect `smoke_completed`, two combat identities and passing checks.

```text
uv run --locked python scripts/run_continuous.py --config configs/runs/evaluation/continuous_qwen2_5_7b_mixed_sft.json --mode smoke --smoke-routes 1
```

This runs one topology with all four seed groups using mixed SFT. **Rough time:
3–15 minutes**, including reward/upgrade search. Its report is
`outputs/generation/qwen2-5-7b-gold-teacher-mixed-sft-act1-continuous-v7-v1/smoke/report.json`.
Expect `smoke_completed`, one route topology, four executions and one arm. Deaths
can still be valid completed executions; inspect metrics for policy performance.

For a full evaluation, omit `--mode smoke` and its size option. Single runs cover
192 combats; continuous runs cover 80 routes per arm. Allow roughly **15–45 minutes
per single policy** or **30–90 minutes per continuous policy**. GPU, CPU search and
combat length affect timing; recorded continuous DPO A took about 41 minutes.
Configs under [runs/evaluation](../../configs/runs/evaluation/) select the policy.
The continuous `base_gold_sft` config contains two arms; `--arm` selects one.
Calling it without that filter combines completed arm caches into a full report.

Formal output is `output_dir/formal/report.json` with `status: completed` and all
declared combats/routes accounted for. Keep the whole output tree, including
inputs and trajectories. Ctrl+C stops a run; repeating the same command validates
completed work and resumes, restarting any interrupted route. Use new outputs for
changed conditions. A short run verifies execution, not policy quality.

## Train new adapters

Use a clean, committed checkout. Preparation connects new SFT/DPO checkpoints and
evaluations in separate outputs; it does not run training (a few seconds):

```text
uv run --locked python scripts/prepare_experiment.py --config configs/experiments/single_7b.json --config configs/experiments/continuous_7b.json --output outputs/reproduction
```

Either experiment list can be used alone. Repeating unchanged preparation preserves
files; use a new output root for changed experiments. Dependency order is:

| Experiment | Order |
| --- | --- |
| single | Gold SFT → Silver DPO |
| continuous | GOLD-only SFT and mixed SFT independently → DPO A/B/C independently from mixed SFT |

For example, run mixed SFT with this block in Linux Bash or Windows Git Bash
(with Windows `uv` on PATH). It stops at the first failure:

```bash
(
set -euo pipefail
training=outputs/reproduction/configs/runs/training/continuous_qwen2_5_7b_mixed_sft.json
recovery="outputs/recovery/mixed-sft-$(date -u +%Y%m%dT%H%M%S)-$$.json"
uv run --locked python scripts/restore_training.py --config "$training" --report "$recovery"
uv run --locked python scripts/run_training.py --config "$training" --mode smoke --recovery-report "$recovery"
uv run --locked python scripts/run_training.py --config "$training" --mode run --recovery-report "$recovery"
uv run --locked python scripts/run_continuous.py --config outputs/reproduction/configs/runs/evaluation/continuous_qwen2_5_7b_mixed_sft.json
)
```

Other arms use their corresponding generated training/evaluation configs and the
same readiness → smoke → training sequence. Single evaluation uses
`run_frozen_policy_panel_evaluation.py`. Evaluate Base under the same conditions,
or reuse a complete matching result; historical scores from a different simulator
are not a substitute. Supplied configs evaluate supplied adapters, while generated
configs evaluate the newly trained ones.

Readiness loads the actual assets and checks the longest-sample backward. Smoke
checks optimizer updates and saving. Allow **roughly 5–20 minutes** for both.
Historical continuous training took 1.3 hours for GOLD SFT, 2.2 hours for mixed SFT
and 3.5–4.1 hours per DPO arm; single 7B SFT/DPO took about 33/4 minutes.
These are historical timings, not guarantees; use current smoke timing to refine
them and add evaluation time separately.

Expected signals are `ready`, `smoke_passed`,
`trained_pending_development_evaluation`, then evaluation `completed`. Training
reports must show complete steps, finite loss and exact fresh-Base reload.
Ctrl+C interrupts; repeating the block checks readiness and resumes compatible
checkpoints. Unsaved steps repeat. Keep code/config/data unchanged while running,
and retain generated configs, recovery receipts, checkpoints and complete
evaluation trees. Final reference adapters contain no optimizer resume state.
See [compatibility boundaries](data_training.md#configuration-and-artifact-reuse-direction).

## Data generation and reconstruction

The included datasets need no reconstruction. To create new continuous data, use
an independent [Teacher pool](../../configs/generation/continuous_teacher_pool.json),
prepare a source-bound selection, then collect, replay and export GOLD. The
[script index](../../scripts/README.md) links entries; [methods](data_training.md#teacher-and-gold)
explains the labels. Collection alone does not produce certified training targets.

Original Teacher pools, imported GOLD trials and replay trajectories are not
distributed. Historical single reconstruction needs completed Teacher V2 exports;
it does not restore the retired certification pipeline. Historical GOLD export
also needs imported V3 trials and a report-bound replay receipt. A sampled receipt
requires explicit `verification_scope: sampled`; matching all exported records
does not establish full replay. Rebuilding with current mechanics can produce new
loadouts and data. The [results](stageresult.md#reconstruction-and-source-evidence)
record what was actually checked locally.

The [asset index](../../assets/README.md) covers included files and picker rebuilds.
Real-game use has a separate [short guide](real_game_testing.md).
