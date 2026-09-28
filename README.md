# STS1 LLM Policy

A small post-training research project: turn public Slay the Spire 1 combat state
into one legal `ACTION_n`, then compare Base, SFT and preference-trained policies.

The two maintained stages are **single** (single-combat information) and
**continuous** (public route context). Source, formal configs, training datasets,
reference LoRA adapters, picker inputs and original reports are included. Base
weights and native builds are obtained separately. Raw Teacher/GOLD collections
and original replay trajectories are not distributed.

```text
public combat state -> ACTION_n -> simulator trajectory
                    -> Teacher data -> SFT -> optional preference training
```

## Current offline results

The experiment stages are [single](docs/open_source/experiments/single.md)
(single-combat information) and [continuous](docs/open_source/experiments/continuous.md)
(route context). Their configurations select the technical versions and run
settings; the [experiment index](docs/open_source/experiments/README.md) connects
each stage's data, training and evaluation, including supporting diagnostics.

The latest completed experiments use Qwen2.5-7B-Instruct with `observation_v7`
on configured Act 1/A0 continuous routes. GOLD-only SFT, mixed GOLD/Teacher SFT,
and three DPO variants have completed training and evaluation. The primary metric
is final Boss victories across 20 route topologies × four seed groups:

| Policy | Boss victories / 80 | Original evaluation report |
|---|---|---|
| Base | 4 | [Base / GOLD SFT](report/evaluation/qwen2_5_7b_base_gold_sft_act1_continuous_v7_v1.json) |
| GOLD-only SFT | 13 | [Base / GOLD SFT](report/evaluation/qwen2_5_7b_base_gold_sft_act1_continuous_v7_v1.json) |
| Mixed GOLD/Teacher SFT | 26 | [Mixed SFT](report/evaluation/qwen2_5_7b_gold_teacher_mixed_sft_act1_continuous_v7_v1.json) |
| DPO A | 31 | [DPO A](report/evaluation/qwen2_5_7b_gold_dpo_a_act1_continuous_v7_v1.json) |
| DPO B | 30 | [DPO B](report/evaluation/qwen2_5_7b_gold_dpo_b_act1_continuous_v7_v1.json) |
| DPO C | 28 | [DPO C](report/evaluation/qwen2_5_7b_gold_dpo_c_act1_continuous_v7_v1.json) |

DPO variants initialize from mixed SFT and share training states and update
budgets, with different preference filtering/weighting. Their definitions are in
the [DPO contract](docs/open_source/data_training.md#matched-v7-dpo-preference-datasets).
These are development-panel results: four seeds share each topology, and the
panel has informed experiment design. Raw win counts do not establish statistical
superiority or independent generalization. See the
[mixed SFT analysis](docs/open_source/stageresult.md#7b-mixed-goldteacher-sft-on-matched-v7-routes)
for its comparisons and limitations.

The real-game default remains the **1.5B Gold SFT V5** profile in
[real_game_gold_sft_v5_session.json](configs/live/real_game_gold_sft_v5_session.json).
V6 card selection is opt-in; the 7B V7 offline results do not replace this default.
Real-game integration is an auxiliary capability; the
[live guide](docs/open_source/real_game_testing.md) describes its setup and tested scope.
Earlier Expanded SFT, topology-DPO and frozen Boss-panel results remain historical
evidence in [reports](report/README.md) and the
[experiment index](docs/open_source/experiments/README.md).

## Setup and verification

Install `uv` and Git, then run commands from the repository root. Use a Git clone
for training: readiness requires a committed HEAD and a clean tracked worktree;
an extracted source archive alone does not satisfy that requirement.
The commands below and the stage pages' `text` blocks work in PowerShell or Bash.
Blocks marked `bash` use Bash on Linux or Git Bash on Windows, with Windows `uv`
on PATH. Run standalone commands in order and continue only after each
succeeds; the Bash experiment blocks stop at the first error.

```text
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

Installation time depends on the PyTorch download and local cache. The default
suite needs no model or GPU; native classes skip when the simulator is absent.
An incomplete or invalid installation fails. See [test prerequisites](tests/README.md).
A fresh Python 3.11 uv environment was checked on Windows x64; that check does not
certify model/native execution or other platforms.

The [CI workflow](.github/workflows/ci.yml) runs the same locked installation and
test discovery on Ubuntu 24.04 and Windows Server 2022 with Python 3.11, then checks
`--help` for every Python entry under `scripts/`. It runs on pull requests, pushes
to `main`/`master`, and manual dispatch. Tests use CPU models and synthetic inputs;
CI does not download Base weights or build the native simulator. Skipped native
tests remain unverified. Logs are available in the repository's Actions tab.
Runtime input validation and [current-machine training readiness](docs/open_source/runtime_and_simulator.md#training-readiness-on-the-current-machine)
still apply to actual experiments; CI success does not establish GPU readiness
or reproduce experimental results.

### Choose a run

For the first model run, obtain the [Base assets](assets/README.md#external-assets)
and meet the [GPU and native build requirements](docs/open_source/runtime_and_simulator.md#execution-requirements-and-recorded-hardware).
The short paths below use included reference adapters and exercise model loading,
simulator execution and saved outputs. They need no training or data reconstruction.

| Goal | single | continuous |
| --- | --- | --- |
| First short run | [Two combats with Gold SFT](docs/open_source/experiments/single.md#first-short-run) | [One topology × four seeds with mixed SFT](docs/open_source/experiments/continuous.md#first-short-run) |
| Full reference evaluation | [Base, SFT and DPO](docs/open_source/experiments/single.md#evaluate-reference-adapters) | [Six policy arms](docs/open_source/experiments/continuous.md#evaluate-reference-adapters) |
| Train and evaluate new models | [Gold SFT → Silver DPO](docs/open_source/experiments/single.md#running-the-7b-comparison) | [GOLD/mixed SFT → DPO A/B/C](docs/open_source/experiments/continuous.md#train-and-evaluate-new-models) |

Each stage gives inputs, commands, time estimates, output checks and resume rules.
Evaluation and training entries perform their input checks automatically;
standalone `--mode preflight` is available for diagnosis before model execution.
Training also requires current-machine readiness and optimizer smoke, as shown in
the stage commands. A short run establishes execution, not policy performance.

## Maintained boundaries

- The student returns one decision-local `ACTION_n`. V7 offline input includes
  public route context, combat card selection and observed draw-prefix memory;
  the live default retains its separate V5 contract.
- Hidden order, native execution IDs, Teacher visits, and rollout values never
  enter student input.
- Duplicate native actions use `action_equivalence_v1`.
- Invalid output receives one retry and then a visible seeded legal fallback.
- Current training and development sources are isolated by source route;
  development trajectories and sealed data are excluded from training.
- Continuous routes use configured reward, upgrade, removal and healing rules;
  they are not full native map/shop/event runs. The native Teacher optimizes local
  combat; GOLD estimates do not certify route-optimal actions.
- Expanded simulator coverage does not establish equivalent real-game coverage.

## Assets and reproduction

| Starting point | Public scope |
| --- | --- |
| Read methods and results | Source, configs, contracts and original reports; historical diagnostics may retain evidence without their retired execution environment. |
| Evaluate or retrain | Included datasets, reference adapters and picker inputs, plus separately obtained Base/native assets. Readiness checks apply to the actual machine. |
| Generate new data | Maintained Teacher/GOLD generation, verification and export entries; use explicit sources, partitions and separate destinations. See [continuous data preparation](docs/open_source/experiments/continuous.md#data-and-labels). |
| Reconstruct historical data or replay original runs | Partial: original source collections, continuation imports and trajectories are not distributed. Hashes cannot replace these inputs. See [single reconstruction](docs/open_source/experiments/single.md#rebuilding-v4-and-v5-datasets) and [continuous prerequisites](docs/open_source/experiments/continuous.md#downloads-and-starting-points). |

### Asset downloads and generation

The [asset index](assets/README.md) owns acquisition, installation paths and
availability. The stage tables connect those assets to their generation entries
and recorded training evidence:
[single](docs/open_source/experiments/single.md#downloads-and-starting-points) and
[continuous](docs/open_source/experiments/continuous.md#downloads-and-starting-points).

Training can produce different adapter bytes. Reproduce the procedure and compare
metrics under the same evaluation conditions. Training completion, exact checkpoint
reload and improved policy performance are distinct results.

Historical source revisions are provenance identifiers; the release snapshot does
not include earlier Git history or all retired entry points. Reports retain their
original identities, with [explicitly marked path redactions](report/README.md).
Their recorded hardware is not a current setup requirement; see
[execution requirements](docs/open_source/runtime_and_simulator.md#execution-requirements-and-recorded-hardware).

Local reconstruction matched ten datasets, while GOLD continuation replay was
sampled (240 of 7,134 complete states). The stage pages describe those different
checks and their limits. Undistributed evidence prevents full independent
verification of the original executions; passing the public tests does not close
that gap.

## Repository

```text
configs/                 runtime, training, and evaluation bindings
assets/                  fixed datasets, reference adapters, picker and evaluation inputs
outputs/                 local generated datasets, training and evaluation runs
docs/open_source/        maintained public contracts and current results
report/                  machine evidence grouped by evidence type
scripts/                 operational entry points
src/sts1_llm_policy/     runtime, data, training, and evaluation code
tests/policy/            policy and observation behavior
tests/simulator/         simulator and Teacher search behavior
tests/live/              real-game adapter and session behavior
tests/training/          training, recovery, and workflow behavior
```

New runs write to `outputs/`; retain the reports, checkpoints and raw trajectories
needed to verify them. Private deployment tools and machine-local artifacts are
excluded. [Scripts](scripts/README.md) lists entry points; the
[documentation index](docs/open_source/README.md) links contracts and results.

## License

Project-authored material is licensed under the [MIT license](LICENSE).
Third-party material retains its own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
Base weights, game assets and raw reconstruction/replay evidence are not included.
Included reference adapters, datasets and picker inputs retain the third-party
terms applicable to their source material, as described in the notices.
