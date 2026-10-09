# STS1 LLM Policy

A small post-training research project: turn public Slay the Spire 1 combat state
into one legal `ACTION_n`, then compare Base, SFT and preference-trained policies.
Here, **Base** means the unadapted Qwen2.5 **Instruct** model, before this
project's SFT or DPO training.

The experiments ask whether simulator-derived supervision improves an LLM's
combat decisions, and whether preference training adds useful discrimination
after supervised fine-tuning. The focus is Ironclad combat: choosing cards,
targets and when to end a turn from structured public information.

## Experiment design

### Tasks and model interface

**single** evaluates independent combats using public combat information (V5).
**continuous** adds public route context and secondary card choices (V7), carrying
HP and resources through configured Act 1 routes toward the final Boss.
In both stages, Base, SFT and DPO receive a text state and legal-action list, then
return one `ACTION_n` per decision. Hidden draw order, future RNG and Teacher
scores are excluded. The stages differ in data and evaluation as well as observation.

> [!NOTE]
> **Details:** [Student inputs and action protocol](docs/open_source/policy_observation.md)
> · [Experiment stages](docs/open_source/experiments/README.md)

### Teacher and training data

The simulator and native search build on
[gamerpuppy/sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed).
Its `BattleScumSearcher2` explores action sequences through tree search and
simulated playouts; this project adds a decision bridge and mechanics adaptations.
This search-based **Teacher** generates trajectories. The continuous **GOLD**
pipeline then compares candidate actions by forcing each one and repeatedly
playing the combat to completion, with hidden-world sampling for continuation
decisions. Wins and carried HP identify a set of acceptable actions rather than
requiring one unique answer. These are estimates of local combat quality, not
proof of the best action for an entire route.

> [!NOTE]
> **Details:** [Teacher continuation and GOLD scoring](docs/open_source/teacher_gold.md#continuation-scoring)
> · [Candidate trajectory collection](docs/open_source/experiments/teacher_candidate_pool_800x4.md)
> · [Upstream attribution and MIT terms](THIRD_PARTY_NOTICES.md#sts_lightspeed)

### From SFT to preference training

**SFT (supervised fine-tuning)** learns action responses from Teacher-derived
targets by training small LoRA adapters.
In the continuous stage, **GOLD-only SFT** uses acceptable-action sets;
**mixed SFT** adds early-combat Teacher demonstrations for denser coverage.
Mixed training also uses more optimizer steps, so its comparison with GOLD-only
SFT includes a training-budget difference.

**DPO (direct preference optimization)** learns to prefer one action over another
relative to a frozen reference model.
The continuous A/B/C variants start independently from the same mixed SFT model
and share training states and optimizer-step budgets:

| Variant | Preference design |
| --- | --- |
| **A — all eligible pairs** | Keep all pairs passing the shared quality filters, with equal total weight per state. |
| **B — larger gaps** | Keep only preferences with a larger carried-HP advantage. |
| **C — gap weighting** | Keep A's pairs, but reduce the contribution of smaller HP differences, including across states. |

> [!NOTE]
> **Details:** [GOLD SFT targets](docs/open_source/data_training.md#executed-gold-v7-sft-groups)
> · [Mixed SFT weighting](docs/open_source/data_training.md#mixed-gold-and-teacher-v7-sft)
> · [DPO A/B/C definitions](docs/open_source/data_training.md#matched-v7-dpo-preference-datasets)

### LoRA and training settings

The 7B continuous experiments freeze the Base weights and train low-rank updates
on the attention query/value projections (`q_proj`, `v_proj`). LoRA uses **rank 8,
alpha 16 and dropout 0**, updating about **2.52 million parameters (0.033%)**.
Both SFT runs start with fresh adapters; DPO continues the mixed SFT adapter.

| Setting | GOLD-only / mixed SFT | DPO A/B/C |
| --- | --- | --- |
| Learning rate | `1e-4` | `2e-5` |
| Training passes | 1 epoch | 1 epoch |
| Effective batch | 8 states | 8 states |
| DPO beta | — | `0.1` |

Each step accumulates gradients from one state at a time, up to eight states;
multiple target actions or preference pairs belong to that same state. Both
recipes use AdamW, zero weight decay and gradient-norm clipping at 1.0. Training
uses BF16 and gradient checkpointing. Inputs are limited to 3,072 tokens without
truncation; only response tokens, including the terminator, contribute to the loss.

> [!NOTE]
> **Settings:** [SFT recipe](configs/profiles/training/sft_lora_v1.json)
> · [DPO recipe](configs/profiles/training/dpo_lora_v1.json)

### Deck building and route control

The LLM controls combat; configured strategies handle the route between fights.
**Reward selection** compares taking each card with skipping, combining Teacher
simulations, pick-statistic priors and deck synergy. **Upgrades** use separate
Teacher combat comparisons; **removal** follows a starter-card rule. Policies
share these rules and route seeds so evaluation focuses on combat decisions.
The routes include relics and healing, but do not reproduce full native map,
shop or event play.

> [!NOTE]
> **Details:** [Reward-picker algorithm](src/sts1_llm_policy/eval/counterfactual_reward_picker_v2.py)
> · [Route and upgrade design](docs/open_source/runtime_and_simulator.md#configured-continuous-act-1-development-routes)
> · [Shared evaluation panel](configs/panels/continuous_act1_development.json)

## Current offline results

The latest completed experiments use Qwen2.5-7B-Instruct with `observation_v7`
on configured Act 1/A0 continuous routes. GOLD-only SFT, mixed GOLD/Teacher SFT,
and three DPO variants have completed training and evaluation. The primary metric
is final Boss victories across 20 route topologies × four seed groups:

| Policy | Boss victories / 80 | Evaluation evidence |
|---|---|---|
| Base | 4 | [Base / GOLD SFT](report/evaluation/qwen2_5_7b_base_gold_sft_act1_continuous_v7_v1.json) |
| GOLD-only SFT | 13 | [Base / GOLD SFT](report/evaluation/qwen2_5_7b_base_gold_sft_act1_continuous_v7_v1.json) |
| Mixed GOLD/Teacher SFT | 26 | [Mixed SFT](report/evaluation/qwen2_5_7b_gold_teacher_mixed_sft_act1_continuous_v7_v1.json) |
| DPO A | 31 | [DPO A](report/evaluation/qwen2_5_7b_gold_dpo_a_act1_continuous_v7_v1.json) |
| DPO B | 30 | [DPO B](report/evaluation/qwen2_5_7b_gold_dpo_b_act1_continuous_v7_v1.json) |
| DPO C | 28 | [DPO C](report/evaluation/qwen2_5_7b_gold_dpo_c_act1_continuous_v7_v1.json) |
| Teacher (historical V6) | 68 | [Recorded Teacher results](docs/open_source/stageresult.md#7b-base-and-teacher-on-continuous-routes) |

The Teacher row predates later simulator corrections and serves as a historical
reference, not a matched V7 control. See the [original evaluation conditions](docs/open_source/experiments/continuous_routes_20x4.md#observation-and-action-contract).

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

## Scope and limitations

- Training and development sources are isolated by source episode or route;
  development trajectories and sealed data are excluded from training.
- Expanded simulator coverage does not establish equivalent real-game coverage.
- Current single evaluation uses corrected simulator mechanics; its scores need
  not reproduce the [historical Boss results](docs/open_source/experiments/frozen_boss_192.md).

## Setup and verification

From the repository root, install dependencies and run the local tests:

```text
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

Installation time depends on downloads and cache. Native tests skip if the
simulator is absent; an invalid installation fails. See [setup and requirements](docs/open_source/runtime_and_simulator.md#setup)
for GPU/native builds and training prerequisites, and [tests](tests/README.md)
for test and CI coverage.

### Choose a run

Obtain the [Base weights](assets/README.md#external-assets) first. Short runs use
included adapters and require no training or data reconstruction.

| Goal | single | continuous |
| --- | --- | --- |
| First short run | [Two combats](docs/open_source/experiments/single.md#first-short-run) | [One topology × four seeds](docs/open_source/experiments/continuous.md#first-short-run) |
| Full reference evaluation | [Base, SFT and DPO](docs/open_source/experiments/single.md#evaluate-reference-adapters) | [Six policy arms](docs/open_source/experiments/continuous.md#evaluate-reference-adapters) |
| Train new models | [Gold SFT → Silver DPO](docs/open_source/experiments/single.md#running-the-7b-comparison) | [SFT → DPO A/B/C](docs/open_source/experiments/continuous.md#train-and-evaluate-new-models) |

The linked pages include commands, time estimates and output checks. A short run
checks execution, not policy performance.

## Assets and reproduction

Source, configs, training datasets, reference LoRA adapters, picker inputs and
original reports are included. Base weights and native builds are obtained
separately; the [asset index](assets/README.md) lists paths and acquisition.
Maintained entries also support [new Teacher/GOLD data generation](docs/open_source/experiments/continuous.md#data-and-labels).

Training can produce different adapter bytes. Reproduce the procedure and compare
metrics under the same evaluation conditions. Training completion, exact checkpoint
reload and improved policy performance are distinct results.

Historical source collections, continuation imports and replay trajectories are
not distributed; complete independent reconstruction of the original experiments
is therefore unavailable. See the [single](docs/open_source/experiments/single.md#rebuilding-v4-and-v5-datasets)
and [continuous](docs/open_source/experiments/continuous.md#downloads-and-starting-points)
pages for retained evidence and local reconstruction checks. Historical commit
hashes identify provenance outside this repository's Git history. Reports keep
their original identities, with [marked path redactions](report/README.md).

## Repository

```text
configs/                 model, training and evaluation settings
assets/                  fixed datasets, reference adapters, picker and evaluation inputs
outputs/                 local generated datasets, training and evaluation runs
docs/open_source/        methods, run guides and result analysis
report/                  original execution reports
scripts/                 operational entry points
src/sts1_llm_policy/     runtime, data, training, and evaluation code
tests/                   policy, simulator, live and training behavior
```

New runs write to `outputs/`; retain the reports, checkpoints and raw trajectories
needed to verify them. Private deployment tools and machine-local artifacts are
excluded. [Scripts](scripts/README.md) lists entry points; the
[documentation index](docs/open_source/README.md) links methods and results.

## License

Project-authored material is licensed under the [MIT license](LICENSE).
Third-party material retains its own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
The STS Metrics picker statistics are redistributed with their author's permission.
Base weights, game assets and raw reconstruction/replay evidence are not included.
Included reference adapters, datasets and picker inputs retain the third-party
terms applicable to their source material, as described in the notices.
