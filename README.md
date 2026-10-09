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

## Retrospective and possible next steps / 实验回顾与可能的后续方向

[English](#english) · [中文](#中文)

### English

This project started with a relatively simple question: can supervision generated
by a simulator help a language model make better combat decisions in Slay the Spire?

I began with a 1.5B model in standalone combats, starting at full HP with preset
decks. The objective was to defeat the enemy while losing as little HP as possible.
Fine-tuning brought a clear improvement over Base, but performance in the real
game remained disappointing, particularly in defense, damage control and survival
at low HP. The model did not seem to fully learn many of the Teacher's ways of
preserving HP. I suspect that full-HP starts did not adequately cover the resource
pressure of consecutive fights, although I did not test that explanation in a
separate controlled experiment. DPO also showed no clear additional benefit over
SFT at this stage. As a pilot experiment, these results helped identify what to
investigate next.

I then moved to Qwen2.5-7B-Instruct and repeated fine-tuning and comparison using
the same training data. The larger Base improved, and its fine-tuned version also
achieved better results. This suggests that model capacity was one limitation,
but increasing model size did not resolve the gaps in data coverage and task design.

Next, I extended the experiment to continuous routes. HP and some relic counters
persisted between combats, while the deck evolved through card rewards, removals
and upgrades. I also revised the deck-building strategy and training-data generation
before training again. Across the current 80 route executions, Boss victories rose
from **4/80** for Base to **13/80** for GOLD-only SFT, **26/80** for mixed SFT, and
**31/80** for DPO A.

The absolute completion rate is still low for Act 1 at Ascension 0. Even so, I am
satisfied with the outcome relative to the project's original question:
simulator-generated supervision helped the model improve its combat performance
on this development panel. DPO A recorded the most victories, although its
additional gain over mixed SFT is not enough to establish a consistent advantage.

For reference, the Teacher, with access to simulator search and internal state,
achieved **68/80** on the historical panel. It can explore alternative actions in
the simulator, somewhat like a player who can repeatedly save and reload ("SL"),
while the model acts step by step using only public information. This result is
clearly insufficient for Ascension 0, Act 1. It makes me suspect that deck building
and route strategy, as well as combat decisions, have considerable room for improvement.

I consider this stage to have answered the main questions I wanted to explore.
There are currently no definite plans to continue the project. The following are
possible directions, also offered to anyone interested in taking it further.

#### 1. Improve card reward, upgrade and removal strategies

This is the direction I would most like to explore first. The current strategies
support the experiments but remain fairly crude. More refined rules, search
methods, or giving these decisions to a model and training it (which I consider
the most suitable approach) are all worth trying.

The main difficulties are data and tooling. Manually organizing card-picking
knowledge is one starting point, but it is time-consuming and can miss important
situations. My preferred approach would be to obtain the creators' permission and
build decision datasets from high-level players' recorded runs on Bilibili or
other video platforms. This would require reconstructing the deck, relics, HP,
offered cards and route context at each decision, while paying attention to
coverage and bias toward successful runs. Recording only the card eventually
chosen would provide incomplete training examples.

The collection, state reconstruction and annotation tools needed for this work
were beyond what I could invest in this lightweight project. That is the main
reason I used simpler strategies here.

#### 2. Try larger models

The move from 1.5B to 7B suggests that further scaling is worth exploring. I do
not yet know what size would be sufficient, and I do not expect parameter count
alone to solve the data and deck-building problems. A more useful experiment
would keep data and evaluation conditions as consistent as possible and measure
how much further improvement model size brings.

#### 3. Improve the tooling

A considerable part of this project went into simulator adaptation, data
generation, state reconstruction and real-game integration. Gaps in the tooling
also limited which experiments were practical.

If I return to the project, I would like to improve these foundations first,
making it easier to add new strategies and data, and to observe and analyze
failures. I hope the existing implementation provides a starting point that saves
others from rebuilding some of these tools.

### 中文

这个项目从一个相对简单的问题开始：通过模拟器生成的监督数据，能否让语言模型更好地完成《杀戮尖塔》的战斗决策？

我首先使用 1.5B 模型，在满血、预设卡组的单场战斗中进行实验，目标是在战胜敌人的同时尽量减少损血。
微调后的模型相较于 Base 有了明显提升，但接入真实游戏后，表现仍然不够理想，尤其是在防御、控损和低血量下的生存决策方面。
Teacher 能够做到的许多控损操作，模型似乎没有充分学会。我怀疑，满血开局的实验设置没有充分覆盖连续作战中的资源压力，
但这并没有通过单独的对照实验验证。此外，这一阶段的 DPO 相较于仅做 SFT，没有显示出明显的额外收益。
作为先导实验，这些结果帮助我看清了下一步的问题。

随后，我换用了 Qwen2.5-7B-Instruct，使用相同训练数据进行微调和比较。更大的 Base 有所改善，微调后也取得了更好的结果。
这说明模型能力可能是此前表现的限制因素之一，但扩大模型并没有消除数据覆盖和任务设置上的不足。

接下来，我将实验扩展到连续路线：血量和部分遗物状态在战斗之间延续，卡组通过沿途的奖励选卡、删卡和强化逐步形成。
同时，我调整了卡组构建策略与训练数据的生成方式，重新进行了微调。在当前的 80 次路线执行中，Boss 胜场数从 Base 的 **4/80**，
提升到 GOLD-only SFT 的 **13/80**、mixed SFT 的 **26/80**，再到 DPO A 的 **31/80**。

对 Act 1、进阶 0 而言，这个绝对通关率仍然偏低，但相对于项目最初的问题，我对这一阶段的结果是满意的：
模拟器生成的监督数据确实帮助模型改善了这个开发面板上的战斗表现。DPO A 取得了最高胜场数，
不过它相较 mixed SFT 的额外收益，仍不足以证明稳定优势。

作为参考，具有模拟搜索和内部状态访问优势的 Teacher，在历史面板上的结果为 **68/80**。
它可以在模拟器中探索多种行动，某种程度上类似于能够反复试错的“SL 视角”；模型执行时则只能依据公开信息逐步行动。
这个数据在 A0、Act 1 中明显是不足的。这让我怀疑，除了战斗决策，卡组构建和路线策略也有较大的改进空间。

我认为，这个阶段已经回答了自己想探索的主要问题。项目目前没有继续推进的明确计划；
下面是一些可能的方向，也供希望接着完善它的人参考。

#### 1. 改善选卡、强化和删卡策略

这是我最希望优先探索的方向。当前策略能够支撑实验，但仍然比较粗糙。
更精细的规则、搜索方法，或者将这些操作也交给模型并进行训练（我认为这是最合适的方案），都值得尝试。

困难主要在于数据和工具。手工整理选卡经验是一个起点，但耗时较多，也容易遗漏场景。
我更倾向于在获得创作者同意的前提下，从高水平玩家在 B站或其他视频站点发布的历史对局中整理决策数据。
不过，这需要还原当时的卡组、遗物、血量、备选牌和路线环境，也需要关注数据覆盖与成功对局偏向。
仅记录“最后选了哪张牌”，很难构成充分的训练样本。

完成这些工作所需的采集、状态还原和标注工具，已经超出了这个轻量项目当时的投入范围。
这也是我目前采用较简单策略的主要原因。

#### 2. 尝试更大的模型

从 1.5B 到 7B 的结果让我认为，继续扩大模型仍然值得尝试。但我目前无法判断多大的模型才足够，
也不认为单纯增加参数量就能解决数据和卡组构建的问题。更有价值的实验，是在尽量一致的数据与评估条件下，
观察模型规模还能带来多少改善。

#### 3. 完善工具链

这个项目有不少时间花在了模拟器适配、数据生成、状态还原和实机接入上。工具链的不足也限制了能够尝试的实验范围。

如果以后继续投入，我希望先改善这些基础工具，让新策略和新数据更容易接入，也让失败行为更容易被观察和分析。
希望这些已有实现能够成为一个起点，减少后来者重复搭建基础工具的工作。

## License

Project-authored material is licensed under the [MIT license](LICENSE).
Third-party material retains its own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
The STS Metrics picker statistics are redistributed with their author's permission.
Base weights, game assets and raw reconstruction/replay evidence are not included.
Included reference adapters, datasets and picker inputs retain the third-party
terms applicable to their source material, as described in the notices.
