# STS1 LLM Policy

A small experiment in teaching language models to play Ironclad combat in
**Slay the Spire 1**: can simulator-generated supervision improve decisions,
and does preference training help beyond supervised fine-tuning?

The model reads public game state and legal actions, then returns one `ACTION_n`.
**Base** means the unadapted Qwen2.5 **Instruct** model. The project compares it
with SFT and DPO policies, from standalone fights to configured Act 1/A0 routes.

## Experiment design

**Tasks.** The initial single-combat experiments use preset decks and public
combat information (V5). Continuous experiments use a 7B model with route context
and secondary card choices (V7): HP and relic counters carry between fights,
and rewards, upgrades and removals change the deck. These are configured routes,
not full native map, shop or event play. Hidden draw order, future RNG and search
scores are never student inputs.

**Teacher.** Search builds on [sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed)
and its `BattleScumSearcher2`. The Teacher explores action sequences in the
simulator; GOLD compares candidate actions through repeated completed combats,
using wins and carried HP to identify acceptable action sets. This estimates
local combat quality, not the best decision for a whole route.

**SFT → DPO.** Supervised fine-tuning (SFT) learns acceptable actions. Mixed SFT
adds early-combat Teacher demonstrations, with loss weights of 70% GOLD, 25% Teacher and 5% forced
end-turn examples. It also uses more training steps than GOLD-only SFT.
Three direct preference optimization (DPO) variants start independently from
mixed SFT, sharing states and state-update budgets: **A** uses all eligible preference pairs, **B** keeps larger
HP gaps, and **C** downweights smaller gaps, including across states.

**Deck building.** The model controls combat. Reward selection combines Teacher
simulations, card-pick statistics and deck synergy; upgrades use separate combat
comparisons, and removal follows a starter-card rule. Policies share these
strategies and route seeds to focus the comparison on combat decisions.

**LoRA.** The 7B runs train `q_proj`/`v_proj` adapters with rank **8**, alpha **16**
and dropout **0**: about **2.52M trainable parameters (0.033%)**. Both SFT runs
start from fresh Base. Training uses BF16, one epoch, effective batches of eight
states and AdamW; learning rates are **1e-4 for SFT** and **2e-5 for DPO**, with
DPO beta **0.1**. Inputs are limited to 3,072 tokens without truncation.

> [!NOTE]
> [Method details](docs/open_source/data_training.md) ·
> [SFT recipe](configs/profiles/training/sft_lora_v1.json) ·
> [DPO recipe](configs/profiles/training/dpo_lora_v1.json)

## Current offline results

Qwen2.5-7B-Instruct on **20 Act 1/A0 route topologies × four seed groups**.
The primary metric is final Boss victories:

| Policy | Boss victories / 80 | Evaluation evidence |
|---|---|---|
| Base | 4 | [Base / GOLD SFT](report/evaluation/qwen2_5_7b_base_gold_sft_act1_continuous_v7_v1.json) |
| GOLD-only SFT | 13 | [Base / GOLD SFT](report/evaluation/qwen2_5_7b_base_gold_sft_act1_continuous_v7_v1.json) |
| Mixed GOLD/Teacher SFT | 26 | [Mixed SFT](report/evaluation/qwen2_5_7b_gold_teacher_mixed_sft_act1_continuous_v7_v1.json) |
| DPO A | 31 | [DPO A](report/evaluation/qwen2_5_7b_gold_dpo_a_act1_continuous_v7_v1.json) |
| DPO B | 30 | [DPO B](report/evaluation/qwen2_5_7b_gold_dpo_b_act1_continuous_v7_v1.json) |
| DPO C | 28 | [DPO C](report/evaluation/qwen2_5_7b_gold_dpo_c_act1_continuous_v7_v1.json) |
| Teacher (historical V6) | 68 | [Recorded Teacher results](docs/open_source/stageresult.md#7b-base-and-teacher-on-continuous-routes) |

The Teacher result predates later simulator corrections and is a historical
reference, not a matched V7 control. Four seeds share each topology, and this
development panel informed experiment design. DPO A has the highest learned-policy
win count, but these counts alone do not establish a stable advantage or independent
generalization. [Result analysis and historical comparisons](docs/open_source/stageresult.md)
provide the evidence and limitations.

## Try it

[Running guide](docs/open_source/README.md) · [Included assets and Base downloads](assets/README.md)
· [Optional real-game integration](docs/open_source/real_game_testing.md)

Datasets, reference LoRA adapters, picker inputs and original reports are included.
Base weights and native builds are obtained separately. Original reconstruction
collections and replay trajectories are not distributed, so complete independent
reconstruction of the historical experiments is unavailable. New training can
also produce different weights. The live default remains the **1.5B Gold SFT V5**
policy; the 7B offline results do not replace it.

## Possible next steps / 后续可能方向

There are no definite plans to continue development. Possible directions:

1. **Improve card rewards, upgrades and removals** by training a model to make
   these decisions, which I consider the most suitable approach. With creators'
   permission, high-level players' recorded runs on Bilibili or other video platforms could supply examples; useful data
   needs the decision context, not just the chosen card, and better collection tools.
2. **Try larger models** under comparable data and evaluation conditions. How much
   capacity is sufficient remains unclear.
3. **Improve the tooling** for data generation, state reconstruction, evaluation
   and real-game integration, making further experiments easier.

目前没有继续开发的明确计划，以下方向供后续探索：

1. **改进选卡、强化和删卡**，尤其是交给模型并进行训练——这是我认为最合适的方案。
   可以在获得创作者同意后，从高水平玩家在 B站或其他视频站点的历史对局中整理数据；
   除了最终选项，还需要还原决策环境，并完善相应的采集工具。
2. **尝试更大的模型**，在可比的数据与评估条件下观察提升；目前还不清楚多大的模型才足够。
3. **完善工具链**，包括数据生成、状态还原、评估和实机接入，让后续实验更容易开展。

## License and data sources

Project-authored material uses the [MIT license](LICENSE). Third-party material
retains its own terms; see [third-party notices](THIRD_PARTY_NOTICES.md), including
sts_lightspeed attribution and STS Metrics data redistributed with its author's
permission. Base weights and game assets are not included.
