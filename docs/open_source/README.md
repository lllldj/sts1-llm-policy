# Project guide

Start with the [project overview](../../README.md#experiment-design) for the ideas
and [current results](../../README.md#current-offline-results) for measured outcomes.
These pages provide a little more detail where useful.

## Understand the experiments

- [Policy inputs and actions](policy_observation.md): what the model sees and controls.
- [Teacher and GOLD](teacher_gold.md): search, action comparisons and sampling.
- [Data and training](data_training.md): SFT targets, mixed demonstrations and DPO A/B/C.
- [Result analysis](stageresult.md): comparisons, uncertainty and limitations, with original reports.

## Try the project

[Setup and requirements](runtime_and_simulator.md) and the [asset index](../../assets/README.md)
cover installation. The [single-combat](experiments/single.md) and
[continuous-route](experiments/continuous.md) pages give short runs and training
commands. [Scripts](../../scripts/README.md) lists the available entry points.

Real-game integration is optional: see the [live guide](real_game_testing.md),
[card-selection extension](combat_card_selection.md) and
[encounter coverage](act1_real_game_encounter_matrix.md).

The [experiment index](experiments/README.md) links earlier experiments and data
collection records. Original reports are in [report/](../../report/README.md).
Historical source revisions identify provenance; their development history and
raw reconstruction collections are not included in this snapshot.
