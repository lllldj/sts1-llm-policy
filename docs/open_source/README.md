# Public documentation

This directory intentionally contains only maintained contracts and current
claim-bearing summaries. Machine evidence remains in `report/`; historical source
revisions are provenance identifiers and are not included in the release snapshot.

Start with [setup and run selection](../../README.md#setup-and-verification),
then the [single short run](experiments/single.md#first-short-run) or
[continuous short run](experiments/continuous.md#first-short-run).
[Assets](../../assets/README.md) owns acquisition and installation;
[current results](../../README.md#current-offline-results) summarizes the measured outcomes.

- [`experiments/README.md`](experiments/README.md): `single` and `continuous`
  experiment stages, configuration naming, complete chain and supporting-work links.
- [`stageresult.md`](stageresult.md): retained Boss-panel and continuous-route
  comparisons, earlier single-stage evidence and GOLD supporting diagnostics,
  with original evidence sources, methods and interpretation limits.
- [`runtime_and_simulator.md`](runtime_and_simulator.md): runtime, simulator,
  action, and real-game boundaries.
- [Script index](../../scripts/README.md): data export, training, evaluation,
  simulator and training-readiness entry points, with their config contracts.
- [`real_game_testing.md`](real_game_testing.md): operator procedure, current
  live-test status, known problems, and next checks.
- [`act1_real_game_encounter_matrix.md`](act1_real_game_encounter_matrix.md):
  BaseMod console commands and the maintained Act 1 terminal-test queue.
- [`policy_observation.md`](policy_observation.md): public student inputs, observation
  versions, legal actions and supervision boundaries.
- [`teacher_gold.md`](teacher_gold.md): Teacher continuation scoring, GOLD collection,
  sampling, storage and replay verification.
- [`data_training.md`](data_training.md): dataset exports, SFT and DPO behavior,
  configuration compatibility and evidence boundaries.
- [`combat_card_selection.md`](combat_card_selection.md): opt-in V6 combat
  card choices, isolated simulator build, and real-game retest procedure.

Stage pages own execution order and output checks; the script index owns entry
lookup; contracts own algorithms, interfaces and compatibility. Result and historical
record pages own measured outcomes, their original conditions and interpretation.
The [experiment index](experiments/README.md) connects supporting and historical work;
[retention rules](data_training.md#retention-and-retirement) explain its availability.
Historical commands and results keep their original implementation boundaries.

`sealed` identifies test data excluded from development and training.
