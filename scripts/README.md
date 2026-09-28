# Script index

Use `uv run --locked python scripts/<entry>.py --help` to inspect an entry.
[Setup and shell conventions](../README.md#setup-and-verification) apply throughout.
Choose [single](../docs/open_source/experiments/single.md#first-short-run) or
[continuous](../docs/open_source/experiments/continuous.md#first-short-run) for a
complete runnable sequence.

## Local experiment workflow

Stage pages own execution order, time estimates, outputs and resume instructions:
[first run and reference evaluation](../README.md#choose-a-run), then optional new
training or data generation. [Assets](../assets/README.md) owns acquisition;
[training readiness](../docs/open_source/runtime_and_simulator.md#training-readiness-on-the-current-machine)
owns current-machine probes; [evidence and tests](../docs/open_source/data_training.md#evidence-and-tests)
defines what their results establish.

Mode-aware entries support subsets of `--mode preflight|smoke|run`; use `--help`
for each entry. Training additionally supports `backward` and requires an explicit
mode. Other mode-aware entries default to `run`. Route/combat smoke requires a
positive `--smoke-routes N` or `--smoke-combats N`; size options are invalid
outside smoke mode. GOLD replay uses `--verify` with `--mode run|smoke` to select
the corresponding artifacts. Invalid argument combinations exit with code 2;
execution failures return nonzero.

### Prepare independent training and evaluation configs

`prepare_experiment.py` accepts repeatable `--config` experiment-member lists and
a required `--output` root. It prepares config copies, connects upstream training
checkpoints and expands continuous panels without executing a run. See the
[preparation contract and command](../docs/open_source/data_training.md#prepare-independent-training-and-evaluation-configs)
for ordering, reference mappings, conflicts and reuse.

## Runtime and simulator

| Entry | Purpose and key arguments | Contract / procedure |
| --- | --- | --- |
| `simulator.py` | `resolve`, `verify`, `build`, or `smoke`; host selection, explicit `--mechanics` and optional `--require` capabilities | [Native selection](../docs/open_source/runtime_and_simulator.md#callable-native-environment-selection) |
| `build_sts_lightspeed.ps1` / `build_sts_lightspeed_linux.py` | Platform builders used by `simulator.py`; direct invocation for an explicit rebuild | [Build prerequisites and recovery](../docs/open_source/runtime_and_simulator.md#native-build-prerequisites) |
| `build_card_selection_bridge.ps1` | Windows corrected-mechanics/selection extension | [Native platform builds](../docs/open_source/runtime_and_simulator.md#native-platform-builds) |
| `generate_combat_panel.py` | Generate independent combat snapshots; default `configs/generation/single_boss_inputs.json` selects corrected mechanics | [Frozen inputs](../docs/open_source/runtime_and_simulator.md#frozen-boss-inputs) |
| `run_continuous.py` | Required `--config` selects Teacher generation or policy evaluation; `--arm ID` filters an arm, `--workers N` applies to Teacher-only collection | [Continuous execution](../docs/open_source/runtime_and_simulator.md#configured-continuous-act-1-development-routes) |
| `run_frozen_policy_panel_evaluation.py` | Required `--config` selects Base/adapter, a frozen V5 panel and mechanics | [Single evaluation](../docs/open_source/runtime_and_simulator.md#frozen-policy-evaluation-inputs-and-reuse) |
| `run_sts_lightspeed_parity.py` | Replay tracked real-game parity fixtures; new `--output` or default `outputs/parity/` report | [Diagnostic commands](../docs/open_source/runtime_and_simulator.md#maintained-commands) |
| `run_random_combat.py` | Capture one guarded CommunicationMod Random Legal combat | [Real-game boundary](../docs/open_source/runtime_and_simulator.md#real-game-boundary) |
| `run_real_game_session.py` | Maintained Gold SFT V5 session or explicit opt-in V6 profile | [Live procedure](../docs/open_source/real_game_testing.md) |

## Data generation and export

| Entry | Purpose and key arguments | Contract / procedure |
| --- | --- | --- |
| `prepare_dataset.py` | V4 Gold/Silver reconstruction; `--config configs/data/single_teacher_v2.json` | [Single reconstruction](../docs/open_source/experiments/single.md#rebuilding-v4-and-v5-datasets) |
| `migrate_observation_v5.py` | V4-to-V5 conversion; `--config configs/data/single_observation_v5.json` | [Single reconstruction](../docs/open_source/experiments/single.md#rebuilding-v4-and-v5-datasets) |
| `rebuild_expanded_sft_summaries.py` | Restore historical source views; explicit `--source-dir`, `--output-dir` or `--verify-only` | [Source reconstruction](../docs/open_source/experiments/single.md#rebuilding-expanded-source-summaries) |
| `verify_teacher_candidate_pool.py` | Verify delivered report, inputs and trajectories without simulator/model execution; `--exclusions` selects source-isolation evidence | [Candidate verification](../docs/open_source/runtime_and_simulator.md#continuous-teacher-candidate-collection) |
| `prepare_gold_collection.py` | Required `--template-config`, `--config-output`, `--output`, `--groups-per-route`; training candidates require `--exclusions` | [Selection and isolation](../docs/open_source/experiments/continuous.md#selection-and-source-isolation) |
| `collect_gold.py` | Required `--config`; replay with `--verify`, new receipt via `--verification-output`, optional state subset via `--verification-selection` | [Execution and replay receipts](../docs/open_source/teacher_gold.md#execution-verification-and-receipts) |
| `export_gold_sft.py` | GOLD-only or mixed SFT selected by `--config`; mixed export also supports its configured smoke subset | [SFT export contracts](../docs/open_source/data_training.md#dataset-exports) |
| `export_gold_dpo.py` | Matched A/B/C preference export selected by `--config` | [DPO export contract](../docs/open_source/data_training.md#matched-v7-dpo-preference-datasets) |

These reconstruction/export commands require their declared source evidence;
the included datasets can be used without rebuilding them. Historical diagnostics
and retired entry points are indexed in [supporting work](../docs/open_source/experiments/README.md#supporting-work).

## Training and recovery

| Entry | Purpose and key arguments | Contract / procedure |
| --- | --- | --- |
| `run_training.py` | Required `--config` and `--mode`; portable execution consumes `--recovery-report` | [Training and checkpoint contracts](../docs/open_source/data_training.md#dataset-and-training-interface) |
| `restore_training.py` | Current checkout, environment, assets, resume state, generation and longest backward; `--previous` permits matching backward reuse, `--report` selects a new receipt | [Readiness](../docs/open_source/runtime_and_simulator.md#training-readiness-on-the-current-machine) |
