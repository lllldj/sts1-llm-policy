# Script index

Start with the [running guide](../docs/open_source/README.md). Each Python entry
supports `--help`. Run from the repository root with `uv run --locked python`.

| Task | Entries |
| --- | --- |
| Native simulator | `simulator.py` (`resolve`, `verify`, `build`, `smoke`); direct rebuilds use `build_sts_lightspeed.ps1`, `build_card_selection_bridge.ps1` or `build_sts_lightspeed_linux.py`. |
| Config preparation | `prepare_experiment.py`: explicit experiment member lists, separate outputs and connected new checkpoints. |
| Readiness / training | `restore_training.py`, then `run_training.py` with `--config`, explicit `--mode` and `--recovery-report`. |
| Single evaluation | `run_frozen_policy_panel_evaluation.py`; new input panels use `generate_combat_panel.py`. |
| Continuous evaluation / Teacher pool | `run_continuous.py --config`; `--arm` selects a policy, `--workers` changes Teacher-only concurrency. |
| GOLD | `prepare_gold_collection.py`, `collect_gold.py`, `export_gold_sft.py`, `export_gold_dpo.py`. |
| Source verification | `verify_teacher_candidate_pool.py`; GOLD replay uses `collect_gold.py --verify --verification-output`. |
| Historical single reconstruction | `rebuild_expanded_sft_summaries.py`, `prepare_dataset.py`, `migrate_observation_v5.py`; undistributed source evidence is required. |
| Live / parity | `run_real_game_session.py`, `run_random_combat.py`, `run_sts_lightspeed_parity.py`. |

Mode-aware entries support subsets of `preflight`, `smoke`, `run`; training also
supports `backward`. Training requires an explicit mode; others default to `run`.
Evaluation smoke needs a positive `--smoke-combats` or `--smoke-routes`, invalid
outside smoke mode. GOLD replay's mode selects which artifacts to verify.
Invalid arguments exit with code 2; execution failures return nonzero.

New GOLD preparation requires explicit template/source/destinations and source-bound
exclusions. It clears inherited imports and old selections. Collection needs the
new source reports and trajectories; exports need report-bound replay evidence.
See [methods](../docs/open_source/data_training.md#teacher-and-gold) and
[asset availability](../assets/README.md). Training from supplied datasets does
not require regenerating them.
