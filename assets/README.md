# Experiment assets

These are ordinary Git files; no LFS or separate asset download is needed.

| Directory | Contents |
| --- | --- |
| [datasets/single/observation-v5/](datasets/single/observation-v5/) | Gold SFT, Silver preferences and separate development data. |
| [datasets/continuous/](datasets/continuous/) | GOLD SFT, mixed SFT, DPO A/B/C and source-exclusion/selection records. |
| [adapters/single/](adapters/single/) | 1.5B live Gold SFT and 7B SFT/DPO reference adapters. |
| [adapters/continuous/](adapters/continuous/) | 7B GOLD-only, mixed SFT and DPO A/B/C adapters. |
| [picker/](picker/) | STS Metrics SQLite statistics plus source CSV and metadata. |
| [eval/](eval/) | Frozen Boss inputs and simulator/live parity fixture. |

Dataset manifests name their compressed JSONL files. Adapters use the project's
[`project_lora_v1` format](../docs/open_source/data_training.md#adapters-and-resume),
require their declared Base, and contain no optimizer state; they are not directly
loadable PEFT adapters. New outputs belong in `outputs/`.

Historical Teacher pools, imported GOLD trials and replay trajectories are not
included. Public selection/exclusion files do not contain those raw inputs;
exclusions apply to their declared source pool. Historical lineage paths identify
provenance, not files required to load the supplied datasets. See
[reconstruction limits](../docs/open_source/README.md#data-generation-and-reconstruction)
and [third-party terms](../THIRD_PARTY_NOTICES.md).

## External assets

Download the complete Base snapshot, including weights, shard index, configuration
and tokenizer files, into `model.snapshot_path` in the corresponding runtime
config. The loader reads that directory directly and checks identity/content;
moving the whole snapshot and updating the path needs no separate model-ID cache.
Download time depends on model size and network speed.

| Asset | Source and configuration |
| --- | --- |
| 7B offline Base | [Qwen2.5-7B-Instruct at the pinned revision](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28); [runtime config](../configs/runtime/base_model_runtime_7b_v2.json). |
| 1.5B live Base | [Qwen2.5-1.5B-Instruct at the pinned revision](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306); [runtime config](../configs/runtime/base_model_runtime_1_5b_v2.json). |
| Simulator / Teacher | [Build from pinned source](../docs/open_source/README.md#build-the-simulator); no prebuilt binary is supplied. |
| Game and mods | Obtain separately for [real-game use](../docs/open_source/real_game_testing.md#prerequisites); offline evaluation needs no running game. |

## Rebuilding the picker database

The included database is ready to use. STS Metrics statistics are redistributed
with their author's permission; [provenance and terms](../THIRD_PARTY_NOTICES.md#sts-metrics-picker-statistics)
cover the CSV and derived database. This command rebuilds from that included
snapshot, not current online statistics. Run from the repository root:

```sh
uv run --locked python -c "import json; from sts1_llm_policy.data.card_pick_metrics import build_card_pick_database; source = 'assets/picker/source/sts_metrics_v0_1_3/20260827T121630Z/'; print(json.dumps(build_card_pick_database(source + 'card_pick_stats.csv', source + 'metadata.json', 'configs/env/d_expansion_v1_scope.json', 'outputs/picker/card_pick_metrics_v1.sqlite3'), indent=2))"
```

This takes a few seconds without a model or simulator. It validates fields, card
keys and base/upgraded coverage, then prints counts and quality statistics; existing
output is not overwritten. Compare logical `card_pick_rates` and `database_metadata`
contents, since SQLite bytes can vary. Set `picker_database` explicitly in a new
panel/generation config; single Boss generation also requires a matching
`expected_picker_database_sha256`. Changed inputs need a fresh output directory.
