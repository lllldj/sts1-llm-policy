# Fixed experiment assets

These files are ordinary Git files, with no LFS or separate download needed.
Paths in the maintained configs select them explicitly. Base model weights and
native builds are obtained separately.

| Directory | Contents | Usage |
| --- | --- | --- |
| [datasets/single/observation-v5/](datasets/single/observation-v5/) | Gold SFT, Silver preferences and separate development data | Select the relevant manifest; each names its compressed JSONL split. |
| [datasets/continuous/](datasets/continuous/) | GOLD SFT, mixed SFT and DPO A/B/C datasets | Each dataset directory contains `manifest.json` and `train.jsonl.gz`. |
| [adapters/single/](adapters/single/) | 1.5B live Gold SFT and 7B Gold SFT/Silver DPO reference adapters | Each directory contains adapter metadata and weights for its declared Base. |
| [adapters/continuous/](adapters/continuous/) | 7B GOLD SFT, mixed SFT and DPO A/B/C reference adapters | Evaluate directly with the matching continuous config. |
| [picker/](picker/) | [SQLite database](picker/card_pick_metrics_v1.sqlite3), [source CSV](picker/source/sts_metrics_v0_1_3/20260827T121630Z/card_pick_stats.csv) and [metadata](picker/source/sts_metrics_v0_1_3/20260827T121630Z/metadata.json) | Consumed by Boss input generation and continuous runs. Source: **STS Metrics mod for Slay the Spire 1**, version 0.1.3; [provenance and terms](../THIRD_PARTY_NOTICES.md#sts-metrics-picker-statistics). |
| [eval/](eval/) | Frozen Boss input panel and simulator/live parity fixture | Used by the corresponding evaluation and parity configs. |

New exports, training checkpoints and evaluation trajectories belong in `outputs/`.
Reference adapters require their exact declared Base and the project's
[`project_lora_v1` loader](../docs/open_source/data_training.md#adapter-checkpoint-format);
they are not directly loadable PEFT adapters and contain no optimizer/resume state.
To train new models and connect their evaluations, use the existing
[experiment preparation command](../docs/open_source/data_training.md#prepare-independent-training-and-evaluation-configs).
Its member lists explicitly connect supplied adapters to the upstream training runs.

The [single](../docs/open_source/experiments/single.md#downloads-and-starting-points)
and [continuous](../docs/open_source/experiments/continuous.md#downloads-and-starting-points)
tables identify the matching configs and generation entries. Rebuilding historical
data requires raw evidence that is not distributed here. Dataset manifests use
the current split-file locations. Machine-specific absolute paths in provenance
are redacted in explicitly marked public copies; see the
[report convention](../report/README.md). Dataset IDs, semantics, counts, split
content hashes and all reference adapter weights remain unchanged.
Historical `lineage` paths identify original inputs, including retired observation
JSON descriptions; dataset loading does not require those historical description files.
See [third-party notices](../THIRD_PARTY_NOTICES.md) for attribution and terms.

## External assets

Download Base at the exact revision below into `model.snapshot_path` in its runtime
config. Include weights, shard index, model/generation config and tokenizer files.
The loader checks local identity and content, then loads both model and tokenizer
directly from that snapshot directory. Moving the complete snapshot and updating
`model.snapshot_path` does not require a separate model-ID cache entry.
The offline stages use 7B; the auxiliary live default uses 1.5B. Obtain the model
selected by the intended config. Allow download time according to snapshot size
and network speed; inference also needs the declared
[execution environment](../docs/open_source/runtime_and_simulator.md#execution-requirements-and-recorded-hardware).

| Asset | Acquisition and installation |
|---|---|
| Qwen2.5-7B-Instruct Base | [Official files at the configured revision](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28). Obtain weights, shard index, model/generation configs and tokenizer files at this revision; install at `model.snapshot_path` in [the runtime config](../configs/runtime/base_model_runtime_7b_v2.json). |
| Qwen2.5-1.5B-Instruct Base | [Official files at the configured revision](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306). Install the complete declared model/tokenizer assets at `model.snapshot_path` in [the runtime config](../configs/runtime/base_model_runtime_1_5b_v2.json). |
| Native simulator / Teacher | Build the pinned upstream source and project patches using the [build guide](../docs/open_source/runtime_and_simulator.md#callable-native-environment-selection). Build manifests and binaries remain under `.simulator/`; the Windows selection extension uses `outputs/card-selection/`. No prebuilt download is supplied. |
| Game and mods | Obtain separately for live sessions; see [live prerequisites](../docs/open_source/real_game_testing.md#prerequisites). Offline evaluation does not require a live game session. |

Teacher/GOLD raw evidence and original replay trajectories are retained locally
and are not distributed. Rebuilding historical exports requires the declared
source reports and trajectories; GOLD additionally requires a report-bound replay
receipt. See [single reconstruction](../docs/open_source/experiments/single.md#rebuilding-v4-and-v5-datasets)
and [continuous starting points](../docs/open_source/experiments/continuous.md#downloads-and-starting-points).
Evaluation with supplied adapters and training with supplied datasets do not
require that historical reconstruction.

The fixed GOLD [selection](datasets/continuous/gold-selection.json) contains ordered
sample positions and their source partition. Collection and mixed SFT export share
this input. Its historical `exclude_configs` values record provenance only; no
consumer opens those retired configs. New training-candidate preparation uses the
[source-bound exclusions](datasets/continuous/gold-exclusions.json) instead.
The source report itself remains a separately required, undistributed input.

Continuous source exclusions are stored in
[`development-exclusions.json`](datasets/continuous/development-exclusions.json).
They declare route identities and closed seed intervals, with historical run/output
provenance; candidate generation consumes them without old execution configs.

## Rebuilding the picker database

Upstream attribution, snapshot provenance and the unverified redistribution
terms are documented in [third-party notices](../THIRD_PARTY_NOTICES.md#sts-metrics-picker-statistics).
The reconstruction below uses the included snapshot; fetching current online
statistics would not establish the same historical input.

The included SQLite database is ready to use. To rebuild it from the included
STS Metrics CSV and metadata, call
[`build_card_pick_database`](../src/sts1_llm_policy/data/card_pick_metrics.py)
from the repository root:

```sh
uv run python -c "import json; from sts1_llm_policy.data.card_pick_metrics import build_card_pick_database; source = 'assets/picker/source/sts_metrics_v0_1_3/20260827T121630Z/'; print(json.dumps(build_card_pick_database(source + 'card_pick_stats.csv', source + 'metadata.json', 'configs/env/d_expansion_v1_scope.json', 'outputs/picker/card_pick_metrics_v1.sqlite3'), indent=2))"
```

Allow a few seconds after the Python environment is installed; no model or native
simulator is needed. The function checks CSV fields, normalized card keys and
base/upgraded coverage for the declared reward-card scope, then writes the database
and prints source, row-count and quality statistics. It refuses to overwrite an
existing output. The same four paths are explicit arguments of the Python API.

The two tables are `card_pick_rates` (selection statistics) and `database_metadata`
(source metadata and input hashes). Compare their logical contents when validating
a rebuild; SQLite file bytes can vary by runtime. A changed scope also changes the
recorded scope hash. For a new experiment, set `picker_database` in a copy of the
relevant generation config or continuous panel to the new repository-relative
path. Single Boss generation also requires its `expected_picker_database_sha256`
to match the rebuilt file. Use a fresh output directory when changing inputs;
the included database and historical reports remain the reference artifacts.
