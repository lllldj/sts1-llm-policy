# Third-party notices

The [MIT license](LICENSE) covers project-authored material. Third-party
material retains its original copyright and license terms.

## sts_lightspeed

The simulator integration uses [gamerpuppy/sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed),
pinned to revision `7476a81954020087da31d41d16fddf475746ec2d` in
[the build configuration](configs/env/sts_lightspeed_build.json).
The patches in `tools/sts_lightspeed/` contain upstream source context.
The upstream copyright and MIT terms are reproduced below for those portions;
the simulator source and compiled binaries are obtained separately.

```text
MIT License

Copyright (c) 2021 gamerpuppy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Included experiment assets

The stage tables in [single](docs/open_source/experiments/single.md#downloads-and-starting-points)
and [continuous](docs/open_source/experiments/continuous.md#downloads-and-starting-points)
identify the included project datasets and reference LoRA adapters. Their manifests,
adapter metadata and original reports preserve source and model identities.
The adapters contain project-trained LoRA parameters, not the Base weights;
the terms accompanying the identified Qwen model snapshots remain applicable.
The project source license does not relicense third-party source material.

## STS Metrics picker statistics

The picker database at `assets/picker/card_pick_metrics_v1.sqlite3` is derived
from the included [CSV](assets/picker/source/sts_metrics_v0_1_3/20260827T121630Z/card_pick_stats.csv)
and [source metadata](assets/picker/source/sts_metrics_v0_1_3/20260827T121630Z/metadata.json).
The metadata identifies **STS Metrics for Slay the Spire 1**, mod version `0.1.3`,
retrieved at `2026-08-27T12:16:30.358Z`: 721 rows, ascension filter 20–99,
vanilla cards only, upgraded cards included, no color filter. Rate fields retain
the two-decimal percentage strings exposed by `CardPickStatData`.
[The extraction report](report/data/card_pick_metrics_v1.json) records the input
hashes and database construction; [rebuild instructions](assets/README.md#rebuilding-the-picker-database)
use these fixed inputs. Original metadata and data are preserved.

The public [STS Metrics mod page](https://steamcommunity.com/sharedfiles/filedetails/?id=3338653921)
credits **PaoPaoYue** and links its
[card-pick dashboard](https://www.defectno4.space/public/dashboard/dee832e7-4edb-476a-85da-56f2c7d69de4)
and [backend repository](https://github.com/PaoPaoYue/sts-service).
These identify the upstream project, not an archived download of this CSV.
The snapshot metadata does not record the acquisition endpoint, mod artifact
identity or a license; its exact association with that published mod version
has not been independently verified.

As of 2026-09-28, the reviewed upstream mod page and backend README provide no
explicit terms for redistribution of this statistics snapshot. The mod page
describes sharing metrics with players and modders; that statement does not
establish a license for the included CSV or derived SQLite database.
Their redistribution permission remains unverified. The project's MIT license
does not grant rights to these third-party statistics.

## Separately obtained dependencies and assets

- Python and native dependencies, including PyTorch, Transformers, nlohmann/json
  and pybind11, retain their own licenses. Preserve their notices when
  redistributing those components or binaries that include them.
- The live adapter targets the protocol provided by
  [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod).
  Game mods are installed separately and retain their own terms.
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  and [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  model assets are obtained separately. This repository's MIT license does not
  replace the licenses accompanying those model snapshots.
- Base weights, full Teacher/GOLD source collections and original replay
  trajectories are not included in this repository.
- Slay the Spire is a game by Mega Crit. This project does not grant rights to
  the game, its assets or trademarks, and is not affiliated with or endorsed by
  Mega Crit. A separate game installation is required for real-game use.
