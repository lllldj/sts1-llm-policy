# Running the project

For a first run, use the [single](experiments/single.md#first-short-run) or
[continuous](experiments/continuous.md#first-short-run) guide. This page covers
installation and the few environment choices that matter.

## Setup

Install Git and `uv`, then run from the repository root:

```text
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

Installation time depends on downloads and cache. Tests need no Base weights or
GPU. Native tests skip when the simulator is absent; a partial or invalid build
fails instead. [CI and test scope](../../tests/README.md) are separate from running
an experiment. Training needs a committed Git checkout with clean tracked files.

Commands marked `text` work in PowerShell or Bash. Blocks marked `bash` require
Linux Bash or Windows Git Bash with Windows `uv` on PATH. Run standalone commands
in order, continuing only after success. Experiment Bash blocks stop on errors.

### Execution requirements and recorded hardware

Model runs use CUDA, BF16, SDPA and unquantized weights on one GPU. CPU inference,
ROCm and multi-GPU sharding are unsupported. The [execution profile](../../configs/runtime/training_cuda_bf16.json)
checks capabilities rather than a GPU model name; actual memory needs depend on
the model and input length. Historical 1.5B work used an RTX 5060, and continuous
7B training used an RTX 5090 D. Those are observations, not minimum requirements.

Get the exact Base snapshot listed in the [asset index](../../assets/README.md#external-assets).
The runtime config selects its local directory and `cuda:N` device. `uv.lock`
provides the recorded environment. If explicitly installing another compatible
PyTorch build, use `uv run --no-sync` to preserve it and repeat training readiness.
Different hardware does not promise identical retraining or trajectories.

### Native build prerequisites

Both hosts need Git and C++17 `g++` with static GCC/C++ runtime libraries; the first
build downloads the pinned upstream source and submodules. Windows requires a
native MinGW toolchain and PowerShell; Git Bash alone does not provide a compiler.
Linux needs pthread support. Building the simulator and running Teacher search
require no GPU.

### Native platform builds

| Platform | Build location | Mechanics |
| --- | --- | --- |
| Windows x86-64 | `.simulator/sts_lightspeed/` plus `outputs/card-selection/` | Historical base plus corrected extension. |
| Linux x86-64 | `.simulator/sts_lightspeed-linux/` | Corrected engine with card selection included. |

## Callable native environment selection

After installing the prerequisites:

```text
uv run --locked python scripts/simulator.py build --require combat_card_selection_v1
```

The command builds missing artifacts, checks their bindings and runs a short
protocol smoke. Success reports `ready` and `smoke.status: passed`. Existing valid
builds are reused. Allow roughly 2–4 minutes for a fresh Linux build, 10–30 seconds
for a Windows extension over an existing base, or several minutes for a fresh
Windows base. Ctrl+C may leave a partial build requiring explicit recovery.

`resolve` shows the selected platform without installed binaries; `verify` checks
bindings without launching native code; `smoke` checks the installed protocol.
`--report` saves a new report. Unsupported hosts or capabilities fail explicitly.
The default is `corrected_v1`; explicit `legacy_v1` supports only the Windows base
without secondary selection. Selecting corrected mechanics does not itself enable
secondary actions in a V5 experiment.

Invalid retained builds are not overwritten automatically. To rebuild explicitly:

- Windows: run `scripts/build_sts_lightspeed.ps1`, then `scripts/build_card_selection_bridge.ps1`.
- Linux: run `uv run --locked python scripts/build_sts_lightspeed_linux.py`.

Use the fresh-build time estimates above, then repeat the shared `build` command.
Old manifests missing mechanics or patch records must be rebuilt, not edited to
pass validation. Both platforms check the search and exhaust patches; the Windows
extension also checks its base before writing output and records its own exhaust
patch. Windows and Linux binary hashes are not expected to match. Preserve the
checkout's declared line endings for source files used by build bindings.

### Training readiness on the current machine

Use the actual training configuration, for example:

```text
uv run --locked python scripts/restore_training.py --config configs/runs/training/single_qwen2_5_7b_gold_sft.json
```

Allow roughly 2–8 minutes for initial 7B readiness. This checks the environment,
assets and training input, loads the model, generates a short response, and runs
backward on the longest sample without an optimizer. Success writes `ready` below
`outputs/recovery/`; `--report` selects a new destination. `nvidia-smi` must work.
Ctrl+C stops the check; choose a new report path if retrying after one was written.

The stage guides pass this receipt to training, then run a short optimizer/save
check before the full run. `--previous` can reuse matching backward evidence;
changed data, model, relevant implementation or environment requires a new check.
Historical acceptance and CI results do not establish current-machine readiness.

## Frozen policy evaluation inputs and reuse

Single evaluation selects a frozen V5 panel, Base or a declared adapter, and
explicit mechanics. Current configs run the retained panel under corrected rules;
their scores need not reproduce historical legacy-engine results. Panel size and
encounter scope come from the configuration, not fixed runner assumptions.

Keep each run's whole output directory: report, inputs, per-combat records and
trajectories. Resume checks completed combats and runs the missing ones. Changed
model, data or simulator conditions require separate outputs. See
[compatibility](data_training.md#current-configuration-and-identity-implementation).

## Combat input generation

### Frozen Boss inputs

[The generator](../../configs/generation/single_boss_inputs.json) uses Teacher-based
reward choices, fixed removal/upgrade rules and route seeds. Corrected mechanics
can change generated loadouts despite keeping the seeds. Generation never replaces
the included frozen panel implicitly; select the new panel, its fingerprint and
counts explicitly for a new evaluation. Historical reconstruction is described in
[the Boss record](experiments/frozen_boss_192.md#outputs-and-reproduction-boundary).

### Configured continuous Act-1 development routes

The [shared panel](../../configs/panels/continuous_act1_development.json) declares
combat categories, rewards, upgrades, removals, relics and healing. Operations run
in that order, carrying HP and persistent relic counters after victory; death ends
the route. Combat-only cards, powers and upgrades do not become permanent state.
Burning Blood heals once, and route healing occurs only at its declared node.

Reward selection compares taking a card with skipping, combining Teacher search,
pick-rate priors and deck options. Upgrades use a separate panel of elites and the
known Boss, without reading the actual next monster or formal combat seed. Separate
random streams keep route construction, combat and upgrade comparisons distinct.
Run configs select policy arms: an omitted checkpoint means Base, while a declared
adapter is actually loaded. Unknown fields fail instead of silently selecting Base.

The current panel has 20 topologies × four seed groups. Smoke selects topologies
and all their groups. A route stops on defeat, final Boss victory or decision limit;
truncation is reported separately from death but stays in the Boss-win denominator.
Completed routes resume after trajectory checks; interrupted routes restart.
Later-stage HP is conditional on reaching that stage, and groups sharing a topology
are not independent routes. Commands are in [continuous](experiments/continuous.md).

### Continuous Teacher candidate collection

The [Teacher pool](experiments/teacher_candidate_pool_800x4.md) retains every
reached decision, including losing routes, and replays each new combat before
finalization. Explicit route and seed exclusions separate it from development data.
Repeated opening hands are not leakage; reusing the same source route is.

`--workers N` changes Teacher-only concurrency without changing seeds or identity.
Use one process per output directory. Keep the complete `formal/` tree;
`verify_teacher_candidate_pool.py` checks delivered artifacts without loading a
model or simulator. Historical exclusion labels require the explicit exclusion
input. Collection alone does not produce certified GOLD labels.

## Simulator and real-game limits

Corrected builds fix several card rules, including hand upgrades, Rage cost,
Iron Wave, Double Tap, Blood for Blood and Disarm+. V2 snapshots preserve repeated
Searing Blow upgrades; older boolean-only snapshots cannot recover lost counts.
V7 also needs native upgrade previews and selection context. Old engines and
observations cannot be treated as interchangeable by renaming their versions.

The narrow real-game parity evidence covers starter-deck Cultist, Jaw Worm and
Two Louse: 42/42 recorded transitions. Expanded simulator coverage does not broaden
that claim. The [live guide](real_game_testing.md) describes the separate V5 default
and opt-in V6 selection. Live trajectories are exploratory evidence, not training
input. The bridge no longer exposes the historical `turn_topology` operations.
