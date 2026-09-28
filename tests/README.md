# Test layout

Tests are grouped by stable code responsibility:

| Directory | Scope |
|---|---|
| `policy/` | Observation serialization, action projection, policy protocol, episodes, and trajectories. |
| `simulator/` | Simulator clients/adapters, installation, parity, mechanics, and reusable Teacher search behavior. |
| `live/` | Communication-mod adaptation, preflight, sessions, and card-selection interaction. |
| `training/` | Data adapters, LoRA, model backend, checkpoints, recovery, configuration, and artifact I/O. |

Run commands remain in the repository [README](../README.md#setup-and-verification).
Shared synthetic inputs and doubles live in the owning package's fixture modules;
`tests/support.py` only writes temporary JSON inputs. Tests import fixtures, not
other `test_*.py` modules. Fixtures contain no test cases or import-time execution.

Native test classes check prerequisites when they run, not during import. A source
checkout without native manifests or binaries skips those classes with an explicit
reason; pure Python tests still run. On Windows, a valid base installation without
the optional card-selection extension also skips selection tests. Partial builds,
invalid bindings, unsupported profiles and protocol errors remain test errors.
Skipped classes do not count as simulator validation. Build instructions are in the
[runtime guide](../docs/open_source/runtime_and_simulator.md#callable-native-environment-selection).

Test ownership and retirement rules live only in the
[configuration contract](../docs/open_source/data_training.md#evidence-and-tests).
