# Machine evidence layout

Published JSON evidence is grouped by what it establishes:

This directory keeps original execution reports and their recorded identities.
Cross-report interpretation is in [result analysis](../docs/open_source/stageresult.md).
Runtime and training checks apply to their recorded run/environment.

| Directory | Evidence |
|---|---|
| `runtime/` | Model or simulator environment acceptance and platform parity. |
| `data/` | Dataset construction, migration, collection, contracts, and data-quality audits. |
| `training/` | Backward preflight, optimizer smoke, training, and checkpoint completion. |
| `evaluation/` | Offline, simulator, held-out, comparative, and policy-quality results. |
| `mechanics/` | Focused interaction and game-mechanics validation. |

Public copies containing machine-specific absolute paths have those paths replaced
by project-relative locations or executable names. Each changed JSON has a
`publication` object with `kind: path_redacted_copy`, `original_file_sha256`, and
the exact `changed_fields` as JSON pointers. Original files are retained locally.
Statistics, experimental parameters, asset hashes and recorded hardware are unchanged.
The public copy has different bytes: its original-file hash and any historical
embedded document hashes identify the original evidence, not the redacted payload.
Public copies must not be substituted for original hash-bound replay evidence.
New simulator descriptions write project-relative paths directly; their commands
are portable build recipes, not transcripts of machine-local executable locations.

Original-run paths and hashes identify provenance. Earlier development commits
and raw reconstruction/replay collections are not distributed; public reports do
not replace those inputs. See [reuse boundaries](../docs/open_source/data_training.md#configuration-and-artifact-reuse-direction).
