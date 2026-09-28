# Machine evidence layout

Published JSON evidence is grouped by what it establishes:

Experiment ownership is indexed under [single and continuous](../docs/open_source/experiments/README.md).
This directory keeps original report filenames and results; stage naming does
not change recorded execution identities. Backward/smoke and runtime checks are
supporting evidence for their declared run or environment, not separate stages.

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

The two 7B cross-run summaries
are recorded in [stage results](../docs/open_source/stageresult.md), which links
the original execution evidence. Interpretation of original-run paths
and hashes, plus retention and reproduction rules, lives in the
[configuration contract](../docs/open_source/data_training.md#retention-and-retirement).
