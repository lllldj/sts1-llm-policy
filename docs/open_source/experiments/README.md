# Experiments

The project has two experiment stages, named by the information available to the
student. Each stage connects data, labels, training, checkpoints and evaluation.

| Stage | Training scope | Evaluation |
| --- | --- | --- |
| [single](single.md) | SFT and DPO using single-combat information; 1.5B and 7B runs | Single-combat panels, including the frozen Boss comparison |
| [continuous](continuous.md) | SFT and DPO using route context; 7B GOLD-only, mixed SFT and DPO A/B/C | Configured continuous routes; final Boss completion |

The Boss panel is an evaluation component of `single`. The Teacher candidate pool
is a data source for `continuous`. Neither is a separate top-level experiment.
The live default remains the `single` 1.5B Gold SFT policy; completing `continuous`
does not replace it.

## Names and configuration

Formal configuration filenames use `single_` or `continuous_`, followed by the
model and role where needed, without a stage version suffix. Configuration contents
select the actual observation, dataset, algorithm, recipe, seeds and checkpoint;
Git records configuration revisions. Shared interface/runtime/recipe versions
retain their technical meanings and are not experiment-stage numbers.

Retained diagnostic and configuration-change trials may use versioned names.
Their role and parent stage are listed below. An execution's recorded `run_id`,
dataset ID, output path and original report retain their historical identity,
including existing version suffixes. Renaming an entry config does not rename its
artifacts or create a new run. Select new run/output identities before executing a
changed experiment; do not overwrite frozen results.

Original reports may name a config's former path. Reproducing that execution
requires its original source revision, which is not included in a release snapshot. Current entry
paths are listed on the stage pages. Fixed training datasets, reference adapters and picker inputs are included in Git.
Other ignored raw reconstruction/replay assets are retained locally and are not distributed. A protocol name alone is insufficient:
early V7 executions predate the public draw-prefix memory correction.

## Supporting work

Historical diagnostic paths under `outputs/` identify locally retained raw evidence,
not files in the source snapshot. No public diagnostic download is currently provided;
formal-stage asset availability is listed on the single and continuous pages.

| Parent | Work | Role and retention boundary |
| --- | --- | --- |
| single | Expanded SFT and topology preference training | Earlier training branches; original data, training and evaluation reports retain their distinct evidence. See [single](single.md#data-and-earlier-branches). |
| single | [Boss panel](frozen_boss_192.md) | Frozen evaluation detail, including input construction and paired comparisons. |
| continuous | [Early route comparisons](continuous_routes_20x4.md) | Historical Base/Teacher and observation/mechanics changes; retained versioned configs identify different execution conditions. |
| continuous | [Teacher pool](teacher_candidate_pool_800x4.md) | Shared source collection, with source isolation and replay evidence. |
| continuous | Action-order diagnostic | Original report (`outputs/inspection/teacher-order-probe-v1/report.json`); dedicated runner, config and tests are retired. Source/config provenance: Git `e1045b0`. Interpretation boundaries are in the [diagnostic record](../stageresult.md#targeted-order-diagnostic). |
| continuous | `teacher_gold_probe_v3` | Executed-continuation diagnostic; its config is retired and its tuning routes are preserved in the fixed exclusion input. The shared GOLD executor is maintained; see the [data contract](../teacher_gold.md#gold-execution-and-replay). |
| continuous | Multi-target budget comparison | Recorded partial standard-budget result (`outputs/inspection/teacher-gold-multitarget-v1/budget-2048/formal/report.json`); this is not a completed budget comparison. Its preparer and the two 8192 configs are retired. The 2048 configs are retired; their tuning routes remain in the fixed exclusion input. |
| continuous | Continuation-policy comparison | Original B (`outputs/inspection/teacher-continuation-policy-v1/arm-b/formal/report.json`) and C (`outputs/inspection/teacher-continuation-policy-v1/arm-c/formal/report.json`) reports; the preparer and dedicated configs are retired. |
| continuous | Sampling-ladder comparison | Original shadow report (`outputs/inspection/teacher-gold-ladder-v1/formal/report.json`); the preparer and diagnostic config are retired. Adaptive collection and its behavior tests remain maintained. |
| continuous | `teacher_gold_collection_v2`, `teacher_gold_collection_v3` | Retired sampling/storage configurations; their reports and raw import sources retain their identities. The formal entry is now [continuous_gold_collection.json](../../../configs/generation/continuous_gold_collection.json); import/source dependencies must be preserved. |
| Shared / corresponding run | Runtime acceptance, backward checks, optimizer smoke, mechanics and parity checks | Support a declared environment, implementation or run; they are not separate experiment stages. The versioned 7B acceptance config remains a preflight/backward entry. |

The maintained execution surface follows the two formal experiment chains.
Historical diagnostics retain original results and necessary interpretation;
their dedicated preparers, run configs and tests do not require ongoing support
when they have no core consumer. Git `4ba3057` retains the three retired panel
preparers and their configs. This is source provenance, not a promise of a fully
reproducible diagnostic environment. Existing raw diagnostic outputs remain local
evidence; they are not removed by retiring their entry points.

Formal data generation, training, evaluation, recovery and local verification
retain their necessary inputs and interfaces. The source-bound exclusion list and original import
assets remain where the formal chain needs them. Reusable behavior tests and
runtime checks remain implementation coverage, not historical experiment runs.

Cross-run interpretation remains in [stage results](../stageresult.md); original
machine evidence remains in [report](../../../report/README.md). These pages index
the evidence rather than replacing it with new summary reports.
