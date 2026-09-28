# Policy and observation contract

Public student inputs, legal actions and supervision boundaries.
For Teacher continuation collection see [Teacher and GOLD](teacher_gold.md);
for exported records and optimization see [data and training](data_training.md).

## Student and supervision boundary

- The student receives the explicitly configured public observation and a
  canonical legal action list, and returns exactly one decision-local `ACTION_n`.
  Historical training uses V5; V6 adds combat card selection; V7 adds remaining-route
  context and retains that selection interface. The V7 correction adds observed
  draw-prefix memory without changing its existing fields or route objective.
- Base, SFT, and preference policies use the same serializer, action projection,
  parser, retry, and visible seeded fallback.
- Hidden draw order, native execution IDs, search visits, rollout values, and
  Teacher actions never enter the student input.
- Train and development records are split by source episode for independent combats,
  and by source route for continuous data. All arms and formal seed groups sharing
  a route's encounter/reward schedule belong to the same split. Test and sealed
  records are not read during development or training.
- Inputs, labels, preference edges, losses, and gradients must be valid and
  finite. Training tokenization does not truncate records.

Equivalent card instances share one model action per target; different targets
remain separate. The projection uses public card semantics, current cost, upgrades
and combat modifiers, and hides instance UUIDs and hand positions. It executes the
lowest-hand-index representative and samples fallback uniformly over model action
classes. Trajectories retain the selected class, equivalent internal IDs and executed
action. New outcome-relevant modifiers must enter the equivalence key; unsupported
mechanics fail closed. These rules live in the action projection and its behavior
tests rather than a second configuration copy.

### Maintained observation versions

| Use | Version | Input contract |
| --- | --- | --- |
| Single-combat datasets, model comparison and default live policy | V5 | Public combat state, legal actions, explicit effects, behavior rules and a self-contained glossary. |
| Opt-in live card selection | V6 | V5 plus one legal secondary-selection action at a time. |
| Continuous GOLD/SFT/DPO and route evaluation | V7 | Combat and secondary-selection state plus the remaining route, upgrade previews, selection context and observed draw-prefix memory. |

V5 includes player HP, Block and energy; visible card piles and current costs;
enemy intent with adjusted damage and hit count; active power amounts; and relic
names, counters and effects. It exposes combat accounting and public monster
behavior, including previous moves and possible following moves, without revealing
the realized future RNG outcome. Calls are independent, without conversation history.
The single-combat objective prioritizes victory, then remaining HP among comparably
successful lines. Native execution IDs are not model-visible.

Card, power, relic, current-intent, possible-next-move and behavior terms contribute
to a deterministic, case-insensitive transitive glossary. Summoning, splitting and
other first-exposure mechanics are explained before they must be acted upon. The
[serializer](../../src/sts1_llm_policy/env/serializer.py),
[mechanics catalog](../../src/sts1_llm_policy/env/mechanics_catalog.py) and
[glossary](../../src/sts1_llm_policy/env/observation_v5_catalog.py) implement this
contract; separate JSON copies of each development revision are not required.

The public comparison is V5 to V7, with V6 retained for the live extension. V1–V4
identifiers in historical reports describe those original inputs. The serializer
retains versions needed by existing fixtures and reconstruction; specifically,
V4-to-V5 reserialization rebuilds the glossary without changing actions or labels.
Those implementation dependencies are not additional recommended experiment stages.
Evaluation validates `observation_version` against its supported protocol directly.

### Continuous-route observation V7

`observation_v7` requires an explicit `CombatRouteContext` on every canonical
combat state, including secondary card-selection decisions. It exposes the current
combat position/type, remaining combat count excluding the current fight, public
Boss identity, and the complete ordered suffix of route operations. Combat nodes
expose encounter families (weak/strong normal, elite, Boss); healing exposes its
amount and max-HP cap. Upgrade, reward card pick, removal and random-relic nodes
expose the operation, without future choices or random outcomes. Future ordinary
monster identities, relic identities, card offers, combat/policy seeds and upgrade
evaluation panels are excluded by projection into typed public operations.

The objective is to maximize final Boss completion probability, balancing current
combat death risk against HP and persistent resources available for the rest of
the route. Current-combat win probability has no absolute lexicographic priority.
Boss victory completes the objective; remaining HP then has no future survival
value. Route operations outside combat remain controlled by the configured
strategies. No additional model action types are introduced.

The simulator attaches context after native state conversion and preserves it
through play, secondary selection and subsequent steps. Initial requests, retry
and trajectory serialization consume that same state. A reset without context
clears the previous route. V7 rejects missing context; earlier versions reject
route-bearing states rather than silently discarding route information. V1–V6
observation text remains unchanged for states without route context.

V7 Armaments choices include each candidate's `AFTER_UPGRADE` card type, current
cost, upgrade count, target, exhaust/ethereal flags and full effect. Normal
Armaments and upgraded Armaments also show `UPGRADE_RESULT` entries before play,
excluding the played instance; the latter upgrades all eligible remaining cards.
Native `hand_upgrade_previews` align to the current hand and are produced by
upgrading copies with the same `CardInstance::upgrade()` operation used by combat.
They preserve per-instance state and repeated Searing Blow upgrades without
executing actions, advancing RNG, or revealing future draw order. Missing previews
fail explicitly in V7; upgrade results are never guessed from card names.

V7 secondary choices also display the public `SOURCE_CARD`. Burning Pact's
pending draw count distinguishes its ordinary and upgraded versions from True
Grit's exhaust-only remainder, with draw restrictions and exhaust triggers noted.
Armaments/True Grit Block, Headbutt damage and Warcry draw are marked already
resolved. Dual Wield specifies copy count, preserved instance state, combat-only
duration and discard overflow. Selection effects and upgrade-result mechanics
participate in the visible glossary. These additions leave V6 prompt text intact.

The V7 public-memory correction adds observed draw-prefix memory, next draw first,
whenever it is available. Existing route context, objective, selection-source and
upgrade-preview details remain unchanged. Full-trace execution saves V7 observations
for public-information checks and review; native Teacher search does not consume
that text. Teacher's single-combat win-rate/expected-carried-HP scoring is independent
of the student's route objective. Public-state sampling preserves route context.
Executed Headbutt and Warcry placements extend the known prefix; drawing/playing a known
card removes it. Shuffling or an unknown-position insertion invalidates the prefix
conservatively. Unknown future cards, actual RNG state and instance IDs are not
serialized. V1–V6 output remains unchanged. Historical V7 route records without
memory remain readable; they do not provide evidence of the corrected memory
behavior. Original reports retain their recorded configuration and code revision.

The native `BattleScumSearcher2` Teacher remains a local combat search baseline.
It does not consume V7 route text or claim to optimize route completion. Logging
its actions beside V7 observations does not certify new route-aware training
labels. A future LLM Teacher would need the same public information boundary and
an explicitly defined supervision objective.

Continuous trajectories carry `route_lineage` metadata with `source_route_id`,
route index, formal seed group, arm and combat index. The source ID excludes
formal combat/policy seeds and is shared across arms and those seed groups;
it is never model input. Route results link each combat trajectory and its final
route outcome. `done` remains combat termination; V7 additionally records
`route_done` and `route_terminal_outcome` for death or final Boss victory. A decision
limit is an abort, not a successful route or a gameplay death.

Development trajectories remain evaluation evidence. Any future training dataset
must declare its separate source manifest, preserve route lineage and split
isolation, and check low-HP, later-combat, Boss and secondary-selection coverage.
No V7 training dataset is produced by the generator. New training observations
must be tokenized without truncation and receive an applicable longest-sample
backward check before training; historical V5 length evidence is insufficient.

The separate `continuous_teacher_pool.json` generation config declares fresh
Teacher candidate evidence through `data_source.evidence_class`. It retains all
decisions from independent continuous routes and excludes the declared historical
development sources during preflight. Four formal seed groups share each source
route ID and must remain together in any later split. The pool is unpartitioned
and uncertified: opportunity selection, public-information plan certification,
training weights and train/validation assignment are later consumers. A Teacher
action in this pool is a proposed label, not an accepted Gold target. Collection
and delivery checks are described in the
[runtime contract](runtime_and_simulator.md#continuous-teacher-candidate-collection).
