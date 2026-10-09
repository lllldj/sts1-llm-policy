# Policy inputs and actions

The policy reads a text description of the public game state and a list of legal
choices, then returns one `ACTION_n`. Base, SFT and DPO use the same input format,
action parser, retry and seeded fallback, so their comparison changes the policy
rather than its interface. Invalid output gets one retry, then a logged legal
fallback if the retry also fails.

## Student and supervision boundary

Inputs describe HP, Block, energy, visible cards, enemy intent, powers and relics.
A glossary explains relevant effects and monster behavior. Calls have no conversation
history. Hidden draw order, future random outcomes, internal execution IDs and
Teacher search scores never appear in student text.

Equivalent copies of a playable card share one action for each target. Different
targets remain separate; current cost, upgrades and combat modifiers matter when
deciding equivalence. The environment executes a representative copy, and fallback
samples uniformly over these action classes. Secondary card choices preserve
individual candidates. Unknown outcome-relevant mechanics are rejected.

Training and development are separated by source combat or route. All variants
and seed groups sharing a route stay together. Test and sealed data are excluded
from development and training. Labels must be legal, numerical values finite,
and training inputs must fit without truncation.

### Maintained observation versions

| Use | Version | Information available |
| --- | --- | --- |
| Single-combat experiments and default live policy | V5 | Public combat state, legal actions, effects and glossary. |
| Opt-in live card selection | V6 | V5 plus choices such as Armaments upgrades or Headbutt discard selection. |
| Continuous-route experiments | V7 | Combat and selection state plus remaining route, upgrade previews and observed draw-prefix memory. |

Single combats prioritize victory, then remaining HP among comparably successful
lines. Continuous routes target final Boss completion. These stages also differ
in data and evaluation conditions; their results do not isolate the effect of
observation version. V1–V4 appear in historical reports and reconstruction code.
V4-to-V5 reconstruction adds the glossary without changing actions or labels.

### Continuous-route observation V7

V7 shows the remaining sequence of route operations, combat categories and public
Boss identity. It shows healing amounts and upcoming reward, upgrade, removal and
relic operations, but hides future ordinary monster identities, card offers,
relic outcomes and random seeds. The model must balance surviving the current
fight against keeping resources for later fights. Configured strategies handle
operations outside combat.

Upgrade previews show the actual effect of upgrading each eligible card, including
repeated Searing Blow upgrades. They come from simulated copies without advancing
combat or RNG. Selection prompts identify the originating card and distinguish
already-resolved effects from pending ones: for example, Burning Pact still draws
after exhaustion, while True Grit does not. Missing route context or required
previews causes an error rather than a guessed prompt.

Draw memory contains only a known prefix, next draw first. Headbutt and Warcry can
establish that knowledge; drawing consumes it. Shuffling or an insertion at an
unknown position clears it conservatively. Historical V7 records without this
memory remain readable, but do not demonstrate the corrected behavior.

The native Teacher searches individual combats and does not read the V7 route
objective. A Teacher action beside a V7 observation is therefore an imitation
label, not proof of the best route decision. The candidate pool retains reached
states from independent routes, including losing runs; later sampling and GOLD
comparison determine training eligibility.

Trajectories retain source-route lineage outside model input. Combat termination
and route termination are separate; a decision limit is an abort, not a victory
or gameplay defeat. Development trajectories remain evaluation evidence.

For implementation details, see the [serializer](../../src/sts1_llm_policy/env/serializer.py)
and [mechanics glossary](../../src/sts1_llm_policy/env/observation_v5_catalog.py).
[Teacher and GOLD](teacher_gold.md) describes target generation;
[data and training](data_training.md) describes exports and optimization.
