# Combat card selection

`combat_card_selection_v1` is an opt-in extension for Ironclad cards that pause
combat to choose another card. Simulator implementation and short native checks
precede the CommunicationMod adapter. Live transport tests use recorded-shape
fixtures; an operator-started real-game retest is still required.

| Card | Base | Upgraded |
|---|---|---|
| Armaments | Gain Block, choose an upgradeable hand card | Upgrade the entire hand automatically |
| Headbutt | Damage, choose discard card to draw-pile top | More damage, same choice |
| Exhume | Return an exhausted card to hand, excluding Exhume | Lower cost, same choice |
| Burning Pact | Choose hand card to exhaust, then draw | Draw more after exhausting |
| True Grit | Exhaust a random hand card | Choose the exhausted card |
| Warcry | Draw, then choose hand card to draw-pile top | Draw more before choosing |
| Dual Wield | Choose an Attack/Power and create one copy | Create two copies of the chosen card |

Exhume selects an eligible card anywhere in the exhaust pile, not only its top.
Cards that upgrade/exhaust all cards or act randomly do not acquire artificial
choice stages. This scope excludes reward selection, potion choices, optional
multi-selection, other characters and mod-added actions.

## Contract and compatibility

The sequence is `PLAY -> environment feedback -> SELECT_CARD -> environment
feedback`. A selection decision has an explicit task, source pile, exact
candidate identities, and (for Dual Wield) copy count. Only candidate actions
are legal; there is no PLAY or END_TURN during selection. Empty or single-option
effects may resolve automatically in the engine. Terminal combat takes
precedence over an outstanding choice.

The model still returns one `ACTION_n` per decision and uses the same parser,
retry and seeded fallback. `observation_v6` extends the V5 rendering with the
seven cards' semantics and selection context. It also renders Searing Blow
beyond its first upgrade after repeated Armaments use. V1–V5 reject extension
states; their existing inputs, catalogs and action ordering remain unchanged.
Ordinary V6 decisions retain the PLAY action equivalence; selection decisions
record `combat_card_selection_v1` and preserve distinct card instances, even
when their public descriptions match. UUIDs, native IDs and pile indices remain
execution metadata; hidden draw order never enters the prompt.

The new canonical types subclass the existing state/action records, so ordinary
historical records gain no fields. Selection trajectories record the task,
source index, exact action mapping and native choose/confirm command trace.
They remain `coverage_collection` and are not training data. The V6 live profile
uses the existing V5 Gold SFT checkpoint for exploratory execution; it does not
claim that the checkpoint was trained on these new decisions.

## Simulator

The pinned engine already implements the seven cards and single-card selection.
The bridge now exposes the supported `CARD_SELECT` tasks, enumerates engine-
validated choices, executes `SINGLE_CARD_SELECT`, and retains stale-decision
checks and terminal accounting. Teacher search can also return a selection
root action. Historical feature-deck recipes and collection manifests remain
frozen; use an explicit `combat_snapshot_v1` deck for this extension.

From the repository root, build a separate bridge from the installed engine objects:

```powershell
.\scripts\build_card_selection_bridge.ps1
uv run python -m unittest discover -s tests -p "*card_selection*.py"
```

The incremental build took 10.85 seconds locally; allow roughly 10–30 seconds
for compilation/linking and 1–5 seconds for the focused tests. It requires the
existing pinned base build and compiler. It writes only
`outputs/card-selection/`, including source, engine-object and binary hashes.
The old binary and `.simulator/sts_lightspeed/build_manifest.json` are unchanged.
The build prints its output path and elapsed time; tests finish with `OK`.
Ctrl+C stops either command; restart from the beginning after interruption.

On Linux, the separately isolated native build includes this current bridge
directly and declares `combat_card_selection_v1`; no `.exe` extension build is
required. The capability resolver verifies the Linux build manifest's bridge
source binding before use. Run the focused native tests against the selected
installation; a missing installation is not a passing native check.

Python callers import `SELECTION` and `resolve_simulator` from
`sts1_llm_policy.env.simulator_execution`, resolve with
`required_capabilities={SELECTION}`, then call `create_environment()` and reset
with an explicit combat snapshot. The resolver verifies binary/source and
base-manifest bindings before enabling selection parsing. `create_client()`
provides the equivalent raw client. The shared resolver defaults to corrected
mechanics; omitting `SELECTION` keeps secondary selection disabled even though
the same binary supports it. Historical `legacy_v1` is an explicit Windows-only
choice without selection. See the [mechanics boundary](runtime_and_simulator.md#callable-native-environment-selection).

Native checks cover both versions of all seven cards, empty/single candidate
cases, exact post-choice effects, stale/illegal commands, lethal Headbutt,
Exhume's exclusion rule, repeated Searing Blow upgrades, and non-mutating
Teacher selection search. The tiny all-exhausted fixture can terminate as an
engine loss; no additional selection is invented for a terminal state.

### Simulator exhaust auto-resolution audit

The follow-up simulator audit verifies actual effects for zero, one and two
exhaust candidates, including both card versions and drawing from an existing
draw pile or after shuffling the discard pile.

| Card | Zero candidates | One candidate | Multiple candidates |
| --- | --- | --- | --- |
| Burning Pact / + | Draw 2 / 3 automatically | Exhaust the sole card, then draw 2 / 3 automatically | Choose one to exhaust, then draw 2 / 3 |
| True Grit | Gain 7 Block automatically | Gain 7 Block and exhaust the sole card automatically | Gain 7 Block and exhaust a random card |
| True Grit+ | Gain 9 Block automatically | Gain 9 Block and exhaust the sole card automatically | Gain 9 Block and choose one to exhaust |

Zero candidates skip exhaustion without suppressing the independent draw or
Block effect. They do not trigger Feel No Pain. Automatically exhausting a
single Sentinel does trigger its energy gain. Zero/one-candidate Burning Pact
also passes through the canonical environment and a stub-backed LLMPolicy with
one policy call and no secondary decision.

This audit adds 29 scenarios across four test methods; the complete suite passes
309 tests with zero failures, errors or skips. Evidence is retained in
`report/mechanics/combat_card_selection_exhaust_edges_v1.json`. No runtime/config changes
were needed, and the original implementation report remains unchanged. These
results validate the simulator path; live validation remains pending.

## Real-game adaptation and first retest

The opt-in profile is `configs/live/real_game_card_selection_session.json`.
The existing V5 session remains the default. The new environment accepts only
combat HAND_SELECT/GRID screens with an audited current action, originating
card, mandatory one-card count and matching candidate UUIDs. GRID UI ordering
is resolved independently from discard/exhaust ordering. Acknowledgement checks
discard repeated snapshots, verify the selected UUID before sending one
confirmation, and require the choice's effect before returning to ordinary play.
Unknown actions and non-combat grids remain guarded stops.

The transport fields and command ordering were checked against upstream
[GameStateConverter](https://github.com/ForgottenArbiter/CommunicationMod/blob/master/src/main/java/communicationmod/GameStateConverter.java)
and [ChoiceScreenUtils](https://github.com/ForgottenArbiter/CommunicationMod/blob/master/src/main/java/communicationmod/ChoiceScreenUtils.java).
These sources describe HAND_SELECT/GRID candidates and their selection and
confirmation behavior. Installed-build behavior still needs live validation.

From the repository root, first run the binding-only preflight
(typically under 5 seconds; no model/game):

```powershell
uv run python .\scripts\run_real_game_session.py --config .\configs\live\real_game_card_selection_session.json --mode preflight
```

Success is `bindings_validated=true` and `preflight_only=true completed=true`
in `outputs/logs/real_game_card_selection.log`. While the game is closed,
run the following in PowerShell from the repository root to print the absolute
child-process command for this checkout (under a second; prints only). Copy its
output into CommunicationMod's command setting:

```powershell
$projectRoot = (Get-Location).Path
'"{0}" --directory "{1}" run python "{1}\scripts\run_real_game_session.py" --config "{1}\configs\live\real_game_card_selection_session.json"' -f (Get-Command uv).Source, $projectRoot
```

Do not launch that command in an unattached terminal. Start with unupgraded
Armaments and at least two upgradeable hand cards in a BaseMod-injected combat.
Then test the other table rows and upgrades, including two discard/exhaust
candidates for Headbutt/Exhume. Use BaseMod's Tab completion to prepare the
cards/encounter. Stop manual combat input while the policy owns decisions;
the operator continues to own all non-combat screens.

Allow roughly 1–3 minutes per prepared combat, or 15–45 minutes for fourteen
card/upgrade cases plus setup. This is a rough estimate based on the previous
33-decision/approximately-70-second live session, not a measured V6 matrix;
model latency, hand preparation, draw order and combat duration dominate.
Progress is logged as decisions and per-combat trajectories under
`outputs/real-game-card-selection-sessions/<session-id>/`. Success requires a
legal selection, return to combat, and a terminal transition. Winning is not
required for interface acceptance. Return to the main menu to finish normally;
closing the game or Ctrl+C stops the session and retains partial evidence.
Sessions do not resume; relaunching creates a new directory. After the run,
report completion or the final error so the retained report can be checked.

No live session, model inference, new training, large evaluation, or sealed
data read was started as part of implementation. Mechanical test results are
recorded in `report/mechanics/combat_card_selection_v1.json`; live status remains pending.
The completed suite has 305 passing tests, zero skips, 32 short native fixtures,
and passing bindings for five historical evaluation configs plus both live profiles.
