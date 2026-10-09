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

Selection trajectories retain the exact executed choice, but remain exploratory
coverage evidence, not training data. The live V6 profile uses the existing V5
Gold SFT adapter; it was not trained on these new choices.

## Simulator

The pinned engine already implements the seven cards and single-card selection.
The bridge now exposes the supported `CARD_SELECT` tasks, enumerates engine-
validated choices, executes `SINGLE_CARD_SELECT`, and retains stale-decision
checks and terminal accounting. Teacher search can also return a selection
root action. Historical feature-deck recipes and collection manifests remain
frozen; use an explicit `combat_snapshot_v1` deck for this extension.

From the repository root, use the shared build entry on Windows or Linux after
installing the [native prerequisites](runtime_and_simulator.md#native-build-prerequisites):

```text
uv run --locked python scripts/simulator.py build --require combat_card_selection_v1
uv run --locked python -m unittest discover -s tests -p "*card_selection*.py"
```

On Windows this builds a missing base, then the separate extension under
`outputs/card-selection/`. The extension reuses the pinned base objects and does
not modify its binary or manifest. Linux builds the bridge directly under
`.simulator/sts_lightspeed-linux/`; no separate extension is needed.
Existing valid builds are reused. Invalid bindings require the
[explicit rebuild procedure](runtime_and_simulator.md#callable-native-environment-selection).

The build includes a bounded protocol smoke and reports `ready`; focused tests
finish with `OK`. Check skips: a missing native installation is not a passing
native check. Allow 1–5 seconds for the focused tests, plus the
[platform build budget](runtime_and_simulator.md#callable-native-environment-selection).
Ctrl+C stops either command; an interrupted build may need explicit recovery
before rerunning.

The shared resolver separates corrected mechanics from enabling secondary choices.
V5 experiments can use the corrected binary without selection actions. Native
checks cover the seven cards, illegal/stale choices, terminal combat, and repeated
Searing Blow upgrades.

Zero/one-candidate effects can resolve automatically: Burning Pact still draws
even when there is nothing to exhaust, and True Grit still grants Block. Automatic
exhaustion triggers effects such as Sentinel's energy; an empty exhaustion does not
trigger Feel No Pain. The [exhaust-edge audit](../../report/mechanics/combat_card_selection_exhaust_edges_v1.json)
records the simulator evidence. It does not validate live behavior.

## Real-game adaptation and first retest

The opt-in profile is `configs/live/real_game_card_selection_session.json`.
The existing V5 session remains the default. The new environment accepts only
combat HAND_SELECT/GRID screens with an audited current action, originating
card, mandatory one-card count and matching candidate UUIDs. GRID UI ordering
is resolved independently from discard/exhaust ordering. Acknowledgement checks
discard repeated snapshots, verify the selected UUID before sending one
confirmation, and require the choice's effect before returning to ordinary play.
Unknown actions and non-combat grids remain guarded stops.

Transport behavior follows CommunicationMod's
[choice handling](https://github.com/ForgottenArbiter/CommunicationMod/blob/master/src/main/java/communicationmod/ChoiceScreenUtils.java).
Fixture checks still need confirmation against the installed game build.

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
Sessions do not resume; relaunching creates a new directory. Inspect the final
session report and retain its trajectories, including any interrupted combat.

The [implementation report](../../report/mechanics/combat_card_selection_v1.json)
contains native/fixture checks, not a real-game selection retest.
