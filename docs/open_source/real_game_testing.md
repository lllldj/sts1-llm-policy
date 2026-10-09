# Real-game testing

CommunicationMod can run the combat policy inside Slay the Spire. This is an
exploratory integration: live success does not establish simulator parity, and
live trajectories are not training data.

## Current binding

The default [session profile](../../configs/live/real_game_gold_sft_v5_session.json)
uses Qwen2.5-1.5B with Gold SFT and V5 observations. The model controls combat;
you handle rewards, map routing, shops, events and other non-combat screens.
The process waits between combats. [V6 card selection](combat_card_selection.md)
is a separate opt-in profile using the same V5-trained adapter.

## Prerequisites

Install Slay the Spire, ModTheSpire, BaseMod and CommunicationMod, and obtain the
model files declared by the session profile. With the game closed, run this
PowerShell block from the repository root. It prints the absolute child-process
command for this checkout; copy the output into CommunicationMod's command setting.

```powershell
$projectRoot = (Get-Location).Path
'"{0}" --directory "{1}" run --locked python "{1}\scripts\run_real_game_session.py"' -f (Get-Command uv).Source, $projectRoot
```

This takes under a second and starts nothing. CommunicationMod must launch the
command; an unrelated terminal does not connect the runner to the game.

## Preflight

From the repository root:

```powershell
uv run python .\scripts\run_real_game_session.py --mode preflight
```

This checks bindings in a few seconds without starting the game or loading the
model. Success writes `bindings_validated=true` and
`preflight_only=true completed=true` to `outputs/logs/real_game_session.log`.

## Running a session

1. Start ModTheSpire with BaseMod and CommunicationMod enabled, then start or
   continue an Ironclad run. The child process can wait at the main menu.
2. Handle non-combat choices yourself. At a stable combat decision, stop manual
   combat input and let the policy play until combat ends or a guarded stop occurs.
3. Resume manual control after combat. The same process takes over the next fight.
4. Finish by returning to the main menu or reaching the configured combat limit.
   Ctrl+C records an interruption. Relaunching starts a new session, not a resume.

Do not start a second runner or play cards concurrently with the policy.

## Outputs and diagnosis

Each launch writes `outputs/real-game-sessions/<session-id>/session_report.json`
and one `combat-NNN.jsonl` trajectory per fight. Read the report first. Normal
statuses are `completed`, `stopped_limit` and `interrupted`.
`stopped_unsupported_mechanic` means the runner rejected an unknown mechanic before
acting; `stopped_safely` records another caught failure. Keep the entire directory
and inspect its `error_type` and `error`. Diagnostics also go to
`outputs/logs/real_game_session.log`; stdout carries the game protocol.

## Current live status

The retained V5 session `real-game-20260905T032357Z-0c7f6b75` defeated The Guardian
in 33 decisions, going from 72 to 32 HP and covering both mode transitions.
The combat ended in victory; the session later reported `stopped_safely` when the
communication channel closed. The earlier non-combat `OSError [Errno 22]` remains
unresolved. Neither issue is claimed fixed by the monster adaptation.

The [encounter matrix](act1_real_game_encounter_matrix.md) records which variants
have terminal evidence. Supported catalog entries and injected debug fixtures do
not prove coverage of every natural encounter. Session directories are local
evidence and are not distributed. V6 card selection still needs a real-game retest.

## Open limitations and next checks

CommunicationMod supplies numeric monster move IDs and no general description for
unknown powers. A mapping reconciles these with simulator semantics—for example,
`Weakened` becomes `Weak`. The installed CJK build supplies useful card/relic text,
but it is not a complete mechanics registry; unknown powers still stop execution.
The model-facing glossary is project-authored.

Legal execution does not imply strong play. Gold SFT has weak later-encounter
results, only 11 selected Flex examples, no selected Rage examples, and seven
legal low-HP opportunities for each. Setup-buff ordering and survival tradeoffs
remain limitations. Live failures can motivate a training category, but new
examples must come from isolated training sources. Further integration checks
should target a missing encounter variant or a reproducible protocol failure.
