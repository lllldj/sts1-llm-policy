# Real-game testing

This document is the maintained operator guide and current status for running
the combat policy against a real Slay the Spire process. It covers exploratory
sessions launched by CommunicationMod. It does not widen the simulator parity
claim or make live trajectories eligible for training.

## Current binding

The maintained session profile is
`configs/live/real_game_gold_sft_v5_session.json`:

- policy: `qwen2_5_1_5b_teacher_v2_gold_sft_observation_v5_v1`;
- model: Qwen2.5 1.5B with the completed Gold SFT V5 adapter, not the DPO
  checkpoint;
- observation: `observation_v5`;
- evidence class: `coverage_collection`;
- live fallback: `communication_mod_descriptions_with_audited_powers_v2`;
- native-ID crosswalk: `communication_mod_to_sts_lightspeed_v7`.

The policy controls combat only. The operator owns Neow choices, rewards, card
selection, map routing, shops, events, rest sites, and every other non-combat
screen. The session waits across those screens and resumes at the next stable
combat decision.

## Prerequisites

1. Install Slay the Spire, ModTheSpire, BaseMod, and the CommunicationMod build
   used by the local game installation.
2. Ensure the model checkpoint and files bound by the session profile are
   present. The configuration validates their hashes before model loading.
3. Close the game before editing CommunicationMod's child-process command.
4. Use absolute paths in the actual CommunicationMod configuration. The
   conceptual command is:

   ```powershell
   uv --directory <project-root> run python <project-root>\scripts\run_real_game_session.py
   ```

CommunicationMod must launch this command. Starting the script in an unrelated
terminal does not attach its stdin/stdout protocol to the game.

## Preflight

From the repository root, run:

```powershell
uv run python .\scripts\run_real_game_session.py --mode preflight
```

This is a short binding check. It does not start the game, connect to
CommunicationMod, load the model onto the GPU, or create a live session. A
successful check appends both `bindings_validated=true` and
`preflight_only=true completed=true` to
`outputs/logs/real_game_session.log`.

## Running a session

1. Complete the preflight and close any existing game process.
2. Confirm that CommunicationMod's configured command points to the maintained
   runner above.
3. Start ModTheSpire with BaseMod and CommunicationMod enabled.
4. Start or continue an Ironclad run. The child process handshakes immediately
   and may wait at the main menu without loading the policy first.
5. Handle every non-combat choice manually.
6. At a stable combat decision, stop manual combat input. The policy will emit
   `PLAY` and `END` commands until combat terminates or a guarded stop occurs.
7. After combat, resume manual control. The same child process waits for and
   takes over the next combat.
8. End the session by returning to the main menu, reaching the configured
   combat limit, or pressing Ctrl+C. Ctrl+C records an interrupted status; it
   does not support resuming the same session directory.

Do not start a second runner while CommunicationMod owns the first process. Do
not manually play cards concurrently with the policy, because the legal-action
projection is tied to the state that produced it.

## Outputs and diagnosis

Each launch creates:

```text
outputs/real-game-sessions/<session-id>/
  session_report.json
  combat-001.jsonl
  combat-002.jsonl
  ...
```

The report is replaced atomically. Every completed combat records encounter,
entry HP, terminal HP, outcome, step count, deck/relic identity, and the bound
policy/fallback identities. Each combat trajectory is append-only and fsynced.
Diagnostic messages go to `outputs/logs/real_game_session.log`; stdout is kept
for the CommunicationMod protocol.

Read `session_report.json` first:

- `completed`: the run returned to its terminal condition normally;
- `stopped_limit`: the configured combat count was reached;
- `interrupted`: the operator stopped it with Ctrl+C;
- `stopped_unsupported_mechanic`: an unknown power, move, behavior, or missing
  description was detected before an action was sent;
- `stopped_safely`: another protocol, state, model, or filesystem exception was
  caught and recorded.

For a failure, retain the entire session directory and report the final
`error_type` and `error`. Do not delete partial trajectories: they are useful
diagnostic evidence even though they are not training data.

## Current live status

As of 2026-09-05, the operator reports Act 1 monster adaptation complete.
The latest retained session, `real-game-20260905T032357Z-0c7f6b75`, completed
The Guardian in 33 decisions, from 72 entry HP to 32 terminal HP. Its 33
trajectory records include the offensive-to-defensive-to-offensive cycle,
native Twin Slam (`4`, `ATTACK_BUFF`), and the previously unobserved native
Whirlwind (`5`, `ATTACK`). The final transition has `done=true` and
`terminal_outcome=victory`. This closes the Guardian crosswalk-v7 retest.

The session itself ended as `stopped_safely` after combat because the
communication channel closed before another JSON state arrived. That status
is distinct from the completed combat. The earlier non-combat `OSError
[Errno 22]` remains unassigned; neither is claimed fixed by monster adaptation.

The encounter matrix retains the exact machine-evidence status of each row.
Some seeded compositions and event variants still lack retained terminal
proof; the operator's adaptation closeout is not a claim that every variant
was measured. Debug injection remains interface evidence, not natural-roll
or simulator-parity evidence.

The live power extension describes `Split` without changing the frozen
simulator/training catalog. Crosswalk v7 includes the audited native power,
monster, move, and intent projections. There is one maintained implementation
of each live protocol; superseded intermediate definitions remain in Git.
The architecture and configuration-retention conclusions are recorded in
[`runtime_and_simulator.md`](runtime_and_simulator.md#shared-policy-boundary-and-historical-compatibility).

The retained policy-quality limitation is independent of integration: Gold
SFT has only 11 selected Flex examples and no selected Rage examples, with
only seven legal low-HP opportunities for each. Live states may identify a
future training error category but are not directly eligible for training.

Previously resolved launch and integration failures are:

- the child process receiving its first stable state at the main menu and never
  reaching policy loading;
- the starter relic ID `Burning Blood` not matching `BURNING_BLOOD`;
- unknown card and relic descriptions stopping otherwise legal live states;
- `Weakened` and `Flex` using different runtime and simulator names.
- The Guardian's native move-byte ordering differing from the canonical
  sts_lightspeed move IDs.
- Sentry and the other supported base-game monsters whose native IDs or move
  constants differ from the canonical simulator catalog.
- Large Acid Slime's native `Split` power being absent even though its Split
  move and behavior were already represented.
- Red Slaver Entangle using native intent `STRONG_DEBUFF` while the canonical
  catalog labels the same non-damaging move `DEBUFF`.
- The Guardian's native defensive-mode transition using `BUFF` while canonical
  move 5 uses `DEFEND` and records the immediately granted Block.
- The Guardian's native Twin Slam using `ATTACK_BUFF` while canonical move 7
  categorizes its visible two-hit action as `ATTACK`.

Act 1 terminal evidence and BaseMod console commands are maintained in
`act1_real_game_encounter_matrix.md`. Adaptation closeout does not start new
Base, SFT, or DPO training or evaluation.

The full unit suite currently contains 292 passing tests, and the maintained
session profile passes its binding-only preflight with fallback v2 and
crosswalk v7.

## Open limitations and next checks

The Guardian retest is complete. Future coverage work should target only an
explicit missing matrix variant or a reproducible protocol failure. Strategy
quality remains separate from integration correctness.

The remaining known limitations are:

- CommunicationMod forwards `power.ID`, but its upstream protocol provides no
  general power description. A truly unknown power must therefore remain a
  fail-closed condition.
- Monster moves are transmitted as monster-specific numeric IDs. Their names,
  effects, and behavior rules still require the maintained monster mapping.
- The installed CommunicationModCJK build can expose card/relic descriptions
  used by the live-only fallback, but this is not a complete mechanics registry.
- Real-game observation coverage is broader than the original P0 parity set;
  exploratory success does not establish simulator parity for a new mechanic.
- The Gold SFT policy has weak development performance on later encounters,
  especially Automaton and Act 2 A20. Legal execution should not be interpreted
  as strong strategic quality.
- Gold supervision contains no selected Rage examples and only 11 selected Flex
  examples, with very sparse low-HP opportunities. The present checkpoint
  should not be expected to reliably order setup buffs before attacks or trade
  immediate damage against combat survival.
- Live development states must not be copied directly into training. They may
  define an error category that is later reconstructed from isolated training
  seeds.

## Namespace and keyword ownership

sts_lightspeed emits status identifiers and display names such as `WEAK` and
`Weak`. The simulator adapter currently uses `Weak` as the canonical power ID.
CommunicationMod instead forwards the base game's runtime ID `Weakened`, so the
live crosswalk performs the necessary `Weakened -> Weak` projection. Guardian
move bytes require an analogous semantic projection because its base-game
constant ordering differs from the simulator's canonical move ordering. The
same rule now covers all discrepancies found by a static comparison of the
currently supported base-game monster classes and canonical catalog.

The model-facing `KEYWORDS` block is a separate layer. Its definitions are
project-owned in `observation_v5_catalog.py` and are selected transitively from
the project-owned card, relic, power, move, and behavior descriptions.
sts_lightspeed supplies the simulated state and mechanics implementation; it
does not directly supply the final keyword glossary text.

## Maintenance rule

After each meaningful live attempt, update only the rolling sections above:

1. replace **Current live status** with the latest retained report and exact
   first failure, or with the completed-combat summary;
2. move fixed failures into the resolved list and remove obsolete wording from
   **Open limitations and next checks**;
3. update the bound policy, fallback, or crosswalk identity whenever its config
   changes;
4. record verification counts only after rerunning the stated checks;
5. keep detailed historical narratives in Git and machine evidence in the
   session directory rather than accumulating an unbounded incident log here.
