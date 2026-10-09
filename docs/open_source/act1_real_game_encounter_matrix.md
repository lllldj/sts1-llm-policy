# Act 1 real-game encounter matrix

This is the maintained terminal-evidence matrix for Act 1 combat integration. It
tracks protocol and observation compatibility, not policy quality. A completed
row means that the current live interface controlled the encounter through its
terminal combat state without an adapter, protocol, or acknowledgement stop.

BaseMod's `fight` command is a debug injection: it creates the selected real
encounter in a synthetic next room, but it is not evidence that the encounter
was reached by the game's natural room roll. It requires an active run with a
valid map node. Encounter IDs are case-sensitive, so type the `fight` prefix and
select the exact token offered by Tab completion rather than retyping the table
entry:

```text
fight <encounter_id>
```

An invalid token (including a case mismatch such as `fight looter` instead of
`fight Looter`) silently makes the base game spawn its `Apology Slime` debug
fallback. That is not a real encounter. The runner intentionally stops before
acting and reports the invalid BaseMod encounter ID.

Status meanings:

- **live complete**: completed in a retained real-game session on the current
  acknowledgement/crosswalk chain;
- **fixed; retest**: a concrete live failure was fixed locally but has not yet
  passed a later live run;
- **partial**: at least one seeded composition completed, but the encounter's
  other possible compositions still need terminal tests;
- **retest**: an older run reached the encounter, but later interface changes
  still need an end-to-end confirmation;
- **untested**: statically represented by the current monster move, power, and
  behavior catalogs but not yet completed live.

## Closeout status (2026-09-05)

The latest retained Guardian trajectory closes its full mode-cycle retest.
Row statuses below describe the retained evidence: partial, retest, and untested
entries still lack terminal evidence for particular compositions or event variants.
Session directories are retained locally, outside the public source snapshot.

## Natural Act 1 encounter pool

| Tier | BaseMod console command | Expected encounter | Live status | Latest evidence / next check |
|---|---|---|---|---|
| Easy | `fight Cultist` | Cultist | live complete | Completed after the acknowledgement fix. |
| Easy | `fight Jaw_Worm` | Jaw Worm | live complete | Completed after the acknowledgement fix. |
| Easy | `fight 2_Louse` | Two seeded Louses | live complete | Completed repeatedly. |
| Easy | `fight Small_Slimes` | Acid Slime (M) + Spike Slime (S), or Acid Slime (S) + Spike Slime (M) | retest | Reached in older runs; one run exposed the duplicate-state race. |
| Hallway | `fight Blue_Slaver` | Blue Slaver | live complete | Reported as `Slaver`; raw monster ID confirmed `SlaverBlue`. |
| Hallway | `fight Gremlin_Gang` | Four Gremlins from the Act 1 pool | partial | Completed once as Sneaky + Fat + Sneaky + Wizard in session `real-game-20260904T193054Z-7afe20f5`; other seeded compositions remain. |
| Hallway | `fight Looter` | Looter | live complete | Completed in session `real-game-20260904T144707Z-ac147216`. |
| Hallway | `fight Large_Slime` | One large Acid or Spike Slime | live complete | Large Acid Slime completed through Split in session `real-game-20260904T192316Z-2da1e5a1`; large Spike Slime completed through Split in 18 decisions in session `real-game-20260904T194859Z-b0fe89bb`. |
| Hallway | `fight Lots_of_Slimes` | Five small slimes | live complete | Five mixed small slimes completed in 17 decisions in session `real-game-20260904T193657Z-a9b0946c`. |
| Hallway | `fight Exordium_Thugs` | One Louse or medium Slime + one Cultist, Slaver, or Looter | partial | Acid Slime (M) + Looter completed in session `real-game-20260904T193657Z-a9b0946c`; other seeded compositions remain. |
| Hallway | `fight Exordium_Wildlife` | One Fungi Beast or Jaw Worm + one Louse or medium Slime | partial | Completed as `Fungi Beast + Louse` and `Jaw Worm + Acid Slime (M)`; other seeded compositions remain. |
| Hallway | `fight Red_Slaver` | Red Slaver | live complete | Completed through native Entangle in 14 decisions in session `real-game-20260904T194859Z-b0fe89bb`; Scrape was exercised before the earlier safe stop in session `real-game-20260904T193657Z-a9b0946c`. |
| Hallway | `fight 3_Louse` | Three seeded Louses | live complete | Completed after crosswalk v3. |
| Hallway | `fight 2_Fungi_Beasts` | Two Fungi Beasts | live complete | Completed after the acknowledgement fix. |
| Elite | `fight Gremlin_Nob` | Gremlin Nob | live complete | Completed in 16 decisions with native `Anger`/canonical `Enrage` and Strength observed in session `real-game-20260904T194859Z-b0fe89bb`. |
| Elite | `fight Lagavulin` | Sleeping Lagavulin | live complete | Completed in 18 decisions through sleep, wake-up STUN projection, attacks, and Siphon Soul in session `real-game-20260904T194859Z-b0fe89bb`. |
| Elite | `fight 3_Sentries` | Three Sentries | live complete | Completed in session `real-game-20260904T144707Z-ac147216` after crosswalk v3. |
| Boss | `fight The_Guardian` | The Guardian | live complete | Session `real-game-20260905T032357Z-0c7f6b75`: 33 decisions, 72 to 32 HP, terminal victory; defensive transition, Twin Slam, and return to offensive Whirlwind exercised under crosswalk v7. |
| Boss | `fight Hexaghost` | Hexaghost | live complete | Completed in 37 decisions through its opening and attack cycle in session `real-game-20260904T194859Z-b0fe89bb`. |
| Boss | `fight Slime_Boss` | Slime Boss | live complete | Completed in 30 decisions through Split and the spawned large/medium slimes in session `real-game-20260904T194859Z-b0fe89bb`. |

## Act 1 event combat variants

| Source | BaseMod console command | Expected encounter | Live status | Latest evidence / next check |
|---|---|---|---|---|
| Dead Adventurer | `fight Lagavulin_Event` | Event Lagavulin | untested | The event can otherwise select a normal Act 1 elite; this row covers its distinct awake Lagavulin encounter ID. |
| Mushrooms | `fight The_Mushroom_Lair` | Three Fungi Beasts | untested | Exercise three targets and all Spore Cloud death triggers. |

The Dead Adventurer's Gremlin Nob and Three Sentries outcomes use the normal
elite encounter implementations and are covered by those rows. This table is
limited to base-game Act 1 combat; Daily modifiers, custom characters,
mod-added enemies, and cross-act console injections are outside it.

## Recording rule

After each terminal test, record only the newest meaningful result in this
table and keep the session directory under `outputs/real-game-sessions/` as the
machine evidence. A strategy mistake does not fail an integration row. An
unknown power/move/behavior, invalid command, duplicate action, failure to
resume, or missing terminal transition does. Debug-injected completion is
interface evidence only; natural encounter evidence is called out explicitly.
