from __future__ import annotations

from .state_schema import MonsterBehaviorState, MonsterState


class UnsupportedMonsterBehaviorError(ValueError):
    """Raised when behavior prior is requested outside the supported catalog."""


def _behavior(
    *,
    phase: str,
    previous_move_id: int | None,
    possible_next_move_ids: tuple[int, ...],
    selection: str,
    rule: str,
) -> MonsterBehaviorState:
    return MonsterBehaviorState(
        phase=phase,
        previous_move_id=previous_move_id,
        possible_next_move_ids=possible_next_move_ids,
        selection=selection,
        rule=rule,
    )


def build_monster_behavior(
    monster: MonsterState,
    *,
    previous_move_id: int | None,
    living_monster_count: int,
    combat_turn: int,
    ascension: int = 0,
    has_used_entangle: bool = False,
) -> MonsterBehaviorState:
    """Resolve public next-move possibilities without sampling future RNG."""

    if combat_turn <= 0:
        raise ValueError("combat_turn must be positive")

    current = monster.move_id
    monster_id = monster.monster_id

    if monster_id == "Apology Slime":
        raise UnsupportedMonsterBehaviorError(
            "BaseMod received an invalid case-sensitive fight encounter ID and "
            "the base game spawned its Apology Slime debug fallback. Use the exact "
            "encounter token offered by Tab completion."
        )

    if monster.is_gone:
        return _behavior(
            phase="gone",
            previous_move_id=previous_move_id,
            possible_next_move_ids=(),
            selection="none",
            rule="A gone monster takes no further actions.",
        )

    if monster_id == "BookOfStabbing":
        if current == 1 and previous_move_id == 1:
            next_moves = (2,)
        elif current == 2:
            next_moves = (1,)
        else:
            next_moves = (1, 2)
        return _behavior(
            phase="escalating_stabs",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule="Chooses Multi-Stab or Single Stab; a third consecutive Multi-Stab and consecutive Single Stabs are forbidden. Multi-Stab gains hits during combat.",
        )

    if monster_id == "GremlinLeader":
        gremlin_count = max(0, living_monster_count - 1)
        if gremlin_count == 0:
            next_moves = (2, 4)
        elif gremlin_count == 1:
            next_moves = (2, 3, 4)
        else:
            next_moves = (3, 4)
        next_moves = tuple(move for move in next_moves if move != current)
        return _behavior(
            phase=f"leader_with_{gremlin_count}_minions",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="stochastic",
            rule="Native choice depends on living Gremlin count: Rally fills open slots, Encourage buffs the group, and Stab attacks; immediate repeats are restricted.",
        )

    if monster_id == "Taskmaster":
        return _behavior(phase="scouring_whip_loop", previous_move_id=previous_move_id,
            possible_next_move_ids=(1,), selection="deterministic",
            rule="Repeats Scouring Whip while alive.")

    if monster_id == "SlimeBoss":
        next_by_move = {4: (1,), 1: (2,), 2: (4,), 3: ()}
        return _behavior(phase="splitting" if current == 3 else "goop_slam_cycle",
            previous_move_id=previous_move_id, possible_next_move_ids=next_by_move[current],
            selection="none" if current == 3 else "deterministic",
            rule="Cycles Goop Spray, Preparing, and Slam until damage at half HP or lower schedules Split into two large slimes.")

    if monster_id == "TheGuardian":
        next_by_move = {1: (2,), 2: (3,), 3: (4,), 4: (1,), 5: (6,), 6: (7,), 7: (4,)}
        defensive = current in {5, 6, 7}
        return _behavior(phase="defensive_mode" if defensive else "offensive_mode",
            previous_move_id=previous_move_id, possible_next_move_ids=next_by_move[current],
            selection="deterministic",
            rule="Follows its fixed cycle; depleting Mode Shift interrupts the offensive cycle with Defensive Mode, then Twin Slam returns to Whirlwind and restores a higher threshold.")

    if monster_id == "Hexaghost":
        next_by_move = {5: (1,), 1: (4,), 2: (4,), 3: (2,), 4: (2, 3, 6), 6: (4,)}
        return _behavior(phase="activate_divider" if current in {5, 1} else "inferno_cycle",
            previous_move_id=previous_move_id, possible_next_move_ids=next_by_move[current],
            selection="conditional" if current == 4 else "deterministic",
            rule="Activate fixes Divider damage from current player HP; afterward a fixed internal orb counter sequences Sear, Tackle, Inflame, and Inferno without exposing future RNG.")

    if monster_id == "BronzeAutomaton":
        next_by_move = {4: (1,), 1: (5,), 5: (1, 2), 2: ((5,) if ascension >= 19 else (3,)), 3: (1,)}
        return _behavior(phase="hyper_beam_cycle", previous_move_id=previous_move_id,
            possible_next_move_ids=next_by_move[current],
            selection="conditional" if current == 5 else "deterministic",
            rule="Spawns two Orbs, then alternates Flail and Boost toward Hyper Beam; Ascension 19+ skips the post-beam Stunned turn.")

    if monster_id == "BronzeOrb":
        next_moves = (1, 2, 3) if current == 1 else (1, 3) if current == 2 else (1, 2)
        return _behavior(phase="stasis_support", previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves, selection="stochastic",
            rule="Chooses Beam, one-time Stasis, or Support Beam subject to native repeat limits; the held Stasis card is public.")

    if monster_id == "TheCollector":
        if current == 5:
            next_moves = (2, 3)
        elif combat_turn == 3:
            next_moves = (4,)
        elif current == 3:
            next_moves = (2,)
        else:
            next_moves = (2, 3, 5)
        return _behavior(phase="summon_control", previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "conditional",
            rule="Opens with Spawn, uses Mega Debuff on its fourth monster turn, and otherwise attacks, buffs, or refills open Torch Head slots subject to repeat limits.")

    if monster_id == "TorchHead":
        return _behavior(phase="tackle_loop", previous_move_id=previous_move_id,
            possible_next_move_ids=(1,), selection="deterministic",
            rule="Repeats Tackle while alive.")

    if monster_id == "TheChamp":
        phase_two = monster.current_hp < monster.max_hp // 2 or current in {3, 7}
        if current == 7:
            next_moves = (3,)
        elif phase_two and current != 3 and previous_move_id != 3:
            next_moves = (3,)
        else:
            next_moves = (1, 2, 4, 5, 6) if not phase_two else (1, 2, 3, 4, 5)
        return _behavior(phase="execute_phase" if phase_two else "champion_phase",
            previous_move_id=previous_move_id, possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "conditional",
            rule="Below half HP, Anger removes debuffs and starts the Execute phase; otherwise native turn count, repeat limits, and limited Defensive Stance uses select the move.")

    if monster_id == "Cultist":
        return _behavior(
            phase="opening_buff" if current == 3 else "dark_strike_loop",
            previous_move_id=previous_move_id,
            possible_next_move_ids=(1,),
            selection="deterministic",
            rule="Opens with Incantation, then repeats Dark Strike.",
        )

    if monster_id == "JawWorm":
        if current == 1:
            next_moves = (2, 3)
            rule = "Chomp cannot occur immediately after Chomp."
        elif current == 2:
            next_moves = (1, 3)
            rule = "Bellow cannot occur immediately after Bellow."
        elif current == 3 and previous_move_id == 3:
            next_moves = (1, 2)
            rule = "A third consecutive Thrash is forbidden."
        else:
            next_moves = (1, 2, 3)
            rule = "Chooses Chomp, Bellow, or Thrash subject to repeat limits."
        return _behavior(
            phase="adaptive_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="stochastic",
            rule=rule,
        )

    if monster_id in {"FuzzyLouseDefensive", "FuzzyLouseNormal"}:
        special = 4
        if current == 3 and previous_move_id == 3:
            next_moves = (special,)
            selection = "deterministic"
            rule = "After two consecutive Bites, the non-attack move is forced."
        elif current == special and previous_move_id == special:
            next_moves = (3,)
            selection = "deterministic"
            rule = "After two consecutive non-attack moves, Bite is forced."
        else:
            next_moves = (3, special)
            selection = "stochastic"
            rule = "Chooses Bite or its non-attack move; neither may occur three times in a row."
        return _behavior(
            phase="bite_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection=selection,
            rule=rule,
        )

    if monster_id == "AcidSlime_S":
        return _behavior(
            phase="lick_tackle_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=((1,) if current == 4 else (4,)),
            selection="deterministic",
            rule="Alternates Lick and Tackle after the seeded opener.",
        )

    if monster_id == "SpikeSlime_S":
        return _behavior(
            phase="tackle_loop",
            previous_move_id=previous_move_id,
            possible_next_move_ids=(1,),
            selection="deterministic",
            rule="Repeats Tackle while alive.",
        )

    if monster_id == "AcidSlime_M":
        repeated = previous_move_id == current
        if current == 1 and repeated:
            next_moves = (2, 4)
        elif current == 2 and repeated:
            next_moves = (1, 4)
        elif current == 4 and (ascension >= 17 or repeated):
            next_moves = (1, 2)
        else:
            next_moves = (1, 2, 4)
        return _behavior(
            phase="three_move_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="stochastic",
            rule="Chooses Corrosive Spit, Tackle, or Lick subject to native repeat limits.",
        )

    if monster_id == "SpikeSlime_M":
        repeated = previous_move_id == current
        forbid_repeat = repeated or (ascension >= 17 and current == 4)
        next_moves = ((4,) if current == 1 else (1,)) if forbid_repeat else (1, 4)
        return _behavior(
            phase="flame_tackle_lick_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule="Chooses Flame Tackle or Lick subject to native repeat limits.",
        )

    if monster_id == "SlaverBlue":
        if current == 1 and previous_move_id == 1:
            next_moves = (4,)
        elif current == 4 and previous_move_id == 4:
            next_moves = (1,)
        else:
            next_moves = (1, 4)
        return _behavior(
            phase="stab_rake_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule="Chooses Stab or Rake; neither may occur three times consecutively.",
        )

    if monster_id == "SlaverRed":
        if not has_used_entangle:
            next_moves = (1, 2) if current == 3 and previous_move_id == 3 else (2, 3)
            rule = "Entangle remains available once; before it is used, Scrape follows ordinary rolls."
        elif current == 1 and previous_move_id == 1:
            next_moves = (3,)
            rule = "After two Stabs, Scrape is forced."
        elif current == 3 and previous_move_id == 3:
            next_moves = (1,)
            rule = "After two Scrapes, Stab is forced."
        else:
            next_moves = (1, 3)
            rule = "After Entangle has been used, chooses Stab or Scrape subject to repeat limits."
        return _behavior(
            phase="pre_entangle" if not has_used_entangle else "post_entangle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule=rule,
        )

    if monster_id == "FungiBeast":
        if current == 1 and previous_move_id == 1:
            next_moves = (2,)
        elif current == 2:
            next_moves = (1,)
        else:
            next_moves = (1, 2)
        return _behavior(
            phase="bite_grow_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule="Grow cannot repeat, and a third consecutive Bite is forbidden.",
        )

    if monster_id == "Looter":
        if current == 1 and combat_turn == 1:
            next_moves = (1,)
            rule = "The first Mug is followed by a second Mug."
        elif current == 1:
            next_moves = (2, 4)
            rule = "After the second Mug, chooses Lunge or Smoke Bomb."
        elif current == 4:
            next_moves = (2,)
            rule = "Lunge is followed by Smoke Bomb."
        elif current == 2:
            next_moves = (3,)
            rule = "Smoke Bomb is followed by Escape."
        else:
            next_moves = ()
            rule = "Escape removes the Looter from combat."
        return _behavior(
            phase="escape_sequence",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection=(
                "none" if not next_moves
                else "deterministic" if len(next_moves) == 1
                else "stochastic"
            ),
            rule=rule,
        )

    if monster_id == "GremlinFat":
        return _behavior(
            phase="smash_loop",
            previous_move_id=previous_move_id,
            possible_next_move_ids=(1,),
            selection="deterministic",
            rule="Repeats Smash while alive.",
        )

    if monster_id == "GremlinWarrior":
        return _behavior(
            phase="scratch_loop",
            previous_move_id=previous_move_id,
            possible_next_move_ids=(1,),
            selection="deterministic",
            rule="Repeats Scratch while alive.",
        )

    if monster_id == "GremlinThief":
        return _behavior(
            phase="puncture_loop",
            previous_move_id=previous_move_id,
            possible_next_move_ids=(1,),
            selection="deterministic",
            rule="Repeats Puncture while alive.",
        )

    if monster_id == "GremlinTsundere":
        if living_monster_count <= 1:
            next_moves = (2,)
            phase = "last_gremlin_attack"
            rule = "When no ally remains, Shield Bash is forced and then repeats."
        else:
            next_moves = (1,)
            phase = "ally_protection"
            rule = "Protects another random living Gremlin while an ally remains."
        return _behavior(
            phase=phase,
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic",
            rule=rule,
        )

    if monster_id == "GremlinWizard":
        if current == 1:
            next_moves = (2,)
            phase = "blast"
            selection = "deterministic"
            rule = "After Ultimate Blast, returns to Charging."
        else:
            next_moves = (2, 1)
            phase = "charging"
            selection = "conditional"
            rule = "Continues Charging until the charge completes, then uses Ultimate Blast."
        return _behavior(
            phase=phase,
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection=selection,
            rule=rule,
        )

    if monster_id == "GremlinNob":
        if current == 3:
            next_moves = (1, 2)
            rule = "Bellow occurs only as the opener; then Rush or Skull Bash follows."
        elif current == 1 and previous_move_id == 1:
            next_moves = (2,)
            rule = "After two consecutive Rushes, Skull Bash is forced."
        else:
            next_moves = (1, 2)
            rule = "Chooses Rush or Skull Bash; a third consecutive Rush is forbidden."
        return _behavior(
            phase="opening_bellow" if current == 3 else "attack_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule=rule,
        )

    if monster_id == "Lagavulin":
        asleep = any(power.power_id == "Asleep" for power in monster.powers)
        if current == 3 and not asleep:
            next_moves = (1,)
            selection = "deterministic"
            phase = "waking"
            rule = "Attack is forced after Lagavulin is awakened."
        elif current == 3 and combat_turn >= 3:
            next_moves = (1,)
            selection = "deterministic"
            phase = "sleep_timeout"
            rule = (
                "The natural sleep timeout is reached: after the current "
                "Sleep, Lagavulin's first Attack is on turn 4."
            )
        elif current == 3:
            next_moves = (3, 1)
            selection = "conditional"
            phase = "asleep"
            remaining = 4 - combat_turn
            rule = (
                "Unblocked Attack damage wakes it immediately. Otherwise it "
                "sleeps through turn 3 and first attacks on turn 4; "
                f"{remaining} Sleep turn(s), including the current turn, remain."
            )
        elif current == 1 and previous_move_id == 1:
            next_moves = (2,)
            selection = "deterministic"
            phase = "attack_cycle"
            rule = "After two consecutive Attacks, Siphon Soul is forced."
        else:
            next_moves = (1,)
            selection = "deterministic"
            phase = "attack_cycle"
            rule = "Siphon Soul is followed by Attack; otherwise Attack repeats twice."
        return _behavior(
            phase=phase,
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection=selection,
            rule=rule,
        )

    if monster_id == "Sentry":
        next_move = 2 if current == 1 else 1
        return _behavior(
            phase="alternating_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=(next_move,),
            selection="deterministic",
            rule="Alternates Beam and Bolt.",
        )

    if monster_id in {"AcidSlime_L", "SpikeSlime_L"}:
        if current == 3:
            return _behavior(
                phase="splitting",
                previous_move_id=previous_move_id,
                possible_next_move_ids=(),
                selection="none",
                rule="Splits into two medium slimes and leaves combat.",
            )
        if monster_id == "AcidSlime_L":
            forbidden = set()
            if current == 1 and previous_move_id == 1:
                forbidden.add(1)
            if current == 2:
                forbidden.add(2)
            if current == 4 and previous_move_id == 4:
                forbidden.add(4)
            next_moves = tuple(move for move in (1, 2, 4) if move not in forbidden)
        elif current == 1 and previous_move_id == 1:
            next_moves = (4,)
        elif current == 4 and (ascension >= 17 or previous_move_id == 4):
            next_moves = (1,)
        else:
            next_moves = (1, 4)
        return _behavior(
            phase="large_slime_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule="Chooses its native attacks/debuff subject to repeat limits; at half HP or lower it instead schedules Split.",
        )

    if monster_id == "Byrd":
        if current == 4:
            next_moves = (5,)
        elif current == 5:
            next_moves = (2,)
        elif current == 2:
            next_moves = (1, 3, 6)
        elif current == 1 and previous_move_id == 1:
            next_moves = (3, 6)
        else:
            next_moves = tuple(move for move in (1, 3, 6) if move != current)
        return _behavior(
            phase="grounded_cycle" if current in {2, 4, 5} else "flight_cycle",
            previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule="While flying, chooses Peck, Swoop, or Caw subject to repeat limits; losing Flight causes Stunned, Headbutt, then Fly.",
        )

    if monster_id == "Centurion":
        support_move = 2 if living_monster_count > 1 else 3
        if current == 1 and previous_move_id == 1:
            next_moves = (support_move,)
        elif current == support_move and previous_move_id == support_move:
            next_moves = (1,)
        else:
            next_moves = (1, support_move)
        return _behavior(phase="paired_cycle", previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves, selection="stochastic",
            rule="Uses Slash or protects Mystic; when alone, Fury replaces Defend.")

    if monster_id == "Mystic":
        return _behavior(phase="support_cycle", previous_move_id=previous_move_id,
            possible_next_move_ids=(1, 2, 3), selection="conditional",
            rule="Heals when either ally is sufficiently injured; otherwise attacks or buffs subject to repeat limits.")

    if monster_id == "Chosen":
        if current == 5 and previous_move_id is None and ascension < 17:
            next_moves = (4,)
        elif current == 4:
            next_moves = (2, 3)
        elif current in {2, 3}:
            next_moves = (1, 5)
        else:
            next_moves = (2, 3)
        return _behavior(phase="hex_attack_cycle", previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves, selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule="Uses its ascension-dependent Hex opener, then alternates strong attacks with Drain or Debilitate.")

    if monster_id == "Mugger":
        if current == 1 and combat_turn == 1:
            next_moves = (1,)
        elif current == 1:
            next_moves = (2, 4)
        elif current == 4:
            next_moves = (2,)
        elif current == 2:
            next_moves = (3,)
        else:
            next_moves = ()
        return _behavior(phase="escape_sequence", previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves,
            selection="none" if not next_moves else "deterministic" if len(next_moves) == 1 else "stochastic",
            rule="Mugs twice, then Lunges or uses Smoke Bomb; Smoke Bomb is followed by Escape.")

    if monster_id == "ShelledParasite":
        if current == 1:
            next_moves = (2, 3)
        elif current == 2 and previous_move_id == 2:
            next_moves = (1, 3)
        elif current == 3 and previous_move_id == 3:
            next_moves = (1, 2)
        else:
            next_moves = (1, 2, 3)
        return _behavior(phase="plated_cycle", previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves, selection="stochastic",
            rule="Chooses Fell, Double Strike, or Suck subject to native repeat limits.")

    if monster_id == "SnakePlant":
        if current == 1 and previous_move_id == 1:
            next_moves = (2,)
        elif current == 2 and (ascension < 17 or previous_move_id == 2):
            next_moves = (1,)
        else:
            next_moves = (1, 2)
        return _behavior(phase="malleable_cycle", previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves, selection="deterministic" if len(next_moves) == 1 else "stochastic",
            rule="Chooses Chomp or Enfeebling Spores subject to ascension-dependent repeat limits.")

    if monster_id == "Snecko":
        next_moves = (3,) if current == 2 and previous_move_id == 2 else (2, 3)
        return _behavior(phase="confusion_cycle", previous_move_id=previous_move_id,
            possible_next_move_ids=next_moves, selection="stochastic",
            rule="Opens with Perplexing Glare, then chooses Bite or Tail Whip; a third Bite is forbidden.")

    if monster_id == "SphericGuardian":
        next_by_move = {2: (4,), 4: (1,), 1: (3,), 3: (1,)}
        return _behavior(phase="fixed_cycle", previous_move_id=previous_move_id,
            possible_next_move_ids=next_by_move[current], selection="deterministic",
            rule="Follows Activate, Attack Debuff, Slam, Harden, Slam, Harden deterministically.")

    raise UnsupportedMonsterBehaviorError(
        f"No behavior prior for supported monster {monster_id!r}"
    )
