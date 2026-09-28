from __future__ import annotations

import unittest

from sts1_llm_policy.env.monster_behavior import build_monster_behavior
from sts1_llm_policy.env.state_schema import MonsterState, PowerState


def _monster(
    monster_id: str,
    move_id: int,
    *,
    powers: tuple[PowerState, ...] = (),
    is_gone: bool = False,
) -> MonsterState:
    return MonsterState(
        monster_id=monster_id,
        name=monster_id,
        current_hp=0 if is_gone else 40,
        max_hp=40,
        block=0,
        intent="ATTACK",
        move_id=move_id,
        move_hits=1,
        move_base_damage=1,
        move_adjusted_damage=1,
        powers=powers,
        is_gone=is_gone,
    )


class MonsterBehaviorTest(unittest.TestCase):
    def test_repeat_limits_resolve_without_future_rng(self) -> None:
        jaw = build_monster_behavior(
            _monster("JawWorm", 3),
            previous_move_id=3,
            living_monster_count=1,
            combat_turn=1,
        )
        nob = build_monster_behavior(
            _monster("GremlinNob", 1),
            previous_move_id=1,
            living_monster_count=1,
            combat_turn=1,
        )

        self.assertEqual(jaw.possible_next_move_ids, (1, 2))
        self.assertEqual(jaw.selection, "stochastic")
        self.assertEqual(nob.possible_next_move_ids, (2,))
        self.assertEqual(nob.selection, "deterministic")

    def test_lagavulin_wake_state_changes_following_move(self) -> None:
        asleep = build_monster_behavior(
            _monster(
                "Lagavulin",
                3,
                powers=(PowerState("Asleep", "Asleep", 1),),
            ),
            previous_move_id=None,
            living_monster_count=1,
            combat_turn=1,
        )
        waking = build_monster_behavior(
            _monster("Lagavulin", 3),
            previous_move_id=None,
            living_monster_count=1,
            combat_turn=1,
        )
        timeout = build_monster_behavior(
            _monster(
                "Lagavulin",
                3,
                powers=(PowerState("Asleep", "Asleep", 1),),
            ),
            previous_move_id=3,
            living_monster_count=1,
            combat_turn=3,
        )

        self.assertEqual(asleep.phase, "asleep")
        self.assertEqual(asleep.possible_next_move_ids, (3, 1))
        self.assertIn("first attacks on turn 4", asleep.rule)
        self.assertEqual(waking.phase, "waking")
        self.assertEqual(waking.possible_next_move_ids, (1,))
        self.assertEqual(timeout.phase, "sleep_timeout")
        self.assertEqual(timeout.possible_next_move_ids, (1,))
        self.assertEqual(timeout.selection, "deterministic")

    def test_deterministic_cycles_and_multi_enemy_condition(self) -> None:
        sentry = build_monster_behavior(
            _monster("Sentry", 2),
            previous_move_id=1,
            living_monster_count=3,
            combat_turn=1,
        )
        shield_with_allies = build_monster_behavior(
            _monster("GremlinTsundere", 1),
            previous_move_id=1,
            living_monster_count=4,
            combat_turn=1,
        )
        shield_alone = build_monster_behavior(
            _monster("GremlinTsundere", 1),
            previous_move_id=1,
            living_monster_count=1,
            combat_turn=1,
        )

        self.assertEqual(sentry.possible_next_move_ids, (1,))
        self.assertEqual(shield_with_allies.possible_next_move_ids, (1,))
        self.assertEqual(shield_alone.possible_next_move_ids, (2,))

    def test_gone_monster_has_no_following_action(self) -> None:
        behavior = build_monster_behavior(
            _monster("Sentry", 1, is_gone=True),
            previous_move_id=2,
            living_monster_count=0,
            combat_turn=1,
        )

        self.assertEqual(behavior.selection, "none")
        self.assertEqual(behavior.possible_next_move_ids, ())

    def test_small_slime_native_repeat_limits_are_ascension_aware(self) -> None:
        acid = build_monster_behavior(
            _monster("AcidSlime_M", 4),
            previous_move_id=1,
            living_monster_count=2,
            combat_turn=1,
            ascension=17,
        )
        spike = build_monster_behavior(
            _monster("SpikeSlime_M", 4),
            previous_move_id=1,
            living_monster_count=2,
            combat_turn=1,
            ascension=17,
        )
        small = build_monster_behavior(
            _monster("AcidSlime_S", 4),
            previous_move_id=None,
            living_monster_count=2,
            combat_turn=1,
        )

        self.assertEqual(acid.possible_next_move_ids, (1, 2))
        self.assertEqual(spike.possible_next_move_ids, (1,))
        self.assertEqual(small.possible_next_move_ids, (1,))

    def test_slaver_fungi_and_looter_public_cycles(self) -> None:
        blue = build_monster_behavior(
            _monster("SlaverBlue", 1),
            previous_move_id=1,
            living_monster_count=1,
            combat_turn=2,
        )
        red_pre = build_monster_behavior(
            _monster("SlaverRed", 1),
            previous_move_id=None,
            living_monster_count=1,
            combat_turn=1,
            has_used_entangle=False,
        )
        red_post = build_monster_behavior(
            _monster("SlaverRed", 1),
            previous_move_id=1,
            living_monster_count=1,
            combat_turn=4,
            has_used_entangle=True,
        )
        fungi = build_monster_behavior(
            _monster("FungiBeast", 2),
            previous_move_id=1,
            living_monster_count=2,
            combat_turn=2,
        )
        looter = build_monster_behavior(
            _monster("Looter", 1),
            previous_move_id=1,
            living_monster_count=1,
            combat_turn=2,
        )

        self.assertEqual(blue.possible_next_move_ids, (4,))
        self.assertEqual(red_pre.possible_next_move_ids, (2, 3))
        self.assertEqual(red_post.possible_next_move_ids, (3,))
        self.assertEqual(fungi.possible_next_move_ids, (1,))
        self.assertEqual(looter.possible_next_move_ids, (2, 4))

    def test_act_two_and_large_slime_public_cycles(self) -> None:
        chosen_open = build_monster_behavior(
            _monster("Chosen", 5), previous_move_id=None,
            living_monster_count=1, combat_turn=1,
        )
        sphere = build_monster_behavior(
            _monster("SphericGuardian", 2), previous_move_id=None,
            living_monster_count=1, combat_turn=1,
        )
        acid = build_monster_behavior(
            _monster("AcidSlime_L", 2), previous_move_id=1,
            living_monster_count=1, combat_turn=2,
        )
        snecko = build_monster_behavior(
            _monster("Snecko", 2), previous_move_id=2,
            living_monster_count=1, combat_turn=3,
        )

        self.assertEqual(chosen_open.possible_next_move_ids, (4,))
        self.assertEqual(sphere.possible_next_move_ids, (4,))
        self.assertEqual(acid.possible_next_move_ids, (1, 4))
        self.assertEqual(snecko.possible_next_move_ids, (3,))

    def test_elite_and_boss_phase_cycles(self) -> None:
        leader = build_monster_behavior(
            _monster("GremlinLeader", 3), previous_move_id=4,
            living_monster_count=3, combat_turn=2,
        )
        guardian = build_monster_behavior(
            _monster("TheGuardian", 5), previous_move_id=4,
            living_monster_count=1, combat_turn=3,
        )
        automaton_a19 = build_monster_behavior(
            _monster("BronzeAutomaton", 2), previous_move_id=5,
            living_monster_count=3, combat_turn=6, ascension=19,
        )
        champ = build_monster_behavior(
            MonsterState(
                monster_id="TheChamp", name="The Champ",
                current_hp=100, max_hp=420, block=0, intent="BUFF",
                move_id=7, move_hits=1, move_base_damage=-1,
                move_adjusted_damage=-1,
            ),
            previous_move_id=1, living_monster_count=1, combat_turn=5,
        )

        self.assertEqual(leader.possible_next_move_ids, (4,))
        self.assertEqual(guardian.possible_next_move_ids, (6,))
        self.assertEqual(automaton_a19.possible_next_move_ids, (5,))
        self.assertEqual(champ.phase, "execute_phase")
        self.assertEqual(champ.possible_next_move_ids, (3,))


if __name__ == "__main__":
    unittest.main()
