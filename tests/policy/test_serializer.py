from dataclasses import replace
import unittest

from sts1_llm_policy.env.action_builder import build_legal_actions
from sts1_llm_policy.env.mechanics_catalog import (
    SUPPORTED_MOVE_EFFECTS,
    SUPPORTED_POWER_EFFECT_TEMPLATES,
    UnsupportedObservationMechanicError,
)
from sts1_llm_policy.env.live_description_fallback import (
    SourceDescribedRelicState,
)
from sts1_llm_policy.env.observation_v5_catalog import upgrade_v4_observation
from sts1_llm_policy.env.serializer import (
    BEHAVIOR_OBSERVATION_SERIALIZER_VERSION,
    RELIC_OBSERVATION_SERIALIZER_VERSION,
    SEMANTIC_OBSERVATION_SERIALIZER_VERSION,
    serialize_observation,
    serialize_state,
)
from sts1_llm_policy.env.state_schema import (
    CanonicalState,
    CardState,
    CombatAccounting,
    CombatState,
    MonsterBehaviorState,
    MonsterState,
    PlayerState,
    PowerState,
    RelicState,
)


class SerializerTest(unittest.TestCase):
    def _single_monster_state(self, monster: MonsterState) -> CanonicalState:
        strike = CardState(
            card_id="Strike_R",
            name="Strike",
            card_type="ATTACK",
            cost=1,
            upgrades=0,
            description="ignored",
            exhausts=False,
            ethereal=False,
            has_target=True,
            is_playable=True,
            uuid="strike",
        )
        return CanonicalState(
            seed=5,
            character="IRONCLAD",
            ascension_level=20,
            act=2,
            floor=33,
            combat=CombatState(
                turn=1,
                player=PlayerState(80, 80, 0, 3),
                monsters=(monster,),
                hand=(strike,),
                draw_pile=(),
                discard_pile=(),
                exhaust_pile=(),
            ),
        )


    def test_future_mechanics_are_self_contained_on_first_exposure(self) -> None:
        self.assertIn(
            "whenever the player plays a Skill",
            SUPPORTED_MOVE_EFFECTS[("GremlinNob", 3)],
        )
        self.assertIn(
            "unplayable; Ethereal",
            SUPPORTED_MOVE_EFFECTS[("Sentry", 2)],
        )
        self.assertIn(
            "owner immediately gains",
            SUPPORTED_POWER_EFFECT_TEMPLATES["Enrage"],
        )


    def test_serializes_env_schema_and_legal_actions(self) -> None:
        card = CardState(
            card_id="Strike_R",
            name="Strike",
            card_type="ATTACK",
            cost=1,
            upgrades=0,
            description="Source-specific dynamic text must stay hidden.",
            exhausts=False,
            ethereal=False,
            has_target=True,
            is_playable=True,
            uuid="card-uuid",
        )
        defend = CardState(
            card_id="Defend_R",
            name="Defend",
            card_type="SKILL",
            cost=1,
            upgrades=0,
            description="Gain 5 Block.",
            exhausts=False,
            ethereal=False,
            has_target=False,
            is_playable=True,
            uuid="defend-uuid",
        )
        bash = CardState(
            card_id="Bash",
            name="Bash",
            card_type="ATTACK",
            cost=2,
            upgrades=0,
            description="Deal 8 damage and apply 2 Vulnerable.",
            exhausts=False,
            ethereal=False,
            has_target=True,
            is_playable=True,
            uuid="bash-uuid",
        )
        state = CanonicalState(
            seed=1,
            character="IRONCLAD",
            ascension_level=0,
            act=1,
            floor=1,
            relics=(
                RelicState(
                    relic_id="PEN_NIB",
                    name="Pen Nib",
                    counter=7,
                ),
            ),
            combat=CombatState(
                turn=1,
                player=PlayerState(
                    current_hp=72,
                    max_hp=80,
                    block=5,
                    energy=3,
                    powers=(
                        PowerState(
                            power_id="Strength",
                            name="Strength",
                            amount=2,
                        ),
                    ),
                ),
                monsters=(
                    MonsterState(
                        monster_id="Cultist",
                        name="Cultist",
                        current_hp=42,
                        max_hp=48,
                        block=0,
                        intent="ATTACK",
                        move_id=1,
                        move_hits=1,
                        move_base_damage=6,
                        move_adjusted_damage=6,
                        powers=(
                            PowerState(
                                power_id="Ritual",
                                name="Ritual",
                                amount=3,
                                damage=6,
                                misc=2,
                                just_applied=True,
                                card=card,
                            ),
                        ),
                        behavior=MonsterBehaviorState(
                            phase="dark_strike_loop",
                            previous_move_id=3,
                            possible_next_move_ids=(1,),
                            selection="deterministic",
                            rule=(
                                "Opens with Incantation, then repeats Dark Strike."
                            ),
                        ),
                    ),
                ),
                hand=(card,),
                draw_pile=(defend, bash, defend),
                discard_pile=(card,),
                exhaust_pile=(),
            ),
        )

        observation = serialize_observation(
            state,
            build_legal_actions(state),
        )

        self.assertIn("HP: 72/80", observation)
        self.assertIn(
            "POWERS: Strength(2): Attack damage is increased by 2 per hit.",
            observation,
        )
        self.assertIn(
            "[0] Cultist | HP 42/48 | BLOCK 0 | INTENT ATTACK"
            " | DAMAGE 6 x 1 | TOTAL 6"
            " | EFFECT Deal 6 base damage.",
            observation,
        )
        self.assertIn(
            "POWERS Ritual(3)"
            "[DAMAGE=6, MISC=2, JUST_APPLIED, CARD=Strike]: "
            "At the end of the owner's turn, gain 3 Strength.",
            observation,
        )
        self.assertIn(
            "ACTION_0: PLAY Strike | COST 1 | UPGRADE 0"
            " -> TARGET_0 | COPIES 1",
            observation,
        )
        self.assertIn(
            "Strike x1 | TYPE ATTACK | COST 1 | UPGRADE 0"
            " | TARGET ENEMY"
            " | EFFECT Deal 6 base damage.",
            observation,
        )
        self.assertNotIn("Source-specific dynamic text", observation)
        self.assertIn("DRAW: 3 | Bash x1, Defend x2", observation)
        self.assertIn("DISCARD: 1 | Strike x1", observation)
        self.assertIn("EXHAUST: 0 | EMPTY", observation)
        self.assertIn("ACTION_1: END_TURN", observation)
        self.assertNotIn("card-uuid", observation)
        self.assertNotIn("BEHAVIOR:", observation)

        behavior_observation = serialize_observation(
            state,
            build_legal_actions(state),
            version=BEHAVIOR_OBSERVATION_SERIALIZER_VERSION,
        )
        self.assertIn(
            "BEHAVIOR: PHASE dark_strike_loop | PREVIOUS Incantation",
            behavior_observation,
        )
        self.assertIn(
            "FOLLOWING_AFTER_CURRENT Dark Strike "
            "[ATTACK; Deal 6 base damage.]",
            behavior_observation,
        )
        self.assertIn("SELECTION DETERMINISTIC", behavior_observation)
        self.assertIn("COMBAT_OBJECTIVE:", behavior_observation)
        self.assertIn(
            "PRIMARY: Maximize the probability of winning this combat.",
            behavior_observation,
        )
        self.assertIn("VULNERABLE: Take 50% more damage", behavior_observation)
        self.assertNotIn("COMBAT_OBJECTIVE:", observation)

        relic_observation = serialize_observation(
            state,
            build_legal_actions(state),
            version=RELIC_OBSERVATION_SERIALIZER_VERSION,
        )
        self.assertIn("RELICS:", relic_observation)
        self.assertIn(
            "Pen Nib | COUNTER 7 | EFFECT Every 10th Attack deals double damage.",
            relic_observation,
        )
        self.assertIn("BEHAVIOR:", relic_observation)
        self.assertNotIn("RELICS:", behavior_observation)

        reordered_state = replace(
            state,
            combat=replace(
                state.combat,
                draw_pile=tuple(reversed(state.combat.draw_pile)),
            ),
        )
        self.assertEqual(
            serialize_state(state),
            serialize_state(reordered_state),
        )

        legacy_observation = serialize_observation(
            state,
            build_legal_actions(state),
            version="observation_v1",
        )
        self.assertIn("[0] Strike | COST 1", legacy_observation)
        self.assertIn("DRAW: 3", legacy_observation)
        self.assertNotIn("DAMAGE 6 x 1", legacy_observation)
        self.assertNotIn("EFFECT Deal 6 base damage", legacy_observation)

    def test_rejects_card_outside_frozen_p0_catalog(self) -> None:
        card = CardState(
            card_id="Headbutt",
            name="Headbutt",
            card_type="ATTACK",
            cost=0,
            upgrades=0,
            description="Deal 6 damage.",
            exhausts=False,
            ethereal=False,
            has_target=True,
            is_playable=True,
            uuid="anger-uuid",
        )
        state = CanonicalState(
            seed=1,
            character="IRONCLAD",
            ascension_level=0,
            act=1,
            floor=1,
            combat=CombatState(
                turn=1,
                player=PlayerState(80, 80, 0, 3),
                monsters=(
                    MonsterState(
                        monster_id="Cultist",
                        name="Cultist",
                        current_hp=48,
                        max_hp=48,
                        block=0,
                        intent="BUFF",
                        move_id=3,
                        move_hits=1,
                        move_base_damage=-1,
                        move_adjusted_damage=-1,
                        behavior=MonsterBehaviorState(
                            phase="opening_buff",
                            previous_move_id=None,
                            possible_next_move_ids=(1,),
                            selection="deterministic",
                            rule="Opens with Incantation, then repeats Dark Strike.",
                        ),
                    ),
                ),
                hand=(card,),
                draw_pile=(),
                discard_pile=(),
                exhaust_pile=(),
            ),
        )

        with self.assertRaisesRegex(
            UnsupportedObservationMechanicError,
            "Headbutt upgrade 0",
        ):
            serialize_observation(state, build_legal_actions(state))

        live_state = replace(
            state,
            relics=(
                SourceDescribedRelicState(
                    relic_id="LIVE_ONLY_RELIC",
                    name="Live Only Relic",
                    source_description="  Gain  1 Strength at combat start.  ",
                ),
            ),
        )
        observation = serialize_observation(
            live_state,
            build_legal_actions(live_state),
            version=SEMANTIC_OBSERVATION_SERIALIZER_VERSION,
            allow_source_descriptions=True,
        )

        self.assertIn("EFFECT Deal 6 damage.", observation)
        self.assertIn(
            "Live Only Relic | COUNTER NONE | EFFECT Gain 1 Strength at combat start.",
            observation,
        )

        missing_description = replace(card, description="  \n ")
        missing_state = replace(
            state,
            combat=replace(state.combat, hand=(missing_description,)),
        )
        with self.assertRaisesRegex(
            UnsupportedObservationMechanicError,
            "Headbutt upgrade 0",
        ):
            serialize_observation(
                missing_state,
                build_legal_actions(missing_state),
                version=SEMANTIC_OBSERVATION_SERIALIZER_VERSION,
                allow_source_descriptions=True,
            )

    def test_behavior_observation_adds_aggregate_accounting_and_keywords(self) -> None:
        carnage = CardState(
            card_id="Carnage",
            name="Carnage",
            card_type="ATTACK",
            cost=2,
            upgrades=1,
            description="ignored",
            exhausts=False,
            ethereal=True,
            has_target=True,
            is_playable=True,
            uuid="carnage",
        )
        battle_trance = CardState(
            card_id="Battle Trance",
            name="Battle Trance",
            card_type="SKILL",
            cost=0,
            upgrades=0,
            description="ignored",
            exhausts=False,
            ethereal=False,
            has_target=False,
            is_playable=True,
            uuid="battle-trance",
        )
        state = CanonicalState(
            seed=4,
            character="IRONCLAD",
            ascension_level=0,
            act=1,
            floor=1,
            combat=CombatState(
                turn=3,
                player=PlayerState(
                    current_hp=65,
                    max_hp=80,
                    block=0,
                    energy=3,
                    powers=(PowerState("Rage", "Rage", 5),),
                ),
                monsters=(
                    MonsterState(
                        monster_id="Cultist",
                        name="Cultist",
                        current_hp=30,
                        max_hp=48,
                        block=0,
                        intent="ATTACK",
                        move_id=1,
                        move_hits=1,
                        move_base_damage=6,
                        move_adjusted_damage=6,
                        behavior=MonsterBehaviorState(
                            phase="dark_strike_loop",
                            previous_move_id=1,
                            possible_next_move_ids=(1,),
                            selection="deterministic",
                            rule="Repeats Dark Strike.",
                        ),
                    ),
                ),
                hand=(carnage, battle_trance),
                draw_pile=(),
                discard_pile=(),
                exhaust_pile=(),
                accounting=CombatAccounting(
                    starting_hp=80,
                    enemy_damage_taken=12,
                    self_hp_loss=3,
                ),
            ),
        )

        observation = serialize_state(
            state,
            version=BEHAVIOR_OBSERVATION_SERIALIZER_VERSION,
        )

        self.assertIn("STARTING_HP: 80", observation)
        self.assertIn("CURRENT_HP: 65", observation)
        self.assertIn("TOTAL_HP_LOSS: 15", observation)
        self.assertIn("ENEMY_DAMAGE_TAKEN: 12", observation)
        self.assertIn("SELF_HP_LOSS: 3", observation)
        self.assertIn("ETHEREAL: If this card remains", observation)
        self.assertIn("EXHAUST: Remove this card", observation)
        self.assertIn("NO_DRAW: The player cannot draw", observation)
        self.assertIn("RAGE: Whenever the player plays", observation)
        self.assertEqual(observation.count("ETHEREAL:"), 1)

    def test_v5_closes_ritual_semantics_on_first_intent(self) -> None:
        state = self._single_monster_state(
            MonsterState(
                monster_id="Cultist",
                name="Cultist",
                current_hp=48,
                max_hp=48,
                block=0,
                intent="BUFF",
                move_id=3,
                move_hits=1,
                move_base_damage=-1,
                move_adjusted_damage=-1,
                behavior=MonsterBehaviorState(
                    phase="opening_buff",
                    previous_move_id=None,
                    possible_next_move_ids=(1,),
                    selection="deterministic",
                    rule="Opens with Incantation, then repeats Dark Strike.",
                ),
            )
        )

        v4 = serialize_state(state, version=RELIC_OBSERVATION_SERIALIZER_VERSION)
        v5 = serialize_state(state, version=SEMANTIC_OBSERVATION_SERIALIZER_VERSION)

        self.assertNotIn("RITUAL:", v4)
        self.assertIn("EFFECT Apply 3 Ritual.", v5)
        self.assertIn(
            "RITUAL: At the end of the owner's turn, gain Strength",
            v5,
        )
        self.assertIn("STRENGTH: Modifies Attack damage", v5)
        self.assertEqual(upgrade_v4_observation(v4), v5)

    def test_v5_explains_future_bronze_orb_without_realized_rng(self) -> None:
        state = self._single_monster_state(
            MonsterState(
                monster_id="BronzeAutomaton",
                name="Bronze Automaton",
                current_hp=300,
                max_hp=300,
                block=0,
                intent="ATTACK",
                move_id=1,
                move_hits=2,
                move_base_damage=8,
                move_adjusted_damage=8,
                behavior=MonsterBehaviorState(
                    phase="hyper_beam_cycle",
                    previous_move_id=None,
                    possible_next_move_ids=(4,),
                    selection="deterministic",
                    rule="Spawns two Orbs, then follows its fixed cycle.",
                ),
            )
        )

        observation = serialize_state(
            state,
            version=SEMANTIC_OBSERVATION_SERIALIZER_VERSION,
        )

        self.assertIn("FOLLOWING_AFTER_CURRENT Spawn Orbs", observation)
        self.assertIn("highest-rarity card", observation)
        self.assertIn("MINION: A summoned enemy", observation)
        self.assertIn("STASIS: A Bronze Orb takes", observation)
        self.assertNotIn("realized future", observation.casefold())

    def test_v5_status_glossary_is_transitively_closed(self) -> None:
        state = self._single_monster_state(
            MonsterState(
                monster_id="Chosen",
                name="Chosen",
                current_hp=95,
                max_hp=95,
                block=0,
                intent="DEBUFF",
                move_id=4,
                move_hits=1,
                move_base_damage=-1,
                move_adjusted_damage=-1,
                behavior=MonsterBehaviorState(
                    phase="hex_cycle",
                    previous_move_id=None,
                    possible_next_move_ids=(1,),
                    selection="deterministic",
                    rule="Uses its public move cycle.",
                ),
            )
        )

        observation = serialize_state(
            state,
            version=SEMANTIC_OBSERVATION_SERIALIZER_VERSION,
        )

        self.assertIn("HEX: Whenever the player plays a non-Attack", observation)
        self.assertIn("DAZED: A Status card that is UNPLAYABLE and ETHEREAL", observation)
        self.assertIn("ETHEREAL: If this card remains", observation)
        self.assertIn("EXHAUST: Remove this card", observation)
        self.assertIn("UNPLAYABLE: This card cannot be played", observation)


if __name__ == "__main__":
    unittest.main()
