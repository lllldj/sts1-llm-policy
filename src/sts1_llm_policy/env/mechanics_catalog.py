from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .state_schema import CardState, MonsterState, PowerState


class UnsupportedObservationMechanicError(ValueError):
    """Raised when a model-facing rule is outside the supported catalog."""


SUPPORTED_CARD_EFFECTS: Mapping[tuple[str, int], str] = MappingProxyType(
    {
        ("AscendersBane", 0): "UNPLAYABLE. ETHEREAL.",
        ("Strike_R", 0): "Deal 6 base damage.",
        ("Strike_R", 1): "Deal 9 base damage.",
        ("Defend_R", 0): "Gain 5 Block.",
        ("Defend_R", 1): "Gain 8 Block.",
        ("Bash", 0): "Deal 8 base damage and apply 2 Vulnerable.",
        ("Bash", 1): "Deal 10 base damage and apply 3 Vulnerable.",
        ("Anger", 0): "Deal 6 base damage. Add an unupgraded copy of Anger to the discard pile.",
        ("Anger", 1): "Deal 8 base damage. Add an upgraded copy of Anger to the discard pile.",
        ("Body Slam", 0): "Deal base damage equal to current Block. Costs 1 Energy.",
        ("Body Slam", 1): "Deal base damage equal to current Block. Costs 0 Energy.",
        ("Clash", 0): "Can only be played if every card in hand is an Attack. Deal 14 base damage.",
        ("Clash", 1): "Can only be played if every card in hand is an Attack. Deal 18 base damage.",
        ("Cleave", 0): "Deal 8 base damage to all enemies.",
        ("Cleave", 1): "Deal 11 base damage to all enemies.",
        ("Carnage", 0): "Deal 20 base damage. ETHEREAL.",
        ("Carnage", 1): "Deal 28 base damage. ETHEREAL.",
        ("Clothesline", 0): "Deal 12 base damage and apply 2 Weak.",
        ("Clothesline", 1): "Deal 14 base damage and apply 3 Weak.",
        ("Iron Wave", 0): "Gain 5 Block. Deal 5 base damage.",
        ("Iron Wave", 1): "Gain 7 Block. Deal 7 base damage.",
        ("Pommel Strike", 0): "Deal 9 base damage. Draw 1 card.",
        ("Pommel Strike", 1): "Deal 10 base damage. Draw 2 cards.",
        ("Thunderclap", 0): "Deal 4 base damage to all enemies and apply 1 Vulnerable to all enemies.",
        ("Thunderclap", 1): "Deal 7 base damage to all enemies and apply 1 Vulnerable to all enemies.",
        ("Uppercut", 0): "Deal 13 base damage and apply 1 Weak and 1 Vulnerable.",
        ("Uppercut", 1): "Deal 13 base damage and apply 2 Weak and 2 Vulnerable.",
        ("Whirlwind", 0): "Spend X Energy. Deal 5 base damage to all enemies X times.",
        ("Whirlwind", 1): "Spend X Energy. Deal 8 base damage to all enemies X times.",
        ("Shrug It Off", 0): "Gain 8 Block. Draw 1 card.",
        ("Shrug It Off", 1): "Gain 11 Block. Draw 1 card.",
        ("Flame Barrier", 0): "Gain 12 Block. Whenever attacked this turn, deal 4 damage back.",
        ("Flame Barrier", 1): "Gain 16 Block. Whenever attacked this turn, deal 6 damage back.",
        ("Ghostly Armor", 0): "Gain 10 Block. ETHEREAL.",
        ("Ghostly Armor", 1): "Gain 13 Block. ETHEREAL.",
        ("Havoc", 0): "Play the top card of the draw pile and EXHAUST it. Costs 1 Energy. The draw-pile order is hidden.",
        ("Havoc", 1): "Play the top card of the draw pile and EXHAUST it. Costs 0 Energy. The draw-pile order is hidden.",
        ("Heavy Blade", 0): "Deal 14 base damage. Strength affects this Attack 3 times.",
        ("Heavy Blade", 1): "Deal 14 base damage. Strength affects this Attack 5 times.",
        ("Power Through", 0): "Add 2 Wounds to your hand. Gain 15 Block.",
        ("Power Through", 1): "Add 2 Wounds to your hand. Gain 20 Block.",
        ("Perfected Strike", 0): "Deal 6 base damage plus 2 for every card containing Strike in all combat piles and hand.",
        ("Perfected Strike", 1): "Deal 6 base damage plus 3 for every card containing Strike in all combat piles and hand.",
        ("Battle Trance", 0): "Draw 3 cards. Apply NO_DRAW for the rest of this turn.",
        ("Battle Trance", 1): "Draw 4 cards. Apply NO_DRAW for the rest of this turn.",
        ("Bloodletting", 0): "Lose 3 HP. Gain 2 Energy.",
        ("Bloodletting", 1): "Lose 3 HP. Gain 3 Energy.",
        ("Blood for Blood", 0): "Deal 18 base damage. Starts at 4 Energy; each HP loss this combat lowers its cost by 1, to a minimum of 0.",
        ("Blood for Blood", 1): "Deal 22 base damage. Starts at 3 Energy; each HP loss this combat lowers its cost by 1, to a minimum of 0.",
        ("Disarm", 0): "Apply -2 Strength to one enemy. EXHAUST.",
        ("Disarm", 1): "Apply -3 Strength to one enemy. EXHAUST.",
        ("Dropkick", 0): "Deal 5 base damage. If the target has Vulnerable, gain 1 Energy and draw 1 card.",
        ("Dropkick", 1): "Deal 8 base damage. If the target has Vulnerable, gain 1 Energy and draw 1 card.",
        ("Entrench", 0): "Double current Block. Costs 2 Energy.",
        ("Entrench", 1): "Double current Block. Costs 1 Energy.",
        ("Fire Breathing", 0): "Whenever a Status or Curse is drawn, deal 6 damage to all enemies.",
        ("Fire Breathing", 1): "Whenever a Status or Curse is drawn, deal 10 damage to all enemies.",
        ("Hemokinesis", 0): "Lose 2 HP. Deal 15 base damage.",
        ("Hemokinesis", 1): "Lose 2 HP. Deal 20 base damage.",
        ("Intimidate", 0): "Apply 1 Weak to all enemies. EXHAUST.",
        ("Intimidate", 1): "Apply 2 Weak to all enemies. EXHAUST.",
        ("Pummel", 0): "Deal 2 base damage 4 times. EXHAUST.",
        ("Pummel", 1): "Deal 2 base damage 5 times. EXHAUST.",
        ("Reckless Charge", 0): "Deal 7 base damage. Shuffle 1 Dazed into the draw pile.",
        ("Reckless Charge", 1): "Deal 10 base damage. Shuffle 1 Dazed into the draw pile.",
        ("Sentinel", 0): "Gain 5 Block. If this card is EXHAUSTED, gain 2 Energy.",
        ("Sentinel", 1): "Gain 8 Block. If this card is EXHAUSTED, gain 3 Energy.",
        ("Sever Soul", 0): "EXHAUST every non-Attack card in hand. Deal 16 base damage.",
        ("Sever Soul", 1): "EXHAUST every non-Attack card in hand. Deal 22 base damage.",
        (
            "Rage",
            0,
        ): "Whenever you play an Attack this turn, gain 3 Block.",
        (
            "Rage",
            1,
        ): "Whenever you play an Attack this turn, gain 5 Block.",
        ("Second Wind", 0): "EXHAUST every non-Attack card in hand and gain 5 Block for each.",
        ("Second Wind", 1): "EXHAUST every non-Attack card in hand and gain 7 Block for each.",
        ("Shockwave", 0): "Apply 3 Weak and 3 Vulnerable to all enemies. EXHAUST.",
        ("Shockwave", 1): "Apply 5 Weak and 5 Vulnerable to all enemies. EXHAUST.",
        ("Spot Weakness", 0): "If the targeted enemy intends to attack, gain 3 Strength.",
        ("Spot Weakness", 1): "If the targeted enemy intends to attack, gain 4 Strength.",
        ("Flex", 0): "Gain 2 Strength this turn.",
        ("Flex", 1): "Gain 4 Strength this turn.",
        ("Combust", 0): "At end of turn, lose 1 HP and deal 5 damage to all enemies.",
        ("Combust", 1): "At end of turn, lose 1 HP and deal 7 damage to all enemies.",
        ("Evolve", 0): "Whenever a Status is drawn, draw 1 additional card.",
        ("Evolve", 1): "Whenever a Status is drawn, draw 2 additional cards.",
        ("Feel No Pain", 0): "Whenever a card is EXHAUSTED, gain 3 Block.",
        ("Feel No Pain", 1): "Whenever a card is EXHAUSTED, gain 4 Block.",
        ("Inflame", 0): "Gain 2 Strength.",
        ("Inflame", 1): "Gain 3 Strength.",
        ("Metallicize", 0): "At end of turn, gain 3 Block.",
        ("Metallicize", 1): "At end of turn, gain 4 Block.",
        ("Sword Boomerang", 0): "Deal 3 base damage to a random enemy 3 times.",
        ("Sword Boomerang", 1): "Deal 3 base damage to a random enemy 4 times.",
        ("Twin Strike", 0): "Deal 5 base damage twice.",
        ("Twin Strike", 1): "Deal 7 base damage twice.",
        ("Wild Strike", 0): "Deal 12 base damage. Shuffle a Wound into the draw pile.",
        ("Wild Strike", 1): "Deal 17 base damage. Shuffle a Wound into the draw pile.",
        ("Dazed", 0): "UNPLAYABLE. ETHEREAL.",
        ("Slimed", 0): "EXHAUST.",
        ("Wound", 0): "UNPLAYABLE.",
        ("Burn", 0): "UNPLAYABLE. At end of turn, take 2 damage.",
        ("Burn", 1): "UNPLAYABLE. At end of turn, take 4 damage.",
        ("Rupture", 0): "Whenever you lose HP from a card, gain 1 Strength.",
        ("Rupture", 1): "Whenever you lose HP from a card, gain 2 Strength.",
        ("Searing Blow", 0): "Deal 12 base damage. Can be upgraded any number of times.",
        ("Searing Blow", 1): "Deal 16 base damage. Can be upgraded any number of times.",
        ("Seeing Red", 0): "Gain 2 Energy. EXHAUST. Costs 1 Energy.",
        ("Seeing Red", 1): "Gain 2 Energy. EXHAUST. Costs 0 Energy.",
        ("Dark Embrace", 0): "Whenever a card is EXHAUSTED, draw 1 card. Costs 2 Energy.",
        ("Dark Embrace", 1): "Whenever a card is EXHAUSTED, draw 1 card. Costs 1 Energy.",
        ("Rampage", 0): "Deal 8 base damage plus this card's current combat bonus. After use, increase that bonus by 5.",
        ("Rampage", 1): "Deal 8 base damage plus this card's current combat bonus. After use, increase that bonus by 8.",
        ("Double Tap", 0): "The next Attack is played twice this turn. EXHAUST.",
        ("Double Tap", 1): "The next 2 Attacks are played twice this turn. EXHAUST.",
        ("Demon Form", 0): "At the start of each turn, gain 2 Strength.",
        ("Demon Form", 1): "At the start of each turn, gain 3 Strength.",
        ("Bludgeon", 0): "Deal 32 base damage.",
        ("Bludgeon", 1): "Deal 42 base damage.",
        ("Feed", 0): "Deal 10 base damage. If fatal, gain 3 Max HP. EXHAUST.",
        ("Feed", 1): "Deal 12 base damage. If fatal, gain 4 Max HP. EXHAUST.",
        ("Limit Break", 0): "Double current Strength. EXHAUST.",
        ("Limit Break", 1): "Double current Strength.",
        ("Corruption", 0): "Skills cost 0 Energy and EXHAUST when played. Costs 3 Energy.",
        ("Corruption", 1): "Skills cost 0 Energy and EXHAUST when played. Costs 2 Energy.",
        ("Barricade", 0): "Block is not removed at the start of your turn. Costs 3 Energy.",
        ("Barricade", 1): "Block is not removed at the start of your turn. Costs 2 Energy.",
        ("Fiend Fire", 0): "EXHAUST every card in hand. Deal 7 base damage once for each card EXHAUSTED. EXHAUST.",
        ("Fiend Fire", 1): "EXHAUST every card in hand. Deal 10 base damage once for each card EXHAUSTED. EXHAUST.",
        ("Berserk", 0): "Gain 2 Vulnerable. Gain 1 additional Energy each turn.",
        ("Berserk", 1): "Gain 1 Vulnerable. Gain 1 additional Energy each turn.",
        ("Impervious", 0): "Gain 30 Block. EXHAUST.",
        ("Impervious", 1): "Gain 40 Block. EXHAUST.",
        ("Juggernaut", 0): "Whenever you gain Block, deal 5 damage to a random enemy.",
        ("Juggernaut", 1): "Whenever you gain Block, deal 7 damage to a random enemy.",
        ("Brutality", 0): "At the start of each turn, lose 1 HP and draw 1 card.",
        ("Brutality", 1): "INNATE. At the start of each turn, lose 1 HP and draw 1 card.",
        ("Reaper", 0): "Deal 4 damage to all enemies and heal HP equal to unblocked damage dealt. EXHAUST.",
        ("Reaper", 1): "Deal 5 damage to all enemies and heal HP equal to unblocked damage dealt. EXHAUST.",
        ("Offering", 0): "Lose 6 HP. Gain 2 Energy and draw 3 cards. EXHAUST.",
        ("Offering", 1): "Lose 6 HP. Gain 2 Energy and draw 5 cards. EXHAUST.",
        ("Immolate", 0): "Deal 21 base damage to all enemies. Add 1 Burn to the discard pile.",
        ("Immolate", 1): "Deal 28 base damage to all enemies. Add 1 Burn to the discard pile.",
    }
)

SUPPORTED_RELIC_EFFECTS: Mapping[str, str] = MappingProxyType(
    {
        "BURNING_BLOOD": "At combat end, heal 6 HP.",
        "VAJRA": "At combat start, gain 1 Strength.",
        "ANCHOR": "At combat start, gain 10 Block.",
        "BAG_OF_MARBLES": "At combat start, apply 1 Vulnerable to all enemies.",
        "BRONZE_SCALES": "Whenever attacked, deal 3 damage back.",
        "LANTERN": "Gain 1 additional Energy on the first turn of combat.",
        "ODDLY_SMOOTH_STONE": "At combat start, gain 1 Dexterity.",
        "ORICHALCUM": "At end of turn, if Block is 0, gain 6 Block.",
        "AKABEKO": "The first Attack each combat deals 8 additional damage.",
        "PRESERVED_INSECT": "Enemies in Elite combats start with 25% less HP.",
        "THE_BOOT": "Unblocked Attack damage from 1 through 4 is increased to 5.",
        "STRIKE_DUMMY": "Cards containing Strike deal 3 additional damage.",
        "PAPER_PHROG": "Vulnerable enemies take 75% more Attack damage instead of 50%.",
        "CHAMPION_BELT": "Applying Vulnerable also applies 1 Weak.",
        "CHEMICAL_X": "X-cost card effects execute 2 additional times.",
        "TUNGSTEN_ROD": "Whenever HP would be lost, lose 1 less HP.",
        "TORII": "Unblocked Attack damage of 5 or less is reduced to 1.",
        "RED_SKULL": "While HP is at or below 50%, gain 3 Strength.",
        "CENTENNIAL_PUZZLE": "The first time HP is lost each combat, draw 3 cards.",
        "GREMLIN_HORN": "Whenever an enemy dies, gain 1 Energy and draw 1 card.",
        "HAPPY_FLOWER": "Every 3 turns, gain 1 Energy.",
        "NUNCHAKU": "Every 10 Attacks played, gain 1 Energy.",
        "PEN_NIB": "Every 10th Attack deals double damage.",
        "ART_OF_WAR": "If no Attack was played last turn, gain 1 Energy this turn.",
        "INK_BOTTLE": "Every 10 cards played, draw 1 card.",
        "KUNAI": "Every 3 Attacks played in one turn, gain 1 Dexterity.",
        "SHURIKEN": "Every 3 Attacks played in one turn, gain 1 Strength.",
        "BUSTED_CROWN": "Gain 1 Energy each turn; card rewards contain 2 fewer cards.",
        "PHILOSOPHERS_STONE": "Gain 1 Energy each turn; enemies gain 1 Strength.",
        "SLAVERS_COLLAR": "Gain 1 Energy each turn during Elite and Boss combats.",
        "VELVET_CHOKER": "Gain 1 Energy each turn; at most 6 cards may be played per turn.",
        "MARK_OF_PAIN": "Gain 1 Energy each turn; shuffle 2 Wounds into the draw pile at combat start.",
    }
)

SUPPORTED_KEYWORDS: Mapping[str, str] = MappingProxyType(
    {
        "ETHEREAL": (
            "If this card remains in hand at end of turn, Exhaust it."
        ),
        "EXHAUST": "Remove this card for the rest of this combat.",
        "NO_DRAW": "The player cannot draw additional cards this turn.",
        "RAGE": (
            "Whenever the player plays an Attack this turn, gain the shown "
            "amount of Block."
        ),
        "UNPLAYABLE": "This card cannot be played.",
        "VULNERABLE": "Take 50% more damage from Attacks.",
        "WEAK": "Attacks deal 25% less damage.",
    }
)

SUPPORTED_MOVE_EFFECTS: Mapping[tuple[str, int], str] = MappingProxyType(
    {
        ("SlaverBlue", 1): "Deal 12 base damage (13 at Ascension 2+).",
        ("SlaverBlue", 4): "Deal 7 base damage (8 at Ascension 2+) and apply 1 Weak (2 at Ascension 17+).",
        ("SlaverRed", 1): "Deal 13 base damage (14 at Ascension 2+).",
        ("SlaverRed", 2): "Apply 1 Entangled; Attacks cannot be played this turn.",
        ("SlaverRed", 3): "Deal 8 base damage (9 at Ascension 2+) and apply 1 Vulnerable (2 at Ascension 17+).",
        ("FungiBeast", 1): "Deal 6 base damage.",
        ("FungiBeast", 2): "Gain 3 Strength (4 at Ascension 2+, 5 at Ascension 17+).",
        ("Looter", 1): "Deal 10 base damage (11 at Ascension 2+) and steal gold.",
        ("Looter", 2): "Gain 6 Block and prepare to Escape.",
        ("Looter", 3): "Escape from combat.",
        ("Looter", 4): "Deal 12 base damage (14 at Ascension 2+) and steal gold, then prepare Smoke Bomb.",
        ("AcidSlime_M", 1): "Deal 7 base damage (8 at Ascension 2+) and add 1 Slimed to the discard pile.",
        ("AcidSlime_M", 2): "Deal 10 base damage (12 at Ascension 2+).",
        ("AcidSlime_M", 4): "Apply 1 Weak.",
        ("AcidSlime_S", 1): "Deal 3 base damage (4 at Ascension 2+).",
        ("AcidSlime_S", 4): "Apply 1 Weak.",
        ("Cultist", 1): "Deal 6 base damage.",
        ("Cultist", 3): "Apply 3 Ritual.",
        (
            "GremlinFat",
            1,
        ): "Deal 4 base damage and apply 1 Weak.",
        ("GremlinNob", 1): "Deal 14 base damage.",
        ("GremlinNob", 2): "Deal 6 base damage and apply 2 Vulnerable.",
        (
            "GremlinNob",
            3,
        ): (
            "Gain Enrage 2: whenever the player plays a Skill, Gremlin Nob "
            "immediately gains 2 Strength."
        ),
        (
            "GremlinWizard",
            1,
        ): "Deal 25 base damage.",
        (
            "GremlinWizard",
            2,
        ): "Advance the charge toward Ultimate Blast.",
        ("JawWorm", 1): "Deal 11 base damage.",
        ("JawWorm", 2): "Gain 6 Block and 3 Strength.",
        ("JawWorm", 3): "Deal 7 base damage and gain 5 Block.",
        ("Lagavulin", 1): "Deal 18 base damage.",
        (
            "Lagavulin",
            2,
        ): "Reduce the player's Strength and Dexterity by 1.",
        ("Lagavulin", 3): "Do nothing while asleep.",
        ("GremlinWarrior", 1): "Deal 4 base damage.",
        (
            "FuzzyLouseDefensive",
            3,
        ): "Deal the encounter-seeded base damage.",
        ("FuzzyLouseDefensive", 4): "Apply 2 Weak.",
        (
            "FuzzyLouseNormal",
            3,
        ): "Deal the encounter-seeded base damage.",
        ("FuzzyLouseNormal", 4): "Gain 3 Strength.",
        ("Sentry", 1): "Deal 9 base damage.",
        (
            "Sentry",
            2,
        ): (
            "Add 2 Dazed (unplayable; Ethereal: exhaust at end of turn if "
            "held) to the discard pile."
        ),
        (
            "GremlinTsundere",
            1,
        ): "Give 7 Block to another random living Gremlin.",
        ("GremlinTsundere", 2): "Deal 6 base damage.",
        ("GremlinThief", 1): "Deal 9 base damage.",
        ("SpikeSlime_M", 1): "Deal 8 base damage (10 at Ascension 2+) and add 1 Slimed to the discard pile.",
        ("SpikeSlime_M", 4): "Apply 1 Frail.",
        ("SpikeSlime_S", 1): "Deal 5 base damage (6 at Ascension 2+).",
        ("AcidSlime_L", 1): "Deal 11 base damage (12 at Ascension 2+) and add 2 Slimed to the discard pile.",
        ("AcidSlime_L", 2): "Deal 16 base damage (18 at Ascension 2+).",
        ("AcidSlime_L", 3): "Split into 2 Acid Slimes (M) at current HP.",
        ("AcidSlime_L", 4): "Apply 2 Weak.",
        ("SpikeSlime_L", 1): "Deal 16 base damage (18 at Ascension 2+) and add 2 Slimed to the discard pile.",
        ("SpikeSlime_L", 3): "Split into 2 Spike Slimes (M) at current HP.",
        ("SpikeSlime_L", 4): "Apply 2 Frail (3 at Ascension 17+).",
        ("Byrd", 1): "Deal 1 base damage 5 times (6 times at Ascension 2+).",
        ("Byrd", 2): "Regain 3 Flight (4 at Ascension 17+).",
        ("Byrd", 3): "Deal 12 base damage (14 at Ascension 2+).",
        ("Byrd", 4): "Do nothing while Stunned.",
        ("Byrd", 5): "Deal 3 base damage, then prepare to Fly.",
        ("Byrd", 6): "Gain 1 Strength.",
        ("Centurion", 1): "Deal 12 base damage (14 at Ascension 2+).",
        ("Centurion", 2): "Give the Mystic 15 Block (20 at Ascension 17+).",
        ("Centurion", 3): "Deal 6 base damage 3 times (7 at Ascension 2+).",
        ("Chosen", 1): "Deal 18 base damage (21 at Ascension 2+).",
        ("Chosen", 2): "Apply 3 Weak and gain 3 Strength.",
        ("Chosen", 3): "Deal 10 base damage (12 at Ascension 2+) and apply 2 Vulnerable.",
        ("Chosen", 4): "Apply Hex: non-Attacks shuffle 1 Dazed into the draw pile.",
        ("Chosen", 5): "Deal 5 base damage twice (6 at Ascension 2+).",
        ("Mugger", 1): "Deal 10 base damage (11 at Ascension 2+) and steal gold.",
        ("Mugger", 2): "Gain 11 Block (17 at Ascension 17+) and prepare to Escape.",
        ("Mugger", 3): "Escape from combat.",
        ("Mugger", 4): "Deal 16 base damage (18 at Ascension 2+) and steal gold, then prepare Smoke Bomb.",
        ("Mystic", 1): "Deal 8 base damage (9 at Ascension 2+) and apply 2 Frail.",
        ("Mystic", 2): "Heal all enemies for 16 HP (20 at Ascension 17+).",
        ("Mystic", 3): "Give all enemies 2 Strength (3 at Ascension 2+, 4 at Ascension 17+).",
        ("ShelledParasite", 1): "Deal 18 base damage (21 at Ascension 2+) and apply 2 Frail.",
        ("ShelledParasite", 2): "Deal 6 base damage twice (7 at Ascension 2+).",
        ("ShelledParasite", 3): "Deal 10 base damage (12 at Ascension 2+) and heal equal to unblocked damage.",
        ("ShelledParasite", 4): "Do nothing while Stunned.",
        ("SnakePlant", 1): "Deal 7 base damage 3 times (8 at Ascension 2+).",
        ("SnakePlant", 2): "Apply 2 Frail and 2 Weak.",
        ("Snecko", 1): "Apply Confused.",
        ("Snecko", 2): "Deal 15 base damage (18 at Ascension 2+).",
        ("Snecko", 3): "Deal 8 base damage (10 at Ascension 2+), apply 2 Vulnerable, and at Ascension 17+ apply 2 Weak.",
        ("SphericGuardian", 1): "Deal 10 base damage twice (11 at Ascension 2+).",
        ("SphericGuardian", 2): "Gain 25 Block (35 at Ascension 17+).",
        ("SphericGuardian", 3): "Gain 15 Block and deal 10 base damage (11 at Ascension 2+).",
        ("SphericGuardian", 4): "Deal 10 base damage (11 at Ascension 2+) and apply 5 Frail.",
        ("BookOfStabbing", 1): "Deal 6 base damage N times (7 at Ascension 3+); N increases during combat.",
        ("BookOfStabbing", 2): "Deal 21 base damage (24 at Ascension 3+).",
        ("BronzeAutomaton", 1): "Deal 7 base damage twice (8 at Ascension 4+).",
        ("BronzeAutomaton", 2): "Deal 45 base damage (50 at Ascension 4+).",
        ("BronzeAutomaton", 3): "Do nothing for one turn after Hyper Beam (skipped at Ascension 19+).",
        ("BronzeAutomaton", 4): "Summon two Bronze Orbs into the open slots.",
        ("BronzeAutomaton", 5): "Gain 3 Strength and 9 Block (4 Strength at Ascension 4+; 12 Block at Ascension 9+).",
        ("BronzeOrb", 1): "Deal 8 base damage.",
        ("BronzeOrb", 2): "Put a highest-rarity card from the draw or discard pile into Stasis; it returns to hand when this Orb dies.",
        ("BronzeOrb", 3): "Give the Bronze Automaton 12 Block.",
        ("GremlinLeader", 2): "Summon two Gremlins into open minion slots.",
        ("GremlinLeader", 3): "Give all living Gremlins and self Strength, and give living Gremlins Block.",
        ("GremlinLeader", 4): "Deal 6 base damage three times.",
        ("Hexaghost", 1): "Deal HP-scaled base damage six times; each hit is floor(current HP / 12) + 1, fixed by Activate.",
        ("Hexaghost", 2): "Deal 5 base damage twice (6 at Ascension 4+).",
        ("Hexaghost", 3): "Gain 12 Block and 2 Strength (3 Strength at Ascension 19+).",
        ("Hexaghost", 4): "Deal 6 base damage and add Burn to discard; count and upgrade depend on Ascension and turn.",
        ("Hexaghost", 5): "Activate and set Divider damage from the player's current HP.",
        ("Hexaghost", 6): "Deal 2 base damage six times (3 at Ascension 4+) and add 3 upgraded Burns to discard.",
        ("SlimeBoss", 1): "Prepare to use Slam.",
        ("SlimeBoss", 2): "Deal 35 base damage (38 at Ascension 4+).",
        ("SlimeBoss", 3): "Split into one Acid Slime (L) and one Spike Slime (L), each at current HP.",
        ("SlimeBoss", 4): "Add 3 Slimed to discard (5 at Ascension 19+).",
        ("Taskmaster", 1): "Deal 7 base damage and add Wounds to discard; count and Strength gain depend on Ascension.",
        ("TheChamp", 1): "Deal 16 base damage (18 at Ascension 4+).",
        ("TheChamp", 2): "Gain Block and Metallicize based on Ascension.",
        ("TheChamp", 3): "Deal 10 base damage twice.",
        ("TheChamp", 4): "Deal 12 base damage (14 at Ascension 4+) and apply 2 Frail and 2 Vulnerable.",
        ("TheChamp", 5): "Gain Strength based on Ascension.",
        ("TheChamp", 6): "Apply 2 Weak and 2 Vulnerable.",
        ("TheChamp", 7): "Remove debuffs and gain substantial Strength when entering phase two.",
        ("TheCollector", 2): "Deal 18 base damage (21 at Ascension 4+).",
        ("TheCollector", 3): "Give all living Torch Heads and self Strength, and gain Block.",
        ("TheCollector", 4): "Apply 3 Weak, 3 Vulnerable, and 3 Frail.",
        ("TheCollector", 5): "Summon Torch Heads into open minion slots.",
        ("TheGuardian", 1): "Gain 9 Block and prepare Fierce Bash.",
        ("TheGuardian", 2): "Deal 32 base damage (36 at Ascension 4+).",
        ("TheGuardian", 3): "Apply 2 Weak and 2 Vulnerable.",
        ("TheGuardian", 4): "Deal 5 base damage four times.",
        ("TheGuardian", 5): "Enter Defensive Mode, gain 20 Block and Sharp Hide, then prepare Roll Attack.",
        ("TheGuardian", 6): "Deal 9 base damage (10 at Ascension 4+).",
        ("TheGuardian", 7): "Deal 8 base damage twice, leave Defensive Mode, and increase the next Mode Shift threshold.",
        ("TorchHead", 1): "Deal 7 base damage.",
        ("OpenMonsterSlot", 0): "No action; this public slot is currently empty and may receive a native summon or split result.",
    }
)

SUPPORTED_MOVE_INTENTS: Mapping[tuple[str, int], str] = MappingProxyType(
    {
        ("SlaverBlue", 1): "ATTACK",
        ("SlaverBlue", 4): "ATTACK_DEBUFF",
        ("SlaverRed", 1): "ATTACK",
        ("SlaverRed", 2): "DEBUFF",
        ("SlaverRed", 3): "ATTACK_DEBUFF",
        ("FungiBeast", 1): "ATTACK",
        ("FungiBeast", 2): "BUFF",
        ("Looter", 1): "ATTACK",
        ("Looter", 2): "DEFEND",
        ("Looter", 3): "ESCAPE",
        ("Looter", 4): "ATTACK",
        ("Cultist", 1): "ATTACK",
        ("Cultist", 3): "BUFF",
        ("GremlinFat", 1): "ATTACK_DEBUFF",
        ("GremlinNob", 1): "ATTACK",
        ("GremlinNob", 2): "ATTACK_DEBUFF",
        ("GremlinNob", 3): "BUFF",
        ("GremlinWizard", 1): "ATTACK",
        ("GremlinWizard", 2): "UNKNOWN",
        ("JawWorm", 1): "ATTACK",
        ("JawWorm", 2): "DEFEND_BUFF",
        ("JawWorm", 3): "ATTACK_DEFEND",
        ("Lagavulin", 1): "ATTACK",
        ("Lagavulin", 2): "STRONG_DEBUFF",
        ("Lagavulin", 3): "SLEEP",
        ("GremlinWarrior", 1): "ATTACK",
        ("FuzzyLouseDefensive", 3): "ATTACK",
        ("FuzzyLouseDefensive", 4): "DEBUFF",
        ("FuzzyLouseNormal", 3): "ATTACK",
        ("FuzzyLouseNormal", 4): "BUFF",
        ("Sentry", 1): "ATTACK",
        ("Sentry", 2): "DEBUFF",
        ("AcidSlime_M", 1): "ATTACK_DEBUFF",
        ("AcidSlime_M", 2): "ATTACK",
        ("AcidSlime_M", 4): "DEBUFF",
        ("AcidSlime_S", 1): "ATTACK",
        ("AcidSlime_S", 4): "DEBUFF",
        ("GremlinTsundere", 1): "DEFEND",
        ("GremlinTsundere", 2): "ATTACK",
        ("GremlinThief", 1): "ATTACK",
        ("SpikeSlime_M", 1): "ATTACK_DEBUFF",
        ("SpikeSlime_M", 4): "DEBUFF",
        ("SpikeSlime_S", 1): "ATTACK",
        ("AcidSlime_L", 1): "ATTACK_DEBUFF",
        ("AcidSlime_L", 2): "ATTACK",
        ("AcidSlime_L", 3): "UNKNOWN",
        ("AcidSlime_L", 4): "DEBUFF",
        ("SpikeSlime_L", 1): "ATTACK_DEBUFF",
        ("SpikeSlime_L", 3): "UNKNOWN",
        ("SpikeSlime_L", 4): "DEBUFF",
        ("Byrd", 1): "ATTACK", ("Byrd", 2): "BUFF",
        ("Byrd", 3): "ATTACK", ("Byrd", 4): "STUN",
        ("Byrd", 5): "ATTACK", ("Byrd", 6): "BUFF",
        ("Centurion", 1): "ATTACK", ("Centurion", 2): "DEFEND", ("Centurion", 3): "ATTACK",
        ("Chosen", 1): "ATTACK", ("Chosen", 2): "DEBUFF", ("Chosen", 3): "ATTACK_DEBUFF", ("Chosen", 4): "DEBUFF", ("Chosen", 5): "ATTACK",
        ("Mugger", 1): "ATTACK", ("Mugger", 2): "DEFEND", ("Mugger", 3): "ESCAPE", ("Mugger", 4): "ATTACK",
        ("Mystic", 1): "ATTACK_DEBUFF", ("Mystic", 2): "HEAL", ("Mystic", 3): "BUFF",
        ("ShelledParasite", 1): "ATTACK_DEBUFF", ("ShelledParasite", 2): "ATTACK", ("ShelledParasite", 3): "ATTACK_BUFF", ("ShelledParasite", 4): "STUN",
        ("SnakePlant", 1): "ATTACK", ("SnakePlant", 2): "STRONG_DEBUFF",
        ("Snecko", 1): "STRONG_DEBUFF", ("Snecko", 2): "ATTACK", ("Snecko", 3): "ATTACK_DEBUFF",
        ("SphericGuardian", 1): "ATTACK", ("SphericGuardian", 2): "DEFEND", ("SphericGuardian", 3): "ATTACK_DEFEND", ("SphericGuardian", 4): "ATTACK_DEBUFF",
        ("BookOfStabbing", 1): "ATTACK", ("BookOfStabbing", 2): "ATTACK",
        ("BronzeAutomaton", 1): "ATTACK", ("BronzeAutomaton", 2): "ATTACK", ("BronzeAutomaton", 3): "STUN", ("BronzeAutomaton", 4): "UNKNOWN", ("BronzeAutomaton", 5): "DEFEND_BUFF",
        ("BronzeOrb", 1): "ATTACK", ("BronzeOrb", 2): "UNKNOWN", ("BronzeOrb", 3): "DEFEND",
        ("GremlinLeader", 2): "UNKNOWN", ("GremlinLeader", 3): "DEFEND_BUFF", ("GremlinLeader", 4): "ATTACK",
        ("Hexaghost", 1): "ATTACK", ("Hexaghost", 2): "ATTACK", ("Hexaghost", 3): "DEFEND_BUFF", ("Hexaghost", 4): "ATTACK_DEBUFF", ("Hexaghost", 5): "UNKNOWN", ("Hexaghost", 6): "ATTACK",
        ("SlimeBoss", 1): "UNKNOWN", ("SlimeBoss", 2): "ATTACK", ("SlimeBoss", 3): "UNKNOWN", ("SlimeBoss", 4): "STRONG_DEBUFF",
        ("Taskmaster", 1): "ATTACK_DEBUFF",
        ("TheChamp", 1): "ATTACK", ("TheChamp", 2): "DEFEND_BUFF", ("TheChamp", 3): "ATTACK", ("TheChamp", 4): "ATTACK_DEBUFF", ("TheChamp", 5): "BUFF", ("TheChamp", 6): "STRONG_DEBUFF", ("TheChamp", 7): "BUFF",
        ("TheCollector", 2): "ATTACK", ("TheCollector", 3): "DEFEND_BUFF", ("TheCollector", 4): "STRONG_DEBUFF", ("TheCollector", 5): "UNKNOWN",
        ("TheGuardian", 1): "DEFEND", ("TheGuardian", 2): "ATTACK", ("TheGuardian", 3): "STRONG_DEBUFF", ("TheGuardian", 4): "ATTACK", ("TheGuardian", 5): "DEFEND", ("TheGuardian", 6): "ATTACK", ("TheGuardian", 7): "ATTACK",
        ("TorchHead", 1): "ATTACK",
        ("OpenMonsterSlot", 0): "UNKNOWN",
    }
)

SUPPORTED_MOVE_NAMES: Mapping[tuple[str, int], str] = MappingProxyType(
    {
        ("SlaverBlue", 1): "Stab",
        ("SlaverBlue", 4): "Rake",
        ("SlaverRed", 1): "Stab",
        ("SlaverRed", 2): "Entangle",
        ("SlaverRed", 3): "Scrape",
        ("FungiBeast", 1): "Bite",
        ("FungiBeast", 2): "Grow",
        ("Looter", 1): "Mug",
        ("Looter", 2): "Smoke Bomb",
        ("Looter", 3): "Escape",
        ("Looter", 4): "Lunge",
        ("Cultist", 1): "Dark Strike",
        ("Cultist", 3): "Incantation",
        ("AcidSlime_M", 1): "Corrosive Spit",
        ("AcidSlime_M", 2): "Tackle",
        ("AcidSlime_M", 4): "Lick",
        ("AcidSlime_S", 1): "Tackle",
        ("AcidSlime_S", 4): "Lick",
        ("GremlinFat", 1): "Smash",
        ("GremlinNob", 1): "Rush",
        ("GremlinNob", 2): "Skull Bash",
        ("GremlinNob", 3): "Bellow",
        ("GremlinWizard", 1): "Ultimate Blast",
        ("GremlinWizard", 2): "Charging",
        ("JawWorm", 1): "Chomp",
        ("JawWorm", 2): "Bellow",
        ("JawWorm", 3): "Thrash",
        ("Lagavulin", 1): "Attack",
        ("Lagavulin", 2): "Siphon Soul",
        ("Lagavulin", 3): "Sleep",
        ("GremlinWarrior", 1): "Scratch",
        ("FuzzyLouseDefensive", 3): "Bite",
        ("FuzzyLouseDefensive", 4): "Spit Web",
        ("FuzzyLouseNormal", 3): "Bite",
        ("FuzzyLouseNormal", 4): "Grow",
        ("Sentry", 1): "Beam",
        ("Sentry", 2): "Bolt",
        ("GremlinTsundere", 1): "Protect",
        ("GremlinTsundere", 2): "Shield Bash",
        ("GremlinThief", 1): "Puncture",
        ("SpikeSlime_M", 1): "Flame Tackle",
        ("SpikeSlime_M", 4): "Lick",
        ("SpikeSlime_S", 1): "Tackle",
        ("AcidSlime_L", 1): "Corrosive Spit", ("AcidSlime_L", 2): "Tackle", ("AcidSlime_L", 3): "Split", ("AcidSlime_L", 4): "Lick",
        ("SpikeSlime_L", 1): "Flame Tackle", ("SpikeSlime_L", 3): "Split", ("SpikeSlime_L", 4): "Lick",
        ("Byrd", 1): "Peck", ("Byrd", 2): "Fly", ("Byrd", 3): "Swoop", ("Byrd", 4): "Stunned", ("Byrd", 5): "Headbutt", ("Byrd", 6): "Caw",
        ("Centurion", 1): "Slash", ("Centurion", 2): "Defend", ("Centurion", 3): "Fury",
        ("Chosen", 1): "Zap", ("Chosen", 2): "Drain", ("Chosen", 3): "Debilitate", ("Chosen", 4): "Hex", ("Chosen", 5): "Poke",
        ("Mugger", 1): "Mug", ("Mugger", 2): "Smoke Bomb", ("Mugger", 3): "Escape", ("Mugger", 4): "Lunge",
        ("Mystic", 1): "Attack Debuff", ("Mystic", 2): "Heal", ("Mystic", 3): "Buff",
        ("ShelledParasite", 1): "Fell", ("ShelledParasite", 2): "Double Strike", ("ShelledParasite", 3): "Suck", ("ShelledParasite", 4): "Stunned",
        ("SnakePlant", 1): "Chomp", ("SnakePlant", 2): "Enfeebling Spores",
        ("Snecko", 1): "Perplexing Glare", ("Snecko", 2): "Bite", ("Snecko", 3): "Tail Whip",
        ("SphericGuardian", 1): "Slam", ("SphericGuardian", 2): "Activate", ("SphericGuardian", 3): "Harden", ("SphericGuardian", 4): "Attack Debuff",
        ("BookOfStabbing", 1): "Multi-Stab", ("BookOfStabbing", 2): "Single Stab",
        ("BronzeAutomaton", 1): "Flail", ("BronzeAutomaton", 2): "Hyper Beam", ("BronzeAutomaton", 3): "Stunned", ("BronzeAutomaton", 4): "Spawn Orbs", ("BronzeAutomaton", 5): "Boost",
        ("BronzeOrb", 1): "Beam", ("BronzeOrb", 2): "Stasis", ("BronzeOrb", 3): "Support Beam",
        ("GremlinLeader", 2): "Rally", ("GremlinLeader", 3): "Encourage", ("GremlinLeader", 4): "Stab",
        ("Hexaghost", 1): "Divider", ("Hexaghost", 2): "Tackle", ("Hexaghost", 3): "Inflame", ("Hexaghost", 4): "Sear", ("Hexaghost", 5): "Activate", ("Hexaghost", 6): "Inferno",
        ("SlimeBoss", 1): "Preparing", ("SlimeBoss", 2): "Slam", ("SlimeBoss", 3): "Split", ("SlimeBoss", 4): "Goop Spray",
        ("Taskmaster", 1): "Scouring Whip",
        ("TheChamp", 1): "Heavy Slash", ("TheChamp", 2): "Defensive Stance", ("TheChamp", 3): "Execute", ("TheChamp", 4): "Face Slap", ("TheChamp", 5): "Gloat", ("TheChamp", 6): "Taunt", ("TheChamp", 7): "Anger",
        ("TheCollector", 2): "Fireball", ("TheCollector", 3): "Buff", ("TheCollector", 4): "Mega Debuff", ("TheCollector", 5): "Spawn",
        ("TheGuardian", 1): "Charging Up", ("TheGuardian", 2): "Fierce Bash", ("TheGuardian", 3): "Vent Steam", ("TheGuardian", 4): "Whirlwind", ("TheGuardian", 5): "Defensive Mode", ("TheGuardian", 6): "Roll Attack", ("TheGuardian", 7): "Twin Slam",
        ("TorchHead", 1): "Tackle",
        ("OpenMonsterSlot", 0): "Empty",
    }
)

SUPPORTED_POWER_EFFECT_TEMPLATES: Mapping[str, str] = MappingProxyType(
    {
        "Angry": (
            "Whenever the owner is hit by an Attack, gain {amount} Strength."
        ),
        "Artifact": "Negate the next {amount} debuff(s).",
        "Asleep": (
            "The owner is asleep; taking unblocked Attack damage wakes it "
            "and removes its Metallicize."
        ),
        "Dexterity": "Block gained from cards is modified by {amount}.",
        "Enrage": (
            "Whenever the player plays a Skill, the owner immediately gains "
            "{amount} Strength."
        ),
        "Entangled": "Attacks cannot be played this turn.",
        "Frail": "Block gained from cards is reduced by 25% for {amount} turn(s).",
        "Metallicize": "At the end of the owner's turn, gain {amount} Block.",
        "Combust": (
            "At the end of the player's turn, lose 1 HP and deal {amount} "
            "damage to all enemies."
        ),
        "Evolve": (
            "Whenever the player draws a Status, draw {amount} additional "
            "card(s)."
        ),
        "Feel No Pain": (
            "Whenever the player Exhausts a card, gain {amount} Block."
        ),
        "Flame Barrier": (
            "Whenever the player is attacked this turn, deal {amount} damage "
            "back to the attacker."
        ),
        "Fire Breathing": (
            "Whenever a Status or Curse is drawn, deal {amount} damage to all enemies."
        ),
        "Lose Strength": (
            "At the end of this turn, lose {amount} Strength."
        ),
        "No Draw": "Cannot draw additional cards this turn.",
        "Rage": (
            "Whenever an Attack is played this turn, gain {amount} Block."
        ),
        "Ritual": "At the end of the owner's turn, gain {amount} Strength.",
        "Strength": "Attack damage is increased by {amount} per hit.",
        "Spore Cloud": (
            "When the owner dies, apply 2 Vulnerable to the player."
        ),
        "Thievery": (
            "Attack moves steal gold; the amount is {amount}. Gold is outside the combat policy state."
        ),
        "Vulnerable": (
            "Take 50% more attack damage for {amount} remaining turn(s)."
        ),
        "Curl Up": (
            "After the first unblocked attack damage received, gain {amount} "
            "Block and remove this power."
        ),
        "Weak": "Attacks deal 25% less damage for {amount} remaining turn(s).",
        "Barricade": "Block is not removed at the start of the owner's turn.",
        "Brutality": "At the start of the player's turn, lose 1 HP and draw {amount} card(s).",
        "Confused": "Whenever a card is drawn, randomize its cost to 0, 1, 2, or 3 Energy for this combat.",
        "Corruption": "Skills cost 0 Energy and Exhaust when played.",
        "Dark Embrace": "Whenever a card is Exhausted, draw {amount} card(s).",
        "Demon Form": "At the start of the player's turn, gain {amount} Strength.",
        "Double Tap": "The next {amount} Attack(s) this turn are played twice.",
        "Flight": "Attack damage is halved; loses 1 stack when hit by an Attack and becomes Stunned at 0.",
        "Hex": "Whenever the player plays a non-Attack, shuffle 1 Dazed into the draw pile.",
        "Juggernaut": "Whenever the player gains Block, deal {amount} damage to a random enemy.",
        "Malleable": "After Attack damage, gain {amount} Block and increase this amount by 1 this turn.",
        "Plated Armor": "At end of turn, gain {amount} Block; unblocked Attack damage reduces it by 1.",
        "Rupture": "Whenever HP is lost from a card, gain {amount} Strength.",
        "Minion": "This enemy is a summoned minion and leaves combat when its leader dies.",
        "Minion Leader": "Killing this enemy also removes its living minions.",
        "Painful Stabs": "Unblocked attack damage from this enemy adds a Wound to the discard pile.",
        "Mode Shift": "After taking {amount} more damage, enter Defensive Mode.",
        "Sharp Hide": "Whenever the player plays an Attack, take {amount} damage.",
        "Stasis": "This enemy is holding one visible card; killing it returns that card to the hand.",
        "Thorns": "Whenever the owner is attacked, deal {amount} damage back to the attacker.",
        "Vigor": "The next Attack deals {amount} additional damage, then this power is consumed.",
    }
)


def describe_card(card: CardState) -> str:
    """Return versioned rules text without trusting source display text."""
    key = (card.card_id, card.upgrades)
    try:
        effect = SUPPORTED_CARD_EFFECTS[key]
        if card.card_id == "Rampage":
            return f"{effect} CURRENT_COMBAT_BONUS={card.special_data}."
        return effect
    except KeyError as error:
        raise UnsupportedObservationMechanicError(
            "Card is outside the frozen P0 observation catalog: "
            f"{card.card_id} upgrade {card.upgrades}"
        ) from error


def describe_relic(relic_id: str) -> str:
    """Return the frozen D-v1 effect text for a supported relic."""
    try:
        return SUPPORTED_RELIC_EFFECTS[relic_id]
    except KeyError as error:
        raise UnsupportedObservationMechanicError(
            f"Relic is outside the frozen D-v1 observation catalog: {relic_id}"
        ) from error


def describe_move(monster: MonsterState) -> str:
    """Return supported rules text for the monster's current move."""
    key = (monster.monster_id, monster.move_id)
    try:
        effect = SUPPORTED_MOVE_EFFECTS[key]
        expected_intent = SUPPORTED_MOVE_INTENTS[key]
    except KeyError as error:
        raise UnsupportedObservationMechanicError(
            "Move is outside the frozen P0 observation catalog: "
            f"{monster.monster_id} move {monster.move_id}"
        ) from error
    if monster.intent != expected_intent:
        raise UnsupportedObservationMechanicError(
            "Move intent disagrees with the frozen P0 observation catalog: "
            f"{monster.monster_id} move {monster.move_id} is "
            f"{monster.intent}, expected {expected_intent}"
        )
    return effect


def describe_move_by_id(monster_id: str, move_id: int) -> tuple[str, str, str]:
    """Return name, intent, and effect for a supported behavior candidate."""

    key = (monster_id, move_id)
    try:
        return (
            SUPPORTED_MOVE_NAMES[key],
            SUPPORTED_MOVE_INTENTS[key],
            SUPPORTED_MOVE_EFFECTS[key],
        )
    except KeyError as error:
        raise UnsupportedObservationMechanicError(
            f"Behavior move is outside support: {monster_id} move {move_id}"
        ) from error


def describe_power(power: PowerState) -> str:
    """Render a supported power template with its current public amount."""
    try:
        template = SUPPORTED_POWER_EFFECT_TEMPLATES[power.power_id]
    except KeyError as error:
        raise UnsupportedObservationMechanicError(
            "Power is outside the frozen P0 observation catalog: "
            f"{power.power_id}"
        ) from error
    return template.format(amount=power.amount)
