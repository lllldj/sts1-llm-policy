#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "combat/BattleContext.h"
#include "constants/Cards.h"
#include "constants/MonsterIds.h"
#include "constants/MonsterMoves.h"
#include "constants/MonsterStatusEffects.h"
#include "constants/PlayerStatusEffects.h"
#include "constants/Rooms.h"
#include "game/GameContext.h"
#include "game/Deck.h"
#include "sim/search/BattleScumSearcher2.h"
#include "public_draw_memory.h"

namespace {

using json = nlohmann::json;
using sts::BattleContext;
using sts::CardInstance;
using sts::InputState;
using sts::MonsterEncounter;
using sts::Outcome;

constexpr int kProtocolVersion = 1;
#ifdef STS_CORRECTED_CARD_MECHANICS
constexpr const char *kCombatSnapshotSchema = "combat_snapshot_v2";
#else
constexpr const char *kCombatSnapshotSchema = "combat_snapshot_v1";
#endif
constexpr std::int64_t kMaxSearchSimulations = 1'000'000;

struct BridgeError : std::runtime_error {
    std::string code;

    BridgeError(std::string codeValue, std::string message)
        : std::runtime_error(std::move(message)), code(std::move(codeValue)) {}
};

struct LegalAction {
    std::string actionId;
    std::string kind;
    int handIndex = -1;
    int targetIndex = -1;
};

struct SemanticCorrectionCounts {
    std::int64_t lagavulinNaturalWake = 0;
    std::int64_t upgradedDisarm = 0;
    std::int64_t redSlaverEntangle = 0;
    std::int64_t philosopherBronzeOrbStrength = 0;
    std::int64_t burningBloodVictoryHeal = 0;
};

struct RootOutcomeStats {
    std::int64_t terminalWins = 0;
    std::int64_t terminalLosses = 0;
    std::int64_t endingHpSum = 0;
    std::int64_t victoryEndingHpSum = 0;
    double evaluationSquareSum = 0.0;
};

const char *inputStateName(InputState value) {
    switch (value) {
        case InputState::EXECUTING_ACTIONS: return "EXECUTING_ACTIONS";
        case InputState::PLAYER_NORMAL: return "PLAYER_NORMAL";
        case InputState::CARD_SELECT: return "CARD_SELECT";
        case InputState::CHOOSE_STANCE_ACTION: return "CHOOSE_STANCE_ACTION";
        case InputState::CHOOSE_TOOLBOX_COLORLESS_CARD: return "CHOOSE_TOOLBOX_COLORLESS_CARD";
        case InputState::CHOOSE_EXHAUST_POTION_CARDS: return "CHOOSE_EXHAUST_POTION_CARDS";
        case InputState::CHOOSE_GAMBLING_CARDS: return "CHOOSE_GAMBLING_CARDS";
        case InputState::CHOOSE_ENTROPIC_BREW_DISCARD_POTIONS: return "CHOOSE_ENTROPIC_BREW_DISCARD_POTIONS";
        case InputState::CHOOSE_DISCARD_CARDS: return "CHOOSE_DISCARD_CARDS";
        case InputState::SCRY: return "SCRY";
        default: return "UNSUPPORTED";
    }
}

MonsterEncounter scenarioEncounter(const std::string &scenarioId) {
    if (scenarioId == "cultist") return MonsterEncounter::CULTIST;
    if (scenarioId == "jaw_worm") return MonsterEncounter::JAW_WORM;
    if (scenarioId == "two_louse") return MonsterEncounter::TWO_LOUSE;
    if (scenarioId == "gremlin_gang") return MonsterEncounter::GREMLIN_GANG;
    if (scenarioId == "gremlin_nob") return MonsterEncounter::GREMLIN_NOB;
    if (scenarioId == "lagavulin") return MonsterEncounter::LAGAVULIN;
    if (scenarioId == "three_sentries") return MonsterEncounter::THREE_SENTRIES;
    if (scenarioId == "small_slimes") return MonsterEncounter::SMALL_SLIMES;
    if (scenarioId == "lots_of_slimes") return MonsterEncounter::LOTS_OF_SLIMES;
    if (scenarioId == "blue_slaver") return MonsterEncounter::BLUE_SLAVER;
    if (scenarioId == "red_slaver") return MonsterEncounter::RED_SLAVER;
    if (scenarioId == "looter") return MonsterEncounter::LOOTER;
    if (scenarioId == "exordium_thugs") return MonsterEncounter::EXORDIUM_THUGS;
    if (scenarioId == "exordium_wildlife") return MonsterEncounter::EXORDIUM_WILDLIFE;
    if (scenarioId == "two_fungi_beasts") return MonsterEncounter::TWO_FUNGI_BEASTS;
    if (scenarioId == "large_slime") return MonsterEncounter::LARGE_SLIME;
    if (scenarioId == "three_louse") return MonsterEncounter::THREE_LOUSE;
    if (scenarioId == "spheric_guardian") return MonsterEncounter::SPHERIC_GUARDIAN;
    if (scenarioId == "chosen") return MonsterEncounter::CHOSEN;
    if (scenarioId == "shell_parasite") return MonsterEncounter::SHELL_PARASITE;
    if (scenarioId == "three_byrds") return MonsterEncounter::THREE_BYRDS;
    if (scenarioId == "two_thieves") return MonsterEncounter::TWO_THIEVES;
    if (scenarioId == "chosen_and_byrds") return MonsterEncounter::CHOSEN_AND_BYRDS;
    if (scenarioId == "sentry_and_sphere") return MonsterEncounter::SENTRY_AND_SPHERE;
    if (scenarioId == "cultist_and_chosen") return MonsterEncounter::CULTIST_AND_CHOSEN;
    if (scenarioId == "three_cultist") return MonsterEncounter::THREE_CULTIST;
    if (scenarioId == "shelled_parasite_and_fungi") return MonsterEncounter::SHELLED_PARASITE_AND_FUNGI;
    if (scenarioId == "snecko") return MonsterEncounter::SNECKO;
    if (scenarioId == "snake_plant") return MonsterEncounter::SNAKE_PLANT;
    if (scenarioId == "centurion_and_healer") return MonsterEncounter::CENTURION_AND_HEALER;
    if (scenarioId == "gremlin_leader") return MonsterEncounter::GREMLIN_LEADER;
    if (scenarioId == "slavers") return MonsterEncounter::SLAVERS;
    if (scenarioId == "book_of_stabbing") return MonsterEncounter::BOOK_OF_STABBING;
    if (scenarioId == "slime_boss") return MonsterEncounter::SLIME_BOSS;
    if (scenarioId == "the_guardian") return MonsterEncounter::THE_GUARDIAN;
    if (scenarioId == "hexaghost") return MonsterEncounter::HEXAGHOST;
    if (scenarioId == "automaton") return MonsterEncounter::AUTOMATON;
    if (scenarioId == "collector") return MonsterEncounter::COLLECTOR;
    if (scenarioId == "champ") return MonsterEncounter::CHAMP;
    throw BridgeError("unknown_scenario", "Unknown scenario_id: " + scenarioId);
}

int scenarioAct(const std::string &scenarioId) {
    static const std::set<std::string> actTwoScenarios = {
        "spheric_guardian", "chosen", "shell_parasite", "three_byrds",
        "two_thieves", "chosen_and_byrds", "sentry_and_sphere",
        "cultist_and_chosen", "three_cultist",
        "shelled_parasite_and_fungi", "snecko", "snake_plant",
        "centurion_and_healer",
        "gremlin_leader", "slavers", "book_of_stabbing",
        "automaton", "collector", "champ",
    };
    return actTwoScenarios.count(scenarioId) == 0 ? 1 : 2;
}

sts::Room scenarioRoom(const std::string &scenarioId) {
    static const std::set<std::string> eliteScenarios = {
        "gremlin_nob", "lagavulin", "three_sentries",
        "gremlin_leader", "slavers", "book_of_stabbing",
    };
    static const std::set<std::string> bossScenarios = {
        "slime_boss", "the_guardian", "hexaghost",
        "automaton", "collector", "champ",
    };
    if (eliteScenarios.count(scenarioId) != 0) return sts::Room::ELITE;
    if (bossScenarios.count(scenarioId) != 0) return sts::Room::BOSS;
    return sts::Room::MONSTER;
}

void applyDeckPreset(sts::GameContext &game, const std::string &deckPreset) {
    if (deckPreset == "starter") return;

    if (deckPreset == "elite_transition") {
        game.obtainCard(sts::Card(sts::CardId::CARNAGE, 1));
        game.obtainCard(sts::Card(sts::CardId::SHRUG_IT_OFF));
        game.obtainCard(sts::Card(sts::CardId::RAGE, 1));
        game.obtainCard(sts::Card(sts::CardId::POMMEL_STRIKE));
        return;
    }

    if (deckPreset == "boss_ready") {
        bool removedStrike = false;
        for (int index = 0; index < game.deck.size(); ++index) {
            if (game.deck.cards[index].getId() == sts::CardId::STRIKE_RED) {
                game.deck.remove(game, index);
                removedStrike = true;
                break;
            }
        }
        if (!removedStrike) {
            throw BridgeError(
                "invalid_deck_preset",
                "boss_ready requires one starter Strike to remove"
            );
        }
        game.obtainCard(sts::Card(sts::CardId::CARNAGE, 1));
        game.obtainCard(sts::Card(sts::CardId::SHRUG_IT_OFF));
        game.obtainCard(sts::Card(sts::CardId::RAGE, 1));
        game.obtainCard(sts::Card(sts::CardId::POMMEL_STRIKE, 1));
        game.obtainCard(sts::Card(sts::CardId::BATTLE_TRANCE));
        game.obtainCard(sts::Card(sts::CardId::BLOODLETTING, 1));
        game.obtainRelic(sts::RelicId::VAJRA);
        return;
    }

    throw BridgeError(
        "unknown_deck_preset",
        "Unknown deck_preset: " + deckPreset
    );
}

struct DeckSpecCard {
    std::string cardId;
    sts::CardId nativeId;
    int upgrades;
    int count;
};

struct ParsedDeckSpec {
    std::string basePreset;
    std::vector<DeckSpecCard> cards;
    json metadata;
};

sts::CardId supportedDeckCardId(const std::string &cardId) {
    if (cardId == "Burning Pact") return sts::CardId::BURNING_PACT;
    if (cardId == "True Grit") return sts::CardId::TRUE_GRIT;
    if (cardId == "Warcry") return sts::CardId::WARCRY;
    if (cardId == "Dual Wield") return sts::CardId::DUAL_WIELD;
    if (cardId == "Armaments") return sts::CardId::ARMAMENTS;
    if (cardId == "Headbutt") return sts::CardId::HEADBUTT;
    if (cardId == "Exhume") return sts::CardId::EXHUME;
    if (cardId == "AscendersBane") return sts::CardId::ASCENDERS_BANE;
    if (cardId == "Strike_R") return sts::CardId::STRIKE_RED;
    if (cardId == "Defend_R") return sts::CardId::DEFEND_RED;
    if (cardId == "Bash") return sts::CardId::BASH;
    if (cardId == "Carnage") return sts::CardId::CARNAGE;
    if (cardId == "Shrug It Off") return sts::CardId::SHRUG_IT_OFF;
    if (cardId == "Rage") return sts::CardId::RAGE;
    if (cardId == "Pommel Strike") return sts::CardId::POMMEL_STRIKE;
    if (cardId == "Battle Trance") return sts::CardId::BATTLE_TRANCE;
    if (cardId == "Bloodletting") return sts::CardId::BLOODLETTING;
    if (cardId == "Blood for Blood") return sts::CardId::BLOOD_FOR_BLOOD;
    if (cardId == "Anger") return sts::CardId::ANGER;
    if (cardId == "Body Slam") return sts::CardId::BODY_SLAM;
    if (cardId == "Clash") return sts::CardId::CLASH;
    if (cardId == "Cleave") return sts::CardId::CLEAVE;
    if (cardId == "Clothesline") return sts::CardId::CLOTHESLINE;
    if (cardId == "Iron Wave") return sts::CardId::IRON_WAVE;
    if (cardId == "Thunderclap") return sts::CardId::THUNDERCLAP;
    if (cardId == "Uppercut") return sts::CardId::UPPERCUT;
    if (cardId == "Whirlwind") return sts::CardId::WHIRLWIND;
    if (cardId == "Flame Barrier") return sts::CardId::FLAME_BARRIER;
    if (cardId == "Fire Breathing") return sts::CardId::FIRE_BREATHING;
    if (cardId == "Ghostly Armor") return sts::CardId::GHOSTLY_ARMOR;
    if (cardId == "Havoc") return sts::CardId::HAVOC;
    if (cardId == "Heavy Blade") return sts::CardId::HEAVY_BLADE;
    if (cardId == "Hemokinesis") return sts::CardId::HEMOKINESIS;
    if (cardId == "Intimidate") return sts::CardId::INTIMIDATE;
    if (cardId == "Power Through") return sts::CardId::POWER_THROUGH;
    if (cardId == "Perfected Strike") return sts::CardId::PERFECTED_STRIKE;
    if (cardId == "Pummel") return sts::CardId::PUMMEL;
    if (cardId == "Disarm") return sts::CardId::DISARM;
    if (cardId == "Dropkick") return sts::CardId::DROPKICK;
    if (cardId == "Entrench") return sts::CardId::ENTRENCH;
    if (cardId == "Reckless Charge") return sts::CardId::RECKLESS_CHARGE;
    if (cardId == "Second Wind") return sts::CardId::SECOND_WIND;
    if (cardId == "Shockwave") return sts::CardId::SHOCKWAVE;
    if (cardId == "Sentinel") return sts::CardId::SENTINEL;
    if (cardId == "Sever Soul") return sts::CardId::SEVER_SOUL;
    if (cardId == "Spot Weakness") return sts::CardId::SPOT_WEAKNESS;
    if (cardId == "Sword Boomerang") return sts::CardId::SWORD_BOOMERANG;
    if (cardId == "Twin Strike") return sts::CardId::TWIN_STRIKE;
    if (cardId == "Flex") return sts::CardId::FLEX;
    if (cardId == "Inflame") return sts::CardId::INFLAME;
    if (cardId == "Metallicize") return sts::CardId::METALLICIZE;
    if (cardId == "Combust") return sts::CardId::COMBUST;
    if (cardId == "Feel No Pain") return sts::CardId::FEEL_NO_PAIN;
    if (cardId == "Evolve") return sts::CardId::EVOLVE;
    if (cardId == "Wild Strike") return sts::CardId::WILD_STRIKE;
    if (cardId == "Rupture") return sts::CardId::RUPTURE;
    if (cardId == "Searing Blow") return sts::CardId::SEARING_BLOW;
    if (cardId == "Seeing Red") return sts::CardId::SEEING_RED;
    if (cardId == "Dark Embrace") return sts::CardId::DARK_EMBRACE;
    if (cardId == "Rampage") return sts::CardId::RAMPAGE;
    if (cardId == "Double Tap") return sts::CardId::DOUBLE_TAP;
    if (cardId == "Demon Form") return sts::CardId::DEMON_FORM;
    if (cardId == "Bludgeon") return sts::CardId::BLUDGEON;
    if (cardId == "Feed") return sts::CardId::FEED;
    if (cardId == "Limit Break") return sts::CardId::LIMIT_BREAK;
    if (cardId == "Corruption") return sts::CardId::CORRUPTION;
    if (cardId == "Barricade") return sts::CardId::BARRICADE;
    if (cardId == "Fiend Fire") return sts::CardId::FIEND_FIRE;
    if (cardId == "Berserk") return sts::CardId::BERSERK;
    if (cardId == "Impervious") return sts::CardId::IMPERVIOUS;
    if (cardId == "Juggernaut") return sts::CardId::JUGGERNAUT;
    if (cardId == "Brutality") return sts::CardId::BRUTALITY;
    if (cardId == "Reaper") return sts::CardId::REAPER;
    if (cardId == "Offering") return sts::CardId::OFFERING;
    if (cardId == "Immolate") return sts::CardId::IMMOLATE;
    if (cardId == "Burn") return sts::CardId::BURN;
    throw BridgeError(
        "unsupported_deck_card",
        "deck_spec contains unsupported card_id: " + cardId
    );
}

bool isPoolCard(const std::string &cardId) {
    return cardId != "Strike_R" && cardId != "Defend_R" && cardId != "Bash";
}

std::string cardKey(const std::string &cardId, int upgrades) {
    return cardId + "#" + std::to_string(upgrades);
}

DeckSpecCard parseDeckSpecCard(const json &value, const std::string &path) {
    if (!value.is_object()) {
        throw BridgeError("invalid_deck_spec", path + " must be an object");
    }
    if (!value.contains("card_id") || !value["card_id"].is_string()) {
        throw BridgeError("invalid_deck_spec", path + ".card_id must be a string");
    }
    if (!value.contains("upgrades") || !value["upgrades"].is_number_integer()) {
        throw BridgeError("invalid_deck_spec", path + ".upgrades must be an integer");
    }
    if (!value.contains("count") || !value["count"].is_number_integer()) {
        throw BridgeError("invalid_deck_spec", path + ".count must be an integer");
    }
    const std::string cardId = value["card_id"].get<std::string>();
    const int upgrades = value["upgrades"].get<int>();
    const int count = value["count"].get<int>();
    if (upgrades < 0 || upgrades > 1 || count <= 0 || count > sts::Deck::MAX_SIZE) {
        throw BridgeError("invalid_deck_spec", path + " has invalid upgrades/count");
    }
    return {cardId, supportedDeckCardId(cardId), upgrades, count};
}

void addExpectedCard(
    std::map<std::string, int> &cards,
    const std::string &cardId,
    int upgrades,
    int count
) {
    cards[cardKey(cardId, upgrades)] += count;
}

std::map<std::string, int> inheritedRecipeCards(const std::string &tier) {
    std::map<std::string, int> cards;
    addExpectedCard(cards, "Bash", 0, 1);
    if (tier == "feature_basic") {
        addExpectedCard(cards, "Strike_R", 0, 5);
        addExpectedCard(cards, "Defend_R", 0, 4);
        return cards;
    }
    addExpectedCard(cards, "Strike_R", 0, 4);
    addExpectedCard(cards, "Defend_R", 0, tier == "feature_elite" ? 4 : 3);
    addExpectedCard(cards, "Carnage", 1, 1);
    addExpectedCard(cards, "Shrug It Off", 0, 1);
    addExpectedCard(cards, "Rage", 1, 1);
    addExpectedCard(cards, "Pommel Strike", tier == "feature_elite" ? 0 : 1, 1);
    if (tier == "feature_boss") {
        addExpectedCard(cards, "Battle Trance", 0, 1);
        addExpectedCard(cards, "Bloodletting", 1, 1);
    }
    return cards;
}

bool validLowerHexHash(const std::string &value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
    });
}

ParsedDeckSpec parseDeckSpec(const json &value) {
    if (!value.is_object()) {
        throw BridgeError("invalid_deck_spec", "deck_spec must be an object");
    }
    const auto requireString = [&value](const char *key) -> std::string {
        if (!value.contains(key) || !value[key].is_string()) {
            throw BridgeError(
                "invalid_deck_spec",
                std::string("deck_spec.") + key + " must be a string"
            );
        }
        return value[key].get<std::string>();
    };
    if (requireString("schema_version") != "feature_loadout_v1") {
        throw BridgeError("invalid_deck_spec", "Unsupported deck_spec schema_version");
    }
    const std::string loadoutId = requireString("loadout_id");
    const std::string tier = requireString("recipe_tier");
    const std::string basePreset = requireString("base_preset");
    const std::string featureFamily = requireString("feature_family");
    const std::string deckHash = requireString("deck_hash");
    if (loadoutId.empty() || !validLowerHexHash(deckHash)) {
        throw BridgeError("invalid_deck_spec", "deck_spec identity/hash is invalid");
    }
    if (!value.contains("loadout_seed") || !value["loadout_seed"].is_number_unsigned()) {
        throw BridgeError("invalid_deck_spec", "deck_spec.loadout_seed must be uint64");
    }

    int additionCount = 0;
    int upgradedAdditionCount = 0;
    int finalSize = 0;
    std::vector<std::string> expectedRelics;
    if (tier == "feature_basic") {
        if (basePreset != "starter") {
            throw BridgeError("invalid_deck_spec", "feature_basic must inherit starter");
        }
        additionCount = 2;
        upgradedAdditionCount = 1;
        finalSize = 12;
        expectedRelics = {"Burning Blood"};
    } else if (tier == "feature_elite") {
        if (basePreset != "elite_transition") {
            throw BridgeError("invalid_deck_spec", "feature_elite must inherit elite_transition");
        }
        additionCount = 3;
        upgradedAdditionCount = 1;
        finalSize = 16;
        expectedRelics = {"Burning Blood"};
    } else if (tier == "feature_boss") {
        if (basePreset != "boss_ready") {
            throw BridgeError("invalid_deck_spec", "feature_boss must inherit boss_ready");
        }
        additionCount = 4;
        upgradedAdditionCount = 2;
        finalSize = 18;
        expectedRelics = {"Burning Blood", "Vajra"};
    } else {
        throw BridgeError("invalid_deck_spec", "Unknown feature recipe tier");
    }

    const std::set<std::string> featureFamilies = {
        "vulnerable_burst", "block_to_damage", "low_cost_order",
        "energy_x_cost", "aoe_target_priority", "persistent_scaling",
        "status_exhaust", "intent_mitigation", "random_mixed",
    };
    if (featureFamilies.count(featureFamily) == 0) {
        throw BridgeError("invalid_deck_spec", "Unknown feature family");
    }
    if (!value.contains("relics") || value["relics"] != expectedRelics) {
        throw BridgeError("invalid_deck_spec", "deck_spec relics disagree with base preset");
    }
    if (!value.contains("additions") || !value["additions"].is_array()) {
        throw BridgeError("invalid_deck_spec", "deck_spec.additions must be an array");
    }
    if (static_cast<int>(value["additions"].size()) != additionCount) {
        throw BridgeError("invalid_deck_spec", "deck_spec has wrong addition count");
    }

    auto expectedCards = inheritedRecipeCards(tier);
    std::set<std::string> inheritedNames;
    for (const auto &[key, count] : expectedCards) {
        (void)count;
        inheritedNames.insert(key.substr(0, key.rfind('#')));
    }
    std::set<std::string> additionNames;
    int observedUpgrades = 0;
    std::tuple<std::string, int> previousAddition{"", -1};
    for (std::size_t index = 0; index < value["additions"].size(); ++index) {
        const auto card = parseDeckSpecCard(
            value["additions"][index],
            "deck_spec.additions[" + std::to_string(index) + "]"
        );
        if (!isPoolCard(card.cardId) || card.count != 1) {
            throw BridgeError("invalid_deck_spec", "additions must be distinct pool cards");
        }
        if (inheritedNames.count(card.cardId) != 0 || !additionNames.insert(card.cardId).second) {
            throw BridgeError("invalid_deck_spec", "addition duplicates base or another addition");
        }
        const auto orderKey = std::make_tuple(card.cardId, card.upgrades);
        if (index > 0 && !(previousAddition < orderKey)) {
            throw BridgeError("invalid_deck_spec", "additions are not canonically ordered");
        }
        previousAddition = orderKey;
        observedUpgrades += card.upgrades;
        expectedCards[cardKey(card.cardId, card.upgrades)] += 1;
    }
    if (observedUpgrades != upgradedAdditionCount) {
        throw BridgeError("invalid_deck_spec", "deck_spec has wrong upgraded addition count");
    }

    if (!value.contains("cards") || !value["cards"].is_array()) {
        throw BridgeError("invalid_deck_spec", "deck_spec.cards must be an array");
    }
    std::vector<DeckSpecCard> cards;
    std::map<std::string, int> actualCards;
    int observedSize = 0;
    std::tuple<std::string, int> previousCard{"", -1};
    for (std::size_t index = 0; index < value["cards"].size(); ++index) {
        auto card = parseDeckSpecCard(
            value["cards"][index],
            "deck_spec.cards[" + std::to_string(index) + "]"
        );
        const auto orderKey = std::make_tuple(card.cardId, card.upgrades);
        if (index > 0 && !(previousCard < orderKey)) {
            throw BridgeError("invalid_deck_spec", "cards are not canonically ordered");
        }
        previousCard = orderKey;
        actualCards[cardKey(card.cardId, card.upgrades)] += card.count;
        observedSize += card.count;
        cards.push_back(std::move(card));
    }
    if (observedSize != finalSize || actualCards != expectedCards) {
        throw BridgeError("invalid_deck_spec", "full deck does not match recipe additions");
    }

    return {
        basePreset,
        std::move(cards),
        {
            {"loadout_id", loadoutId},
            {"recipe_tier", tier},
            {"base_preset", basePreset},
            {"feature_family", featureFamily},
            {"loadout_seed", value["loadout_seed"]},
            {"deck_hash", deckHash},
        },
    };
}

void applyDeckSpec(sts::GameContext &game, const ParsedDeckSpec &deckSpec) {
    applyDeckPreset(game, deckSpec.basePreset);
    game.deck = sts::Deck{};
    for (const auto &card : deckSpec.cards) {
        game.obtainCard(sts::Card(card.nativeId, card.upgrades), card.count);
    }
}

bool playerPowerJustApplied(const sts::Player &player, PlayerStatus status) {
    switch (status) {
        case PS::FRAIL: return player.wasJustApplied<PS::FRAIL>();
        case PS::VULNERABLE: return player.wasJustApplied<PS::VULNERABLE>();
        case PS::WEAK: return player.wasJustApplied<PS::WEAK>();
        default: return false;
    }
}

bool monsterPowerJustApplied(
    const sts::Monster &monster,
    sts::MonsterStatus status
) {
    switch (status) {
        case sts::MS::VULNERABLE:
            return monster.wasJustApplied<sts::MS::VULNERABLE>();
        case sts::MS::WEAK:
            return monster.wasJustApplied<sts::MS::WEAK>();
        case sts::MS::RITUAL:
            return monster.wasJustApplied<sts::MS::RITUAL>();
        default:
            return false;
    }
}

json serializePlayerPowers(const sts::Player &player) {
    json powers = json::array();
    const auto add = [&powers, &player](PlayerStatus status, int amount) {
        if (amount != 0) {
            const int index = static_cast<int>(status);
            powers.push_back({
                {"power_index", index},
                {"id", playerStatusEnumStrings[index]},
                {"name", playerStatusStrings[index]},
                {"amount", amount},
                {"just_applied", playerPowerJustApplied(player, status)},
            });
        }
    };

    add(PS::ARTIFACT, player.artifact);
    add(PS::DEXTERITY, player.dexterity);
    add(PS::FOCUS, player.focus);
    add(PS::STRENGTH, player.strength);
    for (const auto &[status, amount] : player.statusMap) {
        add(status, amount);
    }
    return powers;
}

json serializeMonsterPowers(const sts::Monster &monster) {
    json powers = json::array();
    const int invalid = static_cast<int>(sts::MS::INVALID);
    for (int index = 0; index < invalid; ++index) {
        const auto status = static_cast<sts::MonsterStatus>(index);
        const int amount = monster.getStatusInternal(status);
        if (amount != 0) {
            powers.push_back({
                {"power_index", index},
                {"name", sts::enemyStatusStrings[index]},
                {"amount", amount},
                {"just_applied", monsterPowerJustApplied(monster, status)},
            });
        }
    }
    return powers;
}

json serializeCard(const CardInstance &card, const BattleContext &battle) {
    const int typeIndex = static_cast<int>(card.getType());
    const bool playable =
        battle.outcome == Outcome::UNDECIDED &&
        battle.inputState == InputState::PLAYER_NORMAL &&
        battle.isCardPlayAllowed() &&
        card.canUseOnAnyTarget(battle);

    return {
        {"enum_id", sts::getCardEnumName(card.id)},
        {"string_id", sts::getCardStringId(card.id)},
        {"name", card.getName()},
        {"type", sts::cardTypeStrings[typeIndex]},
        {"cost", static_cast<int>(card.cost)},
        {"cost_for_turn", static_cast<int>(card.costForTurn)},
        {"upgrades", card.getUpgradeCount()},
        {"special_data", static_cast<int>(card.specialData)},
        {"exhausts", card.doesExhaust()},
        {"ethereal", card.isEthereal()},
        {"has_target", card.requiresTarget()},
        {"is_playable", playable},
    };
}

template <typename Container>
json serializeCardPile(const Container &cards, const BattleContext &battle) {
    json result = json::array();
    for (const auto &card : cards) {
        result.push_back(serializeCard(card, battle));
    }
    return result;
}

json serializeHand(const BattleContext &battle) {
    json result = json::array();
    for (int index = 0; index < battle.cards.cardsInHand; ++index) {
        auto value = serializeCard(battle.cards.hand[index], battle);
        value["hand_index"] = index;
        result.push_back(std::move(value));
    }
    return result;
}

std::string selectionTask(const BattleContext &battle) {
    switch (battle.cardSelectInfo.cardSelectTask) {
        case sts::CardSelectTask::EXHAUST_ONE: return "EXHAUST_ONE";
        case sts::CardSelectTask::WARCRY: return "WARCRY";
        case sts::CardSelectTask::DUAL_WIELD: return "DUAL_WIELD";
        case sts::CardSelectTask::ARMAMENTS: return "ARMAMENTS";
        case sts::CardSelectTask::HEADBUTT: return "HEADBUTT";
        case sts::CardSelectTask::EXHUME: return "EXHUME";
        default: throw BridgeError("unsupported_card_selection", "Unaudited card-selection task");
    }
}

std::vector<LegalAction> enumerateLegalActions(const BattleContext &battle) {
    std::vector<LegalAction> result;
    if (battle.outcome != Outcome::UNDECIDED) return result;
    if (battle.inputState == InputState::CARD_SELECT) {
        const auto task = selectionTask(battle);
        for (const auto &action : sts::search::Action::enumerateCardSelectActions(battle)) {
            if (action.getActionType() != sts::search::ActionType::SINGLE_CARD_SELECT ||
                !action.isValidAction(battle)) {
                throw BridgeError("invalid_card_selection", "Expected a legal single-card choice");
            }
            result.push_back({"SELECT:" + std::to_string(action.getSelectIdx()),
                              "SELECT_CARD", action.getSelectIdx(), -1});
        }
        if (result.empty()) throw BridgeError("empty_card_selection", "No legal selection candidates");
        return result;
    }
    if (battle.inputState != InputState::PLAYER_NORMAL) {
        throw BridgeError(
            "unsupported_input_state",
            std::string("P0 bridge cannot act in input state ") +
                inputStateName(battle.inputState)
        );
    }

    if (battle.isCardPlayAllowed()) {
        for (int handIndex = 0; handIndex < battle.cards.cardsInHand; ++handIndex) {
            const auto &card = battle.cards.hand[handIndex];
            if (!card.canUseOnAnyTarget(battle)) continue;

            if (card.requiresTarget()) {
                for (int targetIndex = 0;
                     targetIndex < battle.monsters.monsterCount;
                     ++targetIndex) {
                    if (
                        battle.monsters.arr[targetIndex].isTargetable() &&
                        card.canUse(battle, targetIndex, false)
                    ) {
                        result.push_back({
                            "PLAY:" + std::to_string(handIndex) + ":" +
                                std::to_string(targetIndex),
                            "PLAY_CARD",
                            handIndex,
                            targetIndex,
                        });
                    }
                }
            } else if (card.canUse(battle, 0, false)) {
                result.push_back({
                    "PLAY:" + std::to_string(handIndex) + ":-1",
                    "PLAY_CARD",
                    handIndex,
                    -1,
                });
            }
        }
    }
    result.push_back({"END", "END_TURN", -1, -1});
    return result;
}

std::string bridgeActionId(
    const sts::search::Action &action,
    const BattleContext &battle
) {
    switch (action.getActionType()) {
        case sts::search::ActionType::SINGLE_CARD_SELECT:
            selectionTask(battle);
            return "SELECT:" + std::to_string(action.getSelectIdx());
        case sts::search::ActionType::CARD: {
            const int handIndex = action.getSourceIdx();
            if (handIndex < 0 || handIndex >= battle.cards.cardsInHand) {
                throw BridgeError(
                    "teacher_search_action_mismatch",
                    "Search returned a card action with an invalid hand index"
                );
            }
            const auto &card = battle.cards.hand[handIndex];
            const int targetIndex = card.requiresTarget()
                ? action.getTargetIdx()
                : -1;
            return "PLAY:" + std::to_string(handIndex) + ":" +
                std::to_string(targetIndex);
        }
        case sts::search::ActionType::END_TURN:
            return "END";
        default:
            throw BridgeError(
                "teacher_search_unsupported_action",
                "Search produced an action outside the bridge card/end-turn scope"
            );
    }
}

sts::search::Action searchActionForLegalAction(const LegalAction &action) {
    if (action.kind == "SELECT_CARD") {
        return sts::search::Action(sts::search::ActionType::SINGLE_CARD_SELECT, action.handIndex);
    }
    if (action.kind == "END_TURN") {
        return sts::search::Action(sts::search::ActionType::END_TURN);
    }
    if (action.targetIndex < 0) {
        return sts::search::Action(
            sts::search::ActionType::CARD,
            action.handIndex
        );
    }
    return sts::search::Action(
        sts::search::ActionType::CARD,
        action.handIndex,
        action.targetIndex
    );
}

const LegalAction &requireBridgeLegalAction(
    const std::vector<LegalAction> &legal,
    const std::string &actionId
) {
    const auto match = std::find_if(
        legal.begin(),
        legal.end(),
        [&actionId](const LegalAction &action) {
            return action.actionId == actionId;
        }
    );
    if (match == legal.end()) {
        throw BridgeError(
            "teacher_search_action_mismatch",
            "Search root action is absent from the bridge legal action set: " +
                actionId
        );
    }
    return *match;
}

bool normalizeLagavulinNaturalWake(BattleContext &battle) {
    bool corrected = false;
    for (int index = 0; index < battle.monsters.monsterCount; ++index) {
        auto &monster = battle.monsters.arr[index];
        if (
            monster.id == sts::MonsterId::LAGAVULIN &&
            monster.moveHistory[0] == sts::MonsterMoveId::LAGAVULIN_ATTACK &&
            monster.hasStatus<sts::MS::ASLEEP>()
        ) {
            // The pinned engine schedules the natural turn-4 attack but leaves
            // the sleep powers behind. Match the real wake transition so the
            // state cannot claim that an attacking Lagavulin is still asleep.
            monster.removeStatus<sts::MS::ASLEEP>();
            monster.decrementStatus<sts::MS::METALLICIZE>(8);
            corrected = true;
        }
    }
    return corrected;
}

bool normalizePhilosopherBronzeOrbStrength(BattleContext &battle) {
    if (!battle.player.hasRelic<sts::RelicId::PHILOSOPHERS_STONE>()) {
        return false;
    }
    bool corrected = false;
    for (int index = 0; index < battle.monsters.monsterCount; ++index) {
        auto &monster = battle.monsters.arr[index];
        if (
            monster.id == sts::MonsterId::BRONZE_ORB &&
            monster.getStatus<sts::MS::STRENGTH>() == 2
        ) {
            // The pinned engine buffs empty Automaton summon slots and then
            // applies Philosopher's Stone again when constructing each Orb.
            // The game gives each summoned enemy exactly one Strength.
            monster.decrementStatus<sts::MS::STRENGTH>(1);
            corrected = true;
        }
    }
    return corrected;
}

int normalizeBurningBloodVictoryHeal(BattleContext &battle) {
    if (
        battle.outcome != Outcome::PLAYER_VICTORY ||
        !battle.player.hasRelic<sts::RelicId::BURNING_BLOOD>()
    ) {
        return 0;
    }
    const int hpBefore = battle.player.curHp;
    battle.player.heal(6);
    return battle.player.curHp - hpBefore;
}

int executeCorrectedSearchAction(
    BattleContext &battle,
    const sts::search::Action &action,
    SemanticCorrectionCounts *counts = nullptr
) {
#ifndef STS_CORRECTED_CARD_MECHANICS
    bool upgradedDisarm = false;
    bool targetHadArtifact = false;
    int target = -1;
    if (action.getActionType() == sts::search::ActionType::CARD) {
        const auto &card = battle.cards.hand[action.getSourceIdx()];
        upgradedDisarm =
            card.id == sts::CardId::DISARM && card.isUpgraded();
        if (upgradedDisarm) {
            target = action.getTargetIdx();
            targetHadArtifact =
                battle.monsters.arr[target].hasStatus<sts::MS::ARTIFACT>();
        }
    }

#endif
    bool redSlaverUsesEntangle = false;
    if (action.getActionType() == sts::search::ActionType::END_TURN) {
        for (int index = 0; index < battle.monsters.monsterCount; ++index) {
            auto &monster = battle.monsters.arr[index];
            if (
                monster.id == sts::MonsterId::RED_SLAVER &&
                monster.moveHistory[0] == sts::MonsterMoveId::RED_SLAVER_ENTANGLE &&
                monster.miscInfo == 0
            ) {
                monster.miscInfo = 1;
                redSlaverUsesEntangle = true;
            }
        }
    }

    action.execute(battle);
#ifndef STS_CORRECTED_CARD_MECHANICS
    if (upgradedDisarm && !targetHadArtifact) {
        battle.debuffEnemy<sts::MS::STRENGTH>(target, -1, false);
        if (counts != nullptr) ++counts->upgradedDisarm;
    }
#endif
    if (normalizeLagavulinNaturalWake(battle) && counts != nullptr) {
        ++counts->lagavulinNaturalWake;
    }
    if (normalizePhilosopherBronzeOrbStrength(battle) && counts != nullptr) {
        ++counts->philosopherBronzeOrbStrength;
    }
    const int burningBloodHeal = normalizeBurningBloodVictoryHeal(battle);
    if (burningBloodHeal > 0 && counts != nullptr) {
        ++counts->burningBloodVictoryHeal;
    }
    if (redSlaverUsesEntangle && counts != nullptr) {
        ++counts->redSlaverEntangle;
    }
    return burningBloodHeal;
}

json serializeLegalActions(const BattleContext &battle) {
    json result = json::array();
    for (const auto &action : enumerateLegalActions(battle)) {
        json value = {
            {"action_id", action.actionId},
            {"kind", action.kind},
        };
        if (action.kind == "SELECT_CARD") value["selection_index"] = action.handIndex;
        if (action.kind == "PLAY_CARD") {
            const auto &card = battle.cards.hand[action.handIndex];
            value["hand_index"] = action.handIndex;
            value["target_index"] =
                action.targetIndex < 0 ? json(nullptr) : json(action.targetIndex);
            value["card_id"] = sts::getCardEnumName(card.id);
            value["card_name"] = card.getName();
            value["cost_for_turn"] = static_cast<int>(card.costForTurn);
        }
        result.push_back(std::move(value));
    }
    return result;
}

int currentRelicCounter(
    const BattleContext &battle,
    const sts::RelicInstance &relic
) {
    switch (relic.id) {
        case sts::RelicId::HAPPY_FLOWER:
            return battle.player.happyFlowerCounter;
        case sts::RelicId::INCENSE_BURNER:
            return battle.player.incenseBurnerCounter;
        case sts::RelicId::INK_BOTTLE:
            return battle.player.inkBottleCounter;
        case sts::RelicId::NUNCHAKU:
            return battle.player.nunchakuCounter;
        case sts::RelicId::PEN_NIB:
            return battle.player.penNibCounter;
        case sts::RelicId::SUNDIAL:
            return battle.player.sundialCounter;
        default:
            return relic.data;
    }
}

json serializeRelics(
    const BattleContext &battle,
    const std::vector<sts::RelicInstance> &relics
) {
    json result = json::array();
    for (const auto &relic : relics) {
        const int index = static_cast<int>(relic.id);
        result.push_back({
            {"id", sts::relicEnumNames[index]},
            {"name", sts::getRelicName(relic.id)},
            {"counter", currentRelicCounter(battle, relic)},
        });
    }
    return result;
}

sts::Room rewardRoom(const std::string &room) {
    if (room == "MONSTER") return sts::Room::MONSTER;
    if (room == "ELITE") return sts::Room::ELITE;
    if (room == "BOSS") return sts::Room::BOSS;
    if (room == "EVENT") return sts::Room::EVENT;
    if (room == "REST") return sts::Room::REST;
    throw BridgeError(
        "invalid_reward_room",
        "reward room must be MONSTER, ELITE, BOSS, EVENT, or REST"
    );
}

sts::RelicId rewardRelicId(const std::string &value) {
    const int invalid = static_cast<int>(sts::RelicId::INVALID);
    for (int index = 0; index < invalid; ++index) {
        if (
            value == sts::relicEnumNames[index] ||
            value == sts::relicIds[index]
        ) {
            return static_cast<sts::RelicId>(index);
        }
    }
    throw BridgeError("unknown_reward_relic", "Unknown relic id: " + value);
}

bool isDV1Relic(sts::RelicId relic) {
    static const std::set<std::string> supported = {
        "BURNING_BLOOD", "VAJRA", "ANCHOR", "BAG_OF_MARBLES",
        "BRONZE_SCALES", "LANTERN", "ODDLY_SMOOTH_STONE", "ORICHALCUM",
        "AKABEKO", "PRESERVED_INSECT", "THE_BOOT", "STRIKE_DUMMY",
        "PAPER_PHROG", "CHAMPION_BELT", "CHEMICAL_X", "TUNGSTEN_ROD",
        "TORII", "RED_SKULL", "CENTENNIAL_PUZZLE", "GREMLIN_HORN",
        "HAPPY_FLOWER", "NUNCHAKU", "PEN_NIB", "ART_OF_WAR",
        "INK_BOTTLE", "KUNAI", "SHURIKEN", "BUSTED_CROWN",
        "PHILOSOPHERS_STONE", "SLAVERS_COLLAR", "VELVET_CHOKER",
        "MARK_OF_PAIN",
    };
    return supported.count(sts::relicEnumNames[static_cast<int>(relic)]) != 0;
}

std::string fingerprintBytesFnv1a64(const std::string &payload) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : payload) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

std::string fingerprintFnv1a64(const json &payload) {
    return fingerprintBytesFnv1a64(payload.dump());
}

struct ParsedCombatSnapshot {
    std::uint64_t sourceRewardSeed;
    std::uint64_t sourceRewardId;
    int ascension;
    int act;
    int floor;
    int currentHp;
    int maxHp;
    std::vector<sts::Card> cards;
    std::vector<sts::RelicInstance> relics;
    std::string fingerprint;
};

ParsedCombatSnapshot parseCombatSnapshot(const json &value) {
    const std::string schema = value.is_object() ? value.value("schema_version", "") : "";
    if (schema != "combat_snapshot_v1"
#ifdef STS_CORRECTED_CARD_MECHANICS
        && schema != "combat_snapshot_v2"
#endif
    ) {
        throw BridgeError("invalid_combat_snapshot", "Unsupported combat snapshot schema");
    }
    if (value.value("character", "") != "IRONCLAD") {
        throw BridgeError("invalid_combat_snapshot", "Snapshot character must be IRONCLAD");
    }
    if (!value.contains("fingerprint_fnv1a64") || !value["fingerprint_fnv1a64"].is_string()) {
        throw BridgeError("invalid_combat_snapshot", "Snapshot fingerprint is required");
    }
    const std::string fingerprint = value["fingerprint_fnv1a64"].get<std::string>();
    json unsignedValue = value;
    unsignedValue.erase("fingerprint_fnv1a64");
    if (fingerprint != fingerprintFnv1a64(unsignedValue)) {
        throw BridgeError("combat_snapshot_fingerprint_mismatch", "Snapshot fingerprint does not match its contents");
    }
    const auto requireUnsigned = [&value](const char *key) -> std::uint64_t {
        if (!value.contains(key) || !value[key].is_number_unsigned()) {
            throw BridgeError("invalid_combat_snapshot", std::string("Snapshot ") + key + " must be uint64");
        }
        return value[key].get<std::uint64_t>();
    };
    const auto requireInt = [&value](const char *key) -> int {
        if (!value.contains(key) || !value[key].is_number_integer()) {
            throw BridgeError("invalid_combat_snapshot", std::string("Snapshot ") + key + " must be an integer");
        }
        return value[key].get<int>();
    };
    const int ascension = requireInt("ascension");
    const int act = requireInt("act");
    const int floor = requireInt("floor");
    const int currentHp = requireInt("current_hp");
    const int maxHp = requireInt("max_hp");
    if (ascension < 0 || ascension > 20 || act < 1 || act > 2 || floor < 0 || floor > 54 ||
        maxHp <= 0 || currentHp <= 0 || currentHp > maxHp) {
        throw BridgeError("invalid_combat_snapshot", "Snapshot scalar state is out of range");
    }
    if (!value.contains("deck") || !value["deck"].is_array() || value["deck"].empty() ||
        value["deck"].size() > sts::Deck::MAX_SIZE) {
        throw BridgeError("invalid_combat_snapshot", "Snapshot deck must contain 1 to 96 cards");
    }
    std::vector<sts::Card> cards;
    for (const auto &card : value["deck"]) {
        if (!card.is_object() || !card.contains("string_id") || !card["string_id"].is_string() ||
            !card.contains("upgraded") || !card["upgraded"].is_boolean()) {
            throw BridgeError("invalid_combat_snapshot", "Snapshot deck card is malformed");
        }
        cards.emplace_back(
            supportedDeckCardId(card["string_id"].get<std::string>()),
            card["upgraded"].get<bool>() ? 1 : 0
        );
        if (schema == "combat_snapshot_v2" && cards.back().id == sts::CardId::SEARING_BLOW) {
            if (!card.contains("upgrade_count") || !card["upgrade_count"].is_number_integer()) {
                throw BridgeError("invalid_combat_snapshot", "Searing Blow requires an integer upgrade_count in V2");
            }
            const auto count = card["upgrade_count"].get<std::int64_t>();
            if (count < 0 || count > 32767 || (count > 0) != cards.back().isUpgraded()) {
                throw BridgeError("invalid_combat_snapshot", "Searing Blow upgrade_count is out of range or inconsistent");
            }
            cards.back().misc = static_cast<std::int16_t>(count);
        }
    }
    if (!value.contains("relics") || !value["relics"].is_array()) {
        throw BridgeError("invalid_combat_snapshot", "Snapshot relics must be an array");
    }
    std::vector<sts::RelicInstance> relics;
    std::set<int> seen;
    for (const auto &relic : value["relics"]) {
        if (!relic.is_object() || !relic.contains("id") || !relic["id"].is_string() ||
            !relic.contains("counter") || !relic["counter"].is_number_integer()) {
            throw BridgeError("invalid_combat_snapshot", "Snapshot relic is malformed");
        }
        const auto relicId = rewardRelicId(relic["id"].get<std::string>());
        if (!isDV1Relic(relicId)) {
            throw BridgeError("unsupported_snapshot_relic", "Snapshot contains a relic outside D-v1 support");
        }
        if (!seen.insert(static_cast<int>(relicId)).second) {
            throw BridgeError("invalid_combat_snapshot", "Snapshot contains a duplicate relic");
        }
        relics.push_back({relicId, relic["counter"].get<int>()});
    }
    return {
        requireUnsigned("source_reward_seed"), requireUnsigned("source_reward_id"),
        ascension, act, floor, currentHp, maxHp, std::move(cards), std::move(relics), fingerprint,
    };
}

json serializeRewardCard(const sts::Card &card, int index) {
    const int rarity = static_cast<int>(card.getRarity());
    json result = {
        {"choice", "CARD_" + std::to_string(index)},
        {"enum_id", sts::getCardEnumName(card.id)},
        {"string_id", sts::getCardStringId(card.id)},
        {"name", card.getName()},
        {"rarity", sts::cardRarityStrings[rarity]},
        {"upgraded", card.isUpgraded()},
    };
#ifdef STS_CORRECTED_CARD_MECHANICS
    if (card.id == sts::CardId::SEARING_BLOW) result["upgrade_count"] = card.getUpgraded();
#endif
    return result;
}

json serializeRewardDeck(const sts::GameContext &game) {
    json cards = json::array();
    for (int index = 0; index < game.deck.size(); ++index) {
        const auto &card = game.deck.cards[index];
        cards.push_back({
            {"enum_id", sts::getCardEnumName(card.id)},
            {"string_id", sts::getCardStringId(card.id)},
            {"name", card.getName()},
            {"upgraded", card.isUpgraded()},
        });
#ifdef STS_CORRECTED_CARD_MECHANICS
        if (card.id == sts::CardId::SEARING_BLOW) cards.back()["upgrade_count"] = card.getUpgraded();
#endif
    }
    return cards;
}

json serializeState(
    const BattleContext &battle,
    const std::string &scenarioId,
    const std::string &deckPreset,
    const json &loadoutMetadata,
    int startingHp,
    int enemyDamageTaken,
    int selfHpLoss,
    int decisionId,
    int act,
    const std::vector<sts::RelicInstance> &relics
) {
    json monsters = json::array();
    for (int index = 0; index < battle.monsters.monsterCount; ++index) {
        const auto &monster = battle.monsters.arr[index];
        const int moveIndex = static_cast<int>(monster.moveHistory[0]);
        const int previousMoveIndex = static_cast<int>(monster.moveHistory[1]);
        const bool attacking = monster.isAttacking();
        sts::DamageInfo damage;
        if (attacking) damage = monster.getMoveBaseDamage(battle);

        json serializedMonster = {
            {"index", index},
            {"id", sts::monsterIdStrings[static_cast<int>(monster.id)]},
            {"name", monster.getName()},
            {"current_hp", monster.curHp},
            {"max_hp", monster.maxHp},
            {"block", monster.block},
            {"is_targetable", monster.isTargetable()},
            {"is_gone", monster.isDeadOrEscaped()},
            {"half_dead", monster.isHalfDead()},
            {"move_id", moveIndex},
            {"move_name", sts::monsterMoveStrings[moveIndex]},
            {"previous_move_name", sts::monsterMoveStrings[previousMoveIndex]},
            {"has_used_entangle", (
                monster.id == sts::MonsterId::RED_SLAVER && monster.miscInfo != 0
            )},
            {"is_attacking", attacking},
            {"base_damage", attacking ? damage.damage : 0},
            {"adjusted_damage", attacking
                ? monster.calculateDamageToPlayer(battle, damage.damage)
                : 0},
            {"hits", attacking ? damage.attackCount : 0},
            {"powers", serializeMonsterPowers(monster)},
        };
        if (
            monster.id == sts::MonsterId::BRONZE_ORB &&
            monster.hasStatus<sts::MS::STASIS>()
        ) {
            const auto &card = battle.cards.stasisCards[std::min(index, 1)];
            if (card.id != sts::CardId::INVALID) {
                serializedMonster["stasis_card"] = serializeCard(card, battle);
            }
        }
        monsters.push_back(std::move(serializedMonster));
    }

    const int outcomeIndex = static_cast<int>(battle.outcome);
    json state = {
        {"schema_version", 1},
        {"scenario_id", scenarioId},
        {"deck_preset", deckPreset},
        {"seed", battle.seed},
        {"ascension", battle.ascension},
        {"act", act},
        {"floor", battle.floorNum},
        {"decision_id", decisionId},
        {"turn", battle.turn},
        {"input_state", inputStateName(battle.inputState)},
        {"outcome", sts::battleOutcomeStrings[outcomeIndex]},
        {"terminal", battle.outcome != Outcome::UNDECIDED},
        {"reward", battle.outcome == Outcome::PLAYER_VICTORY
            ? 1.0
            : battle.outcome == Outcome::PLAYER_LOSS ? -1.0 : 0.0},
        {"draw_pile_top", "back"},
        {"combat_accounting", {
            {"starting_hp", startingHp},
            {"total_hp_loss", enemyDamageTaken + selfHpLoss},
            {"enemy_damage_taken", enemyDamageTaken},
            {"self_hp_loss", selfHpLoss},
        }},
        {"player", {
            {"current_hp", battle.player.curHp},
            {"max_hp", battle.player.maxHp},
            {"block", battle.player.block},
            {"energy", battle.player.energy},
            {"powers", serializePlayerPowers(battle.player)},
        }},
        {"relics", serializeRelics(battle, relics)},
        {"monsters", std::move(monsters)},
        {"hand", serializeHand(battle)},
        {"draw_pile", serializeCardPile(battle.cards.drawPile, battle)},
        {"discard_pile", serializeCardPile(battle.cards.discardPile, battle)},
        {"exhaust_pile", serializeCardPile(battle.cards.exhaustPile, battle)},
    };
    if (!loadoutMetadata.is_null()) {
        state["loadout"] = loadoutMetadata;
    }
    bool previewUpgrades = battle.inputState == InputState::CARD_SELECT &&
        battle.cardSelectInfo.cardSelectTask == sts::CardSelectTask::ARMAMENTS;
    for (int i = 0; i < battle.cards.cardsInHand; ++i) {
        previewUpgrades = previewUpgrades || battle.cards.hand[i].id == sts::CardId::ARMAMENTS;
    }
    if (previewUpgrades) {
        auto previews = json::array();
        for (int i = 0; i < battle.cards.cardsInHand; ++i) {
            auto card = battle.cards.hand[i];
            if (card.canUpgrade()) {
                // Exactly the deterministic operation used by chooseArmamentsCard;
                // no battle execution, queue resolution or RNG is involved.
                card.upgrade();
                previews.push_back(serializeCard(card, battle));
            } else {
                previews.push_back(nullptr);
            }
        }
        state["hand_upgrade_previews"] = std::move(previews);
    }
    if (battle.outcome == Outcome::UNDECIDED && battle.inputState == InputState::CARD_SELECT) {
        state["card_selection"] = {{"schema_version", "combat_card_selection_v1"},
                                   {"task", selectionTask(battle)},
                                   {"copies_created", battle.cardSelectInfo.cardSelectTask == sts::CardSelectTask::DUAL_WIELD ? battle.cardSelectInfo.data0 : 0},
                                   {"source_card", serializeCard(battle.curCardQueueItem.card, battle)}};
    }
    return state;
}

class BridgeSession {
public:
    json handle(const json &request) {
        validateEnvelope(request);
        const std::string op = request.at("op").get<std::string>();
        if (op == "hello") return hello();
        if (op == "reset") return reset(request);
        if (op == "legal_actions") return legalActions();
        if (op == "search") return search(request);
        if (op == "public_state") return publicState(request);
        if (op == "step") return step(request);
        if (op == "reward_reset") return rewardReset(request);
        if (op == "sample_card_reward") return sampleCardReward(request);
        if (op == "apply_card_reward_choice") {
            return applyCardRewardChoice(request);
        }
        if (op == "advance_reward_act") return advanceRewardAct(request);
        if (op == "obtain_reward_relic") return obtainRewardRelic(request);
        if (op == "remove_reward_card") return removeRewardCard(request);
        if (op == "upgrade_reward_card") return upgradeRewardCard(request);
        if (op == "export_combat_snapshot") return exportCombatSnapshot();
        if (op == "close") return json::object();
        throw BridgeError("unknown_operation", "Unknown operation: " + op);
    }

    bool isClose(const json &request) const {
        return request.is_object() && request.value("op", "") == "close";
    }

private:
    std::unique_ptr<BattleContext> battle_;
    std::string scenarioId_;
    std::string deckPreset_ = "starter";
    json loadoutMetadata_ = nullptr;
    int startingHp_ = 0;
    int enemyDamageTaken_ = 0;
    int selfHpLoss_ = 0;
    int decisionId_ = 0;
    int act_ = 1;
    std::vector<sts::RelicInstance> relics_;
    std::unique_ptr<sts::GameContext> rewardGame_;
    sts::CardReward pendingCardReward_;
    bool hasPendingCardReward_ = false;
    std::uint64_t rewardId_ = 0;
    int rewardFloor_ = -1;
    int floorRewardIndex_ = 0;
    std::vector<int> knownDrawTop_; // next draw first; populated by executed public movements
    bool publicStateMode_ = false;

    static std::uint32_t publicSeed(const json &request) {
        if (!request.contains("public_state_seed") ||
            !request["public_state_seed"].is_number_unsigned() ||
            request["public_state_seed"].get<std::uint64_t>() > 0xffffffffULL) {
            throw BridgeError("invalid_request", "public_state_seed must be uint32");
        }
        return request["public_state_seed"].get<std::uint32_t>();
    }

    void samplePublicState(BattleContext &state, std::uint32_t seed) const {
        const auto size = state.cards.drawPile.size();
        if (knownDrawTop_.size() > size) throw BridgeError("draw_memory", "Known prefix exceeds draw pile");
        for (std::size_t i = 0; i < knownDrawTop_.size(); ++i) {
            if (state.cards.drawPile[size - 1 - i].getUniqueId() != knownDrawTop_[i]) {
                throw BridgeError("draw_memory", "Known draw prefix does not match executed state");
            }
        }
        auto unknownEnd = state.cards.drawPile.end() - knownDrawTop_.size();
        // Canonicalize before applying a fresh permutation: an inner search seed
        // must not encode the outer world's original hidden draw order.
        std::sort(state.cards.drawPile.begin(), unknownEnd,
            [&state](const CardInstance &a, const CardInstance &b) {
                return serializeCard(a, state).dump() < serializeCard(b, state).dump();
            });
        auto stream = [seed](std::uint64_t domain) {
            return sts::Random::murmurHash3((static_cast<std::uint64_t>(seed) << 32) ^ domain);
        };
        java::Collections::shuffle(state.cards.drawPile.begin(), unknownEnd, java::Random(stream(1)));
        state.aiRng = sts::Random(stream(2));
        state.cardRandomRng = sts::Random(stream(3));
        state.miscRng = sts::Random(stream(4));
        state.monsterHpRng = sts::Random(stream(5));
        state.potionRng = sts::Random(stream(6));
        state.shuffleRng = sts::Random(stream(7));
    }

    json publicState(const json &request) {
        requireBattle();
#ifndef STS_PUBLIC_DRAW_MEMORY
        throw BridgeError("missing_capability", "Rebuild the public draw-memory extension");
#endif
        if (!request.contains("decision_id") || !request["decision_id"].is_number_integer() ||
            request["decision_id"].get<int>() != decisionId_) {
            throw BridgeError("stale_decision", "public_state requires current decision_id");
        }
        if (request.contains("public_state_seed")) samplePublicState(*battle_, publicSeed(request));
        publicStateMode_ = true;
        return observation();
    }

    static void validateEnvelope(const json &request) {
        if (!request.is_object()) {
            throw BridgeError("invalid_request", "Request must be a JSON object");
        }
        if (!request.contains("v") || !request["v"].is_number_integer()) {
            throw BridgeError("invalid_request", "Request v must be integer 1");
        }
        if (request["v"].get<int>() != kProtocolVersion) {
            throw BridgeError("unsupported_version", "Only protocol version 1 is supported");
        }
        if (!request.contains("op") || !request["op"].is_string()) {
            throw BridgeError("invalid_request", "Request op must be a string");
        }
    }

    void requireBattle() const {
        if (!battle_) {
            throw BridgeError("battle_not_initialized", "Call reset before this operation");
        }
    }

    json hello() const {
        return {
            {"backend", "sts_lightspeed"},
            {"protocol_version", kProtocolVersion},
            {"transport", "jsonl_stdio"},
            {"scenarios", {
                "cultist",
                "jaw_worm",
                "two_louse",
                "gremlin_gang",
                "gremlin_nob",
                "lagavulin",
                "three_sentries",
                "small_slimes",
                "lots_of_slimes",
                "blue_slaver",
                "red_slaver",
                "looter",
                "exordium_thugs",
                "exordium_wildlife",
                "two_fungi_beasts",
                "large_slime",
                "three_louse",
                "spheric_guardian",
                "chosen",
                "shell_parasite",
                "three_byrds",
                "two_thieves",
                "chosen_and_byrds",
                "sentry_and_sphere",
                "cultist_and_chosen",
                "three_cultist",
                "shelled_parasite_and_fungi",
                "snecko",
                "snake_plant",
                "centurion_and_healer",
            }},
            {"deck_presets", {
                "starter",
                "elite_transition",
                "boss_ready",
            }},
            {"capabilities", {
                {"reset", true},
                {"player_current_hp_reset", true},
                {"deck_preset_reset", true},
                {"deck_spec_reset", true},
                {"combat_snapshot_v1_reset", true},
#ifdef STS_CORRECTED_CARD_MECHANICS
                {"combat_snapshot_v2_reset", true},
#endif
                {"combat_accounting", true},
                {"legal_actions", true},
                {"battle_scum_searcher2", true},
                {"privileged_root_search", true},
                {"root_action_coverage_v1", true},
                {"root_action_minimum_visits_v1", true},
                {"step", true},
                {"card_play", true},
                {"end_turn", true},
                {"potions", false},
                {"card_selection", true},
                {"native_card_rewards", true},
                {"persistent_reward_context", true},
                {"native_variable_reward_size", true},
                {"native_reward_relic_mutation", true},
                {"native_reward_deck_removal", true},
                {"native_reward_deck_upgrade", true},
                {"export_combat_snapshot", true},
            }},
        };
    }

    json reset(const json &request) {
        if (!request.contains("scenario_id") || !request["scenario_id"].is_string()) {
            throw BridgeError("invalid_request", "reset scenario_id must be a string");
        }
        if (!request.contains("seed") || !request["seed"].is_number_unsigned()) {
            throw BridgeError("invalid_request", "reset seed must be an unsigned integer");
        }
        const std::string scenarioId = request["scenario_id"].get<std::string>();
        const auto encounter = scenarioEncounter(scenarioId);
        const bool hasSnapshot = request.contains("combat_snapshot");
        if (
            (request.contains("deck_spec") && request.contains("deck_preset")) ||
            (hasSnapshot && (
                request.contains("deck_spec") || request.contains("deck_preset") ||
                request.contains("player_current_hp") || request.contains("ascension")
            ))
        ) {
            throw BridgeError(
                "invalid_request",
                "combat_snapshot, deck_spec, deck_preset, player_current_hp, and ascension have incompatible combinations"
            );
        }
        if (
            request.contains("deck_preset") &&
            !request["deck_preset"].is_string()
        ) {
            throw BridgeError(
                "invalid_request",
                "reset deck_preset must be a string"
            );
        }
        std::string deckPreset = request.value("deck_preset", "starter");
        std::unique_ptr<ParsedDeckSpec> parsedDeckSpec;
        std::unique_ptr<ParsedCombatSnapshot> parsedSnapshot;
        if (hasSnapshot) {
            parsedSnapshot = std::make_unique<ParsedCombatSnapshot>(
                parseCombatSnapshot(request["combat_snapshot"])
            );
            deckPreset = request["combat_snapshot"]["schema_version"].get<std::string>();
        }
        if (request.contains("deck_spec")) {
            parsedDeckSpec = std::make_unique<ParsedDeckSpec>(
                parseDeckSpec(request["deck_spec"])
            );
            deckPreset = parsedDeckSpec->basePreset;
        }
        const std::uint64_t seed = request["seed"].get<std::uint64_t>();
        const int ascension = parsedSnapshot
            ? parsedSnapshot->ascension
            : request.value("ascension", 0);
        if (ascension < 0 || ascension > 20) {
            throw BridgeError("invalid_request", "reset ascension must be between 0 and 20");
        }

        sts::GameContext game(sts::CharacterClass::IRONCLAD, seed, ascension);
        if (parsedSnapshot) {
            game.deck = sts::Deck{};
            for (const auto &card : parsedSnapshot->cards) game.deck.obtainRaw(card);
            game.relics = sts::RelicContainer{};
            for (const auto &relic : parsedSnapshot->relics) game.relics.add(relic);
            game.maxHp = parsedSnapshot->maxHp;
            game.curHp = parsedSnapshot->currentHp;
            game.act = parsedSnapshot->act;
        } else if (parsedDeckSpec) {
            applyDeckSpec(game, *parsedDeckSpec);
        } else {
            applyDeckPreset(game, deckPreset);
        }
        if (!parsedSnapshot) game.act = scenarioAct(scenarioId);
        if (request.contains("player_current_hp")) {
            if (!request["player_current_hp"].is_number_integer()) {
                throw BridgeError(
                    "invalid_request",
                    "reset player_current_hp must be an integer"
                );
            }
            const int playerCurrentHp = request["player_current_hp"].get<int>();
            if (playerCurrentHp <= 0 || playerCurrentHp > game.maxHp) {
                throw BridgeError(
                    "invalid_request",
                    "reset player_current_hp must be between 1 and player max HP"
                );
            }
            game.curHp = playerCurrentHp;
        }
        game.curRoom = scenarioRoom(scenarioId);
        // P0 represents the first Act 1 hallway combat. Keeping floor=1 also
        // aligns BattleContext's seed+floor RNG initialization with the game.
        // Mirror GameContext::transitionToMapNode: the constructor seeded these
        // streams for floor 0, but encounter generation consumes miscRng after
        // the floor advances. Louse color selection depends on that stream.
        game.floorNum = parsedSnapshot
            ? parsedSnapshot->floor
            : (scenarioAct(scenarioId) == 2 ? 18 : 1);
        const auto roomRandom = sts::Random(
            seed + static_cast<std::uint64_t>(game.floorNum)
        );
        game.miscRng = roomRandom;
        game.shuffleRng = roomRandom;
        game.cardRandomRng = roomRandom;
        game.curMapNodeX = -100;
        game.curMapNodeY = -100;

        auto nextBattle = std::make_unique<BattleContext>();
        nextBattle->init(game, encounter);
        if (
            nextBattle->outcome == Outcome::UNDECIDED &&
            nextBattle->inputState != InputState::PLAYER_NORMAL
        ) {
            throw BridgeError(
                "unsupported_input_state",
                std::string("reset stopped in unsupported input state ") +
                    inputStateName(nextBattle->inputState)
            );
        }

        battle_ = std::move(nextBattle);
        knownDrawTop_.clear();
        publicStateMode_ = false;
        relics_ = game.relics.relics;
        scenarioId_ = scenarioId;
        deckPreset_ = deckPreset;
        if (parsedSnapshot) {
            loadoutMetadata_ = {
                {"source", deckPreset},
                {"snapshot_fingerprint_fnv1a64", parsedSnapshot->fingerprint},
                {"source_reward_seed", parsedSnapshot->sourceRewardSeed},
                {"source_reward_id", parsedSnapshot->sourceRewardId},
            };
        } else {
            loadoutMetadata_ = parsedDeckSpec ? parsedDeckSpec->metadata : json(nullptr);
        }
        startingHp_ = battle_->player.curHp;
        enemyDamageTaken_ = 0;
        selfHpLoss_ = 0;
        decisionId_ = 0;
        act_ = game.act;
        return observation();
    }

    void requireRewardGame() const {
        if (!rewardGame_) {
            throw BridgeError(
                "reward_not_initialized",
                "Call reward_reset before this operation"
            );
        }
    }

    json rewardState() const {
        requireRewardGame();
        json relics = json::array();
        for (const auto &relic : rewardGame_->relics.relics) {
            const int index = static_cast<int>(relic.id);
            relics.push_back({
                {"id", sts::relicEnumNames[index]},
                {"name", sts::getRelicName(relic.id)},
                {"counter", relic.data},
            });
        }
        return {
            {"seed", rewardGame_->seed},
#ifdef STS_CORRECTED_CARD_MECHANICS
            {"combat_snapshot_schema", kCombatSnapshotSchema},
#endif
            {"ascension", rewardGame_->ascension},
            {"act", rewardGame_->act},
            {"floor", rewardGame_->floorNum},
            {"current_hp", rewardGame_->curHp},
            {"max_hp", rewardGame_->maxHp},
            {"deck", serializeRewardDeck(*rewardGame_)},
            {"relics", std::move(relics)},
        };
    }

    json exportCombatSnapshot() const {
        requireRewardGame();
        if (hasPendingCardReward_) {
            throw BridgeError(
                "reward_choice_pending",
                "Apply or skip the pending reward before exporting a combat snapshot"
            );
        }
        json relics = json::array();
        for (const auto &relic : rewardGame_->relics.relics) {
            relics.push_back({
                {"id", sts::relicEnumNames[static_cast<int>(relic.id)]},
                {"counter", relic.data},
            });
        }
        json snapshot = {
            {"schema_version", kCombatSnapshotSchema},
            {"character", "IRONCLAD"},
            {"source_reward_seed", rewardGame_->seed},
            {"source_reward_id", rewardId_},
            {"ascension", rewardGame_->ascension},
            {"act", rewardGame_->act},
            {"floor", rewardGame_->floorNum},
            {"current_hp", rewardGame_->curHp},
            {"max_hp", rewardGame_->maxHp},
            {"deck", serializeRewardDeck(*rewardGame_)},
            {"relics", std::move(relics)},
        };
        snapshot["fingerprint_fnv1a64"] = fingerprintFnv1a64(snapshot);
        return {{"combat_snapshot", std::move(snapshot)}};
    }

    json rewardReset(const json &request) {
        if (!request.contains("seed") || !request["seed"].is_number_unsigned()) {
            throw BridgeError("invalid_request", "reward_reset seed must be uint64");
        }
        const int ascension = request.value("ascension", 0);
        if (ascension < 0 || ascension > 20) {
            throw BridgeError(
                "invalid_request",
                "reward_reset ascension must be between 0 and 20"
            );
        }
        auto game = std::make_unique<sts::GameContext>(
            sts::CharacterClass::IRONCLAD,
            request["seed"].get<std::uint64_t>(),
            ascension
        );

        if (request.contains("relics")) {
            if (!request["relics"].is_array()) {
                throw BridgeError("invalid_request", "reward_reset relics must be an array");
            }
            game->relics = sts::RelicContainer{};
            std::set<int> seen;
            for (std::size_t index = 0; index < request["relics"].size(); ++index) {
                const auto &value = request["relics"][index];
                if (!value.is_object() || !value.contains("id") || !value["id"].is_string()) {
                    throw BridgeError("invalid_request", "reward_reset relic id must be a string");
                }
                const auto relic = rewardRelicId(value["id"].get<std::string>());
                if (!seen.insert(static_cast<int>(relic)).second) {
                    throw BridgeError("invalid_request", "reward_reset relics contain a duplicate");
                }
                game->obtainRelic(relic);
                if (value.contains("counter")) {
                    if (!value["counter"].is_number_integer()) {
                        throw BridgeError("invalid_request", "reward relic counter must be an integer");
                    }
                    game->relics.getRelicValueRef(relic) = value["counter"].get<int>();
                }
            }
        }

        const int act = request.value("act", 1);
        if (act < 1 || act > 2) {
            throw BridgeError("invalid_request", "reward_reset act must be 1 or 2");
        }
        if (act == 2) game->transitionToAct(2);
        const int floor = request.value("floor", act == 1 ? 0 : 18);
        if (floor < 0 || floor > 54) {
            throw BridgeError("invalid_request", "reward_reset floor is out of range");
        }
        game->floorNum = floor;

        rewardGame_ = std::move(game);
        pendingCardReward_ = sts::CardReward{};
        hasPendingCardReward_ = false;
        rewardId_ = 0;
        rewardFloor_ = -1;
        floorRewardIndex_ = 0;
        return {{"reward_state", rewardState()}};
    }

    json sampleCardReward(const json &request) {
        requireRewardGame();
        if (hasPendingCardReward_) {
            throw BridgeError(
                "reward_choice_pending",
                "Apply or skip the pending reward before sampling another"
            );
        }
        if (!request.contains("room") || !request["room"].is_string()) {
            throw BridgeError("invalid_request", "sample_card_reward room must be a string");
        }
        if (!request.contains("floor") || !request["floor"].is_number_integer()) {
            throw BridgeError("invalid_request", "sample_card_reward floor must be an integer");
        }
        const int floor = request["floor"].get<int>();
        if (floor < rewardGame_->floorNum || floor > 54) {
            throw BridgeError(
                "invalid_reward_floor",
                "sample_card_reward floor must be monotonic and at most 54"
            );
        }
        rewardGame_->floorNum = floor;
        rewardGame_->curRoom = rewardRoom(request["room"].get<std::string>());
        if (rewardFloor_ == floor) {
            ++floorRewardIndex_;
        } else {
            rewardFloor_ = floor;
            floorRewardIndex_ = 1;
        }
        pendingCardReward_ = rewardGame_->createCardReward(rewardGame_->curRoom);
        hasPendingCardReward_ = true;
        ++rewardId_;

        json choices = json::array();
        for (int index = 0; index < pendingCardReward_.size(); ++index) {
            choices.push_back(serializeRewardCard(pendingCardReward_[index], index));
        }
        return {
            {"reward_id", rewardId_},
            {"floor_reward_index", floorRewardIndex_},
            {"choices", std::move(choices)},
            {"can_skip", true},
            {"can_singing_bowl", rewardGame_->hasRelic(sts::RelicId::SINGING_BOWL)},
            {"reward_state", rewardState()},
        };
    }

    json applyCardRewardChoice(const json &request) {
        requireRewardGame();
        if (!hasPendingCardReward_) {
            throw BridgeError("no_pending_reward", "No card reward is pending");
        }
        if (!request.contains("reward_id") || !request["reward_id"].is_number_unsigned()) {
            throw BridgeError("invalid_request", "reward_id must be uint64");
        }
        if (request["reward_id"].get<std::uint64_t>() != rewardId_) {
            throw BridgeError("stale_reward_id", "reward_id does not match the pending reward");
        }
        if (!request.contains("choice") || !request["choice"].is_string()) {
            throw BridgeError("invalid_request", "reward choice must be a string");
        }
        const std::string choice = request["choice"].get<std::string>();
        if (choice == "SKIP") {
            // Native skip leaves the deck and player state unchanged.
        } else if (choice == "SINGING_BOWL") {
            if (!rewardGame_->hasRelic(sts::RelicId::SINGING_BOWL)) {
                throw BridgeError("illegal_reward_choice", "Singing Bowl is not owned");
            }
            rewardGame_->playerIncreaseMaxHp(2);
        } else if (choice.rfind("CARD_", 0) == 0) {
            const std::string suffix = choice.substr(5);
            if (suffix.size() != 1 || suffix[0] < '0' || suffix[0] > '3') {
                throw BridgeError("illegal_reward_choice", "Invalid card reward choice");
            }
            const int index = suffix[0] - '0';
            if (index >= pendingCardReward_.size()) {
                throw BridgeError("illegal_reward_choice", "Card reward choice is out of range");
            }
            rewardGame_->obtainCard(pendingCardReward_[index]);
        } else {
            throw BridgeError("illegal_reward_choice", "Unknown card reward choice");
        }

        hasPendingCardReward_ = false;
        pendingCardReward_ = sts::CardReward{};
        return {
            {"applied_reward_id", rewardId_},
            {"choice", choice},
            {"reward_state", rewardState()},
        };
    }

    json advanceRewardAct(const json &request) {
        requireRewardGame();
        if (hasPendingCardReward_) {
            throw BridgeError(
                "reward_choice_pending",
                "Apply or skip the pending reward before advancing the act"
            );
        }
        if (!request.contains("act") || !request["act"].is_number_integer()) {
            throw BridgeError("invalid_request", "advance_reward_act act must be an integer");
        }
        const int target = request["act"].get<int>();
        if (target != rewardGame_->act + 1 || target > 2) {
            throw BridgeError(
                "invalid_reward_act",
                "D-v1 only supports the native Act 1 to Act 2 transition"
            );
        }
        rewardGame_->transitionToAct(target);
        rewardFloor_ = -1;
        floorRewardIndex_ = 0;
        return {{"reward_state", rewardState()}};
    }

    json obtainRewardRelic(const json &request) {
        requireRewardGame();
        if (hasPendingCardReward_) {
            throw BridgeError("reward_choice_pending", "Resolve the pending reward before obtaining a relic");
        }
        if (!request.contains("relic_id") || !request["relic_id"].is_string()) {
            throw BridgeError("invalid_request", "obtain_reward_relic relic_id must be a string");
        }
        const auto relic = rewardRelicId(request["relic_id"].get<std::string>());
        if (!isDV1Relic(relic)) {
            throw BridgeError("unsupported_reward_relic", "Relic is outside D-v1 support");
        }
        if (rewardGame_->hasRelic(relic)) {
            throw BridgeError("duplicate_reward_relic", "Reward relic is already owned");
        }
        rewardGame_->obtainRelic(relic);
        return {
            {"obtained_relic_id", sts::relicEnumNames[static_cast<int>(relic)]},
            {"reward_state", rewardState()},
        };
    }

    json removeRewardCard(const json &request) {
        requireRewardGame();
        if (hasPendingCardReward_) {
            throw BridgeError("reward_choice_pending", "Resolve the pending reward before removing a card");
        }
        if (!request.contains("card_id") || !request["card_id"].is_string()) {
            throw BridgeError("invalid_request", "remove_reward_card card_id must be a string");
        }
        const auto cardId = supportedDeckCardId(request["card_id"].get<std::string>());
        int selected = -1;
        for (int index = 0; index < rewardGame_->deck.size(); ++index) {
            if (
                rewardGame_->deck.cards[index].getId() == cardId &&
                !rewardGame_->deck.cards[index].isUpgraded()
            ) {
                selected = index;
                break;
            }
        }
        if (selected < 0) {
            for (int index = 0; index < rewardGame_->deck.size(); ++index) {
                if (rewardGame_->deck.cards[index].getId() == cardId) {
                    selected = index;
                    break;
                }
            }
        }
        if (selected < 0) {
            throw BridgeError("reward_card_not_found", "Requested card is not present in the reward deck");
        }
        const auto removed = serializeRewardCard(rewardGame_->deck.cards[selected], selected);
        rewardGame_->deck.remove(*rewardGame_, selected);
        return {
            {"removed_card", removed},
            {"reward_state", rewardState()},
        };
    }

    json upgradeRewardCard(const json &request) {
        requireRewardGame();
        if (hasPendingCardReward_) {
            throw BridgeError("reward_choice_pending", "Resolve the pending reward before upgrading a card");
        }
        if (!request.contains("deck_index") || !request["deck_index"].is_number_integer()) {
            throw BridgeError("invalid_request", "upgrade_reward_card deck_index must be an integer");
        }
        const int index = request["deck_index"].get<int>();
        if (index < 0 || index >= rewardGame_->deck.size()) {
            throw BridgeError("reward_card_not_found", "Upgrade deck_index is out of range");
        }
        if (!rewardGame_->deck.cards[index].canUpgrade()) {
            throw BridgeError("reward_card_not_upgradeable", "Requested reward card cannot be upgraded");
        }
        const auto before = serializeRewardCard(rewardGame_->deck.cards[index], index);
        rewardGame_->deck.upgrade(index);
        const auto after = serializeRewardCard(rewardGame_->deck.cards[index], index);
        return {
            {"upgraded_card_before", before},
            {"upgraded_card_after", after},
            {"reward_state", rewardState()},
        };
    }

    json legalActions() const {
        requireBattle();
        return {
            {"decision_id", decisionId_},
            {"legal_actions", serializeLegalActions(*battle_)},
        };
    }

    json search(const json &request) const {
        requireBattle();
        if (battle_->outcome != Outcome::UNDECIDED) {
            throw BridgeError(
                "battle_terminal",
                "Cannot search a terminal battle"
            );
        }
        if (
            !request.contains("decision_id") ||
            !request["decision_id"].is_number_integer()
        ) {
            throw BridgeError(
                "invalid_request",
                "search decision_id must be an integer"
            );
        }
        if (request["decision_id"].get<int>() != decisionId_) {
            throw BridgeError(
                "stale_decision",
                "search decision_id does not match current state"
            );
        }
        if (
            !request.contains("simulations") ||
            !request["simulations"].is_number_integer()
        ) {
            throw BridgeError(
                "invalid_request",
                "search simulations must be an integer"
            );
        }
        const auto simulations = request["simulations"].get<std::int64_t>();
        if (simulations <= 0 || simulations > kMaxSearchSimulations) {
            throw BridgeError(
                "invalid_request",
                "search simulations must be between 1 and 1000000"
            );
        }
        if (
            !request.contains("search_seed") ||
            !request["search_seed"].is_number_unsigned()
        ) {
            throw BridgeError(
                "invalid_request",
                "search search_seed must be an unsigned 32-bit integer"
            );
        }
        const auto searchSeedValue = request["search_seed"].get<std::uint64_t>();
        if (searchSeedValue > std::numeric_limits<std::uint32_t>::max()) {
            throw BridgeError(
                "invalid_request",
                "search search_seed must be an unsigned 32-bit integer"
            );
        }
        const auto legal = enumerateLegalActions(*battle_);
        const auto searchSeed = static_cast<std::uint32_t>(searchSeedValue);
        bool ensureRootActionCoverage = false;
        if (request.contains("ensure_root_action_coverage")) {
            if (!request["ensure_root_action_coverage"].is_boolean()) {
                throw BridgeError(
                    "invalid_request",
                    "search ensure_root_action_coverage must be boolean"
                );
            }
            ensureRootActionCoverage = request["ensure_root_action_coverage"].get<bool>();
        }
        std::int64_t minimumRootActionVisits = 0;
        if (request.contains("minimum_root_action_visits")) {
            if (!request["minimum_root_action_visits"].is_number_unsigned()) {
                throw BridgeError(
                    "invalid_request",
                    "search minimum_root_action_visits must be an unsigned integer"
                );
            }
            minimumRootActionVisits = request["minimum_root_action_visits"].get<std::int64_t>();
            if (minimumRootActionVisits > simulations) {
                throw BridgeError(
                    "invalid_request",
                    "search minimum_root_action_visits cannot exceed simulations"
                );
            }
        }
        if (ensureRootActionCoverage && minimumRootActionVisits == 0) {
            minimumRootActionVisits = 1;
        }
        if (
            minimumRootActionVisits > 0
            && minimumRootActionVisits * static_cast<std::int64_t>(legal.size()) > simulations
        ) {
            throw BridgeError(
                "invalid_request",
                "search budget cannot satisfy minimum_root_action_visits for every root edge"
            );
        }
        bool hiddenOrderApplied = false;
        std::uint32_t hiddenOrderSeed = 0;
        BattleContext searchRoot(*battle_);
        const bool publicSample = request.contains("public_state_seed");
        if (publicSample) {
            if (!publicStateMode_ || request.contains("hidden_order_seed")) {
                throw BridgeError("invalid_request", "Enable public_state; do not mix sampling protocols");
            }
            samplePublicState(searchRoot, publicSeed(request));
        }
        if (request.contains("hidden_order_seed")) {
            if (!request["hidden_order_seed"].is_number_unsigned()) {
                throw BridgeError(
                    "invalid_request",
                    "search hidden_order_seed must be an unsigned 32-bit integer"
                );
            }
            const auto hiddenOrderSeedValue = request["hidden_order_seed"].get<std::uint64_t>();
            if (hiddenOrderSeedValue > std::numeric_limits<std::uint32_t>::max()) {
                throw BridgeError(
                    "invalid_request",
                    "search hidden_order_seed must be an unsigned 32-bit integer"
                );
            }
            hiddenOrderApplied = true;
            hiddenOrderSeed = static_cast<std::uint32_t>(hiddenOrderSeedValue);
            java::Collections::shuffle(
                searchRoot.cards.drawPile.begin(),
                searchRoot.cards.drawPile.end(),
                java::Random(hiddenOrderSeed)
            );
            searchRoot.shuffleRng = sts::Random(hiddenOrderSeed);
        }
        std::ostringstream hiddenOrderFingerprint;
        for (const auto &card : searchRoot.cards.drawPile) {
            hiddenOrderFingerprint << card.getUniqueId() << ',';
        }
        sts::search::BattleScumSearcher2 searcher(searchRoot);
        searcher.randGen.seed(searchSeed);
        searcher.forceRootActionCoverage = ensureRootActionCoverage;
        searcher.minimumRootActionVisits = static_cast<int>(minimumRootActionVisits);
        SemanticCorrectionCounts correctionCounts;
        std::map<std::string, RootOutcomeStats> outcomeByAction;
        searcher.transitionFnc = [&correctionCounts](
            BattleContext &state,
            const sts::search::Action &action
        ) {
            executeCorrectedSearchAction(state, action, &correctionCounts);
        };
        searcher.simulationObserverFnc = [this, &outcomeByAction](
            const sts::search::Action &rootAction,
            const BattleContext &endState,
            double evaluation
        ) {
            auto &stats = outcomeByAction[bridgeActionId(rootAction, *battle_)];
            if (endState.outcome == Outcome::PLAYER_VICTORY) {
                ++stats.terminalWins;
                stats.victoryEndingHpSum += endState.player.curHp;
            } else if (endState.outcome == Outcome::PLAYER_LOSS) {
                ++stats.terminalLosses;
            } else {
                throw BridgeError(
                    "teacher_search_nonterminal_playout",
                    "BattleScumSearcher2 observer received a nonterminal playout"
                );
            }
            stats.endingHpSum += endState.player.curHp;
            stats.evaluationSquareSum += evaluation * evaluation;
        };

        const auto started = std::chrono::steady_clock::now();
        searcher.search(simulations);
        const auto finished = std::chrono::steady_clock::now();
        sts::search::g_debug_scum_search = nullptr;
        const double elapsedMs = std::chrono::duration<double, std::milli>(
            finished - started
        ).count();

        json rootActions = json::array();
        std::string suggestedActionId;
        double suggestedWinRate = -1.0;
        double suggestedVictoryHp = -1.0;
        std::int64_t suggestedVisits = -1;
        double suggestedMean = -std::numeric_limits<double>::infinity();

        for (const auto &edge : searcher.root.edges) {
            const std::string actionId = bridgeActionId(edge.action, *battle_);
            const auto &bridgeAction = requireBridgeLegalAction(legal, actionId);
            const auto visits = edge.node.simulationCount;
            const auto outcome = outcomeByAction[actionId];
            if (outcome.terminalWins + outcome.terminalLosses != visits) {
                throw BridgeError(
                    "teacher_search_observer_mismatch",
                    "Terminal outcome observations do not match root visits"
                );
            }
            const double meanEvaluation = visits > 0
                ? edge.node.evaluationSum / static_cast<double>(visits)
                : 0.0;
            const double winRate = visits > 0
                ? static_cast<double>(outcome.terminalWins) /
                    static_cast<double>(visits)
                : 0.0;
            const double victoryEndingHpMean = outcome.terminalWins > 0
                ? static_cast<double>(outcome.victoryEndingHpSum) /
                    static_cast<double>(outcome.terminalWins)
                : -1.0;

            json actionRecord = {
                {"action_id", actionId},
                {"kind", bridgeAction.kind},
                {"visits", visits},
                {"evaluation_sum", edge.node.evaluationSum},
                {"evaluation_square_sum", outcome.evaluationSquareSum},
                {"mean_evaluation", visits > 0
                    ? json(meanEvaluation)
                    : json(nullptr)},
                {"terminal_wins", outcome.terminalWins},
                {"terminal_losses", outcome.terminalLosses},
                {"win_rate", visits > 0 ? json(winRate) : json(nullptr)},
                {"ending_hp_sum", outcome.endingHpSum},
                {"ending_hp_mean", visits > 0
                    ? json(static_cast<double>(outcome.endingHpSum) /
                        static_cast<double>(visits))
                    : json(nullptr)},
                {"victory_ending_hp_sum", outcome.victoryEndingHpSum},
                {"victory_ending_hp_mean", outcome.terminalWins > 0
                    ? json(victoryEndingHpMean)
                    : json(nullptr)},
            };
            if (bridgeAction.kind == "PLAY_CARD") {
                actionRecord["hand_index"] = bridgeAction.handIndex;
                actionRecord["target_index"] = bridgeAction.targetIndex < 0
                    ? json(nullptr)
                    : json(bridgeAction.targetIndex);
            }
            rootActions.push_back(std::move(actionRecord));

            if (
                winRate > suggestedWinRate ||
                (
                    winRate == suggestedWinRate &&
                    victoryEndingHpMean > suggestedVictoryHp
                ) ||
                (
                    winRate == suggestedWinRate &&
                    victoryEndingHpMean == suggestedVictoryHp &&
                    visits > suggestedVisits
                ) ||
                (
                    winRate == suggestedWinRate &&
                    victoryEndingHpMean == suggestedVictoryHp &&
                    visits == suggestedVisits &&
                    meanEvaluation > suggestedMean
                ) ||
                (
                    winRate == suggestedWinRate &&
                    victoryEndingHpMean == suggestedVictoryHp &&
                    visits == suggestedVisits &&
                    meanEvaluation == suggestedMean &&
                    (
                        suggestedActionId.empty() ||
                        actionId < suggestedActionId
                    )
                )
            ) {
                suggestedActionId = actionId;
                suggestedWinRate = winRate;
                suggestedVictoryHp = victoryEndingHpMean;
                suggestedVisits = visits;
                suggestedMean = meanEvaluation;
            }
        }
        if (rootActions.empty() || suggestedActionId.empty()) {
            throw BridgeError(
                "teacher_search_empty_root",
                "BattleScumSearcher2 returned no root actions"
            );
        }

        json bestSequenceFirstActionId = nullptr;
        if (!searcher.bestActionSequence.empty()) {
            const std::string actionId = bridgeActionId(
                searcher.bestActionSequence.front(),
                *battle_
            );
            requireBridgeLegalAction(legal, actionId);
            bestSequenceFirstActionId = actionId;
        }

        return {
            {"source", "BattleScumSearcher2"},
            {"decision_id", decisionId_},
            {"simulations", simulations},
            {"search_seed", searchSeed},
            {"search_quality_normalization", "min_max_guarded_v1"},
            {"root_action_coverage_policy", minimumRootActionVisits > 1
                ? "minimum_root_edge_visits_before_ucb_v1"
                : (minimumRootActionVisits == 1
                    ? "visit_every_root_edge_before_ucb_v1"
                    : "legacy_ucb_no_root_coverage_guarantee")},
            {"minimum_root_action_visits", minimumRootActionVisits},
            {"hidden_order_seed", hiddenOrderApplied
                ? json(hiddenOrderSeed)
                : json(nullptr)},
            {"public_state_seed", publicSample ? json(publicSeed(request)) : json(nullptr)},
            {"hidden_order_policy", publicSample
                ? "fresh_future_rng_known_draw_top_v1" : (hiddenOrderApplied
                ? "shuffle_current_draw_pile_and_reseed_shuffle_rng_v1"
                : "exact_privileged_order_v1")},
            {"hidden_order_fingerprint", hiddenOrderFingerprint.str()},
            {"root_simulation_count", searcher.root.simulationCount},
            {"root_actions", std::move(rootActions)},
            {"suggested_action_id", suggestedActionId},
            {"selection_rule", "max_win_rate_then_victory_hp_then_visits_then_mean_evaluation_then_action_id"},
            {"best_sequence_first_action_id", bestSequenceFirstActionId},
            {"best_sequence_length", searcher.bestActionSequence.size()},
            {"best_action_value", searcher.bestActionValue},
            {"min_action_value", searcher.minActionValue},
            {"best_outcome_player_hp", searcher.outcomePlayerHp},
            {"elapsed_ms", elapsedMs},
            {"objective", "upstream_battle_scum_evaluate_end_state_v1"},
            {"privileged_state", {
                {"battle_context_copy", true},
                {"ordered_draw_pile", true},
                {"future_rng_state", true},
            }},
            {"transition_semantics", {
                {"engine", "shared_corrected_action_execute_v1"},
                {"bridge_corrections_applied_inside_rollouts", true},
                {"lagavulin_rejected", false},
                {"upgraded_disarm_decks_rejected", false},
                {"corrections_applied", {
                    {"lagavulin_natural_wake", correctionCounts.lagavulinNaturalWake},
                    {"upgraded_disarm", correctionCounts.upgradedDisarm},
                    {"red_slaver_entangle_once", correctionCounts.redSlaverEntangle},
                    {"philosopher_bronze_orb_strength", correctionCounts.philosopherBronzeOrbStrength},
                    {"burning_blood_victory_heal", correctionCounts.burningBloodVictoryHeal},
                }},
            }},
        };
    }

    json step(const json &request) {
        requireBattle();
        if (battle_->outcome != Outcome::UNDECIDED) {
            throw BridgeError("battle_terminal", "Cannot step a terminal battle");
        }
        if (!request.contains("decision_id") || !request["decision_id"].is_number_integer()) {
            throw BridgeError("invalid_request", "step decision_id must be an integer");
        }
        if (request["decision_id"].get<int>() != decisionId_) {
            throw BridgeError("stale_decision", "step decision_id does not match current state");
        }
        if (!request.contains("action_id") || !request["action_id"].is_string()) {
            throw BridgeError("invalid_request", "step action_id must be a string");
        }

        const std::string actionId = request["action_id"].get<std::string>();
        const auto legal = enumerateLegalActions(*battle_);
        const LegalAction *selected = nullptr;
        for (const auto &action : legal) {
            if (action.actionId == actionId) {
                selected = &action;
                break;
            }
        }
        if (selected == nullptr) {
            throw BridgeError("illegal_action", "action_id is not legal in the current state");
        }

        const int hpBefore = battle_->player.curHp;
        int expectedCardSelfHpLoss = 0;
        int expectedEndTurnSelfHpLoss = 0;
        if (selected->kind == "END_TURN") {
            if (battle_->player.hasStatus<PS::COMBUST>()) {
                expectedEndTurnSelfHpLoss = 1;
            }
        } else if (selected->kind == "PLAY_CARD") {
            const auto card = battle_->cards.hand[selected->handIndex];
            if (card.id == sts::CardId::BLOODLETTING) expectedCardSelfHpLoss = 3;
            if (card.id == sts::CardId::HEMOKINESIS) expectedCardSelfHpLoss = 2;
        }
        sts::public_draw::Scope memoryScope([this](const sts::CardManager *cards,
            sts::public_draw::Event event, int id) {
            if (cards != &battle_->cards) return;
            if (event == sts::public_draw::Event::randomized) knownDrawTop_.clear();
            else if (event == sts::public_draw::Event::top) knownDrawTop_.insert(knownDrawTop_.begin(), id);
            else knownDrawTop_.erase(std::remove(knownDrawTop_.begin(), knownDrawTop_.end(), id), knownDrawTop_.end());
        });
        const int postCombatHeal = executeCorrectedSearchAction(
            *battle_,
            searchActionForLegalAction(*selected)
        );
        const int hpLost = std::max(
            0,
            hpBefore - battle_->player.curHp + postCombatHeal
        );
        if (expectedCardSelfHpLoss > 0) {
            const int selfLoss = std::min(hpLost, expectedCardSelfHpLoss);
            selfHpLoss_ += selfLoss;
            enemyDamageTaken_ += hpLost - selfLoss;
        } else {
            const int selfLoss = std::min(hpLost, expectedEndTurnSelfHpLoss);
            selfHpLoss_ += selfLoss;
            enemyDamageTaken_ += hpLost - selfLoss;
        }
        ++decisionId_;

        if (
            battle_->outcome == Outcome::UNDECIDED &&
            battle_->inputState != InputState::PLAYER_NORMAL &&
            battle_->inputState != InputState::CARD_SELECT
        ) {
            throw BridgeError(
                "unsupported_input_state",
                std::string("step stopped in unsupported input state ") +
                    inputStateName(battle_->inputState)
            );
        }
        return observation();
    }

    json observation() const {
        json result = {
            {"state", serializeState(
                *battle_,
                scenarioId_,
                deckPreset_,
                loadoutMetadata_,
                startingHp_,
                enemyDamageTaken_,
                selfHpLoss_,
                decisionId_,
                act_,
                relics_
            )},
            {"legal_actions", serializeLegalActions(*battle_)},
        };
        if (publicStateMode_) {
            json known = json::array();
            for (std::size_t i = 0; i < knownDrawTop_.size(); ++i) {
                const auto &card = battle_->cards.drawPile[battle_->cards.drawPile.size() - 1 - i];
                if (card.getUniqueId() != knownDrawTop_[i]) throw BridgeError("draw_memory", "Known prefix changed");
                known.push_back(serializeCard(card, *battle_));
            }
            result["state"]["public_draw_memory"] = {
                {"schema_version", "known_draw_top_v1"}, {"next_draw_first", known}
            };
        }
        return result;
    }
};

json responseBase(const json &request) {
    json response = {
        {"v", kProtocolVersion},
        {"id", nullptr},
    };
    if (request.is_object() && request.contains("id")) {
        response["id"] = request["id"];
    }
    if (request.is_object() && request.contains("op") && request["op"].is_string()) {
        response["op"] = request["op"];
    }
    return response;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    BridgeSession session;
    std::string line;

    while (std::getline(std::cin, line)) {
        json request;
        json response;
        bool shouldClose = false;
        try {
            request = json::parse(line);
            response = responseBase(request);
            const json payload = session.handle(request);
            response["ok"] = true;
            for (const auto &[key, value] : payload.items()) {
                response[key] = value;
            }
            shouldClose = session.isClose(request);
        } catch (const BridgeError &error) {
            response = responseBase(request);
            response["ok"] = false;
            response["error"] = {
                {"code", error.code},
                {"message", error.what()},
            };
        } catch (const json::exception &error) {
            response = responseBase(request);
            response["ok"] = false;
            response["error"] = {
                {"code", "invalid_json"},
                {"message", error.what()},
            };
        } catch (const std::exception &error) {
            response = responseBase(request);
            response["ok"] = false;
            response["error"] = {
                {"code", "internal_error"},
                {"message", error.what()},
            };
        }

        std::cout << response.dump() << '\n' << std::flush;
        if (shouldClose) break;
    }
    return 0;
}
