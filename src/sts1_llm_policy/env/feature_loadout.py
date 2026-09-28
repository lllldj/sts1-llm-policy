from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from types import MappingProxyType
from typing import Mapping, Sequence


FEATURE_LOADOUT_SCHEMA_VERSION = "feature_loadout_v1"
RANDOM_MIXED_FEATURE_FAMILY = "random_mixed"


@dataclass(frozen=True, order=True)
class DeckCardSpec:
    card_id: str
    upgrades: int
    count: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "card_id": self.card_id,
            "count": self.count,
            "upgrades": self.upgrades,
        }


@dataclass(frozen=True)
class PoolCard:
    card_id: str
    role: str


@dataclass(frozen=True)
class FeatureRecipe:
    tier: str
    base_preset: str
    removal_card_id: str | None
    addition_count: int
    upgraded_addition_count: int
    final_size: int
    inherited_relics: tuple[str, ...]


@dataclass(frozen=True)
class FeatureLoadout:
    loadout_id: str
    recipe_tier: str
    base_preset: str
    feature_family: str
    loadout_seed: int
    additions: tuple[DeckCardSpec, ...]
    cards: tuple[DeckCardSpec, ...]
    relics: tuple[str, ...]
    deck_hash: str

    def as_deck_spec(self) -> dict[str, object]:
        return {
            "schema_version": FEATURE_LOADOUT_SCHEMA_VERSION,
            "loadout_id": self.loadout_id,
            "recipe_tier": self.recipe_tier,
            "base_preset": self.base_preset,
            "feature_family": self.feature_family,
            "loadout_seed": self.loadout_seed,
            "additions": [card.as_dict() for card in self.additions],
            "cards": [card.as_dict() for card in self.cards],
            "relics": list(self.relics),
            "deck_hash": self.deck_hash,
        }


_STARTER = (
    DeckCardSpec("Strike_R", 0, 5),
    DeckCardSpec("Defend_R", 0, 4),
    DeckCardSpec("Bash", 0),
)

_ELITE_TRANSITION = _STARTER + (
    DeckCardSpec("Carnage", 1),
    DeckCardSpec("Shrug It Off", 0),
    DeckCardSpec("Rage", 1),
    DeckCardSpec("Pommel Strike", 0),
)

_BOSS_READY = (
    DeckCardSpec("Strike_R", 0, 4),
    DeckCardSpec("Defend_R", 0, 4),
    DeckCardSpec("Bash", 0),
    DeckCardSpec("Carnage", 1),
    DeckCardSpec("Shrug It Off", 0),
    DeckCardSpec("Rage", 1),
    DeckCardSpec("Pommel Strike", 1),
    DeckCardSpec("Battle Trance", 0),
    DeckCardSpec("Bloodletting", 1),
)

BASE_PRESET_CARDS: Mapping[str, tuple[DeckCardSpec, ...]] = MappingProxyType(
    {
        "starter": _STARTER,
        "elite_transition": _ELITE_TRANSITION,
        "boss_ready": _BOSS_READY,
    }
)

FEATURE_RECIPES: Mapping[str, FeatureRecipe] = MappingProxyType(
    {
        "feature_basic": FeatureRecipe(
            tier="feature_basic",
            base_preset="starter",
            removal_card_id=None,
            addition_count=2,
            upgraded_addition_count=1,
            final_size=12,
            inherited_relics=("Burning Blood",),
        ),
        "feature_elite": FeatureRecipe(
            tier="feature_elite",
            base_preset="elite_transition",
            removal_card_id="Strike_R",
            addition_count=3,
            upgraded_addition_count=1,
            final_size=16,
            inherited_relics=("Burning Blood",),
        ),
        "feature_boss": FeatureRecipe(
            tier="feature_boss",
            base_preset="boss_ready",
            removal_card_id="Defend_R",
            addition_count=4,
            upgraded_addition_count=2,
            final_size=18,
            inherited_relics=("Burning Blood", "Vajra"),
        ),
    }
)

CARD_POOL: tuple[PoolCard, ...] = (
    PoolCard("Carnage", "attack"),
    PoolCard("Pommel Strike", "attack"),
    PoolCard("Anger", "attack"),
    PoolCard("Body Slam", "attack"),
    PoolCard("Clothesline", "attack"),
    PoolCard("Iron Wave", "attack"),
    PoolCard("Thunderclap", "attack"),
    PoolCard("Uppercut", "attack"),
    PoolCard("Whirlwind", "attack"),
    PoolCard("Shrug It Off", "defense"),
    PoolCard("Flame Barrier", "defense"),
    PoolCard("Ghostly Armor", "defense"),
    PoolCard("Power Through", "defense"),
    PoolCard("Rage", "utility"),
    PoolCard("Battle Trance", "utility"),
    PoolCard("Bloodletting", "utility"),
    PoolCard("Disarm", "utility"),
    PoolCard("Second Wind", "utility"),
    PoolCard("Shockwave", "utility"),
    PoolCard("Spot Weakness", "utility"),
    PoolCard("Flex", "utility"),
    PoolCard("Inflame", "power"),
    PoolCard("Metallicize", "power"),
    PoolCard("Combust", "power"),
    PoolCard("Feel No Pain", "power"),
    PoolCard("Evolve", "power"),
)

POOL_BY_ID: Mapping[str, PoolCard] = MappingProxyType(
    {card.card_id: card for card in CARD_POOL}
)

FEATURE_FAMILY_ANCHORS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "vulnerable_burst": ("Thunderclap", "Carnage"),
        "block_to_damage": ("Body Slam", "Shrug It Off"),
        "low_cost_order": ("Anger", "Rage"),
        "energy_x_cost": ("Whirlwind", "Bloodletting"),
        "aoe_target_priority": ("Thunderclap", "Clothesline"),
        "persistent_scaling": ("Spot Weakness", "Inflame"),
        "status_exhaust": ("Power Through", "Second Wind"),
        "intent_mitigation": ("Disarm", "Flame Barrier"),
    }
)


def _remove_one(
    cards: Sequence[DeckCardSpec],
    card_id: str | None,
) -> tuple[DeckCardSpec, ...]:
    if card_id is None:
        return tuple(cards)
    result: list[DeckCardSpec] = []
    removed = False
    for card in cards:
        if not removed and card.card_id == card_id:
            if card.count > 1:
                result.append(DeckCardSpec(card.card_id, card.upgrades, card.count - 1))
            removed = True
        else:
            result.append(card)
    if not removed:
        raise ValueError(f"Base preset does not contain removable card {card_id!r}")
    return tuple(result)


def _normalized_cards(cards: Sequence[DeckCardSpec]) -> tuple[DeckCardSpec, ...]:
    counts: dict[tuple[str, int], int] = {}
    for card in cards:
        if card.count <= 0 or card.upgrades not in (0, 1):
            raise ValueError(f"Invalid card spec: {card!r}")
        key = (card.card_id, card.upgrades)
        counts[key] = counts.get(key, 0) + card.count
    return tuple(
        DeckCardSpec(card_id, upgrades, count)
        for (card_id, upgrades), count in sorted(counts.items())
    )


def _deck_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _choose_addition_ids(
    recipe: FeatureRecipe,
    feature_family: str,
    rng: random.Random,
) -> tuple[str, ...]:
    if feature_family == RANDOM_MIXED_FEATURE_FAMILY:
        anchors: tuple[str, ...] = ()
    else:
        try:
            anchors = FEATURE_FAMILY_ANCHORS[feature_family]
        except KeyError as error:
            raise ValueError(f"Unknown feature family: {feature_family}") from error

    inherited_ids = {
        card.card_id
        for card in _remove_one(
            BASE_PRESET_CARDS[recipe.base_preset],
            recipe.removal_card_id,
        )
    }
    eligible = [card for card in CARD_POOL if card.card_id not in inherited_ids]
    eligible_ids = {card.card_id for card in eligible}
    chosen = [card_id for card_id in anchors if card_id in eligible_ids]
    if len(chosen) > recipe.addition_count:
        raise ValueError(f"Feature anchors exceed {recipe.tier} addition count")

    target_roles = {
        "feature_basic": 2,
        "feature_elite": 3,
        "feature_boss": 4,
    }[recipe.tier]
    while len(chosen) < recipe.addition_count:
        candidates = [card for card in eligible if card.card_id not in chosen]
        roles = {POOL_BY_ID[card_id].role for card_id in chosen}
        role_expanding = [card for card in candidates if card.role not in roles]
        if len(roles) < target_roles and role_expanding:
            candidates = role_expanding
        chosen.append(rng.choice(sorted(candidates, key=lambda card: card.card_id)).card_id)
    return tuple(chosen)


def build_feature_loadout(
    recipe_tier: str,
    feature_family: str,
    loadout_seed: int,
) -> FeatureLoadout:
    if isinstance(loadout_seed, bool) or not isinstance(loadout_seed, int):
        raise TypeError("loadout_seed must be an integer")
    if not 0 <= loadout_seed < 2**64:
        raise ValueError("loadout_seed must fit uint64")
    try:
        recipe = FEATURE_RECIPES[recipe_tier]
    except KeyError as error:
        raise ValueError(f"Unknown feature recipe: {recipe_tier}") from error

    rng = random.Random(loadout_seed)
    addition_ids = _choose_addition_ids(recipe, feature_family, rng)
    upgraded_ids = set(
        rng.sample(list(addition_ids), recipe.upgraded_addition_count)
    )
    additions = tuple(
        sorted(
            (
                DeckCardSpec(card_id, 1 if card_id in upgraded_ids else 0)
                for card_id in addition_ids
            )
        )
    )
    inherited = _remove_one(
        BASE_PRESET_CARDS[recipe.base_preset],
        recipe.removal_card_id,
    )
    cards = _normalized_cards((*inherited, *additions))
    if sum(card.count for card in cards) != recipe.final_size:
        raise AssertionError("Feature recipe produced the wrong final deck size")

    loadout_id = f"{recipe_tier}-{feature_family}-{loadout_seed:016x}"
    hash_payload: dict[str, object] = {
        "schema_version": FEATURE_LOADOUT_SCHEMA_VERSION,
        "loadout_id": loadout_id,
        "recipe_tier": recipe_tier,
        "base_preset": recipe.base_preset,
        "feature_family": feature_family,
        "loadout_seed": loadout_seed,
        "additions": [card.as_dict() for card in additions],
        "cards": [card.as_dict() for card in cards],
        "relics": list(recipe.inherited_relics),
    }
    return FeatureLoadout(
        loadout_id=loadout_id,
        recipe_tier=recipe_tier,
        base_preset=recipe.base_preset,
        feature_family=feature_family,
        loadout_seed=loadout_seed,
        additions=additions,
        cards=cards,
        relics=recipe.inherited_relics,
        deck_hash=_deck_hash(hash_payload),
    )


def validate_deck_spec(deck_spec: Mapping[str, object]) -> FeatureLoadout:
    required = {
        "schema_version",
        "loadout_id",
        "recipe_tier",
        "base_preset",
        "feature_family",
        "loadout_seed",
        "additions",
        "cards",
        "relics",
        "deck_hash",
    }
    if set(deck_spec) != required:
        raise ValueError("deck_spec fields do not match feature_loadout_v1")
    expected = build_feature_loadout(
        str(deck_spec["recipe_tier"]),
        str(deck_spec["feature_family"]),
        deck_spec["loadout_seed"],  # type: ignore[arg-type]
    )
    if deck_spec != expected.as_deck_spec():
        raise ValueError("deck_spec does not match its deterministic recipe and seed")
    return expected
