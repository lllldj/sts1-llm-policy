from __future__ import annotations

from .route_context import CombatRouteContext

from dataclasses import dataclass


@dataclass(frozen=True)
class PowerState:
    power_id: str
    name: str
    amount: int

    # Optional fields exposed by CommunicationMod for powers whose
    # behavior cannot be represented by amount alone.
    damage: int = 0
    misc: int = 0
    just_applied: bool = False
    card: CardState | None = None


@dataclass(frozen=True)
class CardState:
    card_id: str
    name: str
    card_type: str
    cost: int
    upgrades: int
    description: str

    exhausts: bool
    ethereal: bool
    has_target: bool
    is_playable: bool

    # Internal execution identity.
    # It should normally not be exposed to the LLM serializer.
    uuid: str

    # Native per-instance combat state (currently used by Rampage).
    special_data: int = 0


@dataclass(frozen=True)
class PlayerState:
    current_hp: int
    max_hp: int
    block: int
    energy: int

    powers: tuple[PowerState, ...] = ()


@dataclass(frozen=True)
class MonsterBehaviorState:
    """Public/inferable behavior state without a realized future RNG draw."""

    phase: str
    previous_move_id: int | None
    possible_next_move_ids: tuple[int, ...]
    selection: str
    rule: str


@dataclass(frozen=True)
class MonsterState:
    monster_id: str
    name: str

    current_hp: int
    max_hp: int
    block: int

    intent: str

    move_id: int
    move_hits: int
    move_base_damage: int
    move_adjusted_damage: int

    powers: tuple[PowerState, ...] = ()

    is_gone: bool = False
    half_dead: bool = False
    behavior: MonsterBehaviorState | None = None
    stasis_card: CardState | None = None


@dataclass(frozen=True)
class CombatAccounting:
    """Episode-wide HP-loss totals, without retaining action history."""

    starting_hp: int
    enemy_damage_taken: int
    self_hp_loss: int

    @property
    def total_hp_loss(self) -> int:
        return self.enemy_damage_taken + self.self_hp_loss


@dataclass(frozen=True)
class RelicState:
    """Player-visible relic identity and its native persistent counter."""

    relic_id: str
    name: str
    counter: int | None = None


@dataclass(frozen=True)
class CombatState:
    turn: int

    player: PlayerState
    monsters: tuple[MonsterState, ...]

    hand: tuple[CardState, ...]
    draw_pile: tuple[CardState, ...]
    discard_pile: tuple[CardState, ...]
    exhaust_pile: tuple[CardState, ...]

    accounting: CombatAccounting | None = None


@dataclass(frozen=True)
class CanonicalState:
    seed: int

    character: str
    ascension_level: int
    act: int
    floor: int

    combat: CombatState
    relics: tuple[RelicState, ...] = ()
    route_context: CombatRouteContext | None = None
    # Public deterministic native previews, aligned to the current hand.
    hand_upgrade_previews: tuple[CardState | None, ...] = ()
    # None means unavailable; empty means observed history contains no known prefix.
    known_draw_top: tuple[CardState, ...] | None = None
