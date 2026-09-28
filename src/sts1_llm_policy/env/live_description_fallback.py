from __future__ import annotations

from dataclasses import dataclass

from .mechanics_catalog import (
    SUPPORTED_POWER_EFFECT_TEMPLATES,
    UnsupportedObservationMechanicError,
    describe_card,
    describe_power,
    describe_relic,
)
from .state_schema import CardState, PowerState, RelicState


SOURCE_DESCRIPTION_FALLBACK_ID = (
    "communication_mod_descriptions_with_audited_powers_v2"
)

LIVE_POWER_EFFECT_TEMPLATES = {
    "Split": (
        "This enemy splits into smaller slimes when it uses its Split intent. "
        "Large slimes schedule Split after falling to half HP or lower."
    ),
}


@dataclass(frozen=True)
class SourceDescribedPowerState(PowerState):
    """Live-only power state carrying audited runtime rules text."""

    source_description: str = ""


@dataclass(frozen=True)
class SourceDescribedRelicState(RelicState):
    """Live-only relic state carrying CommunicationMod rules text."""

    source_description: str = ""


def _normalized_source_description(description: str) -> str | None:
    normalized = " ".join(description.split())
    return normalized or None


def resolve_card_effect(
    card: CardState,
    *,
    allow_source_description: bool = False,
) -> str:
    """Resolve frozen semantics, then an explicitly enabled live fallback."""

    from .card_selection import selection_effect, SelectionCardState

    if isinstance(card, SelectionCardState):
        effect = selection_effect(card.card_id, card.upgrades)
        if effect is None:
            raise UnsupportedObservationMechanicError("Unsupported selection card")
        return effect
    try:
        effect = describe_card(card)
        if card.card_id == "Double Tap" and not card.exhausts:
            # Corrected native/live flags override the historical catalog's
            # erroneous exhaust suffix; retained exhaust=true records stay intact.
            return effect.removesuffix(" EXHAUST.")
        return effect
    except UnsupportedObservationMechanicError:
        if allow_source_description:
            fallback = _normalized_source_description(card.description)
            if fallback is not None:
                return fallback
        raise


def resolve_relic_effect(
    relic: RelicState,
    *,
    allow_source_description: bool = False,
) -> str:
    """Resolve frozen semantics, then an explicitly enabled live fallback."""

    try:
        return describe_relic(relic.relic_id)
    except UnsupportedObservationMechanicError:
        if allow_source_description:
            fallback = _normalized_source_description(
                getattr(relic, "source_description", "")
            )
            if fallback is not None:
                return fallback
        raise


def resolve_power_effect(
    power: PowerState,
    *,
    allow_source_description: bool = False,
) -> str:
    """Resolve frozen semantics, then an explicitly enabled live fallback."""

    try:
        return describe_power(power)
    except UnsupportedObservationMechanicError:
        if allow_source_description:
            fallback = _normalized_source_description(
                getattr(power, "source_description", "")
            )
            if fallback is not None:
                return fallback
        raise


def live_power_ids() -> tuple[str, ...]:
    """Return frozen plus audited live-only power identifiers."""

    return tuple(SUPPORTED_POWER_EFFECT_TEMPLATES) + tuple(
        LIVE_POWER_EFFECT_TEMPLATES
    )
