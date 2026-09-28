"""Observable decision opportunities for bounded GOLD state sampling.

Mechanism tags are sampling hints, never labels or a simulation of card effects.
No hidden draw order, outcome or Teacher score participates in selection.
"""
from collections import Counter
import math


VERSION = "event_mixture_v1"
DEFAULT_WEIGHTS = {"turn_start": .4, "event": .3, "random": .3}
EVENTS = ("hand_access", "energy_cost", "secondary_selection", "upgrade_top",
          "hand_exhaust", "rule_change")
DRAW = {"Pommel Strike", "Shrug It Off", "Battle Trance", "Offering", "Warcry", "Burning Pact"}
ENERGY = {"Bloodletting", "Seeing Red", "Offering"}
RULES = {"Rage", "Corruption", "Dark Embrace", "Feel No Pain", "Evolve",
         "Fire Breathing", "Rupture", "Juggernaut", "Double Tap", "Barricade"}


def _normalized(value):
    return value.replace(" ", "").replace("_", "").lower()


def _powers(player):
    return {_normalized(p["power_id"]): p["amount"] for p in player.get("powers", [])}


def _card(state, action):
    if action["action_type"] != "play_card":
        return None
    return state["combat"]["hand"][action["hand_index"]]


def action_events(state, action):
    """Potential mechanisms of a legal action, independent of the recorded choice."""
    if action["action_type"] == "select_card":
        task = state["selection_task"]
        return {"secondary_selection"} | {
            "EXHAUST_ONE": {"hand_exhaust"}, "WARCRY": {"upgrade_top"},
            "HEADBUTT": {"upgrade_top"}, "ARMAMENTS": {"upgrade_top"},
            "DUAL_WIELD": {"hand_access"}, "EXHUME": {"hand_access"},
        }[task]
    card = _card(state, action)
    if card is None:
        return set()
    combat = state["combat"]
    other = [c for i, c in enumerate(combat["hand"]) if i != action["hand_index"]]
    name = card["card_id"]
    tags = set()
    can_draw = _powers(combat["player"]).get("nodraw", 0) <= 0
    if name in DRAW and can_draw:
        tags.add("hand_access")
    if name in {"Power Through", "Dual Wield", "Exhume"}:
        tags.add("hand_access")
    if name in ENERGY or name == "Corruption":
        tags.add("energy_cost")
    if name == "Dropkick":
        target = combat["monsters"][action["target_index"]]
        if _powers(target).get("vulnerable", 0) > 0:
            tags.add("energy_cost")
            if can_draw:
                tags.add("hand_access")
    if name in {"Armaments", "Headbutt", "Warcry"}:
        tags.add("upgrade_top")
    if name in {"Burning Pact", "True Grit", "Fiend Fire"} and other:
        tags.add("hand_exhaust")
    if name in {"Second Wind", "Sever Soul"} and any(c["card_type"] != "ATTACK" for c in other):
        tags.add("hand_exhaust")
    if name in RULES:
        tags.add("rule_change")
    return tags


def transition_events(record, following):
    """Conservative public changes after one action (possibly a card selection).

Simulator UUIDs are position based: compare semantic multisets, not UUIDs.
Normal start-of-turn draw/energy is handled only by the turn-start stratum.
"""
    before = record["canonical_state"]
    after = following["canonical_state"]
    if record["turn"] != following["turn"] or record["action"]["action_type"] == "end_turn":
        return set()
    bc, ac = before["combat"], after["combat"]
    action = record["action"]
    played = _card(before, action)
    remaining = [c for i, c in enumerate(bc["hand"])
                 if played is None or i != action["hand_index"]]
    count = lambda cards: Counter(c["card_id"] for c in cards)
    tags = set()
    if count(ac["hand"]) - count(remaining):
        tags.add("hand_access")
    # Account for spending before generation, including X cost and free skills.
    spent = 0
    if played is not None:
        raw_card = record["raw_state"]["state"]["hand"][action["hand_index"]]
        cost = raw_card["cost_for_turn"]
        spent = bc["player"]["energy"] if cost == -1 else max(0, cost)
        if played["card_type"] == "SKILL" and _powers(bc["player"]).get("corruption", 0):
            spent = 0
    if ac["player"]["energy"] > bc["player"]["energy"] - spent:
        tags.add("energy_cost")
    # Match only unchanged card multiplicities; additions cannot masquerade as discounts.
    for name, n in count(remaining).items():
        old = [c for c in remaining if c["card_id"] == name]
        new = [c for c in ac["hand"] if c["card_id"] == name]
        if len(new) != n:
            continue
        if any(y < x for x, y in zip(sorted(c["cost"] for c in old), sorted(c["cost"] for c in new))):
            tags.add("energy_cost")
        if any(y > x for x, y in zip(sorted(c["upgrades"] for c in old), sorted(c["upgrades"] for c in new))):
            tags.add("upgrade_top")
    if ((after.get("known_draw_top") and after.get("known_draw_top") != before.get("known_draw_top")) or
            (played and played["card_id"] in {"Headbutt", "Warcry"})):
        tags.add("upgrade_top")
    exhausted = count(ac["exhaust_pile"]) - count(bc["exhaust_pile"])
    self_exhausts = played and (played.get("exhausts", False) or
                                (played["card_type"] == "SKILL" and _powers(bc["player"]).get("corruption", 0)))
    if self_exhausts and exhausted[played["card_id"]]:
        exhausted[played["card_id"]] -= 1
    if exhausted & (count(remaining) - count(ac["hand"])):
        tags.add("hand_exhaust")
    bp, ap = _powers(bc["player"]), _powers(ac["player"])
    if any(ap.get(_normalized(name), 0) > bp.get(_normalized(name), 0) for name in RULES):
        tags.add("rule_change")
    if "selection_task" in after:
        tags.add("secondary_selection")
    if "selection_task" in before:
        # Leaving a selection is a decision boundary even when card counts cancel.
        tags |= action_events(before, action)
    return tags


def decision_features(records):
    features = []
    for i, record in enumerate(records):
        state = record["canonical_state"]
        opportunity = set()
        for action in record["legal_actions"]:
            opportunity |= action_events(state, action)
        if "selection_task" in state:
            opportunity.add("secondary_selection")
        result = transition_events(records[i - 1], record) if i else set()
        features.append({"eligible": len(record["model_legal_actions"]) > 1,
                         "turn_start": i == 0 or records[i - 1]["turn"] != record["turn"],
                         "opportunity_events": sorted(opportunity), "result_events": sorted(result),
                         "events": sorted(opportunity | result)})
    return features


def sample_decisions(records, count, rng, *, weights=None):
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if (set(weights) != set(DEFAULT_WEIGHTS) or
            any(not math.isfinite(w) or w < 0 for w in weights.values()) or
            not math.isclose(sum(weights.values()), 1) or weights["random"] <= 0):
        raise ValueError("Sampling weights must sum to one and keep a positive random fraction")
    features = decision_features(records)
    eligible = [i for i, f in enumerate(features) if f["eligible"]]
    frequency = Counter(t for i in eligible for t in features[i]["events"])
    # One bucket per state prevents multiple event tags from multiplying its weight.
    # Prefer its rarest mechanism within this combat, then a stable lexical tie break.
    bucket = {i: min(features[i]["events"], key=lambda t: (frequency[t], t))
              for i in eligible if features[i]["events"]}
    remaining = set(eligible)
    chosen = {}
    for draw in range(min(count, len(eligible))):
        starts = sorted(i for i in remaining if features[i]["turn_start"])
        buckets = {t: sorted(i for i in remaining if bucket.get(i) == t) for t in EVENTS}
        buckets = {t: ids for t, ids in buckets.items() if ids}
        effective = dict(weights)
        for channel, exists in (("turn_start", starts), ("event", buckets)):
            if not exists:
                effective["random"] += effective[channel]
                effective[channel] = 0
        channel = rng.choices(list(effective), weights=list(effective.values()), k=1)[0]
        event_bucket = rng.choice(sorted(buckets)) if channel == "event" else None
        candidates = starts if channel == "turn_start" else buckets[event_bucket] if channel == "event" else sorted(remaining)
        index = rng.choice(candidates)
        probability = effective["random"] / len(remaining)
        if index in starts:
            probability += effective["turn_start"] / len(starts)
        if index in bucket:
            probability += effective["event"] / len(buckets) / len(buckets[bucket[index]])
        chosen[index] = {**features[index], "sampling_channel": channel,
                         "event_bucket": event_bucket, "assigned_event_bucket": bucket.get(index),
                         "draw_index": draw, "conditional_draw_probability": probability}
        remaining.remove(index)
    return chosen, {"eligible_decisions": len(eligible),
                    "excluded_single_action_decisions": len(records) - len(eligible),
                    "turn_start_decisions": sum(features[i]["turn_start"] for i in eligible),
                    "event_decisions": len(bucket), "event_opportunity_counts": dict(sorted(frequency.items()))}
