"""State-equal distributions of legal opportunities and accepted candidates.

Defense tags describe card mechanics, not realized block or causal HP savings.
They overlap with card types (for example Iron Wave is an Attack and blocks).
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import statistics


STATISTICS_VERSION = "state_equal_cards_v1"
BASIC = {"Strike_R", "Defend_R", "Bash"}
DIRECT_BLOCK = {"Defend_R", "Iron Wave", "Shrug It Off", "Flame Barrier", "Ghostly Armor",
                "Power Through", "Sentinel", "Impervious", "True Grit", "Armaments"}
MITIGATION = {"Clothesline", "Uppercut", "Disarm", "Intimidate", "Shockwave"}
CONDITIONAL_BLOCK = {"Rage", "Second Wind", "Entrench", "Feel No Pain", "Metallicize", "Barricade"}
CARD_TYPES = {"ATTACK", "SKILL", "POWER"}


def root_records(space, *, extended=False):
    rows = []
    for action in space.classes:
        row = {"id": action.model_action_id,
               "card": action.card.card_id if action.card else action.representative.action_type.value,
               "target": action.representative.target_index}
        if extended:
            row.update(action_type=action.action_type.value,
                       card_semantics=asdict(action.card) if action.card else None)
        rows.append(row)
    return rows


def categories(action):
    kind = action["action_type"]
    card = action["card_semantics"]
    if kind != "play_card":
        return {kind}
    card_type = card["card_type"].upper()
    if card_type not in CARD_TYPES:
        return {card_type}
    identity = card["card_id"]
    tags = {card_type, "basic" if identity in BASIC else "nonbasic"}
    if identity in {"Strike_R", "Defend_R"}:
        tags.add("strike_or_defend")
    if card["upgrades"]:
        tags.add("upgraded")
    for name, members in (("direct_block", DIRECT_BLOCK), ("mitigation", MITIGATION),
                          ("conditional_or_retained_block", CONDITIONAL_BLOCK)):
        if identity in members:
            tags.update({name, "defense"})
    return tags


def card_key(action):
    """Same public card semantics, ignoring target and indistinguishable copies."""
    return json.dumps(action["card_semantics"], sort_keys=True, separators=(",", ":"))


def playable_cards(actions):
    return {card_key(a): a for a in actions if a["action_type"] == "play_card"
            and a["card_semantics"]["card_type"].upper() in CARD_TYPES}


def distribution(collections):
    tag_mass, card_mass, raw = Counter(), Counter(), Counter()
    nonempty = 0
    for actions in collections:
        if not actions:
            continue
        nonempty += 1
        for action in actions:
            for tag in categories(action):
                tag_mass[tag] += 1 / len(actions)
                raw[tag] += 1
            if action["action_type"] == "play_card":
                card_mass[action["card"]] += 1 / len(actions)
    return {"states_with_items": nonempty, "states_without_items": len(collections) - nonempty,
            "category_state_equal_share": {k: v / nonempty for k, v in sorted(tag_mass.items())},
            "card_state_equal_share": {k: v / nonempty for k, v in sorted(card_mass.items())},
            "category_item_counts": dict(sorted(raw.items()))}


def summarize_group(states):
    legal_cards, accepted_cards, legal_actions, accepted_actions = [], [], [], []
    opportunity, retained, retention_fractions = Counter(), Counter(), defaultdict(list)
    quality = Counter()
    for state in states:
        legal = state["root_actions"]
        candidates = set(state["estimated_acceptable_actions"])
        if not candidates <= {a["id"] for a in legal}:
            raise ValueError("Candidates are outside the legal root action set")
        accepted = [a for a in legal if a["id"] in candidates]
        cards, kept = playable_cards(legal), playable_cards(accepted)
        legal_actions.append(legal)
        accepted_actions.append(accepted)
        legal_cards.append(list(cards.values()))
        accepted_cards.append(list(kept.values()))
        for tag in set().union(*(categories(a) for a in cards.values())):
            available = {k for k, a in cards.items() if tag in categories(a)}
            remaining = available & kept.keys()
            opportunity[tag] += 1
            retained[tag] += bool(remaining)
            retention_fractions[tag].append(len(remaining) / len(available))
        quality["empty_candidate_states"] += not candidates
        quality["single_candidate_states"] += len(candidates) == 1
        quality["all_actions_accepted_states"] += len(candidates) == len(legal)
        quality["single_legal_action_states"] += len(legal) == 1
        quality["secondary_selection_states"] += any(a["action_type"] == "select_card" for a in legal)
        quality["states_with_unpriced_resource_conflicts"] += any(p["unpriced_resource_conflict_trials"] for p in state["paired_comparisons"])
        quality["states_with_candidate_below_win_lcb"] += any(
            a["action"] in candidates and not a["passes_win_floor_lcb"] for a in state["actions"])
    retention = {tag: {"opportunity_states": opportunity[tag], "retained_states": retained[tag],
                       "at_least_one_retained_rate": retained[tag] / opportunity[tag],
                       "mean_within_state_card_retention": statistics.mean(retention_fractions[tag])}
                 for tag in sorted(opportunity)}
    return {"states": len(states), "quality_counts": dict(sorted(quality.items())),
            "primary_metric": retention.get("defense", {}).get("at_least_one_retained_rate"),
            "primary_metric_opportunity_states": opportunity["defense"],
            "category_retention": retention,
            "legal_cards": distribution(legal_cards), "candidate_cards": distribution(accepted_cards),
            "legal_actions": distribution(legal_actions), "candidate_actions": distribution(accepted_actions)}


def collection_statistics(states):
    if any(s["status"] != "completed" for s in states):
        raise ValueError("Card statistics require complete states")
    families = sorted({s["encounter_family"] for s in states})
    return {"version": STATISTICS_VERSION,
            "primary_metric_name": "fraction of defense-opportunity states retaining at least one defense card",
            "weighting": "equal state weight; uniform over distinct card semantics within each nonempty state; action view retains targets",
            "scope": "sampled panel, not an unbiased full-pool decision distribution; defense tags overlap card types",
            "overall": summarize_group(states),
            "by_encounter_family": {f: summarize_group([s for s in states if s["encounter_family"] == f]) for f in families}}
