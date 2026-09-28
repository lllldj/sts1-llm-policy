from __future__ import annotations

from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import random
import re
import sqlite3
from typing import Protocol

from sts1_llm_policy.artifacts import sha256_file


CARD_PICK_DATABASE_SCHEMA = "card_pick_metrics_v1"
RATE_COLUMNS = (
    "pick_rate",
    "first_pick_rate_f1",
    "first_pick_rate_f2",
    "first_pick_rate_f3",
    "duplicate_pick_rate_f1",
    "duplicate_pick_rate_f2",
    "duplicate_pick_rate_f3",
)
_MISSING_PERCENTAGES = frozenset(("", "--", "--%"))


def normalize_card_id(value: str) -> str:
    """Map an STS string id such as ``Blood for Blood`` to an enum-style id."""

    normalized = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    if not normalized:
        raise ValueError("card id must contain at least one letter or digit")
    return normalized


def parse_percentage(value: str) -> float | None:
    stripped = value.strip()
    if stripped in _MISSING_PERCENTAGES:
        return None
    if not stripped.endswith("%"):
        raise ValueError(f"percentage must end with '%': {value!r}")
    probability = float(stripped[:-1]) / 100.0
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"percentage is outside [0%, 100%]: {value!r}")
    return probability


def probability_to_odds(probability: float) -> float:
    """Return the requested p/(1-p) weight (odds, not logarithmic odds)."""

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and inside [0, 1]")
    if probability == 1.0:
        return math.inf
    return probability / (1.0 - probability)


@dataclass(frozen=True)
class CardRate:
    enum_id: str
    upgraded: bool
    metric: str
    probability: float | None
    sample_size: int


class CardRateBackend(Protocol):
    def lookup(self, enum_id: str, upgraded: bool, metric: str) -> CardRate | None: ...


class SqliteCardRateBackend:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def lookup(self, enum_id: str, upgraded: bool, metric: str) -> CardRate | None:
        if metric not in RATE_COLUMNS:
            raise ValueError(f"unsupported card-pick metric: {metric}")
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                f"SELECT {metric}, sample_size FROM card_pick_rates "
                "WHERE enum_id = ? AND upgraded = ? AND found = 1",
                (normalize_card_id(enum_id), int(upgraded)),
            ).fetchone()
        if row is None:
            return None
        return CardRate(
            enum_id=normalize_card_id(enum_id),
            upgraded=upgraded,
            metric=metric,
            probability=row[0],
            sample_size=int(row[1]),
        )


@dataclass(frozen=True)
class OfferedCard:
    enum_id: str
    upgraded: bool = False
    choice_id: str | None = None


@dataclass(frozen=True)
class WeightedRewardChoice:
    choice_id: str
    kind: str
    card: OfferedCard | None
    raw_weight: float
    probability: float
    rate_source: str


def act_rate_metric(act: int, *, duplicate: bool) -> str:
    if act not in (1, 2, 3):
        raise ValueError("act must be 1, 2, or 3")
    prefix = "duplicate_pick_rate" if duplicate else "first_pick_rate"
    return f"{prefix}_f{act}"


class MassNormalizedCardRewardPicker:
    """Apply the v1 probability-mass policy to a native card reward."""

    def __init__(self, backend: CardRateBackend) -> None:
        self.backend = backend

    def distribution(
        self,
        offered_cards: Iterable[OfferedCard],
        *,
        act: int,
        deck_card_ids: Iterable[str],
        epsilon: float = 0.02,
        singing_bowl_available: bool = False,
        minimum_sample_size: int = 1,
        fallback_probability: float | None = None,
    ) -> tuple[WeightedRewardChoice, ...]:
        act_rate_metric(act, duplicate=False)
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon must be finite and inside [0, 1)")
        if minimum_sample_size < 0:
            raise ValueError("minimum_sample_size must be non-negative")
        if fallback_probability is not None:
            probability_to_odds(fallback_probability)

        cards = tuple(offered_cards)
        if not 1 <= len(cards) <= 4:
            raise ValueError("card rewards must contain 1, 2, 3, or 4 offered cards")
        normalized_keys = [(normalize_card_id(card.enum_id), card.upgraded) for card in cards]
        if len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("offered cards must be unique by enum id and upgrade state")
        deck_ids = {normalize_card_id(enum_id) for enum_id in deck_card_ids}

        pending: list[tuple[str, str, OfferedCard | None, float, str]] = []
        for index, (card, (enum_id, upgraded)) in enumerate(
            zip(cards, normalized_keys, strict=True)
        ):
            metric = act_rate_metric(act, duplicate=enum_id in deck_ids)
            rate = self.backend.lookup(enum_id, upgraded, metric)
            if (
                rate is None
                or rate.probability is None
                or rate.sample_size < minimum_sample_size
            ):
                if fallback_probability is None:
                    raise LookupError(
                        f"no usable {metric} rate for {enum_id} upgraded={upgraded}"
                    )
                target_rate = fallback_probability
                source = f"fallback_probability:{metric}"
            else:
                target_rate = rate.probability
                source = f"database:{metric}"
            pending.append(
                (
                    card.choice_id or f"CARD_{index}",
                    "card",
                    card,
                    target_rate,
                    source,
                )
            )

        total_rate = sum(item[3] for item in pending)
        capacity = 1.0 - epsilon
        if total_rate <= capacity:
            card_probabilities = [item[3] for item in pending]
            skip_probability = 1.0 - total_rate
        else:
            card_probabilities = [capacity * item[3] / total_rate for item in pending]
            skip_probability = epsilon

        skip_choice = "SINGING_BOWL" if singing_bowl_available else "SKIP"
        pending.append(
            (
                skip_choice,
                "singing_bowl" if singing_bowl_available else "skip",
                None,
                skip_probability,
                "mass_remainder_or_epsilon",
            )
        )
        final_probabilities = (*card_probabilities, skip_probability)
        return tuple(
            WeightedRewardChoice(
                choice_id=choice_id,
                kind=kind,
                card=card,
                raw_weight=target_rate,
                probability=probability,
                rate_source=source,
            )
            for (choice_id, kind, card, target_rate, source), probability in zip(
                pending, final_probabilities, strict=True
            )
        )

    def sample(
        self,
        offered_cards: Iterable[OfferedCard],
        *,
        rng: random.Random,
        act: int,
        deck_card_ids: Iterable[str],
        epsilon: float = 0.02,
        singing_bowl_available: bool = False,
        minimum_sample_size: int = 1,
        fallback_probability: float | None = None,
    ) -> tuple[WeightedRewardChoice, tuple[WeightedRewardChoice, ...]]:
        distribution = self.distribution(
            offered_cards,
            act=act,
            deck_card_ids=deck_card_ids,
            epsilon=epsilon,
            singing_bowl_available=singing_bowl_available,
            minimum_sample_size=minimum_sample_size,
            fallback_probability=fallback_probability,
        )
        threshold = rng.random()
        cumulative = 0.0
        for choice in distribution:
            cumulative += choice.probability
            if threshold < cumulative:
                return choice, distribution
        return distribution[-1], distribution


def build_card_pick_database(
    csv_path: str | Path,
    metadata_path: str | Path,
    scope_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    csv_file = Path(csv_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    scope_file = Path(scope_path).resolve()
    output_file = Path(output_path).resolve()
    if output_file.exists():
        raise FileExistsError(f"refusing to overwrite existing database: {output_file}")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    scope = json.loads(scope_file.read_text(encoding="utf-8"))
    scope_ids = frozenset(scope["reward_cards"]["included_enum_ids"])
    expected_scope_keys = {(enum_id, upgraded) for enum_id in scope_ids for upgraded in (0, 1)}
    rows: list[dict[str, object]] = []
    with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "card_id", "card_name", "classpath", "upgraded", "color", "rarity",
            "type", "min_ascension", "found", *RATE_COLUMNS, "sample_players",
            "sample_size", "generated_time",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"card-pick CSV is missing columns: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            if raw["upgraded"] not in ("true", "false") or raw["found"] not in ("true", "false"):
                raise ValueError(f"invalid boolean at CSV line {line_number}")
            row: dict[str, object] = {
                "enum_id": normalize_card_id(raw["card_id"]),
                "source_card_id": raw["card_id"],
                "card_name": raw["card_name"],
                "classpath": raw["classpath"],
                "upgraded": int(raw["upgraded"] == "true"),
                "color": raw["color"],
                "rarity": raw["rarity"],
                "card_type": raw["type"],
                "min_ascension": int(raw["min_ascension"]),
                "found": int(raw["found"] == "true"),
                "sample_players": int(raw["sample_players"]),
                "sample_size": int(raw["sample_size"]),
                "generated_time": raw["generated_time"],
            }
            row.update({metric: parse_percentage(raw[metric]) for metric in RATE_COLUMNS})
            rows.append(row)

    if not rows:
        raise ValueError("card-pick CSV contains no data rows")

    keys = [(str(row["enum_id"]), int(row["upgraded"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("normalized card id/upgraded keys are not unique")
    actual_scope_keys = set(keys) & expected_scope_keys
    missing_scope_keys = sorted(expected_scope_keys - actual_scope_keys)
    if missing_scope_keys:
        raise ValueError(f"CSV does not cover the D-v1 scope: {missing_scope_keys}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(sqlite3.connect(output_file)) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE database_metadata (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL
                    );
                    CREATE TABLE card_pick_rates (
                        enum_id TEXT NOT NULL,
                        source_card_id TEXT NOT NULL,
                        card_name TEXT NOT NULL,
                        classpath TEXT NOT NULL,
                        upgraded INTEGER NOT NULL CHECK (upgraded IN (0, 1)),
                        color TEXT NOT NULL,
                        rarity TEXT NOT NULL,
                        card_type TEXT NOT NULL,
                        min_ascension INTEGER NOT NULL,
                        found INTEGER NOT NULL CHECK (found IN (0, 1)),
                        pick_rate REAL,
                        first_pick_rate_f1 REAL,
                        first_pick_rate_f2 REAL,
                        first_pick_rate_f3 REAL,
                        duplicate_pick_rate_f1 REAL,
                        duplicate_pick_rate_f2 REAL,
                        duplicate_pick_rate_f3 REAL,
                        sample_players INTEGER NOT NULL,
                        sample_size INTEGER NOT NULL,
                        generated_time TEXT NOT NULL,
                        PRIMARY KEY (enum_id, upgraded)
                    );
                    CREATE INDEX card_pick_rates_color_idx ON card_pick_rates(color);
                    """
                )
                columns = tuple(rows[0])
                placeholders = ", ".join("?" for _ in columns)
                connection.executemany(
                    f"INSERT INTO card_pick_rates ({', '.join(columns)}) VALUES ({placeholders})",
                    (tuple(row[column] for column in columns) for row in rows),
                )
                db_metadata = {
                    "schema_version": CARD_PICK_DATABASE_SCHEMA,
                    "source_csv_sha256": sha256_file(csv_file),
                    "source_metadata_sha256": sha256_file(metadata_file),
                    "scope_sha256": sha256_file(scope_file),
                    "source_metadata": metadata,
                }
                connection.executemany(
                    "INSERT INTO database_metadata(key, value_json) VALUES (?, ?)",
                    (
                        (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
                        for key, value in db_metadata.items()
                    ),
                )
    except Exception:
        output_file.unlink(missing_ok=True)
        raise

    scope_rows = [row for row in rows if str(row["enum_id"]) in scope_ids]
    unavailable_by_metric = {
        metric: sum(row[metric] is None for row in scope_rows) for metric in RATE_COLUMNS
    }
    boundary_by_metric = {
        metric: sum(row[metric] in (0.0, 1.0) for row in scope_rows if row[metric] is not None)
        for metric in RATE_COLUMNS
    }
    return {
        "schema_version": CARD_PICK_DATABASE_SCHEMA,
        "source": {
            "csv_sha256": sha256_file(csv_file),
            "metadata_sha256": sha256_file(metadata_file),
            "metadata": metadata,
        },
        "rows": {
            "all": len(rows),
            "red": sum(row["color"] == "RED" for row in rows),
            "d_v1_scope": len(scope_rows),
            "d_v1_cards": len(scope_ids),
            "d_v1_base_and_upgraded_complete": len(actual_scope_keys) == len(expected_scope_keys),
        },
        "quality": {
            "unavailable_d_v1_rates_by_metric": unavailable_by_metric,
            "boundary_d_v1_rates_by_metric": boundary_by_metric,
            "d_v1_sample_size_min": min(int(row["sample_size"]) for row in scope_rows),
            "d_v1_sample_size_max": max(int(row["sample_size"]) for row in scope_rows),
        },
        "metric_semantics": {
            "preserved_source_labels": list(RATE_COLUMNS),
            "act_mapping": {"f1": 1, "f2": 2, "f3": 3},
            "active_selection_policy": "mass_normalized_metrics_v1",
            "legacy_available_transform": "p/(1-p)",
        },
    }
