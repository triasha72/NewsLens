"""DuckDB-backed analytical warehouse for validated MIND data."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import pandas as pd

from .mind import load_behaviors, load_news

WAREHOUSE_SCHEMA_VERSION = "1.0.0"


class WarehouseError(RuntimeError):
    """Base exception for NewsLens warehouse operations."""


class WarehouseExistsError(WarehouseError):
    """Raised when a build would replace a warehouse without permission."""


class WarehouseValidationError(WarehouseError):
    """Raised when validated source rows cannot satisfy warehouse constraints."""


@dataclass(frozen=True)
class WarehouseBuildResult:
    """Manifest returned after an atomic warehouse build."""

    database_path: str
    schema_version: str
    split: str
    news_sha256: str | None
    behaviors_sha256: str | None
    articles: int
    behavior_events: int
    history_events: int
    candidate_interactions: int
    clicks: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable build manifest."""

        return asdict(self)


@dataclass(frozen=True)
class WarehouseSummary:
    """Compact SQL-derived warehouse summary."""

    database_path: str
    schema_version: str
    split: str
    articles: int
    behavior_events: int
    users: int
    history_events: int
    candidate_interactions: int
    clicks: int
    first_event_timestamp: str | None
    last_event_timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""

        return asdict(self)


SCHEMA_SQL = files("newslens.data").joinpath("sql/warehouse_schema.sql").read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _source_reference_query(source_expression: str) -> str:
    return f"""
        SELECT DISTINCT source.news_id
        FROM ({source_expression}) AS source
        LEFT JOIN _news_source AS article
            ON article.news_id = source.news_id
        WHERE article.news_id IS NULL
        ORDER BY source.news_id
        LIMIT 10
    """


def _validate_article_references(connection: duckdb.DuckDBPyConnection) -> None:
    candidate_source = """
        SELECT regexp_extract(token, '^(.*)-[01]$', 1) AS news_id
        FROM _behavior_source AS behavior,
        UNNEST(string_split(behavior.impressions, ' ')) AS candidate(token)
    """
    history_source = """
        SELECT history.news_id
        FROM _behavior_source AS behavior,
        UNNEST(string_split(trim(behavior.history), ' ')) AS history(news_id)
        WHERE trim(behavior.history) <> ''
    """

    unknown_candidates = [
        row[0] for row in connection.execute(_source_reference_query(candidate_source)).fetchall()
    ]
    unknown_history = [
        row[0] for row in connection.execute(_source_reference_query(history_source)).fetchall()
    ]

    if unknown_candidates or unknown_history:
        details: list[str] = []
        if unknown_candidates:
            details.append(f"candidate articles: {', '.join(unknown_candidates)}")
        if unknown_history:
            details.append(f"history articles: {', '.join(unknown_history)}")
        raise WarehouseValidationError(
            "Behavior records reference article IDs missing from news.tsv ("
            + "; ".join(details)
            + ")."
        )


def _insert_sources(
    connection: duckdb.DuckDBPyConnection,
    *,
    split: str,
    news_sha256: str | None,
    behaviors_sha256: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO warehouse_metadata (
            schema_version,
            split,
            news_sha256,
            behaviors_sha256
        ) VALUES (?, ?, ?, ?)
        """,
        [
            WAREHOUSE_SCHEMA_VERSION,
            split,
            news_sha256,
            behaviors_sha256,
        ],
    )

    connection.execute(
        """
        INSERT INTO articles
        SELECT
            CAST(news_id AS VARCHAR),
            CAST(category AS VARCHAR),
            CAST(subcategory AS VARCHAR),
            CAST(title AS VARCHAR),
            CAST(abstract AS VARCHAR),
            CAST(url AS VARCHAR),
            CAST(title_entities AS VARCHAR),
            CAST(abstract_entities AS VARCHAR)
        FROM _news_source
        """
    )

    connection.execute(
        """
        INSERT INTO behavior_events
        SELECT
            CAST(impression_id AS VARCHAR),
            CAST(user_id AS VARCHAR),
            CAST(timestamp AS TIMESTAMP),
            CASE
                WHEN trim(history) = '' THEN 0
                ELSE len(string_split(trim(history), ' '))
            END,
            len(string_split(trim(impressions), ' '))
        FROM _behavior_source
        """
    )

    connection.execute(
        """
        INSERT INTO user_history
        SELECT
            CAST(behavior.impression_id AS VARCHAR),
            CAST(history_position AS UINTEGER),
            CAST(news_id AS VARCHAR)
        FROM _behavior_source AS behavior,
        UNNEST(string_split(trim(behavior.history), ' '))
            WITH ORDINALITY AS history(news_id, history_position)
        WHERE trim(behavior.history) <> ''
        """
    )

    connection.execute(
        """
        INSERT INTO candidate_interactions
        SELECT
            CAST(behavior.impression_id AS VARCHAR),
            CAST(candidate_position AS UINTEGER),
            regexp_extract(token, '^(.*)-[01]$', 1),
            regexp_extract(token, '-([01])$', 1) = '1'
        FROM _behavior_source AS behavior,
        UNNEST(string_split(trim(behavior.impressions), ' '))
            WITH ORDINALITY AS candidate(token, candidate_position)
        """
    )


def _count(connection: duckdb.DuckDBPyConnection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def build_mind_warehouse(
    news: pd.DataFrame,
    behaviors: pd.DataFrame,
    database_path: str | Path,
    *,
    split: str,
    overwrite: bool = False,
    news_sha256: str | None = None,
    behaviors_sha256: str | None = None,
) -> WarehouseBuildResult:
    """Build a normalized DuckDB database from validated MIND frames.

    The database is created at a temporary sibling path and atomically moved into
    place only after all constraints and inserts succeed.
    """

    destination = Path(database_path)

    if destination.exists() and not overwrite:
        raise WarehouseExistsError(
            f"Warehouse already exists: {destination}. Pass overwrite=True to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    connection: duckdb.DuckDBPyConnection | None = None

    try:
        connection = duckdb.connect(str(temporary))
        connection.register("_news_source", news)
        connection.register("_behavior_source", behaviors)
        connection.execute("BEGIN TRANSACTION")
        connection.execute(SCHEMA_SQL)
        _validate_article_references(connection)
        _insert_sources(
            connection,
            split=split,
            news_sha256=news_sha256,
            behaviors_sha256=behaviors_sha256,
        )
        connection.execute("COMMIT")

        result = WarehouseBuildResult(
            database_path=str(destination),
            schema_version=WAREHOUSE_SCHEMA_VERSION,
            split=split,
            news_sha256=news_sha256,
            behaviors_sha256=behaviors_sha256,
            articles=_count(connection, "articles"),
            behavior_events=_count(connection, "behavior_events"),
            history_events=_count(connection, "user_history"),
            candidate_interactions=_count(connection, "candidate_interactions"),
            clicks=int(
                connection.execute(
                    "SELECT COUNT(*) FROM candidate_interactions WHERE clicked"
                ).fetchone()[0]
            ),
        )
        connection.close()
        connection = None
        os.replace(temporary, destination)
        return result
    except Exception:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(f"{temporary.suffix}.wal").unlink(missing_ok=True)
        raise


def build_mind_warehouse_from_paths(
    news_path: str | Path,
    behaviors_path: str | Path,
    database_path: str | Path,
    *,
    split: str,
    overwrite: bool = False,
) -> WarehouseBuildResult:
    """Validate source files, hash them, and build a DuckDB warehouse."""

    news_source = Path(news_path)
    behaviors_source = Path(behaviors_path)
    news = load_news(news_source)
    behaviors = load_behaviors(behaviors_source)

    return build_mind_warehouse(
        news,
        behaviors,
        database_path,
        split=split,
        overwrite=overwrite,
        news_sha256=_sha256(news_source),
        behaviors_sha256=_sha256(behaviors_source),
    )


def summarize_warehouse(database_path: str | Path) -> WarehouseSummary:
    """Read core warehouse statistics using SQL."""

    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"DuckDB warehouse was not found: {path}")

    with duckdb.connect(str(path), read_only=True) as connection:
        metadata = connection.execute(
            "SELECT schema_version, split FROM warehouse_metadata"
        ).fetchone()
        event_bounds = connection.execute(
            """
            SELECT min(event_timestamp), max(event_timestamp)
            FROM behavior_events
            """
        ).fetchone()

        return WarehouseSummary(
            database_path=str(path),
            schema_version=str(metadata[0]),
            split=str(metadata[1]),
            articles=_count(connection, "articles"),
            behavior_events=_count(connection, "behavior_events"),
            users=int(
                connection.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM behavior_events"
                ).fetchone()[0]
            ),
            history_events=_count(connection, "user_history"),
            candidate_interactions=_count(connection, "candidate_interactions"),
            clicks=int(
                connection.execute(
                    "SELECT COUNT(*) FROM candidate_interactions WHERE clicked"
                ).fetchone()[0]
            ),
            first_event_timestamp=_serialize_timestamp(event_bounds[0]),
            last_event_timestamp=_serialize_timestamp(event_bounds[1]),
        )


def _serialize_timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def article_training_features(
    database_path: str | Path,
    cutoff_timestamp: str | datetime,
) -> pd.DataFrame:
    """Compute leakage-safe article engagement features before a cutoff."""

    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"DuckDB warehouse was not found: {path}")

    query = """
        WITH training_interactions AS (
            SELECT
                candidate.news_id,
                candidate.clicked,
                behavior.user_id
            FROM candidate_interactions AS candidate
            INNER JOIN behavior_events AS behavior
                ON behavior.impression_id = candidate.impression_id
            WHERE behavior.event_timestamp < CAST(? AS TIMESTAMP)
        )
        SELECT
            article.news_id,
            article.category,
            article.subcategory,
            COUNT(interaction.news_id) AS candidate_exposures,
            COALESCE(
                SUM(CASE WHEN interaction.clicked THEN 1 ELSE 0 END),
                0
            ) AS clicks,
            COUNT(DISTINCT interaction.user_id) AS exposed_users,
            CASE
                WHEN COUNT(interaction.news_id) = 0 THEN 0.0
                ELSE SUM(CASE WHEN interaction.clicked THEN 1 ELSE 0 END)::DOUBLE
                    / COUNT(interaction.news_id)
            END AS click_through_rate
        FROM articles AS article
        LEFT JOIN training_interactions AS interaction
            ON interaction.news_id = article.news_id
        GROUP BY article.news_id, article.category, article.subcategory
        ORDER BY article.news_id
    """

    with duckdb.connect(str(path), read_only=True) as connection:
        return connection.execute(query, [cutoff_timestamp]).fetch_df()
