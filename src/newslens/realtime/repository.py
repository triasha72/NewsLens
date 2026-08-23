"""Repositories for real-time article search."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from .query import QueryIntent
from .ranking import SearchDocument


@runtime_checkable
class ArticleRepository(Protocol):
    """Storage contract used by the real-time search API."""

    def ping(self) -> bool:
        """Return whether the backing store is reachable."""

    def list_candidates(
        self,
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[SearchDocument, ...]:
        """Return a bounded candidate set for ranking."""

    def close(self) -> None:
        """Release repository resources."""


class InMemoryArticleRepository:
    """Deterministic repository for tests and local ranking experiments."""

    def __init__(self, documents: Iterable[SearchDocument]) -> None:
        self._documents = tuple(documents)

    def ping(self) -> bool:
        return True

    def list_candidates(
        self,
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[SearchDocument, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        candidates = self._documents

        if intent.category is not None:
            category_matches = tuple(
                document
                for document in candidates
                if document.category.lower() == intent.category
            )
            if category_matches:
                candidates = category_matches

        return tuple(
            sorted(
                candidates,
                key=lambda document: (
                    -document.published_at.timestamp(),
                    document.article_id,
                ),
            )[:limit]
        )

    def close(self) -> None:
        return None


class PostgresArticleRepository:
    """Read streamed articles from the PostgreSQL ingestion store."""

    def __init__(self, database_url: str) -> None:
        if not database_url.strip():
            raise ValueError("database_url cannot be empty")

        import psycopg

        self._connection = psycopg.connect(database_url, autocommit=True)
        self._database_error = psycopg.Error

    def ping(self) -> bool:
        try:
            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone() == (1,)
        except self._database_error:
            return False

    def list_candidates(
        self,
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[SearchDocument, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        entity_pattern = f"%{intent.entity}%" if intent.entity else None

        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    article_id,
                    title,
                    body,
                    category,
                    published_at,
                    produced_at,
                    indexed_at,
                    popularity
                FROM realtime_articles
                WHERE (CAST(%s AS TEXT) IS NULL OR lower(category) = %s)
                  AND (
                    CAST(%s AS TEXT) IS NULL
                    OR title ILIKE %s
                    OR body ILIKE %s
                  )
                ORDER BY published_at DESC, article_id ASC
                LIMIT %s
                """,
                (
                    intent.category,
                    intent.category,
                    entity_pattern,
                    entity_pattern,
                    entity_pattern,
                    limit,
                ),
            )
            rows = cursor.fetchall()

        return tuple(
            SearchDocument(
                article_id=row[0],
                title=row[1],
                body=row[2],
                category=row[3],
                published_at=row[4],
                produced_at=row[5],
                indexed_at=row[6],
                popularity=row[7],
            )
            for row in rows
        )

    def close(self) -> None:
        self._connection.close()
