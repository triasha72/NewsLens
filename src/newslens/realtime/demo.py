"""Safe, synthetic content for the public NewsLens search demonstration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from .ranking import SearchDocument
from .repository import InMemoryArticleRepository


def create_demo_repository(*, now: datetime | None = None) -> InMemoryArticleRepository:
    """Return a deterministic-in-shape store containing no licensed or user data.

    Timestamps are intentionally relative to startup so the demo can illustrate
    freshness-aware ranking without presenting historical articles as live news.
    """

    reference_time = now or datetime.now(UTC)
    return InMemoryArticleRepository(
        (
            SearchDocument(
                article_id="demo-ai-chip",
                title="Demo: new AI chip benchmark published",
                body="Synthetic technology article used only to demonstrate search ranking.",
                category="technology",
                published_at=reference_time - timedelta(minutes=20),
                produced_at=reference_time - timedelta(minutes=19, seconds=58),
                indexed_at=reference_time - timedelta(minutes=19, seconds=57),
                popularity=42,
            ),
            SearchDocument(
                article_id="demo-energy-grid",
                title="Demo: grid reliability research update",
                body="Synthetic science article for a reproducible public demonstration.",
                category="science",
                published_at=reference_time - timedelta(hours=3),
                produced_at=reference_time - timedelta(hours=3, seconds=-1),
                indexed_at=reference_time - timedelta(hours=3, seconds=-2),
                popularity=17,
            ),
            SearchDocument(
                article_id="demo-market-report",
                title="Demo: market report explains quarterly results",
                body="Synthetic business article for inspecting query intent and ranking signals.",
                category="business",
                published_at=reference_time - timedelta(days=2),
                produced_at=reference_time - timedelta(days=2, seconds=-1),
                indexed_at=reference_time - timedelta(days=2, seconds=-2),
                popularity=91,
            ),
        )
    )
