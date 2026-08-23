"""Tests for transparent freshness-aware ranking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from newslens.realtime import FreshnessRanker, SearchDocument, understand_query


def article(
    article_id: str,
    *,
    title: str,
    age_hours: int,
    popularity: int = 0,
) -> SearchDocument:
    now = datetime(2026, 8, 23, 12, tzinfo=UTC)
    produced_at = now - timedelta(hours=age_hours, milliseconds=120)
    return SearchDocument(
        article_id=article_id,
        title=title,
        body="Apple quarterly earnings and market coverage.",
        category="business",
        published_at=now - timedelta(hours=age_hours),
        produced_at=produced_at,
        indexed_at=produced_at + timedelta(milliseconds=120),
        popularity=popularity,
    )


def test_freshness_intent_prefers_newer_equally_relevant_article() -> None:
    now = datetime(2026, 8, 23, 12, tzinfo=UTC)
    ranker = FreshnessRanker()
    ranked = ranker.rank(
        understand_query("latest Apple earnings"),
        (
            article("old", title="Apple earnings report", age_hours=72),
            article("new", title="Apple earnings report", age_hours=1),
        ),
        top_k=2,
        now=now,
    )

    assert [item.document.article_id for item in ranked] == ["new", "old"]
    assert ranked[0].freshness_score > ranked[1].freshness_score


def test_score_components_and_index_freshness_are_exposed() -> None:
    now = datetime(2026, 8, 23, 12, tzinfo=UTC)
    result = FreshnessRanker().rank(
        understand_query("Apple earnings"),
        (article("N1", title="Apple earnings report", age_hours=2, popularity=10),),
        top_k=1,
        now=now,
    )[0]

    assert 0.0 <= result.score <= 1.0
    assert result.relevance_score > 0.0
    assert result.index_freshness_ms == pytest.approx(120.0)


def test_ranker_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="sum"):
        FreshnessRanker(
            relevance_weight=0.5,
            freshness_weight=0.5,
            popularity_weight=0.5,
        )
