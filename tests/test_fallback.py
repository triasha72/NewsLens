from __future__ import annotations

import pandas as pd
import pytest

from newslens.models import (
    ContentBasedRecommender,
    PopularityRecommender,
)
from newslens.models.fallback import (
    ContentPopularityFallbackRecommender,
    FallbackModelError,
    RecommendationSource,
)


def make_news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "news_id": ["N1", "N2", "N3", "N4"],
            "category": [
                "science",
                "science",
                "sports",
                "finance",
            ],
            "subcategory": [
                "space",
                "space",
                "football",
                "markets",
            ],
            "title": [
                "Mars spacecraft discovers water",
                "Mars mission searches for water",
                "Football team wins championship",
                "Stock markets rise after earnings",
            ],
            "abstract": [
                "Scientists study evidence from the planet.",
                "A spacecraft begins an exploration mission.",
                "The coach celebrates with the players.",
                "Technology companies report strong results.",
            ],
        }
    )


def make_popularity_model() -> PopularityRecommender:
    behaviors = pd.DataFrame(
        {
            "impressions": [
                "N1-0 N2-0 N3-1 N4-0",
                "N1-0 N2-0 N3-1 N4-1",
                "N1-0 N2-0 N3-0 N4-1",
            ]
        }
    )

    return PopularityRecommender().fit(behaviors)


def make_fallback_model() -> ContentPopularityFallbackRecommender:
    content = ContentBasedRecommender().fit(make_news())
    popularity = make_popularity_model()

    return ContentPopularityFallbackRecommender(
        content,
        popularity,
    )


def test_uses_content_for_known_history() -> None:
    model = make_fallback_model()

    results = model.recommend(
        ["N1"],
        candidate_news_ids=["N2", "N3"],
        top_k=2,
    )

    assert results[0].news_id == "N2"
    assert results[0].source == RecommendationSource.CONTENT


def test_empty_history_uses_popularity() -> None:
    model = make_fallback_model()

    results = model.recommend(
        [],
        candidate_news_ids=["N2", "N3", "N4"],
        top_k=3,
    )

    assert results[0].news_id == "N3"
    assert all(result.source == RecommendationSource.POPULARITY for result in results)


def test_unknown_history_uses_popularity() -> None:
    model = make_fallback_model()

    results = model.recommend(
        ["UNKNOWN"],
        candidate_news_ids=["N2", "N3", "N4"],
        top_k=3,
    )

    assert results[0].news_id == "N3"
    assert results[0].source == RecommendationSource.POPULARITY


def test_zero_content_signal_uses_popularity() -> None:
    model = make_fallback_model()

    results = model.recommend(
        ["N1"],
        candidate_news_ids=["N4"],
    )

    assert results[0].news_id == "N4"
    assert results[0].source == RecommendationSource.POPULARITY


def test_fallback_excludes_history_articles() -> None:
    model = make_fallback_model()

    results = model.recommend(
        ["N1"],
        candidate_news_ids=["N1", "N4"],
    )

    assert [result.news_id for result in results] == ["N4"]


def test_requires_fitted_content_model() -> None:
    content = ContentBasedRecommender()
    popularity = make_popularity_model()

    with pytest.raises(
        FallbackModelError,
        match="content model must be fitted",
    ):
        ContentPopularityFallbackRecommender(
            content,
            popularity,
        )


def test_requires_fitted_popularity_model() -> None:
    content = ContentBasedRecommender().fit(make_news())
    popularity = PopularityRecommender()

    with pytest.raises(
        FallbackModelError,
        match="popularity model must be fitted",
    ):
        ContentPopularityFallbackRecommender(
            content,
            popularity,
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_rejects_invalid_top_k(top_k: int) -> None:
    model = make_fallback_model()

    with pytest.raises(
        FallbackModelError,
        match="top_k must be greater than zero",
    ):
        model.recommend(
            [],
            candidate_news_ids=["N1", "N2"],
            top_k=top_k,
        )
