from __future__ import annotations

import pandas as pd
import pytest

from newslens.models import (
    ColdStartUserError,
    ContentBasedRecommender,
    ContentModelError,
    ContentModelNotFittedError,
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
                "A spacecraft begins a new exploration mission.",
                "The coach celebrates with the players.",
                "Technology companies report strong results.",
            ],
        }
    )


def test_recommend_ranks_similar_article_first() -> None:
    model = ContentBasedRecommender().fit(make_news())

    results = model.recommend(
        ["N1"],
        candidate_news_ids=["N2", "N3", "N4"],
        top_k=3,
    )

    assert results[0].news_id == "N2"
    assert results[0].score > results[1].score


def test_recommend_excludes_history_by_default() -> None:
    model = ContentBasedRecommender().fit(make_news())

    results = model.recommend(
        ["N1"],
        candidate_news_ids=["N1", "N2"],
    )

    assert [result.news_id for result in results] == ["N2"]


def test_recommend_can_include_history() -> None:
    model = ContentBasedRecommender().fit(make_news())

    results = model.recommend(
        ["N1"],
        candidate_news_ids=["N1", "N2"],
        exclude_history=False,
    )

    assert results[0].news_id == "N1"


def test_recommend_returns_metadata() -> None:
    model = ContentBasedRecommender().fit(make_news())

    result = model.recommend(
        ["N1"],
        candidate_news_ids=["N2"],
        top_k=1,
    )[0]

    assert result.title == "Mars mission searches for water"
    assert result.category == "science"
    assert 0.0 <= result.score <= 1.0


def test_training_only_articles_define_vocabulary() -> None:
    model = ContentBasedRecommender().fit(
        make_news(),
        vocabulary_news_ids={"N1"},
    )

    results = model.recommend(
        ["N1"],
        candidate_news_ids=["N4"],
    )

    assert model.vocabulary_article_count == 1
    assert results[0].score == 0.0


def test_unknown_history_raises_cold_start_error() -> None:
    model = ContentBasedRecommender().fit(make_news())

    with pytest.raises(
        ColdStartUserError,
        match="no known articles",
    ):
        model.recommend(
            ["UNKNOWN"],
            candidate_news_ids=["N1", "N2"],
        )


def test_recommend_before_fit_raises_error() -> None:
    model = ContentBasedRecommender()

    with pytest.raises(
        ContentModelNotFittedError,
        match="Fit the content model",
    ):
        model.recommend(
            ["N1"],
            candidate_news_ids=["N2"],
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_recommend_rejects_invalid_top_k(top_k: int) -> None:
    model = ContentBasedRecommender().fit(make_news())

    with pytest.raises(
        ContentModelError,
        match="top_k must be greater than zero",
    ):
        model.recommend(
            ["N1"],
            candidate_news_ids=["N2"],
            top_k=top_k,
        )


def test_fit_requires_news_id_and_title() -> None:
    model = ContentBasedRecommender()

    with pytest.raises(
        ContentModelError,
        match="Missing required article columns",
    ):
        model.fit(pd.DataFrame({"abstract": ["Example"]}))


def test_fit_rejects_duplicate_article_ids() -> None:
    news = make_news()
    news.loc[1, "news_id"] = "N1"

    with pytest.raises(
        ContentModelError,
        match="Duplicate article identifiers",
    ):
        ContentBasedRecommender().fit(news)


def test_fit_does_not_modify_input_data() -> None:
    news = make_news()
    original = news.copy(deep=True)

    ContentBasedRecommender().fit(news)

    pd.testing.assert_frame_equal(news, original)
