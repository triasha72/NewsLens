from __future__ import annotations

import pandas as pd
import pytest

from newslens.models import (
    ArticleSearchError,
    ArticleSearchNotFittedError,
    TfidfArticleSearch,
)


def make_news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "news_id": ["N1", "N2", "N3"],
            "category": ["science", "sports", "finance"],
            "subcategory": ["space", "football", "markets"],
            "title": [
                "NASA launches Mars exploration mission",
                "Football team wins national championship",
                "Technology earnings lift stock markets",
            ],
            "abstract": [
                "The spacecraft will search for evidence of water.",
                "The coach praised the players after the final.",
                "Investors responded to strong quarterly results.",
            ],
        }
    )


def test_search_ranks_relevant_article_first() -> None:
    model = TfidfArticleSearch().fit(make_news())

    results = model.search(
        "Mars spacecraft exploration",
        top_k=3,
    )

    assert results
    assert results[0].news_id == "N1"
    assert results[0].score > 0.0
    assert results[0].score <= 1.0


def test_search_uses_article_abstracts() -> None:
    model = TfidfArticleSearch().fit(make_news())

    results = model.search("coach players final")

    assert results
    assert results[0].news_id == "N2"


def test_search_returns_article_metadata() -> None:
    model = TfidfArticleSearch().fit(make_news())

    result = model.search("stock markets", top_k=1)[0]

    assert result.news_id == "N3"
    assert result.title == "Technology earnings lift stock markets"
    assert result.category == "finance"


def test_search_excludes_requested_articles() -> None:
    model = TfidfArticleSearch().fit(make_news())

    results = model.search(
        "Mars spacecraft",
        exclude_news_ids={"N1"},
    )

    assert results == []


def test_search_returns_empty_for_unknown_vocabulary() -> None:
    model = TfidfArticleSearch().fit(make_news())

    results = model.search("xyzzyplugh")

    assert results == []


def test_search_before_fit_raises_error() -> None:
    model = TfidfArticleSearch()

    with pytest.raises(
        ArticleSearchNotFittedError,
        match="Fit the TF-IDF search model",
    ):
        model.search("space")


def test_search_rejects_empty_query() -> None:
    model = TfidfArticleSearch().fit(make_news())

    with pytest.raises(
        ArticleSearchError,
        match="query cannot be empty",
    ):
        model.search("   ")


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_invalid_top_k(top_k: int) -> None:
    model = TfidfArticleSearch().fit(make_news())

    with pytest.raises(
        ArticleSearchError,
        match="top_k must be greater than zero",
    ):
        model.search("space", top_k=top_k)


def test_fit_requires_news_id_and_title() -> None:
    model = TfidfArticleSearch()

    with pytest.raises(
        ArticleSearchError,
        match="Missing required article columns",
    ):
        model.fit(pd.DataFrame({"abstract": ["Example"]}))


def test_fit_rejects_duplicate_news_ids() -> None:
    news = make_news()
    news.loc[1, "news_id"] = "N1"

    with pytest.raises(
        ArticleSearchError,
        match="Duplicate article identifiers",
    ):
        TfidfArticleSearch().fit(news)


def test_fit_does_not_modify_input_data() -> None:
    news = make_news()
    original = news.copy(deep=True)

    TfidfArticleSearch().fit(news)

    pd.testing.assert_frame_equal(news, original)
