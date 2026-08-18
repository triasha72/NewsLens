import numpy as np
import pandas as pd
import pytest

from newslens.features import (
    ArticleTextFeatureEncoder,
    ArticleTextFeatureError,
)


def _news() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "news_id": "n1",
                "title": "alpha rocket launch",
                "abstract": "space propulsion mission",
                "category": "science",
                "subcategory": "space",
            },
            {
                "news_id": "n2",
                "title": "beta aircraft engine",
                "abstract": "aviation turbine performance",
                "category": "science",
                "subcategory": "aviation",
            },
            {
                "news_id": "n3",
                "title": "gamma orbital vehicle",
                "abstract": "rocket flight trajectory",
                "category": "science",
                "subcategory": "space",
            },
            {
                "news_id": "n4",
                "title": "delta turbine maintenance",
                "abstract": "engine reliability aviation",
                "category": "science",
                "subcategory": "aviation",
            },
            {
                "news_id": "future_shared",
                "title": "alpha futureonlytoken",
                "abstract": "rocket futureexclusive",
                "category": "futurecategory",
                "subcategory": "future",
            },
            {
                "news_id": "future_only",
                "title": "futureonlytoken exclusivetoken",
                "abstract": "neverobservedword",
                "category": "unseenfuturecategory",
                "subcategory": "unseenfuture",
            },
        ]
    )


def _encoder() -> ArticleTextFeatureEncoder:
    return ArticleTextFeatureEncoder(
        max_features=100,
        svd_components=2,
        seed=42,
    )


def test_fit_transform_returns_dense_normalized_features() -> None:
    encoder = _encoder()

    batch = encoder.fit_transform(
        _news(),
        fitting_news_ids=[
            "n1",
            "n2",
            "n3",
            "n4",
        ],
    )

    assert batch.values.shape == (
        6,
        2,
    )

    assert batch.values.dtype == np.float32

    norms = np.linalg.norm(
        batch.values,
        axis=1,
    )

    nonzero = norms > 0.0

    np.testing.assert_allclose(
        norms[nonzero],
        np.ones(
            np.count_nonzero(
                nonzero
            )
        ),
        atol=1e-6,
    )


def test_future_only_vocabulary_does_not_influence_fit() -> None:
    encoder = _encoder()

    batch = encoder.fit_transform(
        _news(),
        fitting_news_ids=[
            "n1",
            "n2",
            "n3",
            "n4",
        ],
    )

    mapping = batch.as_mapping()

    assert np.linalg.norm(
        mapping["future_only"]
    ) == pytest.approx(
        0.0,
        abs=1e-8,
    )

    assert np.linalg.norm(
        mapping["future_shared"]
    ) > 0.0


def test_nonzero_mapping_excludes_unrepresentable_articles() -> None:
    encoder = _encoder()

    batch = encoder.fit_transform(
        _news(),
        fitting_news_ids=[
            "n1",
            "n2",
            "n3",
            "n4",
        ],
    )

    mapping = batch.as_mapping(
        include_zero=False
    )

    assert "future_shared" in mapping
    assert "future_only" not in mapping


def test_feature_generation_is_deterministic() -> None:
    first = (
        _encoder()
        .fit_transform(
            _news(),
            fitting_news_ids=[
                "n1",
                "n2",
                "n3",
                "n4",
            ],
        )
    )

    second = (
        _encoder()
        .fit_transform(
            _news(),
            fitting_news_ids=[
                "n1",
                "n2",
                "n3",
                "n4",
            ],
        )
    )

    assert (
        first.news_ids
        == second.news_ids
    )

    np.testing.assert_allclose(
        first.values,
        second.values,
        atol=1e-7,
    )


def test_unknown_fitting_article_is_rejected() -> None:
    with pytest.raises(
        ArticleTextFeatureError,
        match="missing from the catalog",
    ):
        _encoder().fit(
            _news(),
            fitting_news_ids=[
                "n1",
                "missing",
            ],
        )


def test_transform_before_fit_is_rejected() -> None:
    with pytest.raises(
        ArticleTextFeatureError,
        match="Fit the article text encoder",
    ):
        _encoder().transform(
            _news()
        )


def test_duplicate_article_ids_are_rejected() -> None:
    news = _news()

    duplicate = pd.concat(
        [
            news,
            news.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ArticleTextFeatureError,
        match="Duplicate article identifiers",
    ):
        _encoder().fit(
            duplicate,
            fitting_news_ids=[
                "n1",
                "n2",
                "n3",
                "n4",
            ],
        )
