"""Tests for Phase-05 second-stage ranking."""

import numpy as np
import pandas as pd

from newslens.models.popularity import (
    PopularityRecommender,
)
from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
)
from newslens.ranking import (
    SECOND_STAGE_FEATURE_NAMES,
    SecondStageFeatureBuilder,
    SecondStageRanker,
    SecondStageRankerConfig,
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)


def _catalog() -> RetrievalCatalog:
    return RetrievalCatalog(
        news_ids=(
            "N1",
            "N2",
            "N3",
            "N4",
        ),
        vectors=np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def _news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "news_id": [
                "N1",
                "N2",
                "N3",
                "N4",
            ],
            "category": [
                "a",
                "a",
                "b",
                "c",
            ],
            "subcategory": [
                "a1",
                "a2",
                "b1",
                "c1",
            ],
        }
    )


def _popularity() -> PopularityRecommender:
    behaviors = pd.DataFrame(
        {
            "impressions": [
                "N1-1 N2-0 N3-0",
                "N1-0 N2-1 N4-0",
            ]
        }
    )

    return (
        PopularityRecommender()
        .fit(
            behaviors
        )
    )


def _builder() -> SecondStageFeatureBuilder:
    network = TwoTowerNetwork(
        TwoTowerConfig(
            input_dim=2,
            hidden_dim=4,
            embedding_dim=2,
            dropout=0.0,
            temperature=0.07,
        )
    )

    return SecondStageFeatureBuilder(
        catalog=_catalog(),
        network=network,
        news=_news(),
        popularity_model=(
            _popularity()
        ),
        max_history_length=2,
    )


def test_feature_builder_has_fixed_dimension() -> None:
    builder = _builder()

    context = builder.build_context(
        (
            "N1",
            "N2",
        )
    )

    assert context is not None

    matrix = (
        builder.features_for_candidates(
            context,
            (
                "N2",
                "N3",
            ),
        )
    )

    assert matrix.shape == (
        2,
        len(
            SECOND_STAGE_FEATURE_NAMES
        ),
    )

    assert np.isfinite(
        matrix
    ).all()


def test_feature_builder_returns_none_for_unknown_history() -> None:
    builder = _builder()

    assert (
        builder.build_context(
            ("UNKNOWN",)
        )
        is None
    )


def test_ranker_fits_and_ranks() -> None:
    features = np.asarray(
        [
            [0.9] + [0.0] * 10,
            [0.8] + [0.0] * 10,
            [0.1] + [0.0] * 10,
            [0.2] + [0.0] * 10,
            [0.95] + [0.0] * 10,
            [0.05] + [0.0] * 10,
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [
            1,
            1,
            0,
            0,
            1,
            0,
        ],
        dtype=np.int64,
    )

    ranker = SecondStageRanker(
        SecondStageRankerConfig(
            max_iter=10,
            min_samples_leaf=2,
        )
    )

    ranker.fit(
        features,
        labels,
    )

    scores = ranker.score(
        features
    )

    assert scores.shape == (
        6,
    )

    assert np.isfinite(
        scores
    ).all()

    ranking = ranker.rank(
        (
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ),
        features,
        top_k=3,
    )

    assert len(ranking) == 3
    assert len(set(ranking)) == 3
