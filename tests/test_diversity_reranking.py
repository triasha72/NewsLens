"""Tests for deterministic diversity-aware reranking."""

import numpy as np
import pytest

from newslens.reranking import (
    DiversityRerankingError,
    MMRConfig,
    maximal_marginal_relevance,
)


def test_lambda_one_matches_relevance_order() -> None:
    news_ids = (
        "a",
        "b",
        "c",
    )

    scores = np.asarray(
        [
            0.9,
            0.8,
            0.7,
        ],
        dtype=np.float64,
    )

    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = maximal_marginal_relevance(
        news_ids,
        scores,
        vectors,
        top_k=3,
        config=MMRConfig(
            lambda_weight=1.0
        ),
    )

    assert result == (
        "a",
        "b",
        "c",
    )


def test_diversity_can_change_second_position() -> None:
    news_ids = (
        "a",
        "b",
        "c",
    )

    scores = np.asarray(
        [
            1.0,
            0.9,
            0.8,
        ],
        dtype=np.float64,
    )

    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.99995, 0.01],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    vectors /= np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    result = maximal_marginal_relevance(
        news_ids,
        scores,
        vectors,
        top_k=3,
        config=MMRConfig(
            lambda_weight=0.5
        ),
    )

    assert result[0] == "a"
    assert result[1] == "c"


def test_mmr_is_deterministic() -> None:
    news_ids = (
        "z",
        "a",
        "m",
    )

    scores = np.asarray(
        [
            1.0,
            1.0,
            0.5,
        ],
        dtype=np.float64,
    )

    vectors = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    first = maximal_marginal_relevance(
        news_ids,
        scores,
        vectors,
        top_k=3,
        config=MMRConfig(
            lambda_weight=1.0
        ),
    )

    second = maximal_marginal_relevance(
        news_ids,
        scores,
        vectors,
        top_k=3,
        config=MMRConfig(
            lambda_weight=1.0
        ),
    )

    assert first == second
    assert first[:2] == (
        "a",
        "z",
    )


def test_invalid_lambda_rejected() -> None:
    with pytest.raises(
        DiversityRerankingError
    ):
        MMRConfig(
            lambda_weight=1.1
        )


def test_non_normalized_vectors_rejected() -> None:
    with pytest.raises(
        DiversityRerankingError
    ):
        maximal_marginal_relevance(
            (
                "a",
                "b",
            ),
            np.asarray(
                [
                    1.0,
                    0.5,
                ]
            ),
            np.asarray(
                [
                    [2.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            top_k=2,
            config=MMRConfig(
                lambda_weight=0.9
            ),
        )
