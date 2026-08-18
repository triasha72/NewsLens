"""Exact output-parity tests for the Phase-07 MMR optimization."""

import numpy as np
import pytest

from newslens.reranking import (
    MMRConfig,
    maximal_marginal_relevance,
    maximal_marginal_relevance_vectorized,
)


@pytest.mark.parametrize(
    "lambda_weight",
    [
        1.00,
        0.95,
        0.90,
        0.85,
        0.80,
        0.50,
    ],
)
@pytest.mark.parametrize(
    "candidate_count",
    [
        1,
        2,
        10,
        25,
        100,
    ],
)
def test_vectorized_matches_reference_randomized(
    lambda_weight: float,
    candidate_count: int,
) -> None:
    rng = np.random.default_rng(
        42
        + candidate_count
        + int(
            lambda_weight
            * 100
        )
    )

    for trial in range(20):
        vectors = rng.normal(
            size=(
                candidate_count,
                64,
            )
        ).astype(
            np.float32
        )

        vectors /= np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        scores = rng.normal(
            size=candidate_count
        ).astype(
            np.float64
        )

        news_ids = tuple(
            f"N{trial:03d}_{index:04d}"
            for index in range(
                candidate_count
            )
        )

        top_k = min(
            10,
            candidate_count,
        )

        config = MMRConfig(
            lambda_weight=(
                lambda_weight
            )
        )

        reference = (
            maximal_marginal_relevance(
                news_ids,
                scores,
                vectors,
                top_k=top_k,
                config=config,
            )
        )

        optimized = (
            maximal_marginal_relevance_vectorized(
                news_ids,
                scores,
                vectors,
                top_k=top_k,
                config=config,
            )
        )

        assert optimized == reference


def test_vectorized_matches_reference_exact_ties() -> None:
    news_ids = (
        "z",
        "a",
        "m",
        "b",
    )

    scores = np.ones(
        4,
        dtype=np.float64,
    )

    vectors = np.eye(
        4,
        dtype=np.float32,
    )

    config = MMRConfig(
        lambda_weight=1.0
    )

    assert (
        maximal_marginal_relevance_vectorized(
            news_ids,
            scores,
            vectors,
            top_k=4,
            config=config,
        )
        == maximal_marginal_relevance(
            news_ids,
            scores,
            vectors,
            top_k=4,
            config=config,
        )
    )


def test_vectorized_matches_reference_near_ties() -> None:
    news_ids = (
        "z",
        "a",
        "m",
    )

    scores = np.asarray(
        [
            1.0,
            1.0 - 5e-13,
            0.5,
        ],
        dtype=np.float64,
    )

    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    config = MMRConfig(
        lambda_weight=1.0
    )

    assert (
        maximal_marginal_relevance_vectorized(
            news_ids,
            scores,
            vectors,
            top_k=3,
            config=config,
        )
        == maximal_marginal_relevance(
            news_ids,
            scores,
            vectors,
            top_k=3,
            config=config,
        )
    )


def test_vectorized_matches_reference_negative_similarities() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )

    scores = np.asarray(
        [
            1.0,
            0.9,
            0.8,
            0.7,
        ],
        dtype=np.float64,
    )

    news_ids = (
        "a",
        "b",
        "c",
        "d",
    )

    config = MMRConfig(
        lambda_weight=0.80
    )

    assert (
        maximal_marginal_relevance_vectorized(
            news_ids,
            scores,
            vectors,
            top_k=4,
            config=config,
        )
        == maximal_marginal_relevance(
            news_ids,
            scores,
            vectors,
            top_k=4,
            config=config,
        )
    )


def test_vectorized_top_k_larger_than_candidates() -> None:
    news_ids = (
        "a",
        "b",
    )

    scores = np.asarray(
        [
            0.8,
            0.7,
        ],
        dtype=np.float64,
    )

    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    config = MMRConfig(
        lambda_weight=0.80
    )

    assert (
        maximal_marginal_relevance_vectorized(
            news_ids,
            scores,
            vectors,
            top_k=10,
            config=config,
        )
        == maximal_marginal_relevance(
            news_ids,
            scores,
            vectors,
            top_k=10,
            config=config,
        )
    )
