from __future__ import annotations

from math import log2

import pytest

from newslens.evaluation import (
    RankingMetricError,
    catalog_coverage,
    hit_rate_at_k,
    mean_reciprocal_rank_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_hit_rate_at_k_handles_multiple_relevant_items() -> None:
    ranking = ["N1", "N2", "N3", "N4"]
    relevant = {"N3", "N4"}

    assert hit_rate_at_k(ranking, relevant, k=2) == 0.0
    assert hit_rate_at_k(ranking, relevant, k=3) == 1.0


def test_recall_at_k_handles_multiple_relevant_items() -> None:
    ranking = ["N1", "N2", "N3", "N4"]
    relevant = {"N2", "N4"}

    assert recall_at_k(ranking, relevant, k=3) == pytest.approx(0.5)
    assert recall_at_k(ranking, relevant, k=4) == pytest.approx(1.0)


def test_reciprocal_rank_uses_first_relevant_item() -> None:
    ranking = ["N1", "N2", "N3", "N4"]

    assert reciprocal_rank_at_k(
        ranking,
        {"N3", "N4"},
        k=4,
    ) == pytest.approx(1.0 / 3.0)


def test_mean_reciprocal_rank_matches_hand_calculation() -> None:
    rankings = [
        ["N1", "N2", "N3"],
        ["N4", "N5", "N6"],
    ]
    relevant = [
        {"N2"},
        {"N6"},
    ]

    assert mean_reciprocal_rank_at_k(
        rankings,
        relevant,
        k=3,
    ) == pytest.approx(5.0 / 12.0)


def test_ndcg_at_k_matches_hand_calculation() -> None:
    ranking = ["N1", "N2", "N3", "N4"]
    relevant = {"N2", "N4"}
    expected_dcg = (1.0 / log2(3)) + (1.0 / log2(5))
    ideal_dcg = 1.0 + (1.0 / log2(3))

    assert ndcg_at_k(
        ranking,
        relevant,
        k=4,
    ) == pytest.approx(expected_dcg / ideal_dcg)


def test_ndcg_at_k_is_one_for_ideal_ranking() -> None:
    assert ndcg_at_k(
        ["N1", "N2", "N3"],
        {"N1", "N2"},
        k=3,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "metric",
    [
        hit_rate_at_k,
        recall_at_k,
        reciprocal_rank_at_k,
        ndcg_at_k,
    ],
)
def test_metrics_return_zero_when_relevant_items_are_not_retrieved(
    metric,
) -> None:
    assert metric(["N1", "N2"], {"N3"}, k=2) == 0.0


@pytest.mark.parametrize(
    "metric",
    [
        hit_rate_at_k,
        recall_at_k,
        reciprocal_rank_at_k,
        ndcg_at_k,
    ],
)
def test_metrics_reject_empty_relevance(metric) -> None:
    with pytest.raises(
        RankingMetricError,
        match="At least one relevant item",
    ):
        metric(["N1", "N2"], [], k=2)


@pytest.mark.parametrize("invalid_k", [0, -1, 1.5, True])
def test_metrics_reject_invalid_k(invalid_k) -> None:
    with pytest.raises(
        RankingMetricError,
        match="positive integer",
    ):
        hit_rate_at_k(["N1"], {"N1"}, k=invalid_k)


def test_metrics_reject_duplicate_ranked_items() -> None:
    with pytest.raises(
        RankingMetricError,
        match="must not contain duplicates",
    ):
        recall_at_k(["N1", "N1"], {"N1"}, k=2)


def test_mean_reciprocal_rank_rejects_mismatched_lengths() -> None:
    with pytest.raises(
        RankingMetricError,
        match="same number of entries",
    ):
        mean_reciprocal_rank_at_k(
            [["N1"], ["N2"]],
            [{"N1"}],
            k=1,
        )


def test_mean_reciprocal_rank_requires_a_ranking() -> None:
    with pytest.raises(
        RankingMetricError,
        match="At least one ranking",
    ):
        mean_reciprocal_rank_at_k([], [], k=1)


def test_catalog_coverage_counts_unique_recommended_items() -> None:
    recommendations = [
        ["N1", "N2"],
        ["N2", "N3"],
    ]
    catalog = ["N1", "N2", "N3", "N4", "N5"]

    assert catalog_coverage(
        recommendations,
        catalog,
    ) == pytest.approx(0.6)


def test_catalog_coverage_is_zero_without_recommendations() -> None:
    assert catalog_coverage([], ["N1", "N2"]) == 0.0


def test_catalog_coverage_rejects_empty_catalog() -> None:
    with pytest.raises(
        RankingMetricError,
        match="At least one catalog item",
    ):
        catalog_coverage([["N1"]], [])


def test_catalog_coverage_rejects_items_outside_catalog() -> None:
    with pytest.raises(
        RankingMetricError,
        match="outside the catalog",
    ):
        catalog_coverage([["N1", "N3"]], ["N1", "N2"])
