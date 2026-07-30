"""Ranking metrics for offline NewsLens evaluation.

Each per-impression metric requires at least one relevant item. This prevents
impressions without clicks from silently changing the evaluation denominator;
the dataset-level evaluator must skip and report those impressions explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import fsum, log2
from statistics import fmean


class RankingMetricError(ValueError):
    """Raised when ranking-metric input is invalid."""


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise RankingMetricError("k must be a positive integer.")


def _prepare_ranking(ranked_items: Iterable[str]) -> tuple[str, ...]:
    ranking = tuple(ranked_items)

    if len(ranking) != len(set(ranking)):
        raise RankingMetricError("ranked_items must not contain duplicates.")

    return ranking


def _prepare_relevant_items(relevant_items: Iterable[str]) -> frozenset[str]:
    relevant = frozenset(relevant_items)

    if not relevant:
        raise RankingMetricError("At least one relevant item is required.")

    return relevant


def hit_rate_at_k(
    ranked_items: Iterable[str],
    relevant_items: Iterable[str],
    *,
    k: int,
) -> float:
    """Return 1 when at least one relevant item appears in the top-k."""

    _validate_k(k)
    ranking = _prepare_ranking(ranked_items)
    relevant = _prepare_relevant_items(relevant_items)

    return float(any(item in relevant for item in ranking[:k]))


def recall_at_k(
    ranked_items: Iterable[str],
    relevant_items: Iterable[str],
    *,
    k: int,
) -> float:
    """Return the fraction of relevant items retrieved in the top-k."""

    _validate_k(k)
    ranking = _prepare_ranking(ranked_items)
    relevant = _prepare_relevant_items(relevant_items)
    retrieved_relevant = sum(item in relevant for item in ranking[:k])

    return retrieved_relevant / len(relevant)


def reciprocal_rank_at_k(
    ranked_items: Iterable[str],
    relevant_items: Iterable[str],
    *,
    k: int,
) -> float:
    """Return the reciprocal rank of the first relevant top-k item."""

    _validate_k(k)
    ranking = _prepare_ranking(ranked_items)
    relevant = _prepare_relevant_items(relevant_items)

    for rank, item in enumerate(ranking[:k], start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank_at_k(
    rankings: Iterable[Iterable[str]],
    relevant_items_by_ranking: Iterable[Iterable[str]],
    *,
    k: int,
) -> float:
    """Return mean reciprocal rank across a collection of impressions."""

    _validate_k(k)
    ranking_list = list(rankings)
    relevant_list = list(relevant_items_by_ranking)

    if not ranking_list:
        raise RankingMetricError("At least one ranking is required.")

    if len(ranking_list) != len(relevant_list):
        raise RankingMetricError(
            "rankings and relevant_items_by_ranking must contain the same number of entries."
        )

    return fmean(
        reciprocal_rank_at_k(ranking, relevant, k=k)
        for ranking, relevant in zip(ranking_list, relevant_list, strict=True)
    )


def ndcg_at_k(
    ranked_items: Iterable[str],
    relevant_items: Iterable[str],
    *,
    k: int,
) -> float:
    """Return binary normalized discounted cumulative gain at k."""

    _validate_k(k)
    ranking = _prepare_ranking(ranked_items)
    relevant = _prepare_relevant_items(relevant_items)

    dcg = fsum(
        1.0 / log2(rank + 1) for rank, item in enumerate(ranking[:k], start=1) if item in relevant
    )

    ideal_relevant_count = min(k, len(relevant))
    ideal_dcg = fsum(1.0 / log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))

    return dcg / ideal_dcg


def catalog_coverage(
    recommendations: Iterable[Iterable[str]],
    catalog_items: Iterable[str],
) -> float:
    """Return the fraction of catalog items appearing in recommendations."""

    catalog = frozenset(catalog_items)

    if not catalog:
        raise RankingMetricError("At least one catalog item is required.")

    recommended_items: set[str] = set()

    for ranking in recommendations:
        recommended_items.update(_prepare_ranking(ranking))

    unknown_items = recommended_items - catalog

    if unknown_items:
        unknown_preview = ", ".join(sorted(unknown_items)[:3])
        raise RankingMetricError(
            f"Recommendations contain items outside the catalog: {unknown_preview}."
        )

    return len(recommended_items) / len(catalog)
