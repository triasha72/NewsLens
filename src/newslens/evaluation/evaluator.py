"""Model-independent aggregation for offline ranking evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

from .metrics import (
    RankingMetricError,
    catalog_coverage,
    hit_rate_at_k,
    mean_reciprocal_rank_at_k,
    ndcg_at_k,
    recall_at_k,
)


class RankingEvaluationError(ValueError):
    """Raised when an offline ranking evaluation cannot be completed."""


@dataclass(frozen=True)
class RankingExample:
    """One ranked impression and its clicked, relevant articles."""

    impression_id: str
    ranked_items: tuple[str, ...]
    relevant_items: frozenset[str]

    def __post_init__(self) -> None:
        impression_id = str(self.impression_id).strip()

        if not impression_id:
            raise RankingEvaluationError("impression_id must not be empty.")

        if isinstance(self.ranked_items, str):
            raise RankingEvaluationError("ranked_items must be an iterable of item IDs.")

        if isinstance(self.relevant_items, str):
            raise RankingEvaluationError("relevant_items must be an iterable of item IDs.")

        ranking = tuple(self.ranked_items)
        relevant = frozenset(self.relevant_items)

        if any(not isinstance(item, str) or not item.strip() for item in ranking):
            raise RankingEvaluationError("ranked_items must contain non-empty string IDs.")

        if any(not isinstance(item, str) or not item.strip() for item in relevant):
            raise RankingEvaluationError("relevant_items must contain non-empty string IDs.")

        if len(ranking) != len(set(ranking)):
            raise RankingEvaluationError("ranked_items must not contain duplicates.")

        object.__setattr__(self, "impression_id", impression_id)
        object.__setattr__(self, "ranked_items", ranking)
        object.__setattr__(self, "relevant_items", relevant)


@dataclass(frozen=True)
class RankingEvaluationResult:
    """Aggregate top-k metrics and accounting information."""

    k: int
    total_impressions: int
    evaluated_impressions: int
    skipped_no_click_impressions: int
    empty_ranking_impressions: int
    catalog_size: int
    unique_recommended_items: int
    ndcg_at_k: float
    mrr_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    catalog_coverage_at_k: float

    @property
    def evaluated_fraction(self) -> float:
        """Return the fraction included in quality metrics."""

        return self.evaluated_impressions / self.total_impressions

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible representation."""

        return {
            "k": self.k,
            "total_impressions": self.total_impressions,
            "evaluated_impressions": self.evaluated_impressions,
            "skipped_no_click_impressions": (self.skipped_no_click_impressions),
            "empty_ranking_impressions": self.empty_ranking_impressions,
            "evaluated_fraction": self.evaluated_fraction,
            "catalog_size": self.catalog_size,
            "unique_recommended_items": self.unique_recommended_items,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr_at_k": self.mrr_at_k,
            "recall_at_k": self.recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "catalog_coverage_at_k": self.catalog_coverage_at_k,
        }


def evaluate_rankings(
    examples: Iterable[RankingExample],
    catalog_items: Iterable[str],
    *,
    k: int,
) -> RankingEvaluationResult:
    """Aggregate top-k ranking metrics across offline impressions.

    Impressions without relevant items are excluded from quality metrics
    and reported separately. Catalog coverage uses top-k recommendations
    from all impressions, including no-click impressions.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise RankingEvaluationError("k must be a positive integer.")

    example_list = list(examples)

    if not example_list:
        raise RankingEvaluationError("At least one ranking example is required.")

    catalog = frozenset(catalog_items)

    if not catalog:
        raise RankingEvaluationError("At least one catalog item is required.")

    impression_ids = [example.impression_id for example in example_list]

    if len(impression_ids) != len(set(impression_ids)):
        raise RankingEvaluationError("impression_id values must be unique.")

    relevant_items = set().union(*(example.relevant_items for example in example_list))
    unknown_relevant_items = relevant_items - catalog

    if unknown_relevant_items:
        unknown_preview = ", ".join(sorted(unknown_relevant_items)[:3])
        raise RankingEvaluationError(
            f"Relevant items are missing from the catalog: {unknown_preview}."
        )

    evaluated_examples = [example for example in example_list if example.relevant_items]

    if not evaluated_examples:
        raise RankingEvaluationError(
            "At least one ranking example with a relevant item is required."
        )

    top_k_rankings = [example.ranked_items[:k] for example in example_list]
    evaluated_rankings = [example.ranked_items for example in evaluated_examples]
    evaluated_relevance = [example.relevant_items for example in evaluated_examples]

    try:
        coverage = catalog_coverage(
            top_k_rankings,
            catalog,
        )

        ndcg = fmean(
            ndcg_at_k(
                example.ranked_items,
                example.relevant_items,
                k=k,
            )
            for example in evaluated_examples
        )

        mrr = mean_reciprocal_rank_at_k(
            evaluated_rankings,
            evaluated_relevance,
            k=k,
        )

        recall = fmean(
            recall_at_k(
                example.ranked_items,
                example.relevant_items,
                k=k,
            )
            for example in evaluated_examples
        )

        hit_rate = fmean(
            hit_rate_at_k(
                example.ranked_items,
                example.relevant_items,
                k=k,
            )
            for example in evaluated_examples
        )

    except RankingMetricError as error:
        raise RankingEvaluationError(str(error)) from error

    unique_recommended_items = len(set().union(*(set(ranking) for ranking in top_k_rankings)))

    return RankingEvaluationResult(
        k=k,
        total_impressions=len(example_list),
        evaluated_impressions=len(evaluated_examples),
        skipped_no_click_impressions=(len(example_list) - len(evaluated_examples)),
        empty_ranking_impressions=sum(not example.ranked_items for example in example_list),
        catalog_size=len(catalog),
        unique_recommended_items=unique_recommended_items,
        ndcg_at_k=ndcg,
        mrr_at_k=mrr,
        recall_at_k=recall,
        hit_rate_at_k=hit_rate,
        catalog_coverage_at_k=coverage,
    )
