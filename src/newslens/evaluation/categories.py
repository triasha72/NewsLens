"""Model-independent ranking evaluation by clicked article category."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .evaluator import (
    RankingEvaluationError,
    RankingEvaluationResult,
    RankingExample,
    evaluate_rankings,
)


class CategoryEvaluationError(ValueError):
    """Raised when article-category evaluation cannot be completed."""


@dataclass(frozen=True)
class CategoryResult:
    """Relevance and exposure results for one article category."""

    name: str
    catalog_articles: int
    relevant_impressions: int
    relevant_impression_fraction: float
    relevant_item_occurrences: int
    unique_relevant_items: int
    recommended_occurrences_at_k: int
    unique_recommended_items_at_k: int
    catalog_coverage_at_k: float
    meets_minimum_support: bool
    metrics: RankingEvaluationResult | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible category result."""

        return {
            "name": self.name,
            "catalog_articles": self.catalog_articles,
            "relevant_impressions": self.relevant_impressions,
            "relevant_impression_fraction": self.relevant_impression_fraction,
            "relevant_item_occurrences": self.relevant_item_occurrences,
            "unique_relevant_items": self.unique_relevant_items,
            "recommended_occurrences_at_k": self.recommended_occurrences_at_k,
            "unique_recommended_items_at_k": self.unique_recommended_items_at_k,
            "catalog_coverage_at_k": self.catalog_coverage_at_k,
            "meets_minimum_support": self.meets_minimum_support,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class CategoryEvaluationReport:
    """Overall metrics and overlapping clicked-category cohort results."""

    overall_metrics: RankingEvaluationResult
    minimum_relevant_impressions: int
    clicked_impressions: int
    multi_category_clicked_impressions: int
    impression_category_pairs: int
    categories: tuple[CategoryResult, ...]

    @property
    def k(self) -> int:
        """Return the ranking cutoff used throughout the report."""

        return self.overall_metrics.k

    @property
    def total_impressions(self) -> int:
        """Return the number of input impressions."""

        return self.overall_metrics.total_impressions

    @property
    def category_membership_is_overlapping(self) -> bool:
        """Return whether at least one impression belongs to multiple cohorts."""

        return self.multi_category_clicked_impressions > 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible evaluation report."""

        return {
            "k": self.k,
            "total_impressions": self.total_impressions,
            "clicked_impressions": self.clicked_impressions,
            "minimum_relevant_impressions": self.minimum_relevant_impressions,
            "multi_category_clicked_impressions": (self.multi_category_clicked_impressions),
            "impression_category_pairs": self.impression_category_pairs,
            "category_membership_is_overlapping": (self.category_membership_is_overlapping),
            "overall_metrics": self.overall_metrics.to_dict(),
            "categories": [category.to_dict() for category in self.categories],
        }


def _normalize_item_categories(
    item_categories: Mapping[str, str],
) -> dict[str, str]:
    if not item_categories:
        raise CategoryEvaluationError("At least one categorized catalog item is required.")

    normalized: dict[str, str] = {}

    for raw_item_id, raw_category in item_categories.items():
        if not isinstance(raw_item_id, str) or not raw_item_id.strip():
            raise CategoryEvaluationError("Category mappings must use non-empty string item IDs.")

        if not isinstance(raw_category, str) or not raw_category.strip():
            raise CategoryEvaluationError(
                "Category mappings must use non-empty string category names."
            )

        item_id = raw_item_id.strip()
        category = raw_category.strip()

        if item_id in normalized:
            raise CategoryEvaluationError(
                f"Category mapping contains duplicate item ID after normalization: {item_id}."
            )

        normalized[item_id] = category

    return normalized


def _validate_minimum_support(minimum_relevant_impressions: int) -> None:
    if (
        isinstance(minimum_relevant_impressions, bool)
        or not isinstance(minimum_relevant_impressions, int)
        or minimum_relevant_impressions <= 0
    ):
        raise CategoryEvaluationError("minimum_relevant_impressions must be a positive integer.")


def evaluate_article_categories(
    examples: Iterable[RankingExample],
    item_categories: Mapping[str, str],
    *,
    k: int,
    minimum_relevant_impressions: int = 1,
) -> CategoryEvaluationReport:
    """Evaluate original rankings within overlapping clicked-category cohorts.

    An impression belongs to every category represented by its relevant items.
    Category-specific relevance metrics preserve the original global ranking
    positions. Exposure accounting uses top-k recommendations from all input
    impressions and is normalized by each category's catalog size.
    """

    _validate_minimum_support(minimum_relevant_impressions)
    example_list = list(examples)

    if not example_list:
        raise CategoryEvaluationError("At least one ranking example is required.")

    if any(not isinstance(example, RankingExample) for example in example_list):
        raise CategoryEvaluationError("examples must contain only RankingExample values.")

    normalized_categories = _normalize_item_categories(item_categories)
    catalog = frozenset(normalized_categories)
    referenced_items = {
        item
        for example in example_list
        for item in (*example.ranked_items, *example.relevant_items)
    }
    uncategorized_items = referenced_items - catalog

    if uncategorized_items:
        preview = ", ".join(sorted(uncategorized_items)[:3])
        raise CategoryEvaluationError(
            f"Ranking examples reference items without categories: {preview}."
        )

    try:
        overall_metrics = evaluate_rankings(
            example_list,
            catalog,
            k=k,
        )
    except RankingEvaluationError as error:
        raise CategoryEvaluationError(str(error)) from error

    catalog_by_category: dict[str, set[str]] = {}

    for item_id, category in normalized_categories.items():
        catalog_by_category.setdefault(category, set()).add(item_id)

    relevant_categories_by_impression: list[set[str]] = []

    for example in example_list:
        relevant_categories_by_impression.append(
            {normalized_categories[item_id] for item_id in example.relevant_items}
        )

    clicked_impressions = sum(bool(categories) for categories in relevant_categories_by_impression)
    multi_category_clicked_impressions = sum(
        len(categories) > 1 for categories in relevant_categories_by_impression
    )
    impression_category_pairs = sum(
        len(categories) for categories in relevant_categories_by_impression
    )
    results: list[CategoryResult] = []

    for category in sorted(catalog_by_category):
        category_catalog = catalog_by_category[category]
        category_examples: list[RankingExample] = []
        relevant_items: list[str] = []

        for example in example_list:
            category_relevance = frozenset(
                item_id
                for item_id in example.relevant_items
                if normalized_categories[item_id] == category
            )

            if not category_relevance:
                continue

            relevant_items.extend(category_relevance)
            category_examples.append(
                RankingExample(
                    impression_id=example.impression_id,
                    ranked_items=example.ranked_items,
                    relevant_items=category_relevance,
                )
            )

        metrics: RankingEvaluationResult | None = None

        if category_examples:
            try:
                metrics = evaluate_rankings(
                    category_examples,
                    catalog,
                    k=k,
                )
            except RankingEvaluationError as error:
                raise CategoryEvaluationError(str(error)) from error

        recommended_items = [
            item_id
            for example in example_list
            for item_id in example.ranked_items[:k]
            if normalized_categories[item_id] == category
        ]
        unique_recommended_items = set(recommended_items)

        results.append(
            CategoryResult(
                name=category,
                catalog_articles=len(category_catalog),
                relevant_impressions=len(category_examples),
                relevant_impression_fraction=(len(category_examples) / clicked_impressions),
                relevant_item_occurrences=len(relevant_items),
                unique_relevant_items=len(set(relevant_items)),
                recommended_occurrences_at_k=len(recommended_items),
                unique_recommended_items_at_k=len(unique_recommended_items),
                catalog_coverage_at_k=(len(unique_recommended_items) / len(category_catalog)),
                meets_minimum_support=(len(category_examples) >= minimum_relevant_impressions),
                metrics=metrics,
            )
        )

    return CategoryEvaluationReport(
        overall_metrics=overall_metrics,
        minimum_relevant_impressions=minimum_relevant_impressions,
        clicked_impressions=clicked_impressions,
        multi_category_clicked_impressions=multi_category_clicked_impressions,
        impression_category_pairs=impression_category_pairs,
        categories=tuple(results),
    )
