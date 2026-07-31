from __future__ import annotations

import json
from math import log2

import pytest

from newslens.evaluation import (
    CategoryEvaluationError,
    RankingExample,
    evaluate_article_categories,
)


def make_example(
    impression_id: str,
    ranking: list[str],
    relevant: set[str],
) -> RankingExample:
    return RankingExample(
        impression_id=impression_id,
        ranked_items=tuple(ranking),
        relevant_items=frozenset(relevant),
    )


def make_categories() -> dict[str, str]:
    return {
        "A1": "science",
        "A2": "science",
        "B1": "sports",
        "B2": "sports",
        "C1": "finance",
    }


def make_examples() -> list[RankingExample]:
    return [
        make_example("I1", ["A1", "B1", "A2"], {"A1"}),
        make_example("I2", ["B1", "A2", "B2"], {"B2"}),
        make_example("I3", ["A2", "B1", "C1"], {"A2", "B1"}),
    ]


def test_category_metrics_preserve_global_ranking_positions() -> None:
    report = evaluate_article_categories(
        make_examples(),
        make_categories(),
        k=2,
        minimum_relevant_impressions=2,
    )
    categories = {category.name: category for category in report.categories}

    assert report.total_impressions == 3
    assert report.clicked_impressions == 3
    assert report.multi_category_clicked_impressions == 1
    assert report.impression_category_pairs == 4
    assert report.category_membership_is_overlapping

    science = categories["science"]
    assert science.relevant_impressions == 2
    assert science.relevant_impression_fraction == pytest.approx(2 / 3)
    assert science.relevant_item_occurrences == 2
    assert science.unique_relevant_items == 2
    assert science.metrics is not None
    assert science.metrics.ndcg_at_k == pytest.approx(1.0)
    assert science.metrics.mrr_at_k == pytest.approx(1.0)
    assert science.metrics.recall_at_k == pytest.approx(1.0)
    assert science.metrics.hit_rate_at_k == pytest.approx(1.0)

    sports = categories["sports"]
    assert sports.relevant_impressions == 2
    assert sports.metrics is not None
    assert sports.metrics.ndcg_at_k == pytest.approx(1 / (2 * log2(3)))
    assert sports.metrics.mrr_at_k == pytest.approx(0.25)
    assert sports.metrics.recall_at_k == pytest.approx(0.5)
    assert sports.metrics.hit_rate_at_k == pytest.approx(0.5)


def test_category_exposure_uses_category_catalog_denominators() -> None:
    report = evaluate_article_categories(
        make_examples(),
        make_categories(),
        k=2,
        minimum_relevant_impressions=2,
    )
    categories = {category.name: category for category in report.categories}

    assert categories["science"].recommended_occurrences_at_k == 3
    assert categories["science"].unique_recommended_items_at_k == 2
    assert categories["science"].catalog_coverage_at_k == pytest.approx(1.0)

    assert categories["sports"].recommended_occurrences_at_k == 3
    assert categories["sports"].unique_recommended_items_at_k == 1
    assert categories["sports"].catalog_coverage_at_k == pytest.approx(0.5)

    assert categories["finance"].recommended_occurrences_at_k == 0
    assert categories["finance"].unique_recommended_items_at_k == 0
    assert categories["finance"].catalog_coverage_at_k == pytest.approx(0.0)


def test_categories_without_clicks_have_no_quality_metrics() -> None:
    report = evaluate_article_categories(
        make_examples(),
        make_categories(),
        k=2,
        minimum_relevant_impressions=2,
    )
    finance = next(category for category in report.categories if category.name == "finance")

    assert finance.catalog_articles == 1
    assert finance.relevant_impressions == 0
    assert finance.relevant_impression_fraction == pytest.approx(0.0)
    assert not finance.meets_minimum_support
    assert finance.metrics is None


def test_minimum_support_is_reported_without_dropping_categories() -> None:
    report = evaluate_article_categories(
        make_examples(),
        make_categories(),
        k=2,
        minimum_relevant_impressions=3,
    )

    assert [category.name for category in report.categories] == [
        "finance",
        "science",
        "sports",
    ]
    assert all(not category.meets_minimum_support for category in report.categories)
    assert report.categories[1].metrics is not None


def test_report_is_json_serializable() -> None:
    report = evaluate_article_categories(
        make_examples(),
        make_categories(),
        k=2,
    )
    serialized = json.dumps(report.to_dict(), sort_keys=True)

    assert '"category_membership_is_overlapping": true' in serialized
    assert '"name": "science"' in serialized
    assert report.to_dict()["overall_metrics"]["k"] == 2


def test_category_names_and_item_ids_are_trimmed() -> None:
    report = evaluate_article_categories(
        [make_example("I1", ["A1"], {"A1"})],
        {" A1 ": " science "},
        k=1,
    )

    assert report.categories[0].name == "science"


@pytest.mark.parametrize("invalid_minimum", [0, -1, 1.5, True])
def test_evaluation_rejects_invalid_minimum_support(invalid_minimum: object) -> None:
    with pytest.raises(
        CategoryEvaluationError,
        match="minimum_relevant_impressions must be a positive integer",
    ):
        evaluate_article_categories(
            make_examples(),
            make_categories(),
            k=2,
            minimum_relevant_impressions=invalid_minimum,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("categories", "message"),
    [
        ({}, "At least one categorized catalog item"),
        ({"": "science"}, "non-empty string item IDs"),
        ({"A1": ""}, "non-empty string category names"),
    ],
)
def test_evaluation_rejects_invalid_category_mappings(
    categories: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(CategoryEvaluationError, match=message):
        evaluate_article_categories(
            [make_example("I1", ["A1"], {"A1"})],
            categories,
            k=1,
        )


def test_evaluation_rejects_uncategorized_references() -> None:
    with pytest.raises(
        CategoryEvaluationError,
        match="items without categories: B1",
    ):
        evaluate_article_categories(
            [make_example("I1", ["A1", "B1"], {"A1"})],
            {"A1": "science"},
            k=1,
        )


def test_evaluation_rejects_empty_examples() -> None:
    with pytest.raises(
        CategoryEvaluationError,
        match="At least one ranking example",
    ):
        evaluate_article_categories([], make_categories(), k=2)


def test_evaluation_rejects_non_ranking_examples() -> None:
    with pytest.raises(
        CategoryEvaluationError,
        match="only RankingExample values",
    ):
        evaluate_article_categories(
            ["not an example"],  # type: ignore[list-item]
            make_categories(),
            k=1,
        )


def test_evaluation_wraps_ranking_errors() -> None:
    with pytest.raises(
        CategoryEvaluationError,
        match="relevant item is required",
    ):
        evaluate_article_categories(
            [make_example("I1", ["A1"], set())],
            make_categories(),
            k=1,
        )
