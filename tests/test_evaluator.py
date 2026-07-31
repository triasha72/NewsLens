from __future__ import annotations

import json
from math import log2

import pytest

from newslens.evaluation import (
    RankingEvaluationError,
    RankingExample,
    evaluate_rankings,
)


def make_example(
    impression_id: str,
    ranking: list[str],
    relevant: set[str],
) -> RankingExample:
    return RankingExample(
        impression_id=impression_id,
        ranked_items=ranking,
        relevant_items=relevant,
    )


def test_evaluate_rankings_matches_hand_calculation() -> None:
    examples = [
        make_example(
            "I1",
            ["N1", "N2", "N3"],
            {"N2"},
        ),
        make_example(
            "I2",
            ["N4", "N5", "N6"],
            {"N4", "N6"},
        ),
    ]

    result = evaluate_rankings(
        examples,
        ["N1", "N2", "N3", "N4", "N5", "N6"],
        k=2,
    )

    expected_first_ndcg = 1.0 / log2(3)
    expected_second_ndcg = 1.0 / (1.0 + (1.0 / log2(3)))

    assert result.total_impressions == 2
    assert result.evaluated_impressions == 2
    assert result.skipped_no_click_impressions == 0
    assert result.empty_ranking_impressions == 0
    assert result.ndcg_at_k == pytest.approx((expected_first_ndcg + expected_second_ndcg) / 2.0)
    assert result.mrr_at_k == pytest.approx(0.75)
    assert result.recall_at_k == pytest.approx(0.75)
    assert result.hit_rate_at_k == pytest.approx(1.0)
    assert result.unique_recommended_items == 4
    assert result.catalog_coverage_at_k == pytest.approx(4.0 / 6.0)


def test_no_click_impressions_are_skipped_and_reported() -> None:
    examples = [
        make_example(
            "I1",
            ["N1", "N2"],
            {"N2"},
        ),
        make_example(
            "I2",
            ["N3", "N4"],
            set(),
        ),
    ]

    result = evaluate_rankings(
        examples,
        ["N1", "N2", "N3", "N4"],
        k=2,
    )

    assert result.total_impressions == 2
    assert result.evaluated_impressions == 1
    assert result.skipped_no_click_impressions == 1
    assert result.evaluated_fraction == pytest.approx(0.5)
    assert result.catalog_coverage_at_k == pytest.approx(1.0)


def test_empty_ranking_counts_as_zero_quality() -> None:
    examples = [
        make_example(
            "I1",
            [],
            {"N1"},
        ),
        make_example(
            "I2",
            ["N2"],
            {"N2"},
        ),
    ]

    result = evaluate_rankings(
        examples,
        ["N1", "N2"],
        k=1,
    )

    assert result.empty_ranking_impressions == 1
    assert result.hit_rate_at_k == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(0.5)
    assert result.mrr_at_k == pytest.approx(0.5)
    assert result.catalog_coverage_at_k == pytest.approx(0.5)


def test_result_is_json_serializable() -> None:
    result = evaluate_rankings(
        [
            make_example(
                "I1",
                ["N1"],
                {"N1"},
            )
        ],
        ["N1"],
        k=1,
    )

    serialized = json.dumps(result.to_dict())

    assert '"evaluated_impressions": 1' in serialized
    assert result.to_dict()["ndcg_at_k"] == pytest.approx(1.0)


def test_example_rejects_empty_impression_id() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="must not be empty",
    ):
        make_example(
            " ",
            ["N1"],
            {"N1"},
        )


def test_example_rejects_duplicate_ranked_items() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="must not contain duplicates",
    ):
        make_example(
            "I1",
            ["N1", "N1"],
            {"N1"},
        )


def test_evaluator_rejects_empty_examples() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="At least one ranking example",
    ):
        evaluate_rankings(
            [],
            ["N1"],
            k=1,
        )


def test_evaluator_requires_clicked_example() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="with a relevant item",
    ):
        evaluate_rankings(
            [
                make_example(
                    "I1",
                    ["N1"],
                    set(),
                )
            ],
            ["N1"],
            k=1,
        )


def test_evaluator_rejects_duplicate_impression_ids() -> None:
    examples = [
        make_example(
            "I1",
            ["N1"],
            {"N1"},
        ),
        make_example(
            "I1",
            ["N2"],
            {"N2"},
        ),
    ]

    with pytest.raises(
        RankingEvaluationError,
        match="must be unique",
    ):
        evaluate_rankings(
            examples,
            ["N1", "N2"],
            k=1,
        )


def test_evaluator_rejects_relevant_item_outside_catalog() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="missing from the catalog",
    ):
        evaluate_rankings(
            [
                make_example(
                    "I1",
                    ["N1"],
                    {"N2"},
                )
            ],
            ["N1"],
            k=1,
        )


def test_evaluator_rejects_ranked_item_outside_catalog() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="outside the catalog",
    ):
        evaluate_rankings(
            [
                make_example(
                    "I1",
                    ["N2"],
                    {"N1"},
                )
            ],
            ["N1"],
            k=1,
        )


def test_evaluator_rejects_empty_catalog() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="At least one catalog item",
    ):
        evaluate_rankings(
            [
                make_example(
                    "I1",
                    ["N1"],
                    {"N1"},
                )
            ],
            [],
            k=1,
        )


@pytest.mark.parametrize(
    "invalid_k",
    [0, -1, 1.5, True],
)
def test_evaluator_rejects_invalid_k(invalid_k) -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="positive integer",
    ):
        evaluate_rankings(
            [
                make_example(
                    "I1",
                    ["N1"],
                    {"N1"},
                )
            ],
            ["N1"],
            k=invalid_k,
        )
