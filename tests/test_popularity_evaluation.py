from __future__ import annotations

import json
from math import log2

import pandas as pd
import pytest

from newslens.evaluation import (
    PopularityEvaluationError,
    evaluate_popularity_baseline,
)


def make_behaviors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": [
                "I1",
                "I2",
                "I3",
                "I4",
                "I5",
            ],
            "user_id": [
                "U1",
                "U2",
                "U3",
                "U4",
                "U5",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                ]
            ),
            "history": [""] * 5,
            "impressions": [
                "N1-1 N2-0 N3-0",
                "N1-1 N2-0 N3-0",
                "N2-1 N3-0",
                "N2-0 N3-1 N4-0",
                "N1-0 N4-1",
            ],
        }
    )


def test_popularity_evaluation_matches_expected_metrics() -> None:
    report = evaluate_popularity_baseline(
        make_behaviors(),
        ["N1", "N2", "N3", "N4"],
        validation_fraction=0.40,
        k=2,
    )

    expected_ndcg = 1.0 / log2(3)

    assert report.model_name == ("training_click_count_popularity")
    assert report.training_records == 3
    assert report.validation_records == 2
    assert report.cutoff_timestamp == ("2020-01-04T00:00:00")
    assert report.actual_validation_fraction == pytest.approx(0.40)

    assert report.candidate_occurrences == 5
    assert report.unseen_candidate_occurrences == 2
    assert report.unseen_candidate_fraction == pytest.approx(0.40)
    assert report.unseen_validation_impressions == 2

    assert report.metrics.evaluated_impressions == 2
    assert report.metrics.ndcg_at_k == pytest.approx(expected_ndcg)
    assert report.metrics.mrr_at_k == pytest.approx(0.50)
    assert report.metrics.recall_at_k == pytest.approx(1.0)
    assert report.metrics.hit_rate_at_k == pytest.approx(1.0)
    assert report.metrics.catalog_coverage_at_k == (pytest.approx(1.0))


def test_evaluation_does_not_modify_input_data() -> None:
    behaviors = make_behaviors()
    original = behaviors.copy(deep=True)

    evaluate_popularity_baseline(
        behaviors,
        ["N1", "N2", "N3", "N4"],
        validation_fraction=0.40,
        k=2,
    )

    pd.testing.assert_frame_equal(
        behaviors,
        original,
    )


def test_report_is_json_serializable() -> None:
    report = evaluate_popularity_baseline(
        make_behaviors(),
        ["N1", "N2", "N3", "N4"],
        validation_fraction=0.40,
        k=2,
    )

    serialized = json.dumps(report.to_dict())

    assert ('"model_name": "training_click_count_popularity"') in serialized
    assert report.to_dict()["metrics"]["k"] == 2


def test_evaluation_requires_behavior_columns() -> None:
    behaviors = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"])})

    with pytest.raises(
        PopularityEvaluationError,
        match="Missing required behavior columns",
    ):
        evaluate_popularity_baseline(
            behaviors,
            ["N1"],
        )


def test_evaluation_rejects_duplicate_candidates() -> None:
    behaviors = pd.DataFrame(
        {
            "impression_id": ["I1", "I2"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "impressions": [
                "N1-1 N2-0",
                "N1-1 N1-0",
            ],
        }
    )

    with pytest.raises(
        PopularityEvaluationError,
        match="duplicate candidate IDs",
    ):
        evaluate_popularity_baseline(
            behaviors,
            ["N1", "N2"],
            validation_fraction=0.50,
            k=2,
        )


def test_evaluation_rejects_unknown_catalog_candidate() -> None:
    behaviors = pd.DataFrame(
        {
            "impression_id": ["I1", "I2"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "impressions": [
                "N1-1 N2-0",
                "N1-0 NX-1",
            ],
        }
    )

    with pytest.raises(
        PopularityEvaluationError,
        match="missing from the catalog",
    ):
        evaluate_popularity_baseline(
            behaviors,
            ["N1", "N2"],
            validation_fraction=0.50,
            k=2,
        )


def test_evaluation_rejects_empty_catalog() -> None:
    with pytest.raises(
        PopularityEvaluationError,
        match="At least one catalog item",
    ):
        evaluate_popularity_baseline(
            make_behaviors(),
            [],
            validation_fraction=0.40,
            k=2,
        )


def test_evaluation_rejects_catalog_string() -> None:
    with pytest.raises(
        PopularityEvaluationError,
        match="must be an iterable",
    ):
        evaluate_popularity_baseline(
            make_behaviors(),
            "N1 N2 N3 N4",
            validation_fraction=0.40,
            k=2,
        )


@pytest.mark.parametrize(
    "invalid_k",
    [0, -1, 1.5, True],
)
def test_evaluation_rejects_invalid_k(invalid_k) -> None:
    with pytest.raises(
        PopularityEvaluationError,
        match="positive integer",
    ):
        evaluate_popularity_baseline(
            make_behaviors(),
            ["N1", "N2", "N3", "N4"],
            validation_fraction=0.40,
            k=invalid_k,
        )
