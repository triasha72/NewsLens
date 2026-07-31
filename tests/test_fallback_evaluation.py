from __future__ import annotations

import json
from math import log2

import pandas as pd
import pytest

from newslens.evaluation import (
    FallbackEvaluationError,
    evaluate_fallback_baseline,
)


def make_news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "news_id": ["N1", "N2", "N3", "N4", "N5", "N6"],
            "category": [
                "science",
                "science",
                "sports",
                "sports",
                "finance",
                "food",
            ],
            "subcategory": [
                "space",
                "space",
                "football",
                "football",
                "markets",
                "cooking",
            ],
            "title": [
                "Mars mission discovers water",
                "Mars rover searches for water",
                "Football championship begins",
                "Football team wins championship",
                "Stock market earnings rise",
                "Cooking pasta recipe",
            ],
            "abstract": [
                "Spacecraft explores planet",
                "Spacecraft begins exploration",
                "Players prepare for match",
                "Coach celebrates victory",
                "Companies report results",
                "Chef prepares dinner",
            ],
        }
    )


def make_behaviors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": ["I1", "I2", "I3", "I4", "I5"],
            "user_id": ["U1", "U2", "U3", "U4", "U5"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                ]
            ),
            "history": ["", "N1", "N3", "N1", ""],
            "impressions": [
                "N1-1 N2-0 N3-0",
                "N2-1 N3-0 N4-0",
                "N4-1 N1-0 N2-0",
                "N2-1 N3-0 N5-0",
                "N4-0 N6-1",
            ],
        }
    )


def test_fallback_evaluation_matches_expected_routing_and_metrics() -> None:
    report = evaluate_fallback_baseline(
        make_news(),
        make_behaviors(),
        validation_fraction=0.40,
        k=2,
    )

    assert report.model_name == "tfidf_content_with_popularity_fallback"
    assert report.training_records == 3
    assert report.validation_records == 2
    assert report.cutoff_timestamp == "2020-01-04T00:00:00"
    assert report.actual_validation_fraction == pytest.approx(0.40)

    assert report.vocabulary_article_count == 4
    assert report.indexed_article_count == 6
    assert report.vocabulary_size > 0
    assert report.candidate_occurrences == 5

    assert report.content_routed_impressions == 1
    assert report.content_routed_fraction == pytest.approx(0.50)
    assert report.popularity_routed_impressions == 1
    assert report.popularity_routed_fraction == pytest.approx(0.50)
    assert report.empty_history_fallback_impressions == 1
    assert report.unknown_history_fallback_impressions == 0
    assert report.zero_profile_fallback_impressions == 0
    assert report.zero_signal_fallback_impressions == 0
    assert report.recovered_fallback_impressions == 1
    assert report.fallback_recovery_fraction == pytest.approx(1.0)

    expected_ndcg = (1.0 + (1.0 / log2(3))) / 2

    assert report.metrics.evaluated_impressions == 2
    assert report.metrics.empty_ranking_impressions == 0
    assert report.metrics.ndcg_at_k == pytest.approx(expected_ndcg)
    assert report.metrics.mrr_at_k == pytest.approx(0.75)
    assert report.metrics.recall_at_k == pytest.approx(1.0)
    assert report.metrics.hit_rate_at_k == pytest.approx(1.0)
    assert report.metrics.catalog_coverage_at_k == pytest.approx(4 / 6)


def test_unknown_history_routes_to_popularity() -> None:
    behaviors = make_behaviors()
    behaviors.loc[3, "history"] = "UNKNOWN"

    report = evaluate_fallback_baseline(
        make_news(),
        behaviors,
        validation_fraction=0.40,
        k=2,
    )

    assert report.popularity_routed_impressions == 2
    assert report.unknown_history_fallback_impressions == 1
    assert report.empty_history_fallback_impressions == 1


def test_zero_vocabulary_history_routes_to_popularity() -> None:
    behaviors = make_behaviors()
    behaviors.loc[3, "history"] = "N5"

    report = evaluate_fallback_baseline(
        make_news(),
        behaviors,
        validation_fraction=0.40,
        k=2,
    )

    assert report.popularity_routed_impressions == 2
    assert report.zero_profile_fallback_impressions == 1
    assert report.empty_history_fallback_impressions == 1


def test_zero_candidate_similarity_routes_to_popularity() -> None:
    behaviors = make_behaviors()
    behaviors.loc[3, "impressions"] = "N3-1 N5-0"

    report = evaluate_fallback_baseline(
        make_news(),
        behaviors,
        validation_fraction=0.40,
        k=2,
    )

    assert report.popularity_routed_impressions == 2
    assert report.zero_signal_fallback_impressions == 1
    assert report.empty_history_fallback_impressions == 1


def test_evaluation_does_not_modify_inputs() -> None:
    news = make_news()
    behaviors = make_behaviors()
    original_news = news.copy(deep=True)
    original_behaviors = behaviors.copy(deep=True)

    evaluate_fallback_baseline(
        news,
        behaviors,
        validation_fraction=0.40,
        k=2,
    )

    pd.testing.assert_frame_equal(news, original_news)
    pd.testing.assert_frame_equal(behaviors, original_behaviors)


def test_report_is_json_serializable() -> None:
    report = evaluate_fallback_baseline(
        make_news(),
        make_behaviors(),
        validation_fraction=0.40,
        k=2,
    )

    serialized = json.dumps(report.to_dict())

    assert '"model_name": "tfidf_content_with_popularity_fallback"' in serialized
    assert report.to_dict()["metrics"]["k"] == 2


def test_evaluation_requires_behavior_columns() -> None:
    behaviors = make_behaviors().drop(columns="history")

    with pytest.raises(
        FallbackEvaluationError,
        match="Missing required behavior columns",
    ):
        evaluate_fallback_baseline(
            make_news(),
            behaviors,
        )


def test_evaluation_rejects_duplicate_validation_candidates() -> None:
    behaviors = make_behaviors()
    behaviors.loc[3, "impressions"] = "N2-1 N2-0"

    with pytest.raises(
        FallbackEvaluationError,
        match="duplicate candidate IDs",
    ):
        evaluate_fallback_baseline(
            make_news(),
            behaviors,
            validation_fraction=0.40,
            k=2,
        )


def test_evaluation_rejects_unknown_validation_candidate() -> None:
    behaviors = make_behaviors()
    behaviors.loc[3, "impressions"] = "N2-0 NX-1"

    with pytest.raises(
        FallbackEvaluationError,
        match="missing from the catalog",
    ):
        evaluate_fallback_baseline(
            make_news(),
            behaviors,
            validation_fraction=0.40,
            k=2,
        )


@pytest.mark.parametrize(
    "invalid_k",
    [0, -1, 1.5, True],
)
def test_evaluation_rejects_invalid_k(invalid_k: object) -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="k must be a positive integer",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            k=invalid_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_max_features",
    [0, -1, 1.5, True],
)
def test_evaluation_rejects_invalid_max_features(
    invalid_max_features: object,
) -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="max_features must be a positive integer",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            max_features=invalid_max_features,  # type: ignore[arg-type]
        )
