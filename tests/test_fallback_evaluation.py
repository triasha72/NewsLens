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
    assert report.to_dict()["history_segments"]["k"] == 2
    assert report.to_dict()["category_analysis"]["k"] == 2
    assert report.to_dict()["exposure_analysis"]["k"] == 2
    assert report.to_dict()["uncertainty"]["k"] == 2
    assert report.to_dict()["failure_analysis"]["k"] == 2


def test_report_includes_source_specific_high_score_failure_analysis() -> None:
    report = evaluate_fallback_baseline(
        make_news(),
        make_behaviors(),
        validation_fraction=0.40,
        k=1,
        failure_score_quantile=0.50,
        maximum_failures_per_source=5,
    )
    analysis = report.failure_analysis
    thresholds = {threshold.source: threshold for threshold in analysis.source_thresholds}

    assert analysis.total_impressions == report.validation_records
    assert analysis.evaluated_impressions == report.metrics.evaluated_impressions
    assert analysis.score_quantile == pytest.approx(0.50)
    assert analysis.maximum_failures_per_source == 5
    assert analysis.top_k_misses == 1
    assert analysis.high_score_misses == 1

    assert thresholds["content"].eligible_impressions == 1
    assert thresholds["content"].top_k_misses == 0
    assert thresholds["popularity"].eligible_impressions == 1
    assert thresholds["popularity"].top_k_misses == 1
    assert thresholds["popularity"].high_score_misses == 1

    assert len(analysis.failures) == 1
    failure = analysis.failures[0]
    assert failure.impression_id == "I5"
    assert failure.source == "popularity"
    assert failure.history_length == 0
    assert failure.candidate_count == 2
    assert failure.relevant_items == ("N6",)
    assert failure.ranked_items == ("N4",)
    assert failure.ranked_scores == pytest.approx((1.0,))
    assert failure.score_margin is None
    assert failure.score_margin_ratio is None
    assert failure.margin_classification == "single_result"
    assert failure.relevant_articles[0].to_dict() == {
        "news_id": "N6",
        "title": "Cooking pasta recipe",
        "category": "food",
        "subcategory": "cooking",
    }
    assert failure.ranked_articles[0].to_dict() == {
        "news_id": "N4",
        "title": "Football team wins championship",
        "category": "sports",
        "subcategory": "football",
    }


def test_report_includes_deterministic_bootstrap_uncertainty() -> None:
    first = evaluate_fallback_baseline(
        make_news(),
        make_behaviors(),
        validation_fraction=0.40,
        k=2,
        bootstrap_samples=200,
        bootstrap_confidence_level=0.90,
        bootstrap_random_seed=2026,
    )
    second = evaluate_fallback_baseline(
        make_news(),
        make_behaviors(),
        validation_fraction=0.40,
        k=2,
        bootstrap_samples=200,
        bootstrap_confidence_level=0.90,
        bootstrap_random_seed=2026,
    )

    assert first.uncertainty == second.uncertainty
    assert first.uncertainty.bootstrap_samples == 200
    assert first.uncertainty.confidence_level == pytest.approx(0.90)
    assert first.uncertainty.random_seed == 2026
    assert first.uncertainty.total_impressions == first.validation_records
    assert first.uncertainty.evaluated_impressions == first.metrics.evaluated_impressions
    assert first.uncertainty.ndcg_at_k.point_estimate == first.metrics.ndcg_at_k
    assert first.uncertainty.mrr_at_k.point_estimate == first.metrics.mrr_at_k
    assert first.uncertainty.recall_at_k.point_estimate == first.metrics.recall_at_k
    assert first.uncertainty.hit_rate_at_k.point_estimate == first.metrics.hit_rate_at_k


def test_report_includes_exhaustive_history_segments() -> None:
    report = evaluate_fallback_baseline(
        make_news(),
        make_behaviors(),
        validation_fraction=0.40,
        k=2,
    )

    segments = {segment.definition.name: segment for segment in report.history_segments.segments}

    assert report.history_segments.overall_metrics == report.metrics
    assert (
        sum(segment.total_impressions for segment in report.history_segments.segments)
        == report.validation_records
    )
    assert segments["cold_start"].total_impressions == 1
    assert segments["cold_start"].metrics is not None
    assert segments["short_history"].total_impressions == 1
    assert segments["short_history"].metrics is not None
    assert segments["medium_history"].total_impressions == 0
    assert segments["medium_history"].metrics is None
    assert segments["long_history"].total_impressions == 0
    assert segments["long_history"].metrics is None


def test_report_includes_overlapping_clicked_category_analysis() -> None:
    report = evaluate_fallback_baseline(
        make_news(),
        make_behaviors(),
        validation_fraction=0.40,
        k=2,
        minimum_category_impressions=1,
    )
    categories = {category.name: category for category in report.category_analysis.categories}

    assert report.category_analysis.overall_metrics == report.metrics
    assert report.category_analysis.clicked_impressions == report.validation_records
    assert report.category_analysis.multi_category_clicked_impressions == 0
    assert report.category_analysis.impression_category_pairs == 2
    assert not report.category_analysis.category_membership_is_overlapping

    assert categories["science"].relevant_impressions == 1
    assert categories["science"].meets_minimum_support
    assert categories["science"].metrics is not None
    assert categories["science"].metrics.ndcg_at_k == pytest.approx(1.0)

    assert categories["food"].relevant_impressions == 1
    assert categories["food"].meets_minimum_support
    assert categories["food"].metrics is not None
    assert categories["food"].metrics.ndcg_at_k == pytest.approx(1.0 / log2(3))

    assert categories["finance"].relevant_impressions == 0
    assert categories["finance"].metrics is None
    assert categories["sports"].relevant_impressions == 0
    assert categories["sports"].metrics is None


def test_report_includes_training_exposure_analysis() -> None:
    report = evaluate_fallback_baseline(
        make_news(),
        make_behaviors(),
        validation_fraction=0.40,
        k=2,
        minimum_exposure_impressions=1,
    )
    bands = {band.definition.name: band for band in report.exposure_analysis.bands}

    assert report.exposure_analysis.overall_metrics == report.metrics
    assert report.exposure_analysis.clicked_impressions == report.validation_records
    assert report.exposure_analysis.multi_band_clicked_impressions == 0
    assert report.exposure_analysis.impression_band_pairs == 2
    assert not report.exposure_analysis.band_membership_is_overlapping

    unseen = bands["unseen"]
    assert unseen.catalog_articles == 2
    assert unseen.relevant_impressions == 1
    assert unseen.meets_minimum_support
    assert unseen.metrics is not None
    assert unseen.metrics.ndcg_at_k == pytest.approx(1.0 / log2(3))
    assert unseen.metrics.mrr_at_k == pytest.approx(0.5)
    assert unseen.catalog_coverage_at_k == pytest.approx(0.5)

    low = bands["low_exposure"]
    assert low.catalog_articles == 4
    assert low.relevant_impressions == 1
    assert low.meets_minimum_support
    assert low.metrics is not None
    assert low.metrics.ndcg_at_k == pytest.approx(1.0)
    assert low.metrics.mrr_at_k == pytest.approx(1.0)
    assert low.catalog_coverage_at_k == pytest.approx(0.75)

    assert bands["medium_exposure"].catalog_articles == 0
    assert bands["medium_exposure"].metrics is None
    assert bands["high_exposure"].catalog_articles == 0
    assert bands["high_exposure"].metrics is None


def test_evaluation_requires_article_categories() -> None:
    news = make_news().drop(columns="category")

    with pytest.raises(
        FallbackEvaluationError,
        match="Missing required article columns: category",
    ):
        evaluate_fallback_baseline(news, make_behaviors())


def test_evaluation_rejects_empty_article_categories() -> None:
    news = make_news()
    news.loc[0, "category"] = " "

    with pytest.raises(
        FallbackEvaluationError,
        match="Article categories cannot be empty",
    ):
        evaluate_fallback_baseline(news, make_behaviors())


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


@pytest.mark.parametrize(
    "invalid_minimum",
    [0, -1, 1.5, True],
)
def test_evaluation_rejects_invalid_minimum_category_impressions(
    invalid_minimum: object,
) -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="minimum_category_impressions must be a positive integer",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            minimum_category_impressions=invalid_minimum,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_minimum",
    [0, -1, 1.5, True],
)
def test_evaluation_rejects_invalid_minimum_exposure_impressions(
    invalid_minimum: object,
) -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="minimum_exposure_impressions must be a positive integer",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            minimum_exposure_impressions=invalid_minimum,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_samples",
    [0, -1, 1.5, True],
)
def test_evaluation_rejects_invalid_bootstrap_samples(
    invalid_samples: object,
) -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="bootstrap_samples must be a positive integer",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            bootstrap_samples=invalid_samples,  # type: ignore[arg-type]
        )


def test_evaluation_requires_at_least_two_bootstrap_samples() -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="bootstrap_samples must be at least 2",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            bootstrap_samples=1,
        )


@pytest.mark.parametrize(
    "invalid_level",
    [0.0, 1.0, -0.1, 1.1, True, "0.95"],
)
def test_evaluation_rejects_invalid_bootstrap_confidence_level(
    invalid_level: object,
) -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="bootstrap_confidence_level must be between 0 and 1",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            bootstrap_confidence_level=invalid_level,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_seed",
    [-1, 1.5, True],
)
def test_evaluation_rejects_invalid_bootstrap_random_seed(
    invalid_seed: object,
) -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="bootstrap_random_seed must be a non-negative integer",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            bootstrap_random_seed=invalid_seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_quantile",
    [0.0, 1.0, -0.1, 1.1, True, "0.9"],
)
def test_evaluation_rejects_invalid_failure_score_quantile(
    invalid_quantile: object,
) -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="failure_score_quantile must be between 0 and 1",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            failure_score_quantile=invalid_quantile,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_maximum",
    [0, -1, 1.5, True],
)
def test_evaluation_rejects_invalid_maximum_failures_per_source(
    invalid_maximum: object,
) -> None:
    with pytest.raises(
        FallbackEvaluationError,
        match="maximum_failures_per_source must be a positive integer",
    ):
        evaluate_fallback_baseline(
            make_news(),
            make_behaviors(),
            maximum_failures_per_source=invalid_maximum,  # type: ignore[arg-type]
        )
