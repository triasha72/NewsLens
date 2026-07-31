from __future__ import annotations

import json
from math import log2

import pytest

from newslens.evaluation import (
    DEFAULT_TRAINING_EXPOSURE_BANDS,
    ExposureEvaluationError,
    RankingExample,
    TrainingExposureBand,
    evaluate_training_exposure_bands,
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


def make_catalog() -> list[str]:
    return ["U1", "L1", "L2", "M1", "H1"]


def make_exposures() -> dict[str, int]:
    return {
        "L1": 3,
        "L2": 9,
        "M1": 25,
        "H1": 150,
    }


def make_examples() -> list[RankingExample]:
    return [
        make_example("I1", ["L1", "U1", "H1"], {"U1"}),
        make_example("I2", ["M1", "L2", "U1"], {"L2", "M1"}),
        make_example("I3", ["H1", "M1", "L1"], {"H1"}),
    ]


def test_default_exposure_bands_use_log_scale_boundaries() -> None:
    unseen, low, medium, high = DEFAULT_TRAINING_EXPOSURE_BANDS

    assert unseen.contains(0)
    assert not unseen.contains(1)
    assert low.contains(1)
    assert low.contains(9)
    assert medium.contains(10)
    assert medium.contains(99)
    assert high.contains(100)
    assert high.contains(10_000)


def test_band_metrics_preserve_global_positions_and_overlap() -> None:
    report = evaluate_training_exposure_bands(
        make_examples(),
        make_catalog(),
        make_exposures(),
        k=2,
    )
    bands = {band.definition.name: band for band in report.bands}

    assert report.total_impressions == 3
    assert report.clicked_impressions == 3
    assert report.multi_band_clicked_impressions == 1
    assert report.impression_band_pairs == 4
    assert report.band_membership_is_overlapping

    unseen = bands["unseen"]
    assert unseen.relevant_impressions == 1
    assert unseen.relevant_impression_fraction == pytest.approx(1 / 3)
    assert unseen.metrics is not None
    assert unseen.metrics.ndcg_at_k == pytest.approx(1 / log2(3))
    assert unseen.metrics.mrr_at_k == pytest.approx(0.5)
    assert unseen.metrics.recall_at_k == pytest.approx(1.0)
    assert unseen.metrics.hit_rate_at_k == pytest.approx(1.0)

    low = bands["low_exposure"]
    assert low.metrics is not None
    assert low.metrics.ndcg_at_k == pytest.approx(1 / log2(3))
    assert low.metrics.mrr_at_k == pytest.approx(0.5)

    medium = bands["medium_exposure"]
    assert medium.metrics is not None
    assert medium.metrics.ndcg_at_k == pytest.approx(1.0)
    assert medium.metrics.mrr_at_k == pytest.approx(1.0)


def test_overall_metrics_match_the_shared_evaluator() -> None:
    report = evaluate_training_exposure_bands(
        make_examples(),
        make_catalog(),
        make_exposures(),
        k=2,
    )

    assert report.overall_metrics.total_impressions == 3
    assert report.overall_metrics.evaluated_impressions == 3
    assert report.overall_metrics.empty_ranking_impressions == 0


def test_band_coverage_uses_each_band_catalog_denominator() -> None:
    report = evaluate_training_exposure_bands(
        make_examples(),
        make_catalog(),
        make_exposures(),
        k=2,
    )
    bands = {band.definition.name: band for band in report.bands}

    assert bands["unseen"].catalog_articles == 1
    assert bands["unseen"].recommended_occurrences_at_k == 1
    assert bands["unseen"].catalog_coverage_at_k == pytest.approx(1.0)

    assert bands["low_exposure"].catalog_articles == 2
    assert bands["low_exposure"].recommended_occurrences_at_k == 2
    assert bands["low_exposure"].unique_recommended_items_at_k == 2
    assert bands["low_exposure"].catalog_coverage_at_k == pytest.approx(1.0)

    assert bands["medium_exposure"].recommended_occurrences_at_k == 2
    assert bands["medium_exposure"].unique_recommended_items_at_k == 1
    assert bands["high_exposure"].recommended_occurrences_at_k == 1


def test_items_missing_from_training_mapping_are_unseen() -> None:
    report = evaluate_training_exposure_bands(
        [make_example("I1", ["U1"], {"U1"})],
        ["U1", "L1"],
        {"L1": 2},
        k=1,
    )
    unseen = report.bands[0]

    assert unseen.definition.name == "unseen"
    assert unseen.catalog_articles == 1
    assert unseen.relevant_impressions == 1


def test_bands_without_relevant_items_have_no_quality_metrics() -> None:
    report = evaluate_training_exposure_bands(
        [make_example("I1", ["U1"], {"U1"})],
        make_catalog(),
        make_exposures(),
        k=1,
    )
    bands = {band.definition.name: band for band in report.bands}

    assert bands["unseen"].metrics is not None
    assert bands["low_exposure"].metrics is None
    assert bands["medium_exposure"].metrics is None
    assert bands["high_exposure"].metrics is None


def test_minimum_support_is_reported_without_dropping_bands() -> None:
    report = evaluate_training_exposure_bands(
        make_examples(),
        make_catalog(),
        make_exposures(),
        k=2,
        minimum_relevant_impressions=2,
    )

    assert len(report.bands) == 4
    assert all(not band.meets_minimum_support for band in report.bands)
    assert all(band.metrics is not None for band in report.bands)


def test_report_is_json_serializable_and_deterministic() -> None:
    report = evaluate_training_exposure_bands(
        make_examples(),
        make_catalog(),
        make_exposures(),
        k=2,
    )
    first = json.dumps(report.to_dict(), sort_keys=True)
    second = json.dumps(report.to_dict(), sort_keys=True)

    assert first == second
    assert '"band_membership_is_overlapping": true' in first
    assert '"name": "unseen"' in first


@pytest.mark.parametrize("invalid_minimum", [0, -1, 1.5, True])
def test_evaluation_rejects_invalid_minimum_support(invalid_minimum: object) -> None:
    with pytest.raises(
        ExposureEvaluationError,
        match="minimum_relevant_impressions must be a positive integer",
    ):
        evaluate_training_exposure_bands(
            make_examples(),
            make_catalog(),
            make_exposures(),
            k=2,
            minimum_relevant_impressions=invalid_minimum,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_count", [-1, 1.5, True])
def test_evaluation_rejects_invalid_training_exposure_counts(
    invalid_count: object,
) -> None:
    with pytest.raises(
        ExposureEvaluationError,
        match="Training-exposure counts must be non-negative integers",
    ):
        evaluate_training_exposure_bands(
            make_examples(),
            make_catalog(),
            {"L1": invalid_count},  # type: ignore[dict-item]
            k=2,
        )


def test_evaluation_rejects_exposure_items_outside_catalog() -> None:
    with pytest.raises(
        ExposureEvaluationError,
        match="Training-exposure item is missing from the catalog: X1",
    ):
        evaluate_training_exposure_bands(
            make_examples(),
            make_catalog(),
            {**make_exposures(), "X1": 4},
            k=2,
        )


def test_evaluation_rejects_ranked_items_outside_catalog() -> None:
    with pytest.raises(
        ExposureEvaluationError,
        match="Ranking examples reference items missing from the catalog: X1",
    ):
        evaluate_training_exposure_bands(
            [make_example("I1", ["X1"], {"X1"})],
            make_catalog(),
            make_exposures(),
            k=1,
        )


def test_evaluation_requires_examples() -> None:
    with pytest.raises(
        ExposureEvaluationError,
        match="At least one ranking example is required",
    ):
        evaluate_training_exposure_bands([], make_catalog(), make_exposures(), k=2)


def test_evaluation_rejects_duplicate_catalog_ids_after_normalization() -> None:
    with pytest.raises(
        ExposureEvaluationError,
        match="Catalog item identifiers must be unique",
    ):
        evaluate_training_exposure_bands(
            [make_example("I1", ["U1"], {"U1"})],
            ["U1", " U1 "],
            {},
            k=1,
        )


def test_band_definitions_must_be_contiguous() -> None:
    invalid_bands = (
        TrainingExposureBand("unseen", 0, 0),
        TrainingExposureBand("high", 2, None),
    )

    with pytest.raises(
        ExposureEvaluationError,
        match="ordered, contiguous, and non-overlapping",
    ):
        evaluate_training_exposure_bands(
            make_examples(),
            make_catalog(),
            make_exposures(),
            k=2,
            bands=invalid_bands,
        )


def test_band_definitions_require_unique_names() -> None:
    invalid_bands = (
        TrainingExposureBand("same", 0, 0),
        TrainingExposureBand("same", 1, None),
    )

    with pytest.raises(ExposureEvaluationError, match="names must be unique"):
        evaluate_training_exposure_bands(
            make_examples(),
            make_catalog(),
            make_exposures(),
            k=2,
            bands=invalid_bands,
        )


def test_final_band_must_be_open_ended() -> None:
    invalid_bands = (TrainingExposureBand("all", 0, 100),)

    with pytest.raises(
        ExposureEvaluationError,
        match="final exposure band must have no maximum",
    ):
        evaluate_training_exposure_bands(
            make_examples(),
            make_catalog(),
            make_exposures(),
            k=2,
            bands=invalid_bands,
        )
