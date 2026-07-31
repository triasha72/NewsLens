from __future__ import annotations

import json

import pytest

from newslens.evaluation import (
    BootstrapUncertaintyError,
    RankingExample,
    bootstrap_ranking_uncertainty,
    evaluate_rankings,
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


def make_examples() -> list[RankingExample]:
    return [
        make_example("I1", ["N1", "N2", "N3"], {"N1"}),
        make_example("I2", ["N1", "N2", "N3"], {"N2"}),
        make_example("I3", ["N1", "N2", "N3"], {"N3"}),
        make_example("I4", ["N1", "N2", "N3"], {"N4"}),
        make_example("I5", ["N1", "N2", "N3"], set()),
    ]


def test_point_estimates_match_shared_ranking_evaluator() -> None:
    examples = make_examples()
    report = bootstrap_ranking_uncertainty(
        examples,
        k=3,
        bootstrap_samples=200,
        random_seed=17,
    )
    expected = evaluate_rankings(
        examples,
        ["N1", "N2", "N3", "N4"],
        k=3,
    )

    assert report.ndcg_at_k.point_estimate == expected.ndcg_at_k
    assert report.mrr_at_k.point_estimate == expected.mrr_at_k
    assert report.recall_at_k.point_estimate == expected.recall_at_k
    assert report.hit_rate_at_k.point_estimate == expected.hit_rate_at_k


def test_no_click_impressions_are_excluded_and_reported() -> None:
    report = bootstrap_ranking_uncertainty(
        make_examples(),
        k=3,
        bootstrap_samples=20,
    )

    assert report.total_impressions == 5
    assert report.evaluated_impressions == 4
    assert report.skipped_no_click_impressions == 1
    assert report.evaluated_fraction == pytest.approx(0.8)


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    first = bootstrap_ranking_uncertainty(
        make_examples(),
        k=3,
        bootstrap_samples=200,
        random_seed=2026,
    )
    second = bootstrap_ranking_uncertainty(
        make_examples(),
        k=3,
        bootstrap_samples=200,
        random_seed=2026,
    )

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_constant_metric_values_have_zero_width_intervals() -> None:
    examples = [
        make_example("I1", ["N1", "N2"], {"N1"}),
        make_example("I2", ["N2", "N1"], {"N2"}),
        make_example("I3", ["N3", "N1"], {"N3"}),
    ]
    report = bootstrap_ranking_uncertainty(
        examples,
        k=2,
        bootstrap_samples=30,
    )

    for interval in (
        report.ndcg_at_k,
        report.mrr_at_k,
        report.recall_at_k,
        report.hit_rate_at_k,
    ):
        assert interval.point_estimate == pytest.approx(1.0)
        assert interval.lower_bound == pytest.approx(1.0)
        assert interval.upper_bound == pytest.approx(1.0)
        assert interval.standard_error == pytest.approx(0.0)


def test_intervals_and_standard_errors_are_valid() -> None:
    report = bootstrap_ranking_uncertainty(
        make_examples(),
        k=3,
        bootstrap_samples=300,
        confidence_level=0.90,
        random_seed=5,
    )

    for interval in (
        report.ndcg_at_k,
        report.mrr_at_k,
        report.recall_at_k,
        report.hit_rate_at_k,
    ):
        assert 0.0 <= interval.lower_bound <= interval.upper_bound <= 1.0
        assert interval.standard_error >= 0.0


def test_multiple_relevant_items_are_supported() -> None:
    report = bootstrap_ranking_uncertainty(
        [make_example("I1", ["N1", "N2", "N3"], {"N1", "N3"})],
        k=2,
        bootstrap_samples=10,
    )

    assert report.ndcg_at_k.point_estimate == pytest.approx(1.0 / (1.0 + (1.0 / 1.5849625)))
    assert report.mrr_at_k.point_estimate == pytest.approx(1.0)
    assert report.recall_at_k.point_estimate == pytest.approx(0.5)
    assert report.hit_rate_at_k.point_estimate == pytest.approx(1.0)


def test_report_is_json_serializable_and_describes_the_method() -> None:
    report = bootstrap_ranking_uncertainty(
        make_examples(),
        k=3,
        bootstrap_samples=20,
        confidence_level=0.95,
        random_seed=42,
    )
    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["method"] == "nonparametric_percentile_bootstrap"
    assert payload["resampling_unit"] == "evaluated_impression"
    assert payload["bootstrap_samples"] == 20
    assert payload["random_seed"] == 42
    assert '"lower_bound"' in serialized


def test_duplicate_impression_ids_are_rejected() -> None:
    examples = [
        make_example("I1", ["N1"], {"N1"}),
        make_example("I1", ["N2"], {"N2"}),
    ]

    with pytest.raises(
        BootstrapUncertaintyError,
        match="impression_id values must be unique",
    ):
        bootstrap_ranking_uncertainty(examples, k=1, bootstrap_samples=10)


def test_at_least_one_clicked_impression_is_required() -> None:
    with pytest.raises(
        BootstrapUncertaintyError,
        match="with a relevant item",
    ):
        bootstrap_ranking_uncertainty(
            [make_example("I1", ["N1"], set())],
            k=1,
            bootstrap_samples=10,
        )


def test_examples_are_required() -> None:
    with pytest.raises(
        BootstrapUncertaintyError,
        match="At least one ranking example",
    ):
        bootstrap_ranking_uncertainty([], k=1, bootstrap_samples=10)


def test_non_ranking_examples_are_rejected() -> None:
    with pytest.raises(
        BootstrapUncertaintyError,
        match="RankingExample instances",
    ):
        bootstrap_ranking_uncertainty(
            [object()],  # type: ignore[list-item]
            k=1,
            bootstrap_samples=10,
        )


@pytest.mark.parametrize("invalid_k", [0, -1, 1.5, True])
def test_invalid_k_is_rejected(invalid_k: object) -> None:
    with pytest.raises(
        BootstrapUncertaintyError,
        match="k must be a positive integer",
    ):
        bootstrap_ranking_uncertainty(
            make_examples(),
            k=invalid_k,  # type: ignore[arg-type]
            bootstrap_samples=10,
        )


@pytest.mark.parametrize("invalid_samples", [0, -1, 1.5, True])
def test_invalid_bootstrap_sample_count_is_rejected(invalid_samples: object) -> None:
    with pytest.raises(
        BootstrapUncertaintyError,
        match="bootstrap_samples must be a positive integer",
    ):
        bootstrap_ranking_uncertainty(
            make_examples(),
            k=3,
            bootstrap_samples=invalid_samples,  # type: ignore[arg-type]
        )


def test_one_bootstrap_sample_is_rejected() -> None:
    with pytest.raises(
        BootstrapUncertaintyError,
        match="bootstrap_samples must be at least 2",
    ):
        bootstrap_ranking_uncertainty(
            make_examples(),
            k=3,
            bootstrap_samples=1,
        )


@pytest.mark.parametrize("invalid_level", [0.0, 1.0, -0.1, 1.1, True, "0.95"])
def test_invalid_confidence_level_is_rejected(invalid_level: object) -> None:
    with pytest.raises(
        BootstrapUncertaintyError,
        match="confidence_level must be between 0 and 1",
    ):
        bootstrap_ranking_uncertainty(
            make_examples(),
            k=3,
            bootstrap_samples=10,
            confidence_level=invalid_level,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_seed", [-1, 1.5, True])
def test_invalid_random_seed_is_rejected(invalid_seed: object) -> None:
    with pytest.raises(
        BootstrapUncertaintyError,
        match="random_seed must be a non-negative integer",
    ):
        bootstrap_ranking_uncertainty(
            make_examples(),
            k=3,
            bootstrap_samples=10,
            random_seed=invalid_seed,  # type: ignore[arg-type]
        )
