from __future__ import annotations

import json

import pytest

from newslens.evaluation import (
    PairedComparisonError,
    RankingExample,
    paired_bootstrap_ranking_comparison,
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


def make_examples() -> tuple[list[RankingExample], list[RankingExample]]:
    baseline = [
        make_example("I1", ["A", "B"], {"A"}),
        make_example("I2", [], {"B"}),
        make_example("I3", ["C"], set()),
    ]
    candidate = [
        make_example("I1", ["A", "B"], {"A"}),
        make_example("I2", ["B", "A"], {"B"}),
        make_example("I3", ["B"], set()),
    ]
    return baseline, candidate


def compare(
    baseline: list[RankingExample],
    candidate: list[RankingExample],
    *,
    bootstrap_samples: int = 200,
    random_seed: int = 42,
):
    return paired_bootstrap_ranking_comparison(
        baseline,
        candidate,
        baseline_model_name="baseline",
        candidate_model_name="candidate",
        k=2,
        bootstrap_samples=bootstrap_samples,
        confidence_level=0.95,
        random_seed=random_seed,
    )


def test_hand_calculated_paired_differences() -> None:
    baseline, candidate = make_examples()
    report = compare(baseline, candidate)

    assert report.total_impressions == 3
    assert report.evaluated_impressions == 2
    assert report.skipped_no_click_impressions == 1

    for interval in (
        report.ndcg_at_k,
        report.mrr_at_k,
        report.recall_at_k,
        report.hit_rate_at_k,
    ):
        assert interval.baseline_estimate == pytest.approx(0.5)
        assert interval.candidate_estimate == pytest.approx(1.0)
        assert interval.point_difference == pytest.approx(0.5)
        assert interval.lower_bound <= interval.point_difference
        assert interval.upper_bound >= interval.point_difference


def test_input_order_does_not_change_alignment_or_results() -> None:
    baseline, candidate = make_examples()
    first = compare(baseline, candidate)
    second = compare(list(reversed(baseline)), [candidate[1], candidate[2], candidate[0]])

    assert first.to_dict() == second.to_dict()


def test_repeated_seed_produces_identical_report() -> None:
    baseline, candidate = make_examples()
    first = compare(baseline, candidate, bootstrap_samples=500, random_seed=2026)
    second = compare(baseline, candidate, bootstrap_samples=500, random_seed=2026)

    assert first.to_dict() == second.to_dict()


def test_report_is_json_serializable_and_defines_difference_direction() -> None:
    baseline, candidate = make_examples()
    payload = compare(baseline, candidate).to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["method"] == "paired_nonparametric_percentile_bootstrap"
    assert payload["resampling_unit"] == "aligned_evaluated_impression_pair"
    assert payload["difference_direction"] == "candidate_minus_baseline"
    assert payload["evaluated_fraction"] == pytest.approx(2 / 3)
    assert '"point_difference"' in serialized
    assert '"excludes_zero"' in serialized


def test_negative_candidate_difference_is_preserved() -> None:
    baseline = [make_example("I1", ["A"], {"A"})]
    candidate = [make_example("I1", [], {"A"})]
    report = compare(baseline, candidate)

    assert report.hit_rate_at_k.baseline_estimate == pytest.approx(1.0)
    assert report.hit_rate_at_k.candidate_estimate == pytest.approx(0.0)
    assert report.hit_rate_at_k.point_difference == pytest.approx(-1.0)
    assert report.hit_rate_at_k.excludes_zero


def test_impression_sets_must_match() -> None:
    baseline, candidate = make_examples()
    candidate[1] = make_example("OTHER", ["B"], {"B"})

    with pytest.raises(PairedComparisonError, match="impression IDs must match"):
        compare(baseline, candidate)


def test_relevance_labels_must_match_within_each_pair() -> None:
    baseline, candidate = make_examples()
    candidate[1] = make_example("I2", ["B"], {"A"})

    with pytest.raises(PairedComparisonError, match="Relevant items differ"):
        compare(baseline, candidate)


def test_duplicate_impression_ids_are_rejected() -> None:
    baseline, candidate = make_examples()
    baseline.append(make_example("I1", ["A"], {"A"}))

    with pytest.raises(PairedComparisonError, match="must be unique"):
        compare(baseline, candidate)


def test_at_least_one_clicked_pair_is_required() -> None:
    baseline = [make_example("I1", [], set())]
    candidate = [make_example("I1", ["A"], set())]

    with pytest.raises(PairedComparisonError, match="with a relevant item"):
        compare(baseline, candidate)


def test_examples_must_be_nonempty_ranking_examples() -> None:
    _, candidate = make_examples()

    with pytest.raises(PairedComparisonError, match="At least one baseline"):
        compare([], candidate)

    with pytest.raises(PairedComparisonError, match="RankingExample instances"):
        paired_bootstrap_ranking_comparison(
            [object()],  # type: ignore[list-item]
            candidate,
            baseline_model_name="baseline",
            candidate_model_name="candidate",
            k=2,
        )


@pytest.mark.parametrize("invalid_k", [0, -1, 1.5, True])
def test_invalid_k_is_rejected(invalid_k: object) -> None:
    baseline, candidate = make_examples()

    with pytest.raises(PairedComparisonError, match="k must be a positive integer"):
        paired_bootstrap_ranking_comparison(
            baseline,
            candidate,
            baseline_model_name="baseline",
            candidate_model_name="candidate",
            k=invalid_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_samples", [0, 1, -1, 1.5, True])
def test_invalid_bootstrap_samples_are_rejected(invalid_samples: object) -> None:
    baseline, candidate = make_examples()

    with pytest.raises(PairedComparisonError, match="bootstrap_samples must"):
        paired_bootstrap_ranking_comparison(
            baseline,
            candidate,
            baseline_model_name="baseline",
            candidate_model_name="candidate",
            k=2,
            bootstrap_samples=invalid_samples,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_confidence", [0.0, 1.0, -0.1, 1.1, True, "0.95"])
def test_invalid_confidence_level_is_rejected(invalid_confidence: object) -> None:
    baseline, candidate = make_examples()

    with pytest.raises(PairedComparisonError, match="confidence_level must"):
        paired_bootstrap_ranking_comparison(
            baseline,
            candidate,
            baseline_model_name="baseline",
            candidate_model_name="candidate",
            k=2,
            confidence_level=invalid_confidence,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_seed", [-1, 1.5, True, "42"])
def test_invalid_random_seed_is_rejected(invalid_seed: object) -> None:
    baseline, candidate = make_examples()

    with pytest.raises(PairedComparisonError, match="random_seed must"):
        paired_bootstrap_ranking_comparison(
            baseline,
            candidate,
            baseline_model_name="baseline",
            candidate_model_name="candidate",
            k=2,
            random_seed=invalid_seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("baseline_name", "candidate_name", "message"),
    [
        ("", "candidate", "baseline_model_name"),
        ("baseline", "", "candidate_model_name"),
        ("same", "same", "must be distinct"),
    ],
)
def test_model_names_are_validated(
    baseline_name: str,
    candidate_name: str,
    message: str,
) -> None:
    baseline, candidate = make_examples()

    with pytest.raises(PairedComparisonError, match=message):
        paired_bootstrap_ranking_comparison(
            baseline,
            candidate,
            baseline_model_name=baseline_name,
            candidate_model_name=candidate_name,
            k=2,
        )
