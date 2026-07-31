from __future__ import annotations

import json

import pytest

from newslens.evaluation import (
    FailureAnalysisError,
    ScoredRankingExample,
    analyze_high_score_failures,
)


def make_example(
    impression_id: str,
    ranking: list[str],
    scores: list[float],
    relevant: set[str],
    *,
    source: str = "content",
    history_length: int = 3,
    candidate_count: int = 10,
) -> ScoredRankingExample:
    return ScoredRankingExample(
        impression_id=impression_id,
        ranked_items=tuple(ranking),
        ranked_scores=tuple(scores),
        relevant_items=frozenset(relevant),
        source=source,
        history_length=history_length,
        candidate_count=candidate_count,
    )


def make_mixed_source_examples() -> list[ScoredRankingExample]:
    return [
        make_example("C1", ["N1", "N2"], [0.90, 0.60], {"N9"}),
        make_example("C2", ["N1", "N2"], [0.80, 0.50], {"N1"}),
        make_example("C3", ["N1", "N2"], [0.10, 0.05], {"N8"}),
        make_example(
            "P1",
            ["N3", "N4"],
            [100.0, 80.0],
            {"N9"},
            source="popularity",
            history_length=0,
        ),
        make_example(
            "P2",
            ["N3", "N4"],
            [50.0, 40.0],
            {"N3"},
            source="popularity",
            history_length=0,
        ),
        make_example(
            "P3",
            ["N3", "N4"],
            [10.0, 5.0],
            {"N8"},
            source="popularity",
            history_length=0,
        ),
    ]


def test_source_specific_thresholds_find_high_score_misses() -> None:
    report = analyze_high_score_failures(
        make_mixed_source_examples(),
        k=2,
        score_quantile=0.50,
    )

    thresholds = {row.source: row.score_threshold for row in report.source_thresholds}

    assert thresholds == {
        "content": pytest.approx(0.80),
        "popularity": pytest.approx(50.0),
    }
    assert report.total_impressions == 6
    assert report.evaluated_impressions == 6
    assert report.top_k_misses == 4
    assert report.high_score_misses == 2
    assert [failure.impression_id for failure in report.failures] == ["C1", "P1"]


def test_score_scales_are_never_compared_across_sources() -> None:
    report = analyze_high_score_failures(
        make_mixed_source_examples(),
        k=2,
        score_quantile=0.50,
    )
    content, popularity = report.source_thresholds

    assert content.source == "content"
    assert content.eligible_impressions == 3
    assert content.top_k_misses == 2
    assert content.high_score_misses == 1
    assert popularity.source == "popularity"
    assert popularity.eligible_impressions == 3
    assert popularity.top_k_misses == 2
    assert popularity.high_score_misses == 1


def test_no_click_impressions_are_skipped_and_do_not_set_thresholds() -> None:
    examples = make_mixed_source_examples()
    examples.append(make_example("NO_CLICK", ["N1", "N2"], [999.0, 900.0], set()))
    report = analyze_high_score_failures(
        examples,
        k=2,
        score_quantile=0.50,
    )

    assert report.total_impressions == 7
    assert report.evaluated_impressions == 6
    assert report.skipped_no_click_impressions == 1
    assert report.source_thresholds[0].score_threshold == pytest.approx(0.80)


def test_empty_or_non_positive_rankings_remain_visible_but_not_high_score() -> None:
    examples = make_mixed_source_examples()
    examples.extend(
        [
            make_example("EMPTY", [], [], {"N9"}),
            make_example("ZERO", ["N1", "N2"], [0.0, 0.0], {"N9"}),
        ]
    )
    report = analyze_high_score_failures(
        examples,
        k=2,
        score_quantile=0.50,
    )

    assert report.evaluated_impressions == 8
    assert report.score_eligible_impressions == 6
    assert report.non_positive_or_empty_score_impressions == 2
    assert report.top_k_misses == 6
    assert report.high_score_misses == 2


def test_failure_records_include_context_scores_and_margin() -> None:
    report = analyze_high_score_failures(
        make_mixed_source_examples(),
        k=2,
        score_quantile=0.50,
    )
    failure = report.failures[0]

    assert failure.impression_id == "C1"
    assert failure.source == "content"
    assert failure.history_length == 3
    assert failure.candidate_count == 10
    assert failure.relevant_items == ("N9",)
    assert failure.ranked_items == ("N1", "N2")
    assert failure.ranked_scores == pytest.approx((0.90, 0.60))
    assert failure.top_score == pytest.approx(0.90)
    assert failure.score_margin == pytest.approx(0.30)
    assert failure.score_threshold == pytest.approx(0.80)


def test_per_source_limit_is_deterministic() -> None:
    examples = [
        make_example(f"I{index}", ["N1", "N2"], [score, 0.0], {"N9"})
        for index, score in enumerate([0.90, 0.80, 0.70, 0.60], start=1)
    ]
    report = analyze_high_score_failures(
        examples,
        k=2,
        score_quantile=0.25,
        maximum_failures_per_source=2,
    )

    assert report.high_score_misses == 3
    assert [failure.impression_id for failure in report.failures] == ["I1", "I2"]
    assert report.source_thresholds[0].retained_failures == 2


def test_report_is_json_serializable_and_describes_score_semantics() -> None:
    report = analyze_high_score_failures(
        make_mixed_source_examples(),
        k=2,
        score_quantile=0.50,
    )
    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["method"] == "source_specific_top_score_quantile"
    assert payload["score_interpretation"] == "relative_within_recommendation_source"
    assert payload["top_k_miss_fraction"] == pytest.approx(4 / 6)
    assert payload["high_score_miss_fraction"] == pytest.approx(2 / 6)
    assert '"score_margin"' in serialized


def test_duplicate_impression_ids_are_rejected() -> None:
    examples = [
        make_example("I1", ["N1"], [0.9], {"N2"}),
        make_example("I1", ["N2"], [0.8], {"N1"}),
    ]

    with pytest.raises(
        FailureAnalysisError,
        match="impression_id values must be unique",
    ):
        analyze_high_score_failures(examples, k=1)


def test_at_least_one_clicked_impression_is_required() -> None:
    with pytest.raises(
        FailureAnalysisError,
        match="with a relevant item",
    ):
        analyze_high_score_failures(
            [make_example("I1", ["N1"], [0.9], set())],
            k=1,
        )


def test_at_least_one_positive_top_score_is_required() -> None:
    with pytest.raises(
        FailureAnalysisError,
        match="positive top score",
    ):
        analyze_high_score_failures(
            [make_example("I1", ["N1"], [0.0], {"N2"})],
            k=1,
        )


def test_examples_are_required() -> None:
    with pytest.raises(
        FailureAnalysisError,
        match="At least one scored ranking example",
    ):
        analyze_high_score_failures([], k=1)


def test_non_scored_examples_are_rejected() -> None:
    with pytest.raises(
        FailureAnalysisError,
        match="ScoredRankingExample instances",
    ):
        analyze_high_score_failures(
            [object()],  # type: ignore[list-item]
            k=1,
        )


@pytest.mark.parametrize("invalid_k", [0, -1, 1.5, True])
def test_invalid_k_is_rejected(invalid_k: object) -> None:
    with pytest.raises(
        FailureAnalysisError,
        match="k must be a positive integer",
    ):
        analyze_high_score_failures(
            make_mixed_source_examples(),
            k=invalid_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_quantile", [0.0, 1.0, -0.1, 1.1, True, "0.9"])
def test_invalid_quantile_is_rejected(invalid_quantile: object) -> None:
    with pytest.raises(
        FailureAnalysisError,
        match="score_quantile must be between 0 and 1",
    ):
        analyze_high_score_failures(
            make_mixed_source_examples(),
            k=2,
            score_quantile=invalid_quantile,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_maximum", [0, -1, 1.5, True])
def test_invalid_maximum_is_rejected(invalid_maximum: object) -> None:
    with pytest.raises(
        FailureAnalysisError,
        match="maximum_failures_per_source must be a positive integer",
    ):
        analyze_high_score_failures(
            make_mixed_source_examples(),
            k=2,
            maximum_failures_per_source=invalid_maximum,  # type: ignore[arg-type]
        )


def test_ranked_items_and_scores_must_have_equal_lengths() -> None:
    with pytest.raises(
        FailureAnalysisError,
        match="equal lengths",
    ):
        make_example("I1", ["N1", "N2"], [0.9], {"N3"})


def test_scores_must_be_finite_and_descending() -> None:
    with pytest.raises(FailureAnalysisError, match="finite values"):
        make_example("I1", ["N1"], [float("nan")], {"N2"})

    with pytest.raises(FailureAnalysisError, match="highest to lowest"):
        make_example("I1", ["N1", "N2"], [0.5, 0.9], {"N3"})


def test_candidate_count_cannot_be_smaller_than_ranking() -> None:
    with pytest.raises(
        FailureAnalysisError,
        match="at least the number of ranked items",
    ):
        make_example(
            "I1",
            ["N1", "N2"],
            [0.9, 0.8],
            {"N3"},
            candidate_count=1,
        )
