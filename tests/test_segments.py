from __future__ import annotations

import json
from math import log2

import pytest

from newslens.evaluation import (
    DEFAULT_HISTORY_LENGTH_SEGMENTS,
    HistoryLengthSegment,
    HistorySegmentEvaluationError,
    HistorySegmentExample,
    RankingExample,
    evaluate_history_segments,
)


def make_example(
    impression_id: str,
    history_length: int,
    ranking: list[str],
    relevant: set[str],
) -> HistorySegmentExample:
    return HistorySegmentExample(
        ranking=RankingExample(
            impression_id=impression_id,
            ranked_items=ranking,
            relevant_items=relevant,
        ),
        history_length=history_length,
    )


def test_default_segments_are_exhaustive() -> None:
    assert [segment.name for segment in DEFAULT_HISTORY_LENGTH_SEGMENTS] == [
        "cold_start",
        "short_history",
        "medium_history",
        "long_history",
    ]
    assert DEFAULT_HISTORY_LENGTH_SEGMENTS[0].contains(0)
    assert DEFAULT_HISTORY_LENGTH_SEGMENTS[1].contains(1)
    assert DEFAULT_HISTORY_LENGTH_SEGMENTS[1].contains(4)
    assert DEFAULT_HISTORY_LENGTH_SEGMENTS[2].contains(5)
    assert DEFAULT_HISTORY_LENGTH_SEGMENTS[2].contains(9)
    assert DEFAULT_HISTORY_LENGTH_SEGMENTS[3].contains(10)
    assert DEFAULT_HISTORY_LENGTH_SEGMENTS[3].contains(10_000)


def test_history_segment_metrics_match_hand_calculation() -> None:
    report = evaluate_history_segments(
        [
            make_example("I1", 0, ["N1", "N2"], {"N1"}),
            make_example("I2", 2, ["N2", "N1"], {"N1"}),
            make_example("I3", 5, ["N3", "N4"], {"N4"}),
            make_example("I4", 10, ["N4", "N3"], {"N1"}),
        ],
        ["N1", "N2", "N3", "N4"],
        k=2,
    )

    segments = {segment.definition.name: segment for segment in report.segments}

    assert report.k == 2
    assert report.total_impressions == 4
    assert sum(segment.fraction_of_all_impressions for segment in report.segments) == pytest.approx(
        1.0
    )

    assert segments["cold_start"].metrics is not None
    assert segments["cold_start"].metrics.ndcg_at_k == pytest.approx(1.0)
    assert segments["short_history"].metrics is not None
    assert segments["short_history"].metrics.mrr_at_k == pytest.approx(0.5)
    assert segments["medium_history"].metrics is not None
    assert segments["medium_history"].metrics.ndcg_at_k == pytest.approx(1.0 / log2(3))
    assert segments["long_history"].metrics is not None
    assert segments["long_history"].metrics.hit_rate_at_k == pytest.approx(0.0)


def test_empty_and_no_click_segments_preserve_accounting() -> None:
    report = evaluate_history_segments(
        [
            make_example("I1", 0, ["N1"], {"N1"}),
            make_example("I2", 2, ["N2"], set()),
        ],
        ["N1", "N2"],
        k=1,
    )

    segments = {segment.definition.name: segment for segment in report.segments}

    assert segments["short_history"].total_impressions == 1
    assert segments["short_history"].clicked_impressions == 0
    assert segments["short_history"].no_click_impressions == 1
    assert segments["short_history"].metrics is None
    assert segments["medium_history"].total_impressions == 0
    assert segments["medium_history"].metrics is None


def test_report_is_json_serializable() -> None:
    report = evaluate_history_segments(
        [make_example("I1", 0, ["N1"], {"N1"})],
        ["N1"],
        k=1,
    )

    serialized = json.dumps(report.to_dict())

    assert '"name": "cold_start"' in serialized
    assert report.to_dict()["overall_metrics"]["ndcg_at_k"] == pytest.approx(1.0)


@pytest.mark.parametrize("invalid_history_length", [-1, 1.5, True])
def test_example_rejects_invalid_history_length(invalid_history_length: object) -> None:
    with pytest.raises(
        HistorySegmentEvaluationError,
        match="history_length must be a non-negative integer",
    ):
        make_example(
            "I1",
            invalid_history_length,  # type: ignore[arg-type]
            ["N1"],
            {"N1"},
        )


def test_evaluation_rejects_duplicate_segment_names() -> None:
    definitions = (
        HistoryLengthSegment("same", 0, 0),
        HistoryLengthSegment("same", 1, None),
    )

    with pytest.raises(
        HistorySegmentEvaluationError,
        match="names must be unique",
    ):
        evaluate_history_segments(
            [make_example("I1", 0, ["N1"], {"N1"})],
            ["N1"],
            k=1,
            segments=definitions,
        )


def test_evaluation_rejects_segment_gaps() -> None:
    definitions = (
        HistoryLengthSegment("cold", 0, 0),
        HistoryLengthSegment("long", 2, None),
    )

    with pytest.raises(
        HistorySegmentEvaluationError,
        match="ordered, contiguous, and non-overlapping",
    ):
        evaluate_history_segments(
            [make_example("I1", 0, ["N1"], {"N1"})],
            ["N1"],
            k=1,
            segments=definitions,
        )


def test_evaluation_rejects_finite_final_segment() -> None:
    definitions = (HistoryLengthSegment("finite", 0, 10),)

    with pytest.raises(
        HistorySegmentEvaluationError,
        match="final history segment must have no maximum",
    ):
        evaluate_history_segments(
            [make_example("I1", 0, ["N1"], {"N1"})],
            ["N1"],
            k=1,
            segments=definitions,
        )


def test_evaluation_wraps_ranking_errors() -> None:
    with pytest.raises(
        HistorySegmentEvaluationError,
        match="outside the catalog",
    ):
        evaluate_history_segments(
            [make_example("I1", 0, ["UNKNOWN"], {"N1"})],
            ["N1"],
            k=1,
        )
