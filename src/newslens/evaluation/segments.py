"""Model-independent ranking evaluation by user-history length."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .evaluator import (
    RankingEvaluationError,
    RankingEvaluationResult,
    RankingExample,
    evaluate_rankings,
)


class HistorySegmentEvaluationError(ValueError):
    """Raised when history-length segment evaluation cannot be completed."""


@dataclass(frozen=True)
class HistoryLengthSegment:
    """One inclusive history-length interval."""

    name: str
    minimum: int
    maximum: int | None

    def __post_init__(self) -> None:
        name = str(self.name).strip()

        if not name:
            raise HistorySegmentEvaluationError("Segment name must not be empty.")

        if isinstance(self.minimum, bool) or not isinstance(self.minimum, int):
            raise HistorySegmentEvaluationError("Segment minimum must be a non-negative integer.")

        if self.minimum < 0:
            raise HistorySegmentEvaluationError("Segment minimum must be a non-negative integer.")

        if self.maximum is not None:
            if isinstance(self.maximum, bool) or not isinstance(self.maximum, int):
                raise HistorySegmentEvaluationError(
                    "Segment maximum must be a non-negative integer or None."
                )

            if self.maximum < self.minimum:
                raise HistorySegmentEvaluationError(
                    "Segment maximum must be greater than or equal to its minimum."
                )

        object.__setattr__(self, "name", name)

    def contains(self, history_length: int) -> bool:
        """Return whether a history length belongs to this interval."""

        return history_length >= self.minimum and (
            self.maximum is None or history_length <= self.maximum
        )

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-compatible segment definition."""

        return {
            "name": self.name,
            "minimum_history_length": self.minimum,
            "maximum_history_length": self.maximum,
        }


DEFAULT_HISTORY_LENGTH_SEGMENTS: tuple[HistoryLengthSegment, ...] = (
    HistoryLengthSegment("cold_start", 0, 0),
    HistoryLengthSegment("short_history", 1, 4),
    HistoryLengthSegment("medium_history", 5, 9),
    HistoryLengthSegment("long_history", 10, None),
)


@dataclass(frozen=True)
class HistorySegmentExample:
    """A ranking example paired with its user-history length."""

    ranking: RankingExample
    history_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.ranking, RankingExample):
            raise HistorySegmentEvaluationError("ranking must be a RankingExample.")

        if (
            isinstance(self.history_length, bool)
            or not isinstance(self.history_length, int)
            or self.history_length < 0
        ):
            raise HistorySegmentEvaluationError("history_length must be a non-negative integer.")


@dataclass(frozen=True)
class HistorySegmentResult:
    """Accounting and optional ranking metrics for one segment."""

    definition: HistoryLengthSegment
    total_impressions: int
    clicked_impressions: int
    fraction_of_all_impressions: float
    metrics: RankingEvaluationResult | None

    @property
    def no_click_impressions(self) -> int:
        """Return impressions excluded from ranking-quality metrics."""

        return self.total_impressions - self.clicked_impressions

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible segment result."""

        return {
            **self.definition.to_dict(),
            "total_impressions": self.total_impressions,
            "clicked_impressions": self.clicked_impressions,
            "no_click_impressions": self.no_click_impressions,
            "fraction_of_all_impressions": self.fraction_of_all_impressions,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class HistorySegmentEvaluationReport:
    """Overall metrics and history-length subgroup results."""

    overall_metrics: RankingEvaluationResult
    segments: tuple[HistorySegmentResult, ...]

    @property
    def k(self) -> int:
        """Return the ranking cutoff used throughout the report."""

        return self.overall_metrics.k

    @property
    def total_impressions(self) -> int:
        """Return the number of evaluated input impressions."""

        return self.overall_metrics.total_impressions

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible evaluation report."""

        return {
            "k": self.k,
            "total_impressions": self.total_impressions,
            "overall_metrics": self.overall_metrics.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }


def _validate_segment_definitions(
    definitions: tuple[HistoryLengthSegment, ...],
) -> None:
    if not definitions:
        raise HistorySegmentEvaluationError("At least one history segment is required.")

    names = [definition.name for definition in definitions]

    if len(names) != len(set(names)):
        raise HistorySegmentEvaluationError("History segment names must be unique.")

    if definitions[0].minimum != 0:
        raise HistorySegmentEvaluationError("History segments must begin at length zero.")

    for index, definition in enumerate(definitions):
        is_last = index == len(definitions) - 1

        if definition.maximum is None and not is_last:
            raise HistorySegmentEvaluationError(
                "Only the final history segment may have no maximum."
            )

        if is_last:
            if definition.maximum is not None:
                raise HistorySegmentEvaluationError(
                    "The final history segment must have no maximum."
                )
            continue

        next_definition = definitions[index + 1]
        expected_minimum = definition.maximum + 1  # type: ignore[operator]

        if next_definition.minimum != expected_minimum:
            raise HistorySegmentEvaluationError(
                "History segments must be ordered, contiguous, and non-overlapping."
            )


def evaluate_history_segments(
    examples: Iterable[HistorySegmentExample],
    catalog_items: Iterable[str],
    *,
    k: int,
    segments: Iterable[HistoryLengthSegment] = DEFAULT_HISTORY_LENGTH_SEGMENTS,
) -> HistorySegmentEvaluationReport:
    """Evaluate rankings overall and within exhaustive history-length groups."""

    example_list = list(examples)

    if not example_list:
        raise HistorySegmentEvaluationError("At least one segmented example is required.")

    definitions = tuple(segments)
    _validate_segment_definitions(definitions)
    catalog = tuple(catalog_items)

    try:
        overall_metrics = evaluate_rankings(
            [example.ranking for example in example_list],
            catalog,
            k=k,
        )
    except RankingEvaluationError as error:
        raise HistorySegmentEvaluationError(str(error)) from error

    results: list[HistorySegmentResult] = []

    for definition in definitions:
        segment_rankings = [
            example.ranking
            for example in example_list
            if definition.contains(example.history_length)
        ]
        clicked_impressions = sum(bool(example.relevant_items) for example in segment_rankings)
        metrics: RankingEvaluationResult | None = None

        if segment_rankings and clicked_impressions:
            try:
                metrics = evaluate_rankings(
                    segment_rankings,
                    catalog,
                    k=k,
                )
            except RankingEvaluationError as error:
                raise HistorySegmentEvaluationError(str(error)) from error

        results.append(
            HistorySegmentResult(
                definition=definition,
                total_impressions=len(segment_rankings),
                clicked_impressions=clicked_impressions,
                fraction_of_all_impressions=(
                    len(segment_rankings) / overall_metrics.total_impressions
                ),
                metrics=metrics,
            )
        )

    if sum(result.total_impressions for result in results) != overall_metrics.total_impressions:
        raise HistorySegmentEvaluationError(
            "History segment definitions did not assign every example exactly once."
        )

    return HistorySegmentEvaluationReport(
        overall_metrics=overall_metrics,
        segments=tuple(results),
    )
