"""Deterministic inspection of high-score top-k ranking failures.

Raw score scales can differ across recommendation routes. Content similarity
and popularity click counts are therefore thresholded independently rather
than compared as if they represented calibrated probabilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from numbers import Real

import numpy as np


class FailureAnalysisError(ValueError):
    """Raised when high-score failure analysis cannot be completed."""


@dataclass(frozen=True)
class ScoredRankingExample:
    """One ranked impression retaining scores and evaluation context."""

    impression_id: str
    ranked_items: tuple[str, ...]
    ranked_scores: tuple[float, ...]
    relevant_items: frozenset[str]
    source: str
    history_length: int
    candidate_count: int

    def __post_init__(self) -> None:
        impression_id = str(self.impression_id).strip()

        if not impression_id:
            raise FailureAnalysisError("impression_id must not be empty.")

        if isinstance(self.ranked_items, str):
            raise FailureAnalysisError("ranked_items must be an iterable of item IDs.")

        if isinstance(self.ranked_scores, (str, bytes)):
            raise FailureAnalysisError("ranked_scores must be an iterable of numeric scores.")

        if isinstance(self.relevant_items, str):
            raise FailureAnalysisError("relevant_items must be an iterable of item IDs.")

        ranking = tuple(self.ranked_items)
        raw_scores = tuple(self.ranked_scores)
        relevant = frozenset(self.relevant_items)
        source = str(self.source).strip()

        if not source:
            raise FailureAnalysisError("source must not be empty.")

        if any(not isinstance(item, str) or not item.strip() for item in ranking):
            raise FailureAnalysisError("ranked_items must contain non-empty string IDs.")

        if any(not isinstance(item, str) or not item.strip() for item in relevant):
            raise FailureAnalysisError("relevant_items must contain non-empty string IDs.")

        if len(ranking) != len(set(ranking)):
            raise FailureAnalysisError("ranked_items must not contain duplicates.")

        if len(ranking) != len(raw_scores):
            raise FailureAnalysisError("ranked_items and ranked_scores must have equal lengths.")

        if any(isinstance(score, bool) or not isinstance(score, Real) for score in raw_scores):
            raise FailureAnalysisError("ranked_scores must contain numeric values.")

        scores = tuple(float(score) for score in raw_scores)

        if any(not isfinite(score) for score in scores):
            raise FailureAnalysisError("ranked_scores must contain finite values.")

        if any(left < right for left, right in pairwise(scores)):
            raise FailureAnalysisError("ranked_scores must be ordered from highest to lowest.")

        if (
            isinstance(self.history_length, bool)
            or not isinstance(self.history_length, int)
            or self.history_length < 0
        ):
            raise FailureAnalysisError("history_length must be a non-negative integer.")

        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count <= 0
        ):
            raise FailureAnalysisError("candidate_count must be a positive integer.")

        if self.candidate_count < len(ranking):
            raise FailureAnalysisError(
                "candidate_count must be at least the number of ranked items."
            )

        object.__setattr__(self, "impression_id", impression_id)
        object.__setattr__(self, "ranked_items", ranking)
        object.__setattr__(self, "ranked_scores", scores)
        object.__setattr__(self, "relevant_items", relevant)
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class SourceScoreThreshold:
    """A route-specific score threshold and its accounting information."""

    source: str
    score_threshold: float
    eligible_impressions: int
    top_k_misses: int
    high_score_misses: int
    retained_failures: int

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-compatible representation."""

        return {
            "source": self.source,
            "score_threshold": self.score_threshold,
            "eligible_impressions": self.eligible_impressions,
            "top_k_misses": self.top_k_misses,
            "high_score_misses": self.high_score_misses,
            "retained_failures": self.retained_failures,
        }


@dataclass(frozen=True)
class HighScoreFailure:
    """One top-k miss whose top score exceeds its route threshold."""

    impression_id: str
    source: str
    history_length: int
    candidate_count: int
    relevant_items: tuple[str, ...]
    ranked_items: tuple[str, ...]
    ranked_scores: tuple[float, ...]
    top_score: float
    score_margin: float | None
    score_threshold: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "impression_id": self.impression_id,
            "source": self.source,
            "history_length": self.history_length,
            "candidate_count": self.candidate_count,
            "relevant_items": list(self.relevant_items),
            "ranked_items": list(self.ranked_items),
            "ranked_scores": list(self.ranked_scores),
            "top_score": self.top_score,
            "score_margin": self.score_margin,
            "score_threshold": self.score_threshold,
        }


@dataclass(frozen=True)
class HighScoreFailureReport:
    """Route-aware high-score failure records and denominator accounting."""

    k: int
    score_quantile: float
    maximum_failures_per_source: int
    total_impressions: int
    evaluated_impressions: int
    skipped_no_click_impressions: int
    score_eligible_impressions: int
    non_positive_or_empty_score_impressions: int
    top_k_misses: int
    high_score_misses: int
    source_thresholds: tuple[SourceScoreThreshold, ...]
    failures: tuple[HighScoreFailure, ...]

    @property
    def top_k_miss_fraction(self) -> float:
        """Return the top-k miss fraction among clicked impressions."""

        return self.top_k_misses / self.evaluated_impressions

    @property
    def high_score_miss_fraction(self) -> float:
        """Return the high-score miss fraction among score-eligible impressions."""

        return self.high_score_misses / self.score_eligible_impressions

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "method": "source_specific_top_score_quantile",
            "score_interpretation": "relative_within_recommendation_source",
            "k": self.k,
            "score_quantile": self.score_quantile,
            "maximum_failures_per_source": self.maximum_failures_per_source,
            "total_impressions": self.total_impressions,
            "evaluated_impressions": self.evaluated_impressions,
            "skipped_no_click_impressions": self.skipped_no_click_impressions,
            "score_eligible_impressions": self.score_eligible_impressions,
            "non_positive_or_empty_score_impressions": (
                self.non_positive_or_empty_score_impressions
            ),
            "top_k_misses": self.top_k_misses,
            "top_k_miss_fraction": self.top_k_miss_fraction,
            "high_score_misses": self.high_score_misses,
            "high_score_miss_fraction": self.high_score_miss_fraction,
            "source_thresholds": [threshold.to_dict() for threshold in self.source_thresholds],
            "failures": [failure.to_dict() for failure in self.failures],
        }


def _validate_configuration(
    *,
    k: object,
    score_quantile: object,
    maximum_failures_per_source: object,
) -> tuple[int, float, int]:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise FailureAnalysisError("k must be a positive integer.")

    if (
        isinstance(score_quantile, bool)
        or not isinstance(score_quantile, Real)
        or not 0.0 < float(score_quantile) < 1.0
    ):
        raise FailureAnalysisError("score_quantile must be between 0 and 1.")

    if (
        isinstance(maximum_failures_per_source, bool)
        or not isinstance(maximum_failures_per_source, int)
        or maximum_failures_per_source <= 0
    ):
        raise FailureAnalysisError("maximum_failures_per_source must be a positive integer.")

    return k, float(score_quantile), maximum_failures_per_source


def _is_top_k_miss(example: ScoredRankingExample, *, k: int) -> bool:
    return example.relevant_items.isdisjoint(example.ranked_items[:k])


def _score_margin(scores: tuple[float, ...]) -> float | None:
    if len(scores) < 2:
        return None

    return scores[0] - scores[1]


def analyze_high_score_failures(
    examples: Iterable[ScoredRankingExample],
    *,
    k: int,
    score_quantile: float = 0.90,
    maximum_failures_per_source: int = 25,
) -> HighScoreFailureReport:
    """Find high-score top-k misses without comparing unlike score scales.

    Thresholds are empirical quantiles of positive top scores among clicked
    impressions and are calculated independently for each recommendation
    source. The scores are ranking signals, not calibrated probabilities.
    """

    validated_k, validated_quantile, validated_maximum = _validate_configuration(
        k=k,
        score_quantile=score_quantile,
        maximum_failures_per_source=maximum_failures_per_source,
    )
    example_list = list(examples)

    if not example_list:
        raise FailureAnalysisError("At least one scored ranking example is required.")

    if any(not isinstance(example, ScoredRankingExample) for example in example_list):
        raise FailureAnalysisError("examples must contain ScoredRankingExample instances.")

    impression_ids = [example.impression_id for example in example_list]

    if len(impression_ids) != len(set(impression_ids)):
        raise FailureAnalysisError("impression_id values must be unique.")

    evaluated_examples = [example for example in example_list if example.relevant_items]

    if not evaluated_examples:
        raise FailureAnalysisError(
            "At least one scored ranking example with a relevant item is required."
        )

    eligible_examples = [
        example
        for example in evaluated_examples
        if example.ranked_scores and example.ranked_scores[0] > 0.0
    ]

    if not eligible_examples:
        raise FailureAnalysisError(
            "At least one evaluated example with a positive top score is required."
        )

    scores_by_source: dict[str, list[float]] = {}

    for example in eligible_examples:
        scores_by_source.setdefault(example.source, []).append(example.ranked_scores[0])

    thresholds = {
        source: float(np.quantile(scores, validated_quantile))
        for source, scores in scores_by_source.items()
    }
    top_k_miss_examples = [
        example for example in evaluated_examples if _is_top_k_miss(example, k=validated_k)
    ]
    high_score_by_source: dict[str, list[ScoredRankingExample]] = {
        source: [] for source in thresholds
    }

    for example in top_k_miss_examples:
        if not example.ranked_scores or example.ranked_scores[0] <= 0.0:
            continue

        threshold = thresholds.get(example.source)

        if threshold is not None and example.ranked_scores[0] >= threshold:
            high_score_by_source[example.source].append(example)

    failures: list[HighScoreFailure] = []
    source_summaries: list[SourceScoreThreshold] = []

    for source in sorted(thresholds):
        source_failures = sorted(
            high_score_by_source[source],
            key=lambda example: (
                -example.ranked_scores[0],
                -(_score_margin(example.ranked_scores) or 0.0),
                example.impression_id,
            ),
        )
        retained = source_failures[:validated_maximum]

        for example in retained:
            failures.append(
                HighScoreFailure(
                    impression_id=example.impression_id,
                    source=example.source,
                    history_length=example.history_length,
                    candidate_count=example.candidate_count,
                    relevant_items=tuple(sorted(example.relevant_items)),
                    ranked_items=example.ranked_items[:validated_k],
                    ranked_scores=example.ranked_scores[:validated_k],
                    top_score=example.ranked_scores[0],
                    score_margin=_score_margin(example.ranked_scores),
                    score_threshold=thresholds[source],
                )
            )

        source_summaries.append(
            SourceScoreThreshold(
                source=source,
                score_threshold=thresholds[source],
                eligible_impressions=sum(example.source == source for example in eligible_examples),
                top_k_misses=sum(
                    example.source == source
                    for example in top_k_miss_examples
                    if example.ranked_scores and example.ranked_scores[0] > 0.0
                ),
                high_score_misses=len(source_failures),
                retained_failures=len(retained),
            )
        )

    return HighScoreFailureReport(
        k=validated_k,
        score_quantile=validated_quantile,
        maximum_failures_per_source=validated_maximum,
        total_impressions=len(example_list),
        evaluated_impressions=len(evaluated_examples),
        skipped_no_click_impressions=len(example_list) - len(evaluated_examples),
        score_eligible_impressions=len(eligible_examples),
        non_positive_or_empty_score_impressions=(len(evaluated_examples) - len(eligible_examples)),
        top_k_misses=len(top_k_miss_examples),
        high_score_misses=sum(len(values) for values in high_score_by_source.values()),
        source_thresholds=tuple(source_summaries),
        failures=tuple(failures),
    )
