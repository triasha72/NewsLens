"""Paired bootstrap comparison of two ranking systems.

Both systems must be evaluated on the same impressions and relevance labels.
Each bootstrap replicate resamples aligned impression pairs once and therefore
preserves the within-impression dependence between model results.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from statistics import fmean

import numpy as np

from .evaluator import RankingExample
from .metrics import (
    RankingMetricError,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


class PairedComparisonError(ValueError):
    """Raised when an aligned ranking comparison cannot be completed."""


@dataclass(frozen=True)
class PairedMetricInterval:
    """Point estimates and an interval for candidate minus baseline."""

    baseline_estimate: float
    candidate_estimate: float
    point_difference: float
    lower_bound: float
    upper_bound: float
    standard_error: float

    @property
    def excludes_zero(self) -> bool:
        """Return whether the percentile interval excludes zero."""

        return self.lower_bound > 0.0 or self.upper_bound < 0.0

    def to_dict(self) -> dict[str, float | bool]:
        """Return a JSON-compatible representation."""

        return {
            "baseline_estimate": self.baseline_estimate,
            "candidate_estimate": self.candidate_estimate,
            "point_difference": self.point_difference,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "standard_error": self.standard_error,
            "excludes_zero": self.excludes_zero,
        }


@dataclass(frozen=True)
class PairedBootstrapComparisonReport:
    """Paired percentile-bootstrap differences for ranking metrics."""

    baseline_model_name: str
    candidate_model_name: str
    k: int
    confidence_level: float
    bootstrap_samples: int
    random_seed: int
    total_impressions: int
    evaluated_impressions: int
    skipped_no_click_impressions: int
    ndcg_at_k: PairedMetricInterval
    mrr_at_k: PairedMetricInterval
    recall_at_k: PairedMetricInterval
    hit_rate_at_k: PairedMetricInterval

    @property
    def evaluated_fraction(self) -> float:
        """Return the fraction included in the paired resampling population."""

        return self.evaluated_impressions / self.total_impressions

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "method": "paired_nonparametric_percentile_bootstrap",
            "resampling_unit": "aligned_evaluated_impression_pair",
            "difference_direction": "candidate_minus_baseline",
            "baseline_model_name": self.baseline_model_name,
            "candidate_model_name": self.candidate_model_name,
            "k": self.k,
            "confidence_level": self.confidence_level,
            "bootstrap_samples": self.bootstrap_samples,
            "random_seed": self.random_seed,
            "total_impressions": self.total_impressions,
            "evaluated_impressions": self.evaluated_impressions,
            "skipped_no_click_impressions": self.skipped_no_click_impressions,
            "evaluated_fraction": self.evaluated_fraction,
            "metrics": {
                "ndcg_at_k": self.ndcg_at_k.to_dict(),
                "mrr_at_k": self.mrr_at_k.to_dict(),
                "recall_at_k": self.recall_at_k.to_dict(),
                "hit_rate_at_k": self.hit_rate_at_k.to_dict(),
            },
        }


def _validate_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PairedComparisonError(f"{name} must be a positive integer.")

    return value


def _validate_configuration(
    *,
    baseline_model_name: object,
    candidate_model_name: object,
    k: object,
    bootstrap_samples: object,
    confidence_level: object,
    random_seed: object,
) -> tuple[str, str, int, int, float, int]:
    baseline_name = str(baseline_model_name).strip()
    candidate_name = str(candidate_model_name).strip()

    if not baseline_name:
        raise PairedComparisonError("baseline_model_name must not be empty.")

    if not candidate_name:
        raise PairedComparisonError("candidate_model_name must not be empty.")

    if baseline_name == candidate_name:
        raise PairedComparisonError("Model names must be distinct.")

    validated_k = _validate_positive_integer(k, name="k")
    validated_samples = _validate_positive_integer(
        bootstrap_samples,
        name="bootstrap_samples",
    )

    if validated_samples < 2:
        raise PairedComparisonError("bootstrap_samples must be at least 2.")

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise PairedComparisonError("confidence_level must be between 0 and 1.")

    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise PairedComparisonError("random_seed must be a non-negative integer.")

    return (
        baseline_name,
        candidate_name,
        validated_k,
        validated_samples,
        float(confidence_level),
        random_seed,
    )


def _validate_examples(
    examples: Iterable[RankingExample],
    *,
    label: str,
) -> list[RankingExample]:
    example_list = list(examples)

    if not example_list:
        raise PairedComparisonError(f"At least one {label} ranking example is required.")

    if any(not isinstance(example, RankingExample) for example in example_list):
        raise PairedComparisonError(f"{label} examples must contain RankingExample instances.")

    impression_ids = [example.impression_id for example in example_list]

    if len(impression_ids) != len(set(impression_ids)):
        raise PairedComparisonError(f"{label} impression_id values must be unique.")

    return example_list


def _metric_row(example: RankingExample, *, k: int) -> tuple[float, float, float, float]:
    return (
        ndcg_at_k(example.ranked_items, example.relevant_items, k=k),
        reciprocal_rank_at_k(example.ranked_items, example.relevant_items, k=k),
        recall_at_k(example.ranked_items, example.relevant_items, k=k),
        hit_rate_at_k(example.ranked_items, example.relevant_items, k=k),
    )


def _paired_interval(
    *,
    baseline_estimate: float,
    candidate_estimate: float,
    bootstrap_differences: np.ndarray,
    confidence_level: float,
) -> PairedMetricInterval:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        bootstrap_differences,
        [alpha / 2.0, 1.0 - (alpha / 2.0)],
    )
    point_difference = candidate_estimate - baseline_estimate

    return PairedMetricInterval(
        baseline_estimate=float(baseline_estimate),
        candidate_estimate=float(candidate_estimate),
        point_difference=float(point_difference),
        lower_bound=float(lower),
        upper_bound=float(upper),
        standard_error=float(np.std(bootstrap_differences, ddof=1)),
    )


def paired_bootstrap_ranking_comparison(
    baseline_examples: Iterable[RankingExample],
    candidate_examples: Iterable[RankingExample],
    *,
    baseline_model_name: str,
    candidate_model_name: str,
    k: int,
    bootstrap_samples: int = 1_000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> PairedBootstrapComparisonReport:
    """Compare aligned ranking systems using paired impression resampling.

    Metric differences are always candidate minus baseline. Catalog coverage is
    intentionally excluded because it is a set-level statistic rather than an
    impression-level mean that can be paired row by row.
    """

    (
        validated_baseline_name,
        validated_candidate_name,
        validated_k,
        validated_samples,
        validated_confidence,
        validated_seed,
    ) = _validate_configuration(
        baseline_model_name=baseline_model_name,
        candidate_model_name=candidate_model_name,
        k=k,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    baseline_list = _validate_examples(baseline_examples, label="baseline")
    candidate_list = _validate_examples(candidate_examples, label="candidate")
    baseline_by_id = {example.impression_id: example for example in baseline_list}
    candidate_by_id = {example.impression_id: example for example in candidate_list}

    if baseline_by_id.keys() != candidate_by_id.keys():
        missing_candidate = sorted(baseline_by_id.keys() - candidate_by_id.keys())
        missing_baseline = sorted(candidate_by_id.keys() - baseline_by_id.keys())
        details: list[str] = []

        if missing_candidate:
            details.append(f"missing candidate IDs: {', '.join(missing_candidate[:3])}")

        if missing_baseline:
            details.append(f"missing baseline IDs: {', '.join(missing_baseline[:3])}")

        raise PairedComparisonError(
            "Baseline and candidate impression IDs must match; " + "; ".join(details) + "."
        )

    aligned_pairs: list[tuple[RankingExample, RankingExample]] = []

    for impression_id in sorted(baseline_by_id):
        baseline = baseline_by_id[impression_id]
        candidate = candidate_by_id[impression_id]

        if baseline.relevant_items != candidate.relevant_items:
            raise PairedComparisonError(f"Relevant items differ for impression '{impression_id}'.")

        aligned_pairs.append((baseline, candidate))

    evaluated_pairs = [pair for pair in aligned_pairs if pair[0].relevant_items]

    if not evaluated_pairs:
        raise PairedComparisonError(
            "At least one aligned impression with a relevant item is required."
        )

    baseline_rows: list[tuple[float, float, float, float]] = []
    candidate_rows: list[tuple[float, float, float, float]] = []

    try:
        for baseline, candidate in evaluated_pairs:
            baseline_rows.append(_metric_row(baseline, k=validated_k))
            candidate_rows.append(_metric_row(candidate, k=validated_k))
    except RankingMetricError as error:
        raise PairedComparisonError(str(error)) from error

    baseline_matrix = np.asarray(baseline_rows, dtype=np.float64)
    candidate_matrix = np.asarray(candidate_rows, dtype=np.float64)
    paired_difference_matrix = candidate_matrix - baseline_matrix

    if not np.isfinite(paired_difference_matrix).all():
        raise PairedComparisonError("Paired metric differences must be finite.")

    baseline_estimates = tuple(
        fmean(row[column] for row in baseline_rows) for column in range(baseline_matrix.shape[1])
    )
    candidate_estimates = tuple(
        fmean(row[column] for row in candidate_rows) for column in range(candidate_matrix.shape[1])
    )
    random_generator = np.random.default_rng(validated_seed)
    evaluated_count = len(evaluated_pairs)
    bootstrap_differences = np.empty(
        (validated_samples, paired_difference_matrix.shape[1]),
        dtype=np.float64,
    )

    for sample_index in range(validated_samples):
        sampled_indices = random_generator.integers(
            0,
            evaluated_count,
            size=evaluated_count,
        )
        bootstrap_differences[sample_index] = paired_difference_matrix[sampled_indices].mean(axis=0)

    intervals = tuple(
        _paired_interval(
            baseline_estimate=baseline_estimates[column],
            candidate_estimate=candidate_estimates[column],
            bootstrap_differences=bootstrap_differences[:, column],
            confidence_level=validated_confidence,
        )
        for column in range(paired_difference_matrix.shape[1])
    )

    if any(
        not isfinite(value)
        for interval in intervals
        for value in (
            interval.baseline_estimate,
            interval.candidate_estimate,
            interval.point_difference,
            interval.lower_bound,
            interval.upper_bound,
            interval.standard_error,
        )
    ):
        raise PairedComparisonError("Paired comparison results must be finite.")

    return PairedBootstrapComparisonReport(
        baseline_model_name=validated_baseline_name,
        candidate_model_name=validated_candidate_name,
        k=validated_k,
        confidence_level=validated_confidence,
        bootstrap_samples=validated_samples,
        random_seed=validated_seed,
        total_impressions=len(aligned_pairs),
        evaluated_impressions=evaluated_count,
        skipped_no_click_impressions=len(aligned_pairs) - evaluated_count,
        ndcg_at_k=intervals[0],
        mrr_at_k=intervals[1],
        recall_at_k=intervals[2],
        hit_rate_at_k=intervals[3],
    )
