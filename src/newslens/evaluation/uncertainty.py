"""Deterministic bootstrap uncertainty for offline ranking metrics.

The resampling unit is one evaluated impression. Impressions without a clicked,
relevant article are excluded from quality metrics in the same way as the shared
ranking evaluator, but they remain visible in the accounting fields.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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


class BootstrapUncertaintyError(ValueError):
    """Raised when a bootstrap uncertainty analysis cannot be completed."""


@dataclass(frozen=True)
class BootstrapInterval:
    """A point estimate and percentile-bootstrap uncertainty summary."""

    point_estimate: float
    lower_bound: float
    upper_bound: float
    standard_error: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-compatible representation."""

        return {
            "point_estimate": self.point_estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "standard_error": self.standard_error,
        }


@dataclass(frozen=True)
class BootstrapUncertaintyReport:
    """Bootstrap intervals plus reproducibility and denominator metadata."""

    k: int
    confidence_level: float
    bootstrap_samples: int
    random_seed: int
    total_impressions: int
    evaluated_impressions: int
    skipped_no_click_impressions: int
    ndcg_at_k: BootstrapInterval
    mrr_at_k: BootstrapInterval
    recall_at_k: BootstrapInterval
    hit_rate_at_k: BootstrapInterval

    @property
    def evaluated_fraction(self) -> float:
        """Return the fraction included in the resampling population."""

        return self.evaluated_impressions / self.total_impressions

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "method": "nonparametric_percentile_bootstrap",
            "resampling_unit": "evaluated_impression",
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
        raise BootstrapUncertaintyError(f"{name} must be a positive integer.")

    return value


def _validate_configuration(
    *,
    k: object,
    bootstrap_samples: object,
    confidence_level: object,
    random_seed: object,
) -> tuple[int, int, float, int]:
    validated_k = _validate_positive_integer(k, name="k")
    validated_samples = _validate_positive_integer(
        bootstrap_samples,
        name="bootstrap_samples",
    )

    if validated_samples < 2:
        raise BootstrapUncertaintyError("bootstrap_samples must be at least 2.")

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise BootstrapUncertaintyError("confidence_level must be between 0 and 1.")

    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise BootstrapUncertaintyError("random_seed must be a non-negative integer.")

    return (
        validated_k,
        validated_samples,
        float(confidence_level),
        random_seed,
    )


def _interval(
    *,
    point_estimate: float,
    bootstrap_estimates: np.ndarray,
    confidence_level: float,
) -> BootstrapInterval:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        bootstrap_estimates,
        [alpha / 2.0, 1.0 - (alpha / 2.0)],
    )

    return BootstrapInterval(
        point_estimate=float(point_estimate),
        lower_bound=float(lower),
        upper_bound=float(upper),
        standard_error=float(np.std(bootstrap_estimates, ddof=1)),
    )


def bootstrap_ranking_uncertainty(
    examples: Iterable[RankingExample],
    *,
    k: int,
    bootstrap_samples: int = 1_000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapUncertaintyReport:
    """Estimate percentile-bootstrap intervals for impression-level metrics.

    Each replicate samples the evaluated impressions with replacement and keeps
    the original evaluated sample size. Catalog coverage is intentionally not
    included because it is a set-level statistic rather than the mean of an
    impression-level quantity.
    """

    (
        validated_k,
        validated_samples,
        validated_confidence,
        validated_seed,
    ) = _validate_configuration(
        k=k,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )

    example_list = list(examples)

    if not example_list:
        raise BootstrapUncertaintyError("At least one ranking example is required.")

    if any(not isinstance(example, RankingExample) for example in example_list):
        raise BootstrapUncertaintyError("examples must contain RankingExample instances.")

    impression_ids = [example.impression_id for example in example_list]

    if len(impression_ids) != len(set(impression_ids)):
        raise BootstrapUncertaintyError("impression_id values must be unique.")

    evaluated_examples = [example for example in example_list if example.relevant_items]

    if not evaluated_examples:
        raise BootstrapUncertaintyError(
            "At least one ranking example with a relevant item is required."
        )

    metric_rows: list[tuple[float, float, float, float]] = []

    try:
        for example in evaluated_examples:
            metric_rows.append(
                (
                    ndcg_at_k(
                        example.ranked_items,
                        example.relevant_items,
                        k=validated_k,
                    ),
                    reciprocal_rank_at_k(
                        example.ranked_items,
                        example.relevant_items,
                        k=validated_k,
                    ),
                    recall_at_k(
                        example.ranked_items,
                        example.relevant_items,
                        k=validated_k,
                    ),
                    hit_rate_at_k(
                        example.ranked_items,
                        example.relevant_items,
                        k=validated_k,
                    ),
                )
            )
    except RankingMetricError as error:
        raise BootstrapUncertaintyError(str(error)) from error

    metric_matrix = np.asarray(metric_rows, dtype=np.float64)
    point_estimates = tuple(
        fmean(row[column] for row in metric_rows) for column in range(metric_matrix.shape[1])
    )
    bootstrap_estimates = np.empty((validated_samples, metric_matrix.shape[1]), dtype=np.float64)
    random_generator = np.random.default_rng(validated_seed)
    evaluated_count = len(evaluated_examples)

    for sample_index in range(validated_samples):
        sampled_indices = random_generator.integers(
            0,
            evaluated_count,
            size=evaluated_count,
        )
        bootstrap_estimates[sample_index] = metric_matrix[sampled_indices].mean(axis=0)

    intervals = tuple(
        _interval(
            point_estimate=point_estimates[column],
            bootstrap_estimates=bootstrap_estimates[:, column],
            confidence_level=validated_confidence,
        )
        for column in range(metric_matrix.shape[1])
    )

    return BootstrapUncertaintyReport(
        k=validated_k,
        confidence_level=validated_confidence,
        bootstrap_samples=validated_samples,
        random_seed=validated_seed,
        total_impressions=len(example_list),
        evaluated_impressions=evaluated_count,
        skipped_no_click_impressions=len(example_list) - evaluated_count,
        ndcg_at_k=intervals[0],
        mrr_at_k=intervals[1],
        recall_at_k=intervals[2],
        hit_rate_at_k=intervals[3],
    )
