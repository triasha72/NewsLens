"""Chronological evaluation of the training-only popularity baseline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from ..data import MindDataValidationError, parse_impressions
from ..models import (
    PopularityModelError,
    PopularityRecommender,
)
from .evaluator import (
    RankingEvaluationError,
    RankingEvaluationResult,
    RankingExample,
    evaluate_rankings,
)
from .split import (
    ChronologicalSplitError,
    chronological_train_validation_split,
)


class PopularityEvaluationError(ValueError):
    """Raised when popularity evaluation cannot be completed."""


@dataclass(frozen=True)
class PopularityEvaluationReport:
    """Results and accounting for one popularity evaluation."""

    model_name: str
    training_records: int
    validation_records: int
    requested_validation_fraction: float
    actual_validation_fraction: float
    cutoff_timestamp: str
    candidate_occurrences: int
    unseen_candidate_occurrences: int
    unseen_validation_impressions: int
    metrics: RankingEvaluationResult

    @property
    def unseen_candidate_fraction(self) -> float:
        """Return the fraction unseen during model fitting."""

        if self.candidate_occurrences == 0:
            return 0.0

        return self.unseen_candidate_occurrences / self.candidate_occurrences

    def to_dict(
        self,
    ) -> dict[
        str,
        str | int | float | dict[str, int | float],
    ]:
        """Return a JSON-compatible evaluation report."""

        return {
            "model_name": self.model_name,
            "training_records": self.training_records,
            "validation_records": self.validation_records,
            "requested_validation_fraction": (self.requested_validation_fraction),
            "actual_validation_fraction": (self.actual_validation_fraction),
            "cutoff_timestamp": self.cutoff_timestamp,
            "candidate_occurrences": self.candidate_occurrences,
            "unseen_candidate_occurrences": (self.unseen_candidate_occurrences),
            "unseen_candidate_fraction": (self.unseen_candidate_fraction),
            "unseen_validation_impressions": (self.unseen_validation_impressions),
            "metrics": self.metrics.to_dict(),
        }


def _prepare_catalog(
    catalog_items: Iterable[str],
) -> frozenset[str]:
    if isinstance(catalog_items, str):
        raise PopularityEvaluationError("catalog_items must be an iterable of article IDs.")

    catalog = frozenset(catalog_items)

    if not catalog:
        raise PopularityEvaluationError("At least one catalog item is required.")

    if any(not isinstance(item, str) or not item.strip() for item in catalog):
        raise PopularityEvaluationError("catalog_items must contain non-empty string IDs.")

    return catalog


def _build_ranking_examples(
    validation_behaviors: pd.DataFrame,
    model: PopularityRecommender,
    catalog: frozenset[str],
    *,
    k: int,
) -> tuple[
    list[RankingExample],
    int,
    int,
    int,
]:
    examples: list[RankingExample] = []
    candidate_occurrences = 0
    unseen_candidate_occurrences = 0
    unseen_validation_impressions = 0

    for row in validation_behaviors.itertuples(index=False):
        impression_id = str(row.impression_id)

        try:
            parsed_impressions = parse_impressions(str(row.impressions))
        except MindDataValidationError as error:
            raise PopularityEvaluationError(
                f"Invalid validation impression '{impression_id}': {error}"
            ) from error

        candidate_ids = [news_id for news_id, _ in parsed_impressions]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise PopularityEvaluationError(
                f"Validation impression '{impression_id}' contains duplicate candidate IDs."
            )

        unknown_candidates = set(candidate_ids) - catalog

        if unknown_candidates:
            unknown_preview = ", ".join(sorted(unknown_candidates)[:3])
            raise PopularityEvaluationError(
                f"Validation impression '{impression_id}' "
                "contains candidates missing from the catalog: "
                f"{unknown_preview}."
            )

        relevant_items = frozenset(news_id for news_id, label in parsed_impressions if label == 1)

        unseen_candidates = [
            news_id for news_id in candidate_ids if model.statistics(news_id).exposures == 0
        ]

        candidate_occurrences += len(candidate_ids)
        unseen_candidate_occurrences += len(unseen_candidates)

        if unseen_candidates:
            unseen_validation_impressions += 1

        ranked_items = tuple(
            model.rank_candidates(
                candidate_ids,
                top_k=k,
            )
        )

        examples.append(
            RankingExample(
                impression_id=impression_id,
                ranked_items=ranked_items,
                relevant_items=relevant_items,
            )
        )

    return (
        examples,
        candidate_occurrences,
        unseen_candidate_occurrences,
        unseen_validation_impressions,
    )


def evaluate_popularity_baseline(
    behaviors: pd.DataFrame,
    catalog_items: Iterable[str],
    *,
    validation_fraction: float = 0.20,
    k: int = 10,
) -> PopularityEvaluationReport:
    """Fit on chronological training data and evaluate validation."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise PopularityEvaluationError("k must be a positive integer.")

    required_columns = {
        "impression_id",
        "timestamp",
        "impressions",
    }
    missing_columns = required_columns.difference(behaviors.columns)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise PopularityEvaluationError(f"Missing required behavior columns: {formatted}.")

    catalog = _prepare_catalog(catalog_items)

    try:
        split = chronological_train_validation_split(
            behaviors,
            validation_fraction=validation_fraction,
        )
        model = PopularityRecommender().fit(split.train)

    except (
        ChronologicalSplitError,
        PopularityModelError,
    ) as error:
        raise PopularityEvaluationError(str(error)) from error

    (
        examples,
        candidate_occurrences,
        unseen_candidate_occurrences,
        unseen_validation_impressions,
    ) = _build_ranking_examples(
        split.validation,
        model,
        catalog,
        k=k,
    )

    try:
        metrics = evaluate_rankings(
            examples,
            catalog,
            k=k,
        )
    except RankingEvaluationError as error:
        raise PopularityEvaluationError(str(error)) from error

    return PopularityEvaluationReport(
        model_name="training_click_count_popularity",
        training_records=len(split.train),
        validation_records=len(split.validation),
        requested_validation_fraction=validation_fraction,
        actual_validation_fraction=(split.actual_validation_fraction),
        cutoff_timestamp=split.cutoff.isoformat(),
        candidate_occurrences=candidate_occurrences,
        unseen_candidate_occurrences=(unseen_candidate_occurrences),
        unseen_validation_impressions=(unseen_validation_impressions),
        metrics=metrics,
    )
