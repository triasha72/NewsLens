"""Chronological evaluation of the TF-IDF history recommender."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data import MindDataValidationError, parse_impressions
from ..models import (
    ColdStartUserError,
    ContentBasedRecommender,
    ContentModelError,
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


class ContentEvaluationError(ValueError):
    """Raised when content-based evaluation cannot be completed."""


@dataclass(frozen=True)
class ContentEvaluationReport:
    """Results and accounting for one content-based evaluation."""

    model_name: str
    training_records: int
    validation_records: int
    requested_validation_fraction: float
    actual_validation_fraction: float
    cutoff_timestamp: str
    vocabulary_article_count: int
    indexed_article_count: int
    vocabulary_size: int
    candidate_occurrences: int
    content_ranked_impressions: int
    cold_start_impressions: int
    empty_history_impressions: int
    unknown_history_impressions: int
    zero_profile_impressions: int
    zero_signal_impressions: int
    metrics: RankingEvaluationResult

    @property
    def content_ranked_fraction(self) -> float:
        """Return the fraction receiving a meaningful content ranking."""

        return self.content_ranked_impressions / self.validation_records

    @property
    def cold_start_fraction(self) -> float:
        """Return the fraction without a usable content profile."""

        return self.cold_start_impressions / self.validation_records

    @property
    def zero_signal_fraction(self) -> float:
        """Return the fraction whose candidates have no positive similarity."""

        return self.zero_signal_impressions / self.validation_records

    @property
    def abstained_impressions(self) -> int:
        """Return the number for which content produces no ranking."""

        return self.cold_start_impressions + self.zero_signal_impressions

    @property
    def abstained_fraction(self) -> float:
        """Return the fraction for which content produces no ranking."""

        return self.abstained_impressions / self.validation_records

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
            "requested_validation_fraction": self.requested_validation_fraction,
            "actual_validation_fraction": self.actual_validation_fraction,
            "cutoff_timestamp": self.cutoff_timestamp,
            "vocabulary_article_count": self.vocabulary_article_count,
            "indexed_article_count": self.indexed_article_count,
            "vocabulary_size": self.vocabulary_size,
            "candidate_occurrences": self.candidate_occurrences,
            "content_ranked_impressions": self.content_ranked_impressions,
            "content_ranked_fraction": self.content_ranked_fraction,
            "cold_start_impressions": self.cold_start_impressions,
            "cold_start_fraction": self.cold_start_fraction,
            "empty_history_impressions": self.empty_history_impressions,
            "unknown_history_impressions": self.unknown_history_impressions,
            "zero_profile_impressions": self.zero_profile_impressions,
            "zero_signal_impressions": self.zero_signal_impressions,
            "zero_signal_fraction": self.zero_signal_fraction,
            "abstained_impressions": self.abstained_impressions,
            "abstained_fraction": self.abstained_fraction,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class _ContentExampleBuildResult:
    examples: list[RankingExample]
    candidate_occurrences: int
    content_ranked_impressions: int
    cold_start_impressions: int
    empty_history_impressions: int
    unknown_history_impressions: int
    zero_profile_impressions: int
    zero_signal_impressions: int


def _prepare_catalog(news: pd.DataFrame) -> frozenset[str]:
    required_columns = {"news_id", "title"}
    missing_columns = required_columns.difference(news.columns)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise ContentEvaluationError(f"Missing required article columns: {formatted}.")

    if news.empty:
        raise ContentEvaluationError("At least one article is required.")

    news_ids = news["news_id"].fillna("").astype(str).str.strip()

    if news_ids.eq("").any():
        raise ContentEvaluationError("Article identifiers cannot be empty.")

    duplicate_ids = news_ids[news_ids.duplicated(keep=False)].unique()

    if len(duplicate_ids) > 0:
        raise ContentEvaluationError(
            f"Duplicate article identifiers found: {', '.join(duplicate_ids)}"
        )

    return frozenset(news_ids)


def _parse_history(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    return str(value).split()


def _training_vocabulary_news_ids(
    training_behaviors: pd.DataFrame,
    catalog: frozenset[str],
) -> frozenset[str]:
    referenced_ids: set[str] = set()

    for row in training_behaviors.itertuples(index=False):
        impression_id = str(row.impression_id)
        history_ids = _parse_history(row.history)

        try:
            parsed_impressions = parse_impressions(str(row.impressions))
        except MindDataValidationError as error:
            raise ContentEvaluationError(
                f"Invalid training impression '{impression_id}': {error}"
            ) from error

        candidate_ids = [news_id for news_id, _ in parsed_impressions]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ContentEvaluationError(
                f"Training impression '{impression_id}' contains duplicate candidate IDs."
            )

        referenced_ids.update(history_ids)
        referenced_ids.update(candidate_ids)

    unknown_ids = referenced_ids - catalog

    if unknown_ids:
        unknown_preview = ", ".join(sorted(unknown_ids)[:3])
        raise ContentEvaluationError(
            f"Training interactions reference articles missing from the catalog: {unknown_preview}."
        )

    if not referenced_ids:
        raise ContentEvaluationError(
            "Training interactions did not reference any catalog articles."
        )

    return frozenset(referenced_ids)


def _build_ranking_examples(
    validation_behaviors: pd.DataFrame,
    model: ContentBasedRecommender,
    catalog: frozenset[str],
    *,
    k: int,
) -> _ContentExampleBuildResult:
    examples: list[RankingExample] = []
    candidate_occurrences = 0
    content_ranked_impressions = 0
    cold_start_impressions = 0
    empty_history_impressions = 0
    unknown_history_impressions = 0
    zero_profile_impressions = 0
    zero_signal_impressions = 0

    for row in validation_behaviors.itertuples(index=False):
        impression_id = str(row.impression_id)
        history_ids = _parse_history(row.history)

        try:
            parsed_impressions = parse_impressions(str(row.impressions))
        except MindDataValidationError as error:
            raise ContentEvaluationError(
                f"Invalid validation impression '{impression_id}': {error}"
            ) from error

        candidate_ids = [news_id for news_id, _ in parsed_impressions]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ContentEvaluationError(
                f"Validation impression '{impression_id}' contains duplicate candidate IDs."
            )

        unknown_candidates = set(candidate_ids) - catalog

        if unknown_candidates:
            unknown_preview = ", ".join(sorted(unknown_candidates)[:3])
            raise ContentEvaluationError(
                f"Validation impression '{impression_id}' "
                "contains candidates missing from the catalog: "
                f"{unknown_preview}."
            )

        relevant_items = frozenset(news_id for news_id, label in parsed_impressions if label == 1)
        candidate_occurrences += len(candidate_ids)

        if not history_ids:
            empty_history_impressions += 1
            cold_start_impressions += 1
            ranked_items: tuple[str, ...] = ()
        elif not any(news_id in catalog for news_id in history_ids):
            unknown_history_impressions += 1
            cold_start_impressions += 1
            ranked_items = ()
        else:
            try:
                recommendations = model.recommend(
                    history_ids,
                    candidate_news_ids=candidate_ids,
                    top_k=k,
                    exclude_history=False,
                )
            except ColdStartUserError:
                zero_profile_impressions += 1
                cold_start_impressions += 1
                ranked_items = ()
            else:
                if recommendations and any(
                    recommendation.score > 0.0 for recommendation in recommendations
                ):
                    content_ranked_impressions += 1
                    ranked_items = tuple(
                        recommendation.news_id for recommendation in recommendations
                    )
                else:
                    zero_signal_impressions += 1
                    ranked_items = ()

        examples.append(
            RankingExample(
                impression_id=impression_id,
                ranked_items=ranked_items,
                relevant_items=relevant_items,
            )
        )

    return _ContentExampleBuildResult(
        examples=examples,
        candidate_occurrences=candidate_occurrences,
        content_ranked_impressions=content_ranked_impressions,
        cold_start_impressions=cold_start_impressions,
        empty_history_impressions=empty_history_impressions,
        unknown_history_impressions=unknown_history_impressions,
        zero_profile_impressions=zero_profile_impressions,
        zero_signal_impressions=zero_signal_impressions,
    )


def evaluate_content_baseline(
    news: pd.DataFrame,
    behaviors: pd.DataFrame,
    *,
    validation_fraction: float = 0.20,
    k: int = 10,
    max_features: int = 50_000,
) -> ContentEvaluationReport:
    """Fit on training-referenced text and evaluate later interactions."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ContentEvaluationError("k must be a positive integer.")

    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features <= 0:
        raise ContentEvaluationError("max_features must be a positive integer.")

    required_columns = {
        "impression_id",
        "timestamp",
        "history",
        "impressions",
    }
    missing_columns = required_columns.difference(behaviors.columns)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise ContentEvaluationError(f"Missing required behavior columns: {formatted}.")

    catalog = _prepare_catalog(news)

    try:
        split = chronological_train_validation_split(
            behaviors,
            validation_fraction=validation_fraction,
        )
    except ChronologicalSplitError as error:
        raise ContentEvaluationError(str(error)) from error

    vocabulary_news_ids = _training_vocabulary_news_ids(
        split.train,
        catalog,
    )

    try:
        model = ContentBasedRecommender(max_features=max_features).fit(
            news,
            vocabulary_news_ids=vocabulary_news_ids,
        )
    except ContentModelError as error:
        raise ContentEvaluationError(str(error)) from error

    build_result = _build_ranking_examples(
        split.validation,
        model,
        catalog,
        k=k,
    )

    try:
        metrics = evaluate_rankings(
            build_result.examples,
            catalog,
            k=k,
        )
    except RankingEvaluationError as error:
        raise ContentEvaluationError(str(error)) from error

    return ContentEvaluationReport(
        model_name="tfidf_history_content",
        training_records=len(split.train),
        validation_records=len(split.validation),
        requested_validation_fraction=validation_fraction,
        actual_validation_fraction=split.actual_validation_fraction,
        cutoff_timestamp=split.cutoff.isoformat(),
        vocabulary_article_count=model.vocabulary_article_count,
        indexed_article_count=model.indexed_article_count,
        vocabulary_size=model.vocabulary_size,
        candidate_occurrences=build_result.candidate_occurrences,
        content_ranked_impressions=build_result.content_ranked_impressions,
        cold_start_impressions=build_result.cold_start_impressions,
        empty_history_impressions=build_result.empty_history_impressions,
        unknown_history_impressions=build_result.unknown_history_impressions,
        zero_profile_impressions=build_result.zero_profile_impressions,
        zero_signal_impressions=build_result.zero_signal_impressions,
        metrics=metrics,
    )
