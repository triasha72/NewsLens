"""Chronological evaluation of content ranking with popularity fallback."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data import MindDataValidationError, parse_impressions
from ..models import (
    ColdStartUserError,
    ContentBasedRecommender,
    ContentModelError,
    ContentPopularityFallbackRecommender,
    FallbackModelError,
    PopularityModelError,
    PopularityRecommender,
    RecommendationSource,
)
from .content import (
    ContentEvaluationError,
    _parse_history,
    _prepare_catalog,
    _training_vocabulary_news_ids,
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


class FallbackEvaluationError(ValueError):
    """Raised when fallback evaluation cannot be completed."""


@dataclass(frozen=True)
class FallbackEvaluationReport:
    """Results and routing accounting for one fallback evaluation."""

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
    content_routed_impressions: int
    popularity_routed_impressions: int
    empty_history_fallback_impressions: int
    unknown_history_fallback_impressions: int
    zero_profile_fallback_impressions: int
    zero_signal_fallback_impressions: int
    metrics: RankingEvaluationResult

    @property
    def content_routed_fraction(self) -> float:
        """Return the fraction ranked by content similarity."""

        return self.content_routed_impressions / self.validation_records

    @property
    def popularity_routed_fraction(self) -> float:
        """Return the fraction routed to popularity."""

        return self.popularity_routed_impressions / self.validation_records

    @property
    def recovered_fallback_impressions(self) -> int:
        """Return fallback-routed impressions receiving a non-empty ranking."""

        return self.popularity_routed_impressions - self.metrics.empty_ranking_impressions

    @property
    def fallback_recovery_fraction(self) -> float:
        """Return the fraction of fallback routes producing a ranking."""

        if self.popularity_routed_impressions == 0:
            return 0.0

        return self.recovered_fallback_impressions / self.popularity_routed_impressions

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
            "content_routed_impressions": self.content_routed_impressions,
            "content_routed_fraction": self.content_routed_fraction,
            "popularity_routed_impressions": self.popularity_routed_impressions,
            "popularity_routed_fraction": self.popularity_routed_fraction,
            "empty_history_fallback_impressions": (self.empty_history_fallback_impressions),
            "unknown_history_fallback_impressions": (self.unknown_history_fallback_impressions),
            "zero_profile_fallback_impressions": (self.zero_profile_fallback_impressions),
            "zero_signal_fallback_impressions": self.zero_signal_fallback_impressions,
            "recovered_fallback_impressions": self.recovered_fallback_impressions,
            "fallback_recovery_fraction": self.fallback_recovery_fraction,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class _FallbackExampleBuildResult:
    examples: list[RankingExample]
    candidate_occurrences: int
    content_routed_impressions: int
    popularity_routed_impressions: int
    empty_history_fallback_impressions: int
    unknown_history_fallback_impressions: int
    zero_profile_fallback_impressions: int
    zero_signal_fallback_impressions: int


def _classify_fallback_reason(
    history_ids: list[str],
    candidate_ids: list[str],
    content_model: ContentBasedRecommender,
    catalog: frozenset[str],
    *,
    k: int,
) -> str:
    if not history_ids:
        return "empty_history"

    if not any(news_id in catalog for news_id in history_ids):
        return "unknown_history"

    try:
        content_results = content_model.recommend(
            history_ids,
            candidate_news_ids=candidate_ids,
            top_k=k,
            exclude_history=True,
        )
    except ColdStartUserError:
        return "zero_profile"

    if not content_results or not any(result.score > 0.0 for result in content_results):
        return "zero_signal"

    raise FallbackEvaluationError("Fallback routing disagreed with the fitted content model.")


def _build_ranking_examples(
    validation_behaviors: pd.DataFrame,
    fallback_model: ContentPopularityFallbackRecommender,
    content_model: ContentBasedRecommender,
    catalog: frozenset[str],
    *,
    k: int,
) -> _FallbackExampleBuildResult:
    examples: list[RankingExample] = []
    candidate_occurrences = 0
    content_routed_impressions = 0
    popularity_routed_impressions = 0
    empty_history_fallback_impressions = 0
    unknown_history_fallback_impressions = 0
    zero_profile_fallback_impressions = 0
    zero_signal_fallback_impressions = 0

    for row in validation_behaviors.itertuples(index=False):
        impression_id = str(row.impression_id)
        history_ids = _parse_history(row.history)

        try:
            parsed_impressions = parse_impressions(str(row.impressions))
        except MindDataValidationError as error:
            raise FallbackEvaluationError(
                f"Invalid validation impression '{impression_id}': {error}"
            ) from error

        candidate_ids = [news_id for news_id, _ in parsed_impressions]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise FallbackEvaluationError(
                f"Validation impression '{impression_id}' contains duplicate candidate IDs."
            )

        unknown_candidates = set(candidate_ids) - catalog

        if unknown_candidates:
            unknown_preview = ", ".join(sorted(unknown_candidates)[:3])
            raise FallbackEvaluationError(
                f"Validation impression '{impression_id}' "
                "contains candidates missing from the catalog: "
                f"{unknown_preview}."
            )

        relevant_items = frozenset(news_id for news_id, label in parsed_impressions if label == 1)
        candidate_occurrences += len(candidate_ids)

        try:
            recommendations = fallback_model.recommend(
                history_ids,
                candidate_news_ids=candidate_ids,
                top_k=k,
            )
        except FallbackModelError as error:
            raise FallbackEvaluationError(str(error)) from error

        sources = {recommendation.source for recommendation in recommendations}

        if len(sources) > 1:
            raise FallbackEvaluationError(
                f"Impression '{impression_id}' mixed content and popularity routes."
            )

        if sources == {RecommendationSource.CONTENT}:
            content_routed_impressions += 1
        elif sources == {RecommendationSource.POPULARITY} or not recommendations:
            popularity_routed_impressions += 1
            fallback_reason = _classify_fallback_reason(
                history_ids,
                candidate_ids,
                content_model,
                catalog,
                k=k,
            )

            if fallback_reason == "empty_history":
                empty_history_fallback_impressions += 1
            elif fallback_reason == "unknown_history":
                unknown_history_fallback_impressions += 1
            elif fallback_reason == "zero_profile":
                zero_profile_fallback_impressions += 1
            else:
                zero_signal_fallback_impressions += 1
        else:
            raise FallbackEvaluationError(
                f"Impression '{impression_id}' returned an unknown recommendation source."
            )

        examples.append(
            RankingExample(
                impression_id=impression_id,
                ranked_items=tuple(recommendation.news_id for recommendation in recommendations),
                relevant_items=relevant_items,
            )
        )

    return _FallbackExampleBuildResult(
        examples=examples,
        candidate_occurrences=candidate_occurrences,
        content_routed_impressions=content_routed_impressions,
        popularity_routed_impressions=popularity_routed_impressions,
        empty_history_fallback_impressions=empty_history_fallback_impressions,
        unknown_history_fallback_impressions=unknown_history_fallback_impressions,
        zero_profile_fallback_impressions=zero_profile_fallback_impressions,
        zero_signal_fallback_impressions=zero_signal_fallback_impressions,
    )


def evaluate_fallback_baseline(
    news: pd.DataFrame,
    behaviors: pd.DataFrame,
    *,
    validation_fraction: float = 0.20,
    k: int = 10,
    max_features: int = 50_000,
) -> FallbackEvaluationReport:
    """Evaluate content ranking with training-only popularity fallback."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise FallbackEvaluationError("k must be a positive integer.")

    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features <= 0:
        raise FallbackEvaluationError("max_features must be a positive integer.")

    required_columns = {
        "impression_id",
        "timestamp",
        "history",
        "impressions",
    }
    missing_columns = required_columns.difference(behaviors.columns)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise FallbackEvaluationError(f"Missing required behavior columns: {formatted}.")

    try:
        catalog = _prepare_catalog(news)
        split = chronological_train_validation_split(
            behaviors,
            validation_fraction=validation_fraction,
        )
        vocabulary_news_ids = _training_vocabulary_news_ids(
            split.train,
            catalog,
        )
        content_model = ContentBasedRecommender(max_features=max_features).fit(
            news,
            vocabulary_news_ids=vocabulary_news_ids,
        )
        popularity_model = PopularityRecommender().fit(split.train)
        fallback_model = ContentPopularityFallbackRecommender(
            content_model,
            popularity_model,
        )
    except (
        ChronologicalSplitError,
        ContentEvaluationError,
        ContentModelError,
        PopularityModelError,
        FallbackModelError,
    ) as error:
        raise FallbackEvaluationError(str(error)) from error

    build_result = _build_ranking_examples(
        split.validation,
        fallback_model,
        content_model,
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
        raise FallbackEvaluationError(str(error)) from error

    return FallbackEvaluationReport(
        model_name="tfidf_content_with_popularity_fallback",
        training_records=len(split.train),
        validation_records=len(split.validation),
        requested_validation_fraction=validation_fraction,
        actual_validation_fraction=split.actual_validation_fraction,
        cutoff_timestamp=split.cutoff.isoformat(),
        vocabulary_article_count=content_model.vocabulary_article_count,
        indexed_article_count=content_model.indexed_article_count,
        vocabulary_size=content_model.vocabulary_size,
        candidate_occurrences=build_result.candidate_occurrences,
        content_routed_impressions=build_result.content_routed_impressions,
        popularity_routed_impressions=build_result.popularity_routed_impressions,
        empty_history_fallback_impressions=(build_result.empty_history_fallback_impressions),
        unknown_history_fallback_impressions=(build_result.unknown_history_fallback_impressions),
        zero_profile_fallback_impressions=(build_result.zero_profile_fallback_impressions),
        zero_signal_fallback_impressions=(build_result.zero_signal_fallback_impressions),
        metrics=metrics,
    )
