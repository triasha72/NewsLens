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
from .categories import (
    CategoryEvaluationError,
    CategoryEvaluationReport,
    evaluate_article_categories,
)
from .comparison import (
    PairedBootstrapComparisonReport,
    PairedComparisonError,
    paired_bootstrap_ranking_comparison,
)
from .content import (
    ContentEvaluationError,
    _parse_history,
    _prepare_catalog,
    _training_vocabulary_news_ids,
)
from .content import (
    _build_ranking_examples as _build_content_ranking_examples,
)
from .evaluator import (
    RankingEvaluationError,
    RankingEvaluationResult,
    RankingExample,
    evaluate_rankings,
)
from .exposure import (
    ExposureEvaluationError,
    ExposureEvaluationReport,
    evaluate_training_exposure_bands,
)
from .failures import (
    FailureAnalysisError,
    FailureArticle,
    HighScoreFailureReport,
    ScoredRankingExample,
    analyze_high_score_failures,
)
from .segments import (
    HistorySegmentEvaluationError,
    HistorySegmentEvaluationReport,
    HistorySegmentExample,
    evaluate_history_segments,
)
from .split import (
    ChronologicalSplitError,
    chronological_train_validation_split,
)
from .uncertainty import (
    BootstrapUncertaintyError,
    BootstrapUncertaintyReport,
    bootstrap_ranking_uncertainty,
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
    uncertainty: BootstrapUncertaintyReport
    history_segments: HistorySegmentEvaluationReport
    category_analysis: CategoryEvaluationReport
    exposure_analysis: ExposureEvaluationReport
    failure_analysis: HighScoreFailureReport
    paired_comparison: PairedBootstrapComparisonReport

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
    ) -> dict[str, object]:
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
            "uncertainty": self.uncertainty.to_dict(),
            "history_segments": self.history_segments.to_dict(),
            "category_analysis": self.category_analysis.to_dict(),
            "exposure_analysis": self.exposure_analysis.to_dict(),
            "failure_analysis": self.failure_analysis.to_dict(),
            "paired_comparison": self.paired_comparison.to_dict(),
        }


@dataclass(frozen=True)
class _FallbackExampleBuildResult:
    examples: list[RankingExample]
    scored_examples: list[ScoredRankingExample]
    history_segment_examples: list[HistorySegmentExample]
    candidate_occurrences: int
    content_routed_impressions: int
    popularity_routed_impressions: int
    empty_history_fallback_impressions: int
    unknown_history_fallback_impressions: int
    zero_profile_fallback_impressions: int
    zero_signal_fallback_impressions: int


def _prepare_item_categories(news: pd.DataFrame) -> dict[str, str]:
    required_columns = {"news_id", "category"}
    missing_columns = required_columns.difference(news.columns)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise FallbackEvaluationError(f"Missing required article columns: {formatted}.")

    news_ids = news["news_id"].fillna("").astype(str).str.strip()
    categories = news["category"].fillna("").astype(str).str.strip()

    if categories.eq("").any():
        raise FallbackEvaluationError("Article categories cannot be empty.")

    return dict(zip(news_ids, categories, strict=True))


def _prepare_failure_article_metadata(
    news: pd.DataFrame,
) -> dict[str, FailureArticle]:
    required_columns = {"news_id", "title", "category"}
    missing_columns = required_columns.difference(news.columns)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise FallbackEvaluationError(f"Missing required article columns: {formatted}.")

    news_ids = news["news_id"].fillna("").astype(str).str.strip()
    titles = news["title"].fillna("").astype(str).str.strip()
    categories = news["category"].fillna("").astype(str).str.strip()

    if "subcategory" in news.columns:
        subcategories = news["subcategory"].fillna("").astype(str).str.strip()
    else:
        subcategories = pd.Series("", index=news.index, dtype="object")

    articles = (
        FailureArticle(
            news_id=news_id,
            title=title,
            category=category,
            subcategory=subcategory,
        )
        for news_id, title, category, subcategory in zip(
            news_ids,
            titles,
            categories,
            subcategories,
            strict=True,
        )
    )

    return {article.news_id: article for article in articles}


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
    scored_examples: list[ScoredRankingExample] = []
    history_segment_examples: list[HistorySegmentExample] = []
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
            recommendation_source = RecommendationSource.CONTENT.value
        elif sources == {RecommendationSource.POPULARITY} or not recommendations:
            popularity_routed_impressions += 1
            recommendation_source = RecommendationSource.POPULARITY.value
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

        ranking_example = RankingExample(
            impression_id=impression_id,
            ranked_items=tuple(recommendation.news_id for recommendation in recommendations),
            relevant_items=relevant_items,
        )
        examples.append(ranking_example)
        scored_examples.append(
            ScoredRankingExample(
                impression_id=impression_id,
                ranked_items=ranking_example.ranked_items,
                ranked_scores=tuple(recommendation.score for recommendation in recommendations),
                relevant_items=relevant_items,
                source=recommendation_source,
                history_length=len(history_ids),
                candidate_count=len(candidate_ids),
            )
        )
        history_segment_examples.append(
            HistorySegmentExample(
                ranking=ranking_example,
                history_length=len(history_ids),
            )
        )

    return _FallbackExampleBuildResult(
        examples=examples,
        scored_examples=scored_examples,
        history_segment_examples=history_segment_examples,
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
    minimum_category_impressions: int = 100,
    minimum_exposure_impressions: int = 100,
    bootstrap_samples: int = 1_000,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_random_seed: int = 42,
    failure_score_quantile: float = 0.90,
    maximum_failures_per_source: int = 25,
) -> FallbackEvaluationReport:
    """Evaluate content ranking with training-only popularity fallback."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise FallbackEvaluationError("k must be a positive integer.")

    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features <= 0:
        raise FallbackEvaluationError("max_features must be a positive integer.")

    if (
        isinstance(minimum_category_impressions, bool)
        or not isinstance(minimum_category_impressions, int)
        or minimum_category_impressions <= 0
    ):
        raise FallbackEvaluationError("minimum_category_impressions must be a positive integer.")

    if (
        isinstance(minimum_exposure_impressions, bool)
        or not isinstance(minimum_exposure_impressions, int)
        or minimum_exposure_impressions <= 0
    ):
        raise FallbackEvaluationError("minimum_exposure_impressions must be a positive integer.")

    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise FallbackEvaluationError("bootstrap_samples must be a positive integer.")

    if bootstrap_samples < 2:
        raise FallbackEvaluationError("bootstrap_samples must be at least 2.")

    if (
        isinstance(bootstrap_confidence_level, bool)
        or not isinstance(bootstrap_confidence_level, (int, float))
        or not 0.0 < float(bootstrap_confidence_level) < 1.0
    ):
        raise FallbackEvaluationError("bootstrap_confidence_level must be between 0 and 1.")

    if (
        isinstance(bootstrap_random_seed, bool)
        or not isinstance(bootstrap_random_seed, int)
        or bootstrap_random_seed < 0
    ):
        raise FallbackEvaluationError("bootstrap_random_seed must be a non-negative integer.")

    if (
        isinstance(failure_score_quantile, bool)
        or not isinstance(failure_score_quantile, (int, float))
        or not 0.0 < float(failure_score_quantile) < 1.0
    ):
        raise FallbackEvaluationError("failure_score_quantile must be between 0 and 1.")

    if (
        isinstance(maximum_failures_per_source, bool)
        or not isinstance(maximum_failures_per_source, int)
        or maximum_failures_per_source <= 0
    ):
        raise FallbackEvaluationError("maximum_failures_per_source must be a positive integer.")

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
        item_categories = _prepare_item_categories(news)
        failure_article_metadata = _prepare_failure_article_metadata(news)
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
        training_exposures = {
            news_id: popularity_model.statistics(news_id).exposures for news_id in catalog
        }
        fallback_model = ContentPopularityFallbackRecommender(
            content_model,
            popularity_model,
        )
    except (
        ChronologicalSplitError,
        ContentEvaluationError,
        ContentModelError,
        FailureAnalysisError,
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
    content_build_result = _build_content_ranking_examples(
        split.validation,
        content_model,
        catalog,
        k=k,
    )

    if content_build_result.candidate_occurrences != build_result.candidate_occurrences:
        raise FallbackEvaluationError("Content and fallback candidate accounting must match.")

    try:
        metrics = evaluate_rankings(
            build_result.examples,
            catalog,
            k=k,
        )
        history_segments = evaluate_history_segments(
            build_result.history_segment_examples,
            catalog,
            k=k,
        )
        category_analysis = evaluate_article_categories(
            build_result.examples,
            item_categories,
            k=k,
            minimum_relevant_impressions=minimum_category_impressions,
        )
        exposure_analysis = evaluate_training_exposure_bands(
            build_result.examples,
            catalog,
            training_exposures,
            k=k,
            minimum_relevant_impressions=minimum_exposure_impressions,
        )
        uncertainty = bootstrap_ranking_uncertainty(
            build_result.examples,
            k=k,
            bootstrap_samples=bootstrap_samples,
            confidence_level=bootstrap_confidence_level,
            random_seed=bootstrap_random_seed,
        )
        failure_analysis = analyze_high_score_failures(
            build_result.scored_examples,
            k=k,
            score_quantile=failure_score_quantile,
            maximum_failures_per_source=maximum_failures_per_source,
            article_metadata=failure_article_metadata,
        )
        paired_comparison = paired_bootstrap_ranking_comparison(
            content_build_result.examples,
            build_result.examples,
            baseline_model_name="tfidf_history_content",
            candidate_model_name="tfidf_content_with_popularity_fallback",
            k=k,
            bootstrap_samples=bootstrap_samples,
            confidence_level=bootstrap_confidence_level,
            random_seed=bootstrap_random_seed,
        )
    except (
        BootstrapUncertaintyError,
        CategoryEvaluationError,
        ExposureEvaluationError,
        FailureAnalysisError,
        HistorySegmentEvaluationError,
        PairedComparisonError,
        RankingEvaluationError,
    ) as error:
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
        uncertainty=uncertainty,
        history_segments=history_segments,
        category_analysis=category_analysis,
        exposure_analysis=exposure_analysis,
        failure_analysis=failure_analysis,
        paired_comparison=paired_comparison,
    )
