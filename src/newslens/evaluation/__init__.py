"""Evaluation and leakage-prevention utilities for NewsLens."""

from .categories import (
    CategoryEvaluationError,
    CategoryEvaluationReport,
    CategoryResult,
    evaluate_article_categories,
)
from .content import (
    ContentEvaluationError,
    ContentEvaluationReport,
    evaluate_content_baseline,
)
from .evaluator import (
    RankingEvaluationError,
    RankingEvaluationResult,
    RankingExample,
    evaluate_rankings,
)
from .exposure import (
    DEFAULT_TRAINING_EXPOSURE_BANDS,
    ExposureBandResult,
    ExposureEvaluationError,
    ExposureEvaluationReport,
    TrainingExposureBand,
    evaluate_training_exposure_bands,
)
from .failures import (
    FailureAnalysisError,
    FailureArticle,
    HighScoreFailure,
    HighScoreFailureReport,
    ScoredRankingExample,
    SourceScoreThreshold,
    analyze_high_score_failures,
)
from .fallback import (
    FallbackEvaluationError,
    FallbackEvaluationReport,
    evaluate_fallback_baseline,
)
from .metrics import (
    RankingMetricError,
    catalog_coverage,
    hit_rate_at_k,
    mean_reciprocal_rank_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from .popularity import (
    PopularityEvaluationError,
    PopularityEvaluationReport,
    evaluate_popularity_baseline,
)
from .segments import (
    DEFAULT_HISTORY_LENGTH_SEGMENTS,
    HistoryLengthSegment,
    HistorySegmentEvaluationError,
    HistorySegmentEvaluationReport,
    HistorySegmentExample,
    HistorySegmentResult,
    evaluate_history_segments,
)
from .split import (
    ChronologicalSplit,
    ChronologicalSplitError,
    chronological_train_validation_split,
)
from .uncertainty import (
    BootstrapInterval,
    BootstrapUncertaintyError,
    BootstrapUncertaintyReport,
    bootstrap_ranking_uncertainty,
)

__all__ = [
    "DEFAULT_HISTORY_LENGTH_SEGMENTS",
    "DEFAULT_TRAINING_EXPOSURE_BANDS",
    "BootstrapInterval",
    "BootstrapUncertaintyError",
    "BootstrapUncertaintyReport",
    "CategoryEvaluationError",
    "CategoryEvaluationReport",
    "CategoryResult",
    "ChronologicalSplit",
    "ChronologicalSplitError",
    "ContentEvaluationError",
    "ContentEvaluationReport",
    "ExposureBandResult",
    "ExposureEvaluationError",
    "ExposureEvaluationReport",
    "FailureAnalysisError",
    "FailureArticle",
    "FallbackEvaluationError",
    "FallbackEvaluationReport",
    "HighScoreFailure",
    "HighScoreFailureReport",
    "HistoryLengthSegment",
    "HistorySegmentEvaluationError",
    "HistorySegmentEvaluationReport",
    "HistorySegmentExample",
    "HistorySegmentResult",
    "PopularityEvaluationError",
    "PopularityEvaluationReport",
    "RankingEvaluationError",
    "RankingEvaluationResult",
    "RankingExample",
    "RankingMetricError",
    "ScoredRankingExample",
    "SourceScoreThreshold",
    "TrainingExposureBand",
    "analyze_high_score_failures",
    "bootstrap_ranking_uncertainty",
    "catalog_coverage",
    "chronological_train_validation_split",
    "evaluate_article_categories",
    "evaluate_content_baseline",
    "evaluate_fallback_baseline",
    "evaluate_history_segments",
    "evaluate_popularity_baseline",
    "evaluate_rankings",
    "evaluate_training_exposure_bands",
    "hit_rate_at_k",
    "mean_reciprocal_rank_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]
