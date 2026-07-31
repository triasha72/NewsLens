"""Evaluation and leakage-prevention utilities for NewsLens."""

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
from .split import (
    ChronologicalSplit,
    ChronologicalSplitError,
    chronological_train_validation_split,
)

__all__ = [
    "ChronologicalSplit",
    "ChronologicalSplitError",
    "ContentEvaluationError",
    "ContentEvaluationReport",
    "PopularityEvaluationError",
    "PopularityEvaluationReport",
    "RankingEvaluationError",
    "RankingEvaluationResult",
    "RankingExample",
    "RankingMetricError",
    "catalog_coverage",
    "chronological_train_validation_split",
    "evaluate_content_baseline",
    "evaluate_popularity_baseline",
    "evaluate_rankings",
    "hit_rate_at_k",
    "mean_reciprocal_rank_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]
