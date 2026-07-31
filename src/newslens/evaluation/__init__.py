"""Evaluation and leakage-prevention utilities for NewsLens."""

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
from .split import (
    ChronologicalSplit,
    ChronologicalSplitError,
    chronological_train_validation_split,
)

__all__ = [
    "ChronologicalSplit",
    "ChronologicalSplitError",
    "RankingEvaluationError",
    "RankingEvaluationResult",
    "RankingExample",
    "RankingMetricError",
    "catalog_coverage",
    "chronological_train_validation_split",
    "evaluate_rankings",
    "hit_rate_at_k",
    "mean_reciprocal_rank_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]
