"""Evaluation and leakage-prevention utilities for NewsLens."""

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
    "RankingMetricError",
    "catalog_coverage",
    "chronological_train_validation_split",
    "hit_rate_at_k",
    "mean_reciprocal_rank_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]
