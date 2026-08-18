"""Core contracts and validation for vector retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


class RetrievalError(ValueError):
    """Raised when retrieval inputs or configuration are invalid."""


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One ranked vector-retrieval result."""

    news_id: str
    score: float
    rank: int


def validate_top_k(
    top_k: int,
) -> int:
    """Validate and return a positive retrieval cutoff."""

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise RetrievalError(
            "top_k must be a positive integer."
        )

    return top_k


def normalize_query_vector(
    query_vector: np.ndarray,
    *,
    embedding_dim: int,
) -> np.ndarray:
    """Validate and L2-normalize one query vector."""

    values = np.asarray(
        query_vector,
        dtype=np.float32,
    )

    if values.ndim != 1:
        raise RetrievalError(
            "query_vector must be one-dimensional."
        )

    if values.shape[0] != embedding_dim:
        raise RetrievalError(
            "query_vector dimension mismatch: "
            f"expected {embedding_dim}, "
            f"received {values.shape[0]}."
        )

    if not np.isfinite(values).all():
        raise RetrievalError(
            "query_vector contains NaN or infinite values."
        )

    norm = float(
        np.linalg.norm(values)
    )

    if norm <= 0.0:
        raise RetrievalError(
            "query_vector cannot be a zero vector."
        )

    normalized = values / norm

    return np.ascontiguousarray(
        normalized,
        dtype=np.float32,
    )


def prepare_exclusions(
    exclude_news_ids: Iterable[str],
) -> frozenset[str]:
    """Normalize a collection of excluded article identifiers."""

    if isinstance(
        exclude_news_ids,
        str,
    ):
        values = (
            exclude_news_ids.split()
        )
    else:
        values = exclude_news_ids

    return frozenset(
        str(news_id)
        for news_id
        in values
    )
