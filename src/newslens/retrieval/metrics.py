"""Metrics for comparing vector-retrieval systems."""

from __future__ import annotations

from collections.abc import Iterable

from .base import RetrievalError


def retrieval_recall_at_k(
    reference_ids: Iterable[str],
    candidate_ids: Iterable[str],
    *,
    k: int,
) -> float:
    """Measure candidate-set recall against an exact retrieval result."""

    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or k <= 0
    ):
        raise RetrievalError(
            "k must be a positive integer."
        )

    reference = list(
        reference_ids
    )[:k]

    candidate = list(
        candidate_ids
    )[:k]

    reference_set = set(
        reference
    )

    if not reference_set:
        return 1.0

    candidate_set = set(
        candidate
    )

    return (
        len(
            reference_set
            & candidate_set
        )
        / len(
            reference_set
        )
    )
