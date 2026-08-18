"""Exact NumPy inner-product retrieval."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .base import (
    RetrievalHit,
    normalize_query_vector,
    prepare_exclusions,
    validate_top_k,
)
from .catalog import RetrievalCatalog


class ExactInnerProductRetriever:
    """Exact deterministic retrieval over normalized article embeddings."""

    def __init__(
        self,
        catalog: RetrievalCatalog,
    ) -> None:
        self.catalog = catalog

    def retrieve(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        exclude_news_ids: Iterable[str] = (),
    ) -> list[RetrievalHit]:
        """Return exact top-k inner-product neighbors."""

        validate_top_k(
            top_k
        )

        query = normalize_query_vector(
            query_vector,
            embedding_dim=(
                self.catalog.embedding_dim
            ),
        )

        exclusions = prepare_exclusions(
            exclude_news_ids
        )

        scores = (
            self.catalog.vectors
            @ query
        )

        candidates = [
            index
            for index, news_id in enumerate(
                self.catalog.news_ids
            )
            if news_id not in exclusions
        ]

        if not candidates:
            return []

        effective_k = min(
            top_k,
            len(candidates),
        )

        ordered = sorted(
            candidates,
            key=lambda index: (
                -float(
                    scores[index]
                ),
                self.catalog.news_ids[
                    index
                ],
            ),
        )[
            :effective_k
        ]

        return [
            RetrievalHit(
                news_id=(
                    self.catalog.news_ids[
                        index
                    ]
                ),
                score=float(
                    scores[index]
                ),
                rank=rank,
            )
            for rank, index in enumerate(
                ordered,
                start=1,
            )
        ]
