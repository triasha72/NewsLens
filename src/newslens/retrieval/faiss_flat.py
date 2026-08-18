"""Exact FAISS inner-product retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .base import (
    RetrievalError,
    RetrievalHit,
    normalize_query_vector,
    prepare_exclusions,
    validate_top_k,
)
from .catalog import RetrievalCatalog


def _require_faiss():
    """Import FAISS lazily so it remains an optional retrieval dependency."""

    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "FAISS is required for this retrieval backend."
        ) from error

    return faiss


class FaissFlatIPRetriever:
    """Exact FAISS IndexFlatIP retrieval over the frozen catalog."""

    def __init__(
        self,
        catalog: RetrievalCatalog,
    ) -> None:
        faiss = _require_faiss()

        self.catalog = catalog

        self._id_to_position = (
            catalog.id_to_position
        )

        self._index = faiss.IndexFlatIP(
            catalog.embedding_dim
        )

        self._index.add(
            np.ascontiguousarray(
                catalog.vectors,
                dtype=np.float32,
            )
        )

        if (
            self._index.ntotal
            != catalog.article_count
        ):
            raise RetrievalError(
                "FAISS index article count does not match the catalog."
            )

    @property
    def index(self):
        """Return the underlying FAISS index."""

        return self._index

    def retrieve(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        exclude_news_ids: Iterable[str] = (),
    ) -> list[RetrievalHit]:
        """Return exact FAISS neighbors with history filtering."""

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

        known_exclusion_count = sum(
            news_id
            in self._id_to_position
            for news_id
            in exclusions
        )

        available_count = (
            self.catalog.article_count
            - known_exclusion_count
        )

        if available_count <= 0:
            return []

        effective_k = min(
            top_k,
            available_count,
        )

        # IndexFlatIP is exact. We request enough extra positions to
        # compensate for known articles that will be removed afterward.
        search_k = min(
            self.catalog.article_count,
            effective_k
            + known_exclusion_count,
        )

        scores, positions = (
            self._index.search(
                query.reshape(
                    1,
                    -1,
                ),
                search_k,
            )
        )

        candidates: list[
            tuple[
                str,
                float,
            ]
        ] = []

        for score, position in zip(
            scores[0],
            positions[0],
            strict=True,
        ):
            if position < 0:
                continue

            news_id = (
                self.catalog.news_ids[
                    int(position)
                ]
            )

            if news_id in exclusions:
                continue

            candidates.append(
                (
                    news_id,
                    float(score),
                )
            )

        ordered = sorted(
            candidates,
            key=lambda row: (
                -row[1],
                row[0],
            ),
        )[
            :effective_k
        ]

        return [
            RetrievalHit(
                news_id=news_id,
                score=score,
                rank=rank,
            )
            for rank, (
                news_id,
                score,
            ) in enumerate(
                ordered,
                start=1,
            )
        ]

    def save(
        self,
        path: Path,
    ) -> None:
        """Persist the FAISS index."""

        faiss = _require_faiss()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self._index,
            str(path),
        )

    @classmethod
    def load(
        cls,
        catalog: RetrievalCatalog,
        path: Path,
    ) -> FaissFlatIPRetriever:
        """Restore an existing exact FAISS index."""

        faiss = _require_faiss()

        instance = cls.__new__(
            cls
        )

        instance.catalog = catalog

        instance._id_to_position = (
            catalog.id_to_position
        )

        instance._index = (
            faiss.read_index(
                str(path)
            )
        )

        if (
            instance._index.d
            != catalog.embedding_dim
        ):
            raise RetrievalError(
                "Loaded FAISS index dimension does not match the catalog."
            )

        if (
            instance._index.ntotal
            != catalog.article_count
        ):
            raise RetrievalError(
                "Loaded FAISS index article count does not match the catalog."
            )

        if (
            instance._index.metric_type
            != faiss.METRIC_INNER_PRODUCT
        ):
            raise RetrievalError(
                "Loaded FAISS index does not use inner-product similarity."
            )

        return instance
