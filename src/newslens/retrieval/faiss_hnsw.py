"""Approximate FAISS HNSW inner-product retrieval."""

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
    """Import FAISS lazily so it remains optional."""

    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "FAISS is required for HNSW retrieval."
        ) from error

    return faiss


def _validate_positive_int(
    value: int,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise RetrievalError(
            f"{name} must be a positive integer."
        )

    return value


class FaissHNSWRetriever:
    """FAISS HNSW approximate inner-product retrieval."""

    def __init__(
        self,
        catalog: RetrievalCatalog,
        *,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        _validate_positive_int(
            m,
            name="m",
        )

        _validate_positive_int(
            ef_construction,
            name="ef_construction",
        )

        _validate_positive_int(
            ef_search,
            name="ef_search",
        )

        faiss = _require_faiss()

        self.catalog = catalog
        self._id_to_position = (
            catalog.id_to_position
        )

        self._index = faiss.IndexHNSWFlat(
            catalog.embedding_dim,
            m,
            faiss.METRIC_INNER_PRODUCT,
        )

        self._index.hnsw.efConstruction = (
            ef_construction
        )

        self._index.hnsw.efSearch = (
            ef_search
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
                "HNSW index article count does not match catalog."
            )

    @property
    def index(self):
        """Return underlying FAISS index."""

        return self._index

    @property
    def ef_search(self) -> int:
        """Return current search breadth."""

        return int(
            self._index.hnsw.efSearch
        )

    @ef_search.setter
    def ef_search(
        self,
        value: int,
    ) -> None:
        _validate_positive_int(
            value,
            name="ef_search",
        )

        self._index.hnsw.efSearch = (
            value
        )

    def retrieve(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        exclude_news_ids: Iterable[str] = (),
    ) -> list[RetrievalHit]:
        """Retrieve ANN neighbors with bounded over-retrieval."""

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

        search_k = min(
            self.catalog.article_count,
            max(
                effective_k * 2,
                effective_k
                + known_exclusion_count,
            ),
        )

        candidates: list[
            tuple[str, float]
        ] = []

        while True:
            scores, positions = (
                self._index.search(
                    query.reshape(
                        1,
                        -1,
                    ),
                    search_k,
                )
            )

            candidates = []

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

            if (
                len(candidates)
                >= effective_k
            ):
                break

            if (
                search_k
                >= self.catalog.article_count
            ):
                break

            search_k = min(
                self.catalog.article_count,
                search_k * 2,
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
        """Persist HNSW index."""

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
        *,
        ef_search: int,
    ) -> FaissHNSWRetriever:
        """Restore persisted HNSW index."""

        _validate_positive_int(
            ef_search,
            name="ef_search",
        )

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
                "Loaded HNSW dimension does not match catalog."
            )

        if (
            instance._index.ntotal
            != catalog.article_count
        ):
            raise RetrievalError(
                "Loaded HNSW article count does not match catalog."
            )

        if (
            instance._index.metric_type
            != faiss.METRIC_INNER_PRODUCT
        ):
            raise RetrievalError(
                "Loaded HNSW index is not inner-product."
            )

        instance.ef_search = (
            ef_search
        )

        return instance
