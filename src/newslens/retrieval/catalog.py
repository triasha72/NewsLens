"""Frozen retrieval embedding catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import RetrievalError


@dataclass(frozen=True, slots=True)
class RetrievalCatalog:
    """Article identifiers aligned with normalized retrieval vectors."""

    news_ids: tuple[str, ...]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        news_ids = tuple(
            str(news_id)
            for news_id in self.news_ids
        )

        vectors = np.ascontiguousarray(
            np.asarray(
                self.vectors,
                dtype=np.float32,
            )
        )

        if vectors.ndim != 2:
            raise RetrievalError(
                "Retrieval vectors must be a two-dimensional matrix."
            )

        if len(news_ids) != vectors.shape[0]:
            raise RetrievalError(
                "Article identifiers and vector rows must have equal length."
            )

        if not news_ids:
            raise RetrievalError(
                "Retrieval catalog cannot be empty."
            )

        if len(set(news_ids)) != len(news_ids):
            raise RetrievalError(
                "Retrieval catalog contains duplicate article identifiers."
            )

        if vectors.shape[1] <= 0:
            raise RetrievalError(
                "Retrieval vectors must have positive dimension."
            )

        if not np.isfinite(vectors).all():
            raise RetrievalError(
                "Retrieval catalog contains NaN or infinite values."
            )

        norms = np.linalg.norm(
            vectors,
            axis=1,
        )

        if np.any(norms <= 0.0):
            raise RetrievalError(
                "Retrieval catalog contains zero vectors."
            )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-4,
            rtol=1e-4,
        ):
            raise RetrievalError(
                "Retrieval catalog vectors must be L2-normalized."
            )

        object.__setattr__(
            self,
            "news_ids",
            news_ids,
        )

        object.__setattr__(
            self,
            "vectors",
            vectors,
        )

    @property
    def article_count(self) -> int:
        """Return the number of indexed articles."""

        return len(self.news_ids)

    @property
    def embedding_dim(self) -> int:
        """Return the retrieval embedding dimension."""

        return int(
            self.vectors.shape[1]
        )

    @property
    def id_to_position(
        self,
    ) -> dict[str, int]:
        """Return deterministic article-ID to matrix-row mapping."""

        return {
            news_id: index
            for index, news_id in enumerate(
                self.news_ids
            )
        }

    def save_npz(
        self,
        path: Path,
    ) -> None:
        """Persist the retrieval catalog as a compressed NumPy artifact."""

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            news_ids=np.asarray(
                self.news_ids,
                dtype=np.str_,
            ),
            vectors=self.vectors,
        )

    @classmethod
    def load_npz(
        cls,
        path: Path,
    ) -> RetrievalCatalog:
        """Restore a persisted retrieval catalog."""

        with np.load(
            path,
            allow_pickle=False,
        ) as payload:
            news_ids = tuple(
                str(news_id)
                for news_id in payload[
                    "news_ids"
                ].tolist()
            )

            vectors = np.asarray(
                payload["vectors"],
                dtype=np.float32,
            )

        return cls(
            news_ids=news_ids,
            vectors=vectors,
        )
