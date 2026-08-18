"""Deterministic diversity-aware reranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class DiversityRerankingError(ValueError):
    """Raised when diversity reranking input is invalid."""


@dataclass(frozen=True, slots=True)
class MMRConfig:
    """Frozen Maximal Marginal Relevance configuration."""

    lambda_weight: float

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.lambda_weight,
                bool,
            )
            or not isinstance(
                self.lambda_weight,
                (int, float),
            )
            or not 0.0
            <= float(
                self.lambda_weight
            )
            <= 1.0
        ):
            raise DiversityRerankingError(
                "lambda_weight must be between 0 and 1."
            )

        object.__setattr__(
            self,
            "lambda_weight",
            float(
                self.lambda_weight
            ),
        )


def maximal_marginal_relevance(
    candidate_news_ids: tuple[str, ...],
    relevance_scores: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    top_k: int,
    config: MMRConfig,
) -> tuple[str, ...]:
    """Return deterministic MMR ranking."""

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise DiversityRerankingError(
            "top_k must be a positive integer."
        )

    news_ids = tuple(
        str(news_id)
        for news_id
        in candidate_news_ids
    )

    if not news_ids:
        return ()

    if len(news_ids) != len(
        set(news_ids)
    ):
        raise DiversityRerankingError(
            "candidate_news_ids must be unique."
        )

    scores = np.asarray(
        relevance_scores,
        dtype=np.float64,
    )

    vectors = np.asarray(
        candidate_vectors,
        dtype=np.float32,
    )

    if scores.ndim != 1:
        raise DiversityRerankingError(
            "relevance_scores must be one-dimensional."
        )

    if vectors.ndim != 2:
        raise DiversityRerankingError(
            "candidate_vectors must be two-dimensional."
        )

    if (
        len(news_ids)
        != scores.shape[0]
        or len(news_ids)
        != vectors.shape[0]
    ):
        raise DiversityRerankingError(
            "Candidate IDs, scores, and vectors must align."
        )

    if not np.isfinite(
        scores
    ).all():
        raise DiversityRerankingError(
            "relevance_scores must be finite."
        )

    if not np.isfinite(
        vectors
    ).all():
        raise DiversityRerankingError(
            "candidate_vectors must be finite."
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
        rtol=1e-4,
    ):
        raise DiversityRerankingError(
            "candidate_vectors must be L2-normalized."
        )

    effective_k = min(
        top_k,
        len(news_ids),
    )

    remaining = set(
        range(
            len(news_ids)
        )
    )

    selected: list[int] = []

    while (
        remaining
        and len(selected)
        < effective_k
    ):
        best_index: int | None = None
        best_score: float | None = None

        for index in remaining:
            if selected:
                similarity = float(
                    np.max(
                        vectors[
                            selected
                        ]
                        @ vectors[
                            index
                        ]
                    )
                )
            else:
                similarity = 0.0

            mmr_score = (
                config.lambda_weight
                * float(
                    scores[index]
                )
                - (
                    1.0
                    - config.lambda_weight
                )
                * similarity
            )

            if (
                best_index is None
                or mmr_score
                > float(
                    best_score
                )
                or (
                    np.isclose(
                        mmr_score,
                        float(
                            best_score
                        ),
                        atol=1e-12,
                        rtol=0.0,
                    )
                    and news_ids[
                        index
                    ]
                    < news_ids[
                        best_index
                    ]
                )
            ):
                best_index = index
                best_score = (
                    mmr_score
                )

        assert best_index is not None

        selected.append(
            best_index
        )

        remaining.remove(
            best_index
        )

    return tuple(
        news_ids[index]
        for index in selected
    )
