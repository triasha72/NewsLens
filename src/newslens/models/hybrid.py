"""Support-aware fusion of content and collaborative recommendation scores."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .collaborative import CollaborativeRecommender
from .content import (
    ColdStartUserError,
    ContentBasedRecommender,
)
from .popularity import PopularityRecommender


class HybridModelError(ValueError):
    """Raised when hybrid-model configuration is invalid."""


@dataclass(frozen=True, slots=True)
class HybridRecommendation:
    """One recommendation with component-level score provenance."""

    news_id: str
    score: float
    source: str
    content_score: float | None
    normalized_content_score: float | None
    collaborative_score: float | None
    normalized_collaborative_score: float | None
    collaborative_supported: bool


def _minmax_normalize(
    scores: dict[str, float],
) -> dict[str, float]:
    """Normalize one impression's scores to [0, 1].

    Constant score sets receive 0.5 so that the component contributes
    no arbitrary ranking preference after centering.
    """

    if not scores:
        return {}

    minimum = min(scores.values())
    maximum = max(scores.values())

    if maximum == minimum:
        return {
            news_id: 0.5
            for news_id in scores
        }

    scale = maximum - minimum

    return {
        news_id: (score - minimum) / scale
        for news_id, score in scores.items()
    }


class HybridRecommender:
    """Content-dominant recommender with a gated BPR residual.

    Content remains the primary signal. Collaborative scores are used
    only when enough candidates in the current impression are scoreable
    by BPR.

    The gate depends only on information available at recommendation
    time: user identity and candidate embedding support.
    """

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        collaborative_model: CollaborativeRecommender,
        popularity_model: PopularityRecommender,
        *,
        collaborative_weight: float = 0.10,
        minimum_supported_candidates: int = 0,
    ) -> None:
        if not content_model.is_fitted:
            raise HybridModelError(
                "The content model must be fitted first."
            )

        if not collaborative_model.is_fitted:
            raise HybridModelError(
                "The collaborative model must be fitted first."
            )

        if not popularity_model.is_fitted:
            raise HybridModelError(
                "The popularity model must be fitted first."
            )

        if not 0.0 <= collaborative_weight <= 1.0:
            raise HybridModelError(
                "collaborative_weight must be between 0 and 1."
            )

        if (
            isinstance(
                minimum_supported_candidates,
                bool,
            )
            or not isinstance(
                minimum_supported_candidates,
                int,
            )
            or minimum_supported_candidates < 0
        ):
            raise HybridModelError(
                "minimum_supported_candidates "
                "must be a non-negative integer."
            )

        self._content_model = content_model
        self._collaborative_model = collaborative_model
        self._popularity_model = popularity_model

        self.collaborative_weight = (
            collaborative_weight
        )

        self.minimum_supported_candidates = (
            minimum_supported_candidates
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether all component models are fitted."""

        return (
            self._content_model.is_fitted
            and self._collaborative_model.is_fitted
            and self._popularity_model.is_fitted
        )

    @staticmethod
    def _materialize_ids(
        values: Iterable[str],
    ) -> list[str]:
        if isinstance(values, str):
            return values.split()

        return list(values)

    def supported_candidate_count(
        self,
        user_id: str,
        candidate_news_ids: Iterable[str],
    ) -> int:
        """Return candidate count scoreable by BPR for this user."""

        if (
            user_id
            not in self._collaborative_model.user_to_index
        ):
            return 0

        return sum(
            news_id
            in self._collaborative_model.item_to_index
            for news_id in candidate_news_ids
        )

    def collaborative_gate_open(
        self,
        user_id: str,
        candidate_news_ids: Iterable[str],
    ) -> bool:
        """Return whether the collaborative residual may be used."""

        candidate_ids = self._materialize_ids(
            candidate_news_ids
        )

        if (
            user_id
            not in self._collaborative_model.user_to_index
        ):
            return False

        return (
            self.supported_candidate_count(
                user_id,
                candidate_ids,
            )
            >= self.minimum_supported_candidates
        )

    def _popularity_fallback(
        self,
        history_ids: list[str],
        candidate_ids: list[str],
        *,
        top_k: int,
    ) -> list[HybridRecommendation]:
        ranked_ids = (
            self._popularity_model.rank_candidates(
                candidate_ids,
                top_k=top_k,
                exclude_news_ids=history_ids,
            )
        )

        return [
            HybridRecommendation(
                news_id=news_id,
                score=self._popularity_model.score(
                    news_id
                ),
                source="popularity",
                content_score=None,
                normalized_content_score=None,
                collaborative_score=None,
                normalized_collaborative_score=None,
                collaborative_supported=False,
            )
            for news_id in ranked_ids
        ]

    def recommend_for_user(
        self,
        user_id: str,
        history_news_ids: Iterable[str],
        *,
        candidate_news_ids: Iterable[str],
        top_k: int = 10,
    ) -> list[HybridRecommendation]:
        """Rank candidates using content plus a gated BPR residual."""

        if top_k <= 0:
            raise HybridModelError(
                "top_k must be greater than zero."
            )

        history_ids = self._materialize_ids(
            history_news_ids
        )

        candidate_ids = sorted(
            set(
                self._materialize_ids(
                    candidate_news_ids
                )
            )
        )

        if not candidate_ids:
            return []

        gate_open = (
            self.collaborative_gate_open(
                user_id,
                candidate_ids,
            )
        )

        try:
            content_results = (
                self._content_model.recommend(
                    history_ids,
                    candidate_news_ids=candidate_ids,
                    top_k=len(candidate_ids),
                    exclude_history=True,
                )
            )
        except ColdStartUserError:
            return self._popularity_fallback(
                history_ids,
                candidate_ids,
                top_k=top_k,
            )

        if (
            not content_results
            or not any(
                result.score > 0.0
                for result in content_results
            )
        ):
            return self._popularity_fallback(
                history_ids,
                candidate_ids,
                top_k=top_k,
            )

        content_scores = {
            result.news_id: float(result.score)
            for result in content_results
        }

        normalized_content_scores = (
            _minmax_normalize(
                content_scores
            )
        )

        if gate_open:
            collaborative_results = (
                self._collaborative_model
                .recommend_for_user(
                    user_id,
                    candidate_news_ids=(
                        content_scores
                    ),
                    top_k=len(content_scores),
                )
            )

            collaborative_scores = {
                result.news_id: float(
                    result.score
                )
                for result
                in collaborative_results
            }

            normalized_collaborative_scores = (
                _minmax_normalize(
                    collaborative_scores
                )
            )
        else:
            collaborative_scores = {}
            normalized_collaborative_scores = {}

        recommendations: list[
            HybridRecommendation
        ] = []

        for (
            news_id,
            content_score,
        ) in content_scores.items():
            normalized_content = (
                normalized_content_scores[
                    news_id
                ]
            )

            collaborative_score = (
                collaborative_scores.get(
                    news_id
                )
            )

            normalized_collaborative = (
                normalized_collaborative_scores.get(
                    news_id
                )
            )

            supported = (
                gate_open
                and collaborative_score
                is not None
                and normalized_collaborative
                is not None
            )

            if supported:
                assert (
                    normalized_collaborative
                    is not None
                )

                residual = (
                    self.collaborative_weight
                    * (
                        normalized_collaborative
                        - 0.5
                    )
                )

                score = (
                    normalized_content
                    + residual
                )

                source = (
                    "hybrid"
                    if self.collaborative_weight
                    > 0.0
                    else "content"
                )
            else:
                score = normalized_content
                source = "content"

            recommendations.append(
                HybridRecommendation(
                    news_id=news_id,
                    score=float(score),
                    source=source,
                    content_score=(
                        content_score
                    ),
                    normalized_content_score=(
                        normalized_content
                    ),
                    collaborative_score=(
                        collaborative_score
                    ),
                    normalized_collaborative_score=(
                        normalized_collaborative
                    ),
                    collaborative_supported=(
                        supported
                    ),
                )
            )

        ranked = sorted(
            recommendations,
            key=lambda result: (
                -result.score,
                result.news_id,
            ),
        )

        return ranked[:top_k]
