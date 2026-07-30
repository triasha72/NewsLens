"""Cold-start fallback for content-based recommendations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .content import (
    ColdStartUserError,
    ContentBasedRecommender,
)
from .popularity import PopularityRecommender


class FallbackModelError(ValueError):
    """Raised when fallback-model configuration is invalid."""


class RecommendationSource(StrEnum):
    """Strategy responsible for a recommendation."""

    CONTENT = "content"
    POPULARITY = "popularity"


@dataclass(frozen=True)
class FallbackRecommendation:
    """A recommendation and the strategy that produced it."""

    news_id: str
    score: float
    source: RecommendationSource


class ContentPopularityFallbackRecommender:
    """Use content similarity with popularity for cold-start cases."""

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        popularity_model: PopularityRecommender,
    ) -> None:
        if not content_model.is_fitted:
            raise FallbackModelError("The content model must be fitted first.")

        if not popularity_model.is_fitted:
            raise FallbackModelError("The popularity model must be fitted first.")

        self._content_model = content_model
        self._popularity_model = popularity_model

    @staticmethod
    def _materialize_ids(
        values: Iterable[str],
    ) -> list[str]:
        if isinstance(values, str):
            return values.split()

        return list(values)

    def _popularity_fallback(
        self,
        history_ids: list[str],
        candidate_ids: list[str],
        top_k: int,
    ) -> list[FallbackRecommendation]:
        ranked_ids = self._popularity_model.rank_candidates(
            candidate_ids,
            top_k=top_k,
            exclude_news_ids=history_ids,
        )

        return [
            FallbackRecommendation(
                news_id=news_id,
                score=self._popularity_model.score(news_id),
                source=RecommendationSource.POPULARITY,
            )
            for news_id in ranked_ids
        ]

    def recommend(
        self,
        history_news_ids: Iterable[str],
        *,
        candidate_news_ids: Iterable[str],
        top_k: int = 10,
    ) -> list[FallbackRecommendation]:
        """Recommend with content or fall back to popularity."""

        if top_k <= 0:
            raise FallbackModelError("top_k must be greater than zero.")

        history_ids = self._materialize_ids(history_news_ids)
        candidate_ids = self._materialize_ids(candidate_news_ids)

        try:
            content_results = self._content_model.recommend(
                history_ids,
                candidate_news_ids=candidate_ids,
                top_k=top_k,
                exclude_history=True,
            )
        except ColdStartUserError:
            return self._popularity_fallback(
                history_ids,
                candidate_ids,
                top_k,
            )

        if not content_results or not any(result.score > 0.0 for result in content_results):
            return self._popularity_fallback(
                history_ids,
                candidate_ids,
                top_k,
            )

        return [
            FallbackRecommendation(
                news_id=result.news_id,
                score=result.score,
                source=RecommendationSource.CONTENT,
            )
            for result in content_results
        ]
