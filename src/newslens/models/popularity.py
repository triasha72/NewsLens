"""Training-only popularity recommendation baseline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from ..data import parse_impressions


class PopularityModelError(ValueError):
    """Raised when popularity-model input is invalid."""


class PopularityModelNotFittedError(RuntimeError):
    """Raised when recommendations are requested before fitting."""


@dataclass(frozen=True)
class PopularityStatistics:
    """Observed training statistics for one news article."""

    exposures: int
    clicks: int

    @property
    def click_through_rate(self) -> float:
        """Return clicks divided by candidate exposures."""

        if self.exposures == 0:
            return 0.0

        return self.clicks / self.exposures


class PopularityRecommender:
    """Rank articles using click counts observed in training data."""

    def __init__(self) -> None:
        self._statistics: dict[str, PopularityStatistics] | None = None

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""

        return self._statistics is not None

    def fit(self, behaviors: pd.DataFrame) -> PopularityRecommender:
        """Learn article popularity from labeled training impressions."""

        if "impressions" not in behaviors.columns:
            raise PopularityModelError("The behaviors data must contain an 'impressions' column.")

        exposure_counts: dict[str, int] = {}
        click_counts: dict[str, int] = {}

        for value in behaviors["impressions"]:
            for news_id, label in parse_impressions(str(value)):
                exposure_counts[news_id] = exposure_counts.get(news_id, 0) + 1
                click_counts[news_id] = click_counts.get(news_id, 0) + label

        if not exposure_counts:
            raise PopularityModelError("At least one candidate impression is required.")

        self._statistics = {
            news_id: PopularityStatistics(
                exposures=exposures,
                clicks=click_counts.get(news_id, 0),
            )
            for news_id, exposures in exposure_counts.items()
        }

        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise PopularityModelNotFittedError(
                "Fit the popularity model before requesting recommendations."
            )

    def statistics(self, news_id: str) -> PopularityStatistics:
        """Return training statistics for an article."""

        self._require_fitted()
        assert self._statistics is not None

        return self._statistics.get(
            news_id,
            PopularityStatistics(exposures=0, clicks=0),
        )

    def score(self, news_id: str) -> float:
        """Return the article's training click-count score."""

        return float(self.statistics(news_id).clicks)

    def rank_candidates(
        self,
        candidate_news_ids: Iterable[str],
        *,
        top_k: int | None = None,
        exclude_news_ids: Iterable[str] = (),
    ) -> list[str]:
        """Rank a candidate set by training popularity."""

        self._require_fitted()

        if top_k is not None and top_k <= 0:
            raise PopularityModelError("top_k must be greater than zero.")

        excluded = set(exclude_news_ids)
        unique_candidates = {news_id for news_id in candidate_news_ids if news_id not in excluded}

        ranked = sorted(
            unique_candidates,
            key=lambda news_id: (
                -self.statistics(news_id).clicks,
                news_id,
            ),
        )

        if top_k is None:
            return ranked

        return ranked[:top_k]

    def recommend(
        self,
        *,
        top_k: int = 10,
        exclude_news_ids: Iterable[str] = (),
    ) -> list[str]:
        """Return globally popular articles observed during training."""

        self._require_fitted()
        assert self._statistics is not None

        return self.rank_candidates(
            self._statistics,
            top_k=top_k,
            exclude_news_ids=exclude_news_ids,
        )
