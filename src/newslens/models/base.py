"""Common recommendation-model contracts used by NewsLens serving pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One ranked recommendation returned by a NewsLens model."""

    news_id: str
    score: float
    source: str


@runtime_checkable
class RecommendationModel(Protocol):
    """Structural contract for models that can rank a provided candidate set."""

    def recommend(
        self,
        history_news_ids: Iterable[str],
        *,
        candidate_news_ids: Iterable[str],
        top_k: int = 10,
    ) -> list[Recommendation]:
        """Return at most ``top_k`` ranked recommendations."""
        ...
