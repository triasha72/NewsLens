"""Transparent lexical and freshness-aware ranking for streamed articles."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from .query import FRESHNESS_TERMS, QueryIntent

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


@dataclass(frozen=True, slots=True)
class SearchDocument:
    """One article available to the real-time search ranker."""

    article_id: str
    title: str
    body: str
    category: str
    published_at: datetime
    produced_at: datetime
    indexed_at: datetime
    popularity: int = 0


@dataclass(frozen=True, slots=True)
class RankedArticle:
    """Ranked article with inspectable score components."""

    document: SearchDocument
    score: float
    relevance_score: float
    freshness_score: float
    popularity_score: float

    @property
    def index_freshness_ms(self) -> float:
        """Return event-produced to indexed latency in milliseconds."""

        return max(
            0.0,
            (self.document.indexed_at - self.document.produced_at).total_seconds()
            * 1_000,
        )


class FreshnessRanker:
    """Rank documents with lexical relevance and an explicit freshness boost."""

    def __init__(
        self,
        *,
        relevance_weight: float = 0.75,
        freshness_weight: float = 0.20,
        popularity_weight: float = 0.05,
        freshness_half_life_hours: float = 24.0,
    ) -> None:
        weights = (relevance_weight, freshness_weight, popularity_weight)

        if any(weight < 0.0 for weight in weights):
            raise ValueError("ranking weights cannot be negative")

        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("ranking weights must sum to 1.0")

        if freshness_half_life_hours <= 0.0:
            raise ValueError("freshness_half_life_hours must be positive")

        self.relevance_weight = relevance_weight
        self.freshness_weight = freshness_weight
        self.popularity_weight = popularity_weight
        self.freshness_half_life_hours = freshness_half_life_hours

    @staticmethod
    def _relevance(intent: QueryIntent, document: SearchDocument) -> float:
        query_tokens = tuple(
            token for token in intent.tokens if token not in FRESHNESS_TERMS
        )

        if not query_tokens:
            query_tokens = intent.tokens

        title_tokens = {
            token.lower() for token in TOKEN_PATTERN.findall(document.title)
        }
        body_tokens = {token.lower() for token in TOKEN_PATTERN.findall(document.body)}
        document_tokens = title_tokens | body_tokens | {document.category.lower()}

        matched_weight = sum(
            2.0 if token in title_tokens else 1.0
            for token in query_tokens
            if token in document_tokens
        )
        maximum_weight = 2.0 * len(query_tokens)

        if maximum_weight == 0.0:
            return 0.0

        score = matched_weight / maximum_weight

        if intent.category and document.category.lower() == intent.category:
            score = min(1.0, score + 0.15)

        if intent.entity and intent.entity.lower() in document.title.lower():
            score = min(1.0, score + 0.20)

        return score

    def _freshness(self, document: SearchDocument, *, now: datetime) -> float:
        age_hours = max(0.0, (now - document.published_at).total_seconds() / 3_600)
        return math.exp(-math.log(2.0) * age_hours / self.freshness_half_life_hours)

    def rank(
        self,
        intent: QueryIntent,
        documents: tuple[SearchDocument, ...],
        *,
        top_k: int,
        now: datetime | None = None,
    ) -> tuple[RankedArticle, ...]:
        """Rank candidates and return score components for inspection."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        ranking_time = now or datetime.now(UTC)
        max_popularity = max((document.popularity for document in documents), default=0)
        scored: list[RankedArticle] = []

        for document in documents:
            relevance = self._relevance(intent, document)
            freshness = self._freshness(document, now=ranking_time)
            popularity = (
                math.log1p(max(0, document.popularity)) / math.log1p(max_popularity)
                if max_popularity > 0
                else 0.0
            )
            active_freshness_weight = (
                self.freshness_weight if intent.prefers_freshness else 0.0
            )
            active_relevance_weight = self.relevance_weight + (
                self.freshness_weight - active_freshness_weight
            )
            score = (
                active_relevance_weight * relevance
                + active_freshness_weight * freshness
                + self.popularity_weight * popularity
            )
            scored.append(
                RankedArticle(
                    document=document,
                    score=score,
                    relevance_score=relevance,
                    freshness_score=freshness,
                    popularity_score=popularity,
                )
            )

        scored.sort(
            key=lambda item: (
                -item.score,
                -item.document.published_at.timestamp(),
                item.document.article_id,
            )
        )
        return tuple(scored[:top_k])
