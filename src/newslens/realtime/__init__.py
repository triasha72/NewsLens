"""Real-time query understanding, storage, and freshness-aware ranking."""

from .demo import create_demo_repository
from .query import QueryIntent, understand_query
from .ranking import FreshnessRanker, RankedArticle, SearchDocument
from .repository import (
    ArticleRepository,
    InMemoryArticleRepository,
    PostgresArticleRepository,
)

__all__ = [
    "ArticleRepository",
    "FreshnessRanker",
    "InMemoryArticleRepository",
    "PostgresArticleRepository",
    "QueryIntent",
    "RankedArticle",
    "SearchDocument",
    "create_demo_repository",
    "understand_query",
]
