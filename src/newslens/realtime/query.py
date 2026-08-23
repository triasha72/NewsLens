"""Deterministic query understanding for the real-time search path."""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

FRESHNESS_TERMS = frozenset(
    {
        "breaking",
        "current",
        "latest",
        "live",
        "new",
        "recent",
        "today",
        "tonight",
    }
)

CATEGORY_TERMS: dict[str, frozenset[str]] = {
    "business": frozenset(
        {
            "business",
            "earnings",
            "finance",
            "market",
            "markets",
            "shares",
            "stock",
            "stocks",
        }
    ),
    "sports": frozenset(
        {
            "baseball",
            "basketball",
            "football",
            "game",
            "match",
            "score",
            "soccer",
            "sports",
        }
    ),
    "technology": frozenset(
        {
            "ai",
            "app",
            "chip",
            "software",
            "tech",
            "technology",
        }
    ),
    "science": frozenset(
        {
            "climate",
            "mars",
            "nasa",
            "research",
            "science",
            "space",
        }
    ),
    "health": frozenset(
        {
            "health",
            "hospital",
            "medicine",
            "vaccine",
        }
    ),
}

ENTITY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
    }
) | FRESHNESS_TERMS | frozenset().union(*CATEGORY_TERMS.values())

ENTITY_ALLOWLIST = frozenset({"mars", "nasa"})


@dataclass(frozen=True, slots=True)
class QueryIntent:
    """Structured query information used by retrieval and ranking."""

    normalized_query: str
    tokens: tuple[str, ...]
    category: str | None
    entity: str | None
    prefers_freshness: bool


def _detect_category(tokens: tuple[str, ...]) -> str | None:
    token_set = set(tokens)
    matches = [
        (
            category,
            len(token_set & terms),
            min(index for index, token in enumerate(tokens) if token in terms),
        )
        for category, terms in CATEGORY_TERMS.items()
        if token_set & terms
    ]

    if not matches:
        return None

    return max(matches, key=lambda item: (item[1], -item[2], item[0]))[0]


def _detect_entity(query: str) -> str | None:
    candidates: list[str] = []

    for token in TOKEN_PATTERN.findall(query):
        normalized = token.lower()

        if normalized in ENTITY_STOPWORDS and normalized not in ENTITY_ALLOWLIST:
            continue

        if token.isupper() or token[:1].isupper():
            candidates.append(token)

    if not candidates:
        return None

    return " ".join(candidates[:3])


def understand_query(query: str) -> QueryIntent:
    """Normalize a query and extract category, entity, and freshness intent."""

    normalized_query = " ".join(query.strip().split())

    if not normalized_query:
        raise ValueError("query cannot be empty")

    tokens = tuple(token.lower() for token in TOKEN_PATTERN.findall(normalized_query))

    if not tokens:
        raise ValueError("query must contain at least one searchable token")

    return QueryIntent(
        normalized_query=normalized_query,
        tokens=tokens,
        category=_detect_category(tokens),
        entity=_detect_entity(normalized_query),
        prefers_freshness=bool(set(tokens) & FRESHNESS_TERMS),
    )
