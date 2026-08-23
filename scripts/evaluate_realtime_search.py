#!/usr/bin/env python3
"""Evaluate deterministic query understanding and freshness-aware ranking."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from newslens.realtime import FreshnessRanker, SearchDocument, understand_query


@dataclass(frozen=True)
class QueryCase:
    query: str
    category: str | None
    entity: str | None
    prefers_freshness: bool
    relevance: dict[str, int]


def documents(now: datetime) -> tuple[SearchDocument, ...]:
    def article(
        article_id: str,
        title: str,
        body: str,
        category: str,
        age_hours: float,
        popularity: int,
    ) -> SearchDocument:
        published_at = now - timedelta(hours=age_hours)
        return SearchDocument(
            article_id=article_id,
            title=title,
            body=body,
            category=category,
            published_at=published_at,
            produced_at=published_at + timedelta(minutes=1),
            indexed_at=published_at + timedelta(minutes=1, seconds=2),
            popularity=popularity,
        )

    return (
        article("tech-new", "Apple launches AI chip today", "New performance details.", "technology", 1, 15),
        article("tech-old", "Apple AI chip performance analysis", "Detailed chip review.", "technology", 168, 900),
        article("space-new", "NASA shares latest Mars mission update", "Live mission status.", "science", 2, 20),
        article("space-old", "NASA Mars mission research archive", "Long-form mission research.", "science", 240, 700),
        article("sport-new", "Falcons football score tonight", "The current game score.", "sports", 0.5, 30),
        article("sport-old", "Falcons football season analysis", "Full season analysis.", "sports", 120, 600),
        article("market-new", "Microsoft shares rise in latest market", "Fresh earnings reaction.", "business", 3, 25),
        article("market-old", "Microsoft market earnings analysis", "Detailed historical analysis.", "business", 96, 800),
        article("health-new", "Pfizer vaccine update today", "New hospital guidance.", "health", 4, 10),
        article("health-old", "Pfizer vaccine research review", "A detailed medicine review.", "health", 336, 500),
    )


CASES = (
    QueryCase("latest Apple AI chip", "technology", "Apple", True, {"tech-new": 3, "tech-old": 2}),
    QueryCase("Apple AI chip analysis", "technology", "Apple", False, {"tech-old": 3, "tech-new": 2}),
    QueryCase("NASA recent Mars research", "science", "NASA Mars", True, {"space-new": 3, "space-old": 2}),
    QueryCase("Falcons football score tonight", "sports", "Falcons", True, {"sport-new": 3, "sport-old": 1}),
    QueryCase("latest Microsoft market earnings", "business", "Microsoft", True, {"market-new": 3, "market-old": 2}),
    QueryCase("Pfizer vaccine research", "health", "Pfizer", False, {"health-old": 3, "health-new": 2}),
)


def dcg(grades: list[float]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def ndcg(order: list[str], relevance: dict[str, float], cutoff: int = 10) -> float:
    observed = [relevance.get(article_id, 0.0) for article_id in order[:cutoff]]
    ideal = sorted(relevance.values(), reverse=True)[:cutoff]
    denominator = dcg(ideal)
    return dcg(observed) / denominator if denominator else 0.0


def reciprocal_rank(order: list[str], relevance: dict[str, int], cutoff: int = 10) -> float:
    for rank, article_id in enumerate(order[:cutoff], start=1):
        if relevance.get(article_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def recall(order: list[str], relevance: dict[str, int], cutoff: int = 10) -> float:
    relevant = {article_id for article_id, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(order[:cutoff])) / len(relevant)


def evaluate_ranker(
    ranker: FreshnessRanker,
    corpus: tuple[SearchDocument, ...],
    now: datetime,
) -> dict[str, float]:
    ndcgs: list[float] = []
    mrrs: list[float] = []
    recalls: list[float] = []
    freshness_ndcgs: list[float] = []
    by_id = {document.article_id: document for document in corpus}
    for case in CASES:
        intent = understand_query(case.query)
        order = [item.document.article_id for item in ranker.rank(intent, corpus, top_k=10, now=now)]
        ndcgs.append(ndcg(order, case.relevance))
        mrrs.append(reciprocal_rank(order, case.relevance))
        recalls.append(recall(order, case.relevance))
        freshness_relevance = {
            article_id: grade
            * math.exp(
                -math.log(2)
                * max(0.0, (now - by_id[article_id].published_at).total_seconds() / 3600)
                / 24.0
            )
            for article_id, grade in case.relevance.items()
        }
        freshness_ndcgs.append(ndcg(order, freshness_relevance))
    return {
        "ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "mrr_at_10": sum(mrrs) / len(mrrs),
        "recall_at_10": sum(recalls) / len(recalls),
        "freshness_weighted_ndcg_at_10": sum(freshness_ndcgs) / len(freshness_ndcgs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/realtime_search_evaluation_v0_1.json"))
    args = parser.parse_args()
    now = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)
    corpus = documents(now)

    intent_results = []
    correct_fields = 0
    total_fields = 0
    for case in CASES:
        actual = understand_query(case.query)
        expected = {
            "category": case.category,
            "entity": case.entity,
            "prefers_freshness": case.prefers_freshness,
        }
        observed = {
            "category": actual.category,
            "entity": actual.entity,
            "prefers_freshness": actual.prefers_freshness,
        }
        correct_fields += sum(expected[key] == observed[key] for key in expected)
        total_fields += len(expected)
        intent_results.append({"query": case.query, "expected": expected, "observed": observed})

    relevance_only = FreshnessRanker(
        relevance_weight=0.95,
        freshness_weight=0.0,
        popularity_weight=0.05,
    )
    freshness_aware = FreshnessRanker()
    report = {
        "schema_version": "newslens.realtime-search-evaluation.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture": {
            "queries": len(CASES),
            "articles": len(corpus),
            "reference_time": now.isoformat(),
            "source": "small, hand-labeled deterministic contract fixture",
        },
        "query_understanding": {
            "field_accuracy": correct_fields / total_fields,
            "correct_fields": correct_fields,
            "total_fields": total_fields,
            "cases": intent_results,
        },
        "ranking": {
            "relevance_only": evaluate_ranker(relevance_only, corpus, now),
            "freshness_aware": evaluate_ranker(freshness_aware, corpus, now),
        },
        "ranking_configuration": {
            "relevance_only": {"relevance": 0.95, "freshness": 0.0, "popularity": 0.05},
            "freshness_aware": {"relevance": 0.75, "freshness": 0.20, "popularity": 0.05},
        },
        "cases": [asdict(case) for case in CASES],
        "limitations": [
            "This fixture verifies ranking behavior; it is not evidence of generalization to live traffic.",
            "A larger independently judged query set is still needed before product-quality claims.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
