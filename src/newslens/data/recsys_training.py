"""Leakage-safe interaction extraction for recommender training."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import duckdb

from newslens.models.collaborative import InteractionTriple


def load_bpr_triples(
    database: str | Path,
    *,
    cutoff_timestamp: datetime | str,
    max_negatives_per_positive: int = 1,
    seed: int = 42,
) -> list[InteractionTriple]:
    """Create BPR triples from pre-cutoff clicked/non-clicked candidates.

    Negatives are sampled from the same impression whenever possible.
    """
    if max_negatives_per_positive <= 0:
        raise ValueError("max_negatives_per_positive must be positive.")

    rng = random.Random(seed)
    query = """
        SELECT
            b.user_id,
            b.impression_id,
            c.news_id,
            c.clicked
        FROM behavior_events AS b
        JOIN candidate_interactions AS c
          ON b.impression_id = c.impression_id
        WHERE b.event_timestamp < ?
        ORDER BY b.impression_id, c.candidate_position
    """

    with duckdb.connect(str(database), read_only=True) as connection:
        rows = connection.execute(query, [str(cutoff_timestamp)]).fetchall()

    by_impression: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    for user_id, impression_id, news_id, clicked in rows:
        by_impression.setdefault((str(user_id), str(impression_id)), []).append(
            (str(news_id), bool(clicked))
        )

    triples: list[InteractionTriple] = []
    for (user_id, _), interactions in by_impression.items():
        positives = [item for item, clicked in interactions if clicked]
        negatives = [item for item, clicked in interactions if not clicked]
        if not positives or not negatives:
            continue

        for positive in positives:
            sample_size = min(max_negatives_per_positive, len(negatives))
            for negative in rng.sample(negatives, sample_size):
                triples.append(
                    InteractionTriple(
                        user_id=user_id,
                        positive_news_id=positive,
                        negative_news_id=negative,
                    )
                )

    return triples
