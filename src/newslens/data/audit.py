"""Aggregate, non-sensitive quality statistics for a loaded MIND split."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .mind import parse_impressions


@dataclass(frozen=True)
class MindDatasetAudit:
    """Summary statistics for one MIND dataset split."""

    split: str
    news_articles: int
    categories: int
    behavior_records: int
    unique_users: int
    candidate_impressions: int
    clicks: int
    non_clicks: int
    click_through_rate: float
    empty_histories: int
    average_history_length: float
    average_candidates_per_impression: float
    missing_titles: int
    missing_abstracts: int
    referenced_news_missing_metadata: int
    first_timestamp: str
    last_timestamp: str
    top_categories: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def audit_dataset(
    news: pd.DataFrame,
    behaviors: pd.DataFrame,
    split: str,
) -> MindDatasetAudit:
    """Calculate deterministic quality and interaction statistics."""

    candidate_count = 0
    click_count = 0
    history_item_count = 0
    empty_history_count = 0
    referenced_news_ids: set[str] = set()

    for row in behaviors.itertuples(index=False):
        history_ids = str(row.history).split()

        if history_ids:
            history_item_count += len(history_ids)
            referenced_news_ids.update(history_ids)
        else:
            empty_history_count += 1

        candidates = parse_impressions(str(row.impressions))
        candidate_count += len(candidates)

        for news_id, label in candidates:
            referenced_news_ids.add(news_id)
            click_count += label

    behavior_count = len(behaviors)
    known_news_ids = set(news["news_id"].astype(str))
    missing_metadata_count = len(referenced_news_ids - known_news_ids)

    category_counts = news["category"].value_counts().head(10)
    top_categories = {str(category): int(count) for category, count in category_counts.items()}

    first_timestamp = behaviors["timestamp"].min().isoformat()
    last_timestamp = behaviors["timestamp"].max().isoformat()

    return MindDatasetAudit(
        split=split,
        news_articles=len(news),
        categories=int(news["category"].nunique()),
        behavior_records=behavior_count,
        unique_users=int(behaviors["user_id"].nunique()),
        candidate_impressions=candidate_count,
        clicks=click_count,
        non_clicks=candidate_count - click_count,
        click_through_rate=round(click_count / candidate_count, 6),
        empty_histories=empty_history_count,
        average_history_length=round(
            history_item_count / behavior_count,
            6,
        ),
        average_candidates_per_impression=round(
            candidate_count / behavior_count,
            6,
        ),
        missing_titles=int(news["title"].str.strip().eq("").sum()),
        missing_abstracts=int(news["abstract"].str.strip().eq("").sum()),
        referenced_news_missing_metadata=missing_metadata_count,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        top_categories=top_categories,
    )
