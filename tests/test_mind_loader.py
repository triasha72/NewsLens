from __future__ import annotations

from pathlib import Path

import pytest

from newslens.data.mind import (
    BEHAVIOR_COLUMNS,
    NEWS_COLUMNS,
    MindDataValidationError,
    load_behaviors,
    load_news,
    parse_impressions,
)


def make_news_row(news_id: str = "N1") -> str:
    fields = [
        news_id,
        "sports",
        "football",
        "Example news title",
        "Example news abstract",
        "https://example.com/article",
        "[]",
        "[]",
    ]
    return "\t".join(fields)


def make_behavior_row(
    timestamp: str = "11/15/2019 10:22:32 AM",
    history: str = "N1 N2",
    impressions: str = "N3-0 N4-1",
) -> str:
    fields = [
        "1",
        "U1",
        timestamp,
        history,
        impressions,
    ]
    return "\t".join(fields)


def test_load_news_returns_expected_schema(tmp_path: Path) -> None:
    news_path = tmp_path / "news.tsv"
    news_path.write_text(make_news_row(), encoding="utf-8")

    news = load_news(news_path)

    assert tuple(news.columns) == NEWS_COLUMNS
    assert len(news) == 1
    assert news.loc[0, "news_id"] == "N1"
    assert news.loc[0, "title"] == "Example news title"


def test_load_news_rejects_wrong_column_count(tmp_path: Path) -> None:
    news_path = tmp_path / "news.tsv"
    news_path.write_text("N1\tsports\tfootball\n", encoding="utf-8")

    with pytest.raises(
        MindDataValidationError,
        match="expected 8 tab-separated fields",
    ):
        load_news(news_path)


def test_load_news_rejects_duplicate_ids(tmp_path: Path) -> None:
    news_path = tmp_path / "news.tsv"
    news_path.write_text(
        f"{make_news_row()}\n{make_news_row()}\n",
        encoding="utf-8",
    )

    with pytest.raises(MindDataValidationError, match="Duplicate news IDs"):
        load_news(news_path)


def test_load_behaviors_parses_timestamp(tmp_path: Path) -> None:
    behaviors_path = tmp_path / "behaviors.tsv"
    behaviors_path.write_text(
        make_behavior_row(),
        encoding="utf-8",
    )

    behaviors = load_behaviors(behaviors_path)

    assert tuple(behaviors.columns) == BEHAVIOR_COLUMNS
    assert behaviors.loc[0, "user_id"] == "U1"
    assert behaviors.loc[0, "timestamp"].year == 2019
    assert behaviors.loc[0, "timestamp"].month == 11


def test_load_behaviors_allows_empty_history(tmp_path: Path) -> None:
    behaviors_path = tmp_path / "behaviors.tsv"
    behaviors_path.write_text(
        make_behavior_row(history=""),
        encoding="utf-8",
    )

    behaviors = load_behaviors(behaviors_path)

    assert behaviors.loc[0, "history"] == ""


def test_parse_impressions_returns_click_labels() -> None:
    parsed = parse_impressions("N3-0 N4-1")

    assert parsed == (("N3", 0), ("N4", 1))


def test_load_behaviors_rejects_invalid_label(tmp_path: Path) -> None:
    behaviors_path = tmp_path / "behaviors.tsv"
    behaviors_path.write_text(
        make_behavior_row(impressions="N3-2"),
        encoding="utf-8",
    )

    with pytest.raises(MindDataValidationError, match="label must be 0 or 1"):
        load_behaviors(behaviors_path)


def test_load_behaviors_rejects_invalid_timestamp(tmp_path: Path) -> None:
    behaviors_path = tmp_path / "behaviors.tsv"
    behaviors_path.write_text(
        make_behavior_row(timestamp="not-a-timestamp"),
        encoding="utf-8",
    )

    with pytest.raises(MindDataValidationError, match="Invalid MIND timestamp"):
        load_behaviors(behaviors_path)
