from pathlib import Path

import pytest

from newslens.data.mind import (
    NEWS_COLUMNS,
    MindDataValidationError,
    load_news,
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