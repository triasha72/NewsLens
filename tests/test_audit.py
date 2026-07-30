from __future__ import annotations

from pathlib import Path

from newslens.data import audit_dataset, load_behaviors, load_news


def write_fixture_split(directory: Path) -> None:
    directory.mkdir(parents=True)

    news_rows = [
        ("N1\tsports\tfootball\tFirst title\tFirst abstract\thttps://example.com/1\t[]\t[]"),
        ("N2\tfinance\tmarkets\tSecond title\t\thttps://example.com/2\t[]\t[]"),
    ]

    behavior_rows = [
        "1\tU1\t11/15/2019 10:22:32 AM\tN1 N2\tN1-0 N2-1",
        "2\tU2\t11/16/2019 11:30:00 AM\t\tN2-0",
    ]

    (directory / "news.tsv").write_text(
        f"{'\n'.join(news_rows)}\n",
        encoding="utf-8",
    )
    (directory / "behaviors.tsv").write_text(
        f"{'\n'.join(behavior_rows)}\n",
        encoding="utf-8",
    )


def test_audit_dataset_calculates_expected_statistics(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "MINDsmall_train"
    write_fixture_split(split_path)

    news = load_news(split_path / "news.tsv")
    behaviors = load_behaviors(split_path / "behaviors.tsv")
    audit = audit_dataset(news, behaviors, split="train")

    assert audit.news_articles == 2
    assert audit.categories == 2
    assert audit.behavior_records == 2
    assert audit.unique_users == 2
    assert audit.candidate_impressions == 3
    assert audit.clicks == 1
    assert audit.non_clicks == 2
    assert audit.click_through_rate == 0.333333
    assert audit.empty_histories == 1
    assert audit.average_history_length == 1.0
    assert audit.average_candidates_per_impression == 1.5
    assert audit.missing_titles == 0
    assert audit.missing_abstracts == 1
    assert audit.referenced_news_missing_metadata == 0
