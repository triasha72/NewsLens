from __future__ import annotations

import json
from pathlib import Path

from newslens.cli import main


def test_audit_command_writes_json_report(
    tmp_path: Path,
    capsys: object,
) -> None:
    data_dir = tmp_path / "data"
    split_path = data_dir / "MINDsmall_train"
    split_path.mkdir(parents=True)

    (split_path / "news.tsv").write_text(
        ("N1\tsports\tfootball\tExample title\tExample abstract\thttps://example.com/1\t[]\t[]\n"),
        encoding="utf-8",
    )
    (split_path / "behaviors.tsv").write_text(
        ("1\tU1\t11/15/2019 10:22:32 AM\t\tN1-1\n"),
        encoding="utf-8",
    )

    output = tmp_path / "reports" / "audit.json"

    main(
        [
            "audit-data",
            "--data-dir",
            str(data_dir),
            "--split",
            "train",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert report["news_articles"] == 1
    assert report["clicks"] == 1
    assert "Audit report written" in captured.out


def _write_popularity_fixture(
    data_dir: Path,
) -> None:
    split_path = data_dir / "MINDsmall_train"
    split_path.mkdir(parents=True)

    news_rows = [
        ("N1\tnews\tgeneral\tArticle one\tAbstract one\thttps://example.com/1\t[]\t[]\n"),
        ("N2\tnews\tgeneral\tArticle two\tAbstract two\thttps://example.com/2\t[]\t[]\n"),
        ("N3\tnews\tgeneral\tArticle three\tAbstract three\thttps://example.com/3\t[]\t[]\n"),
        ("N4\tnews\tgeneral\tArticle four\tAbstract four\thttps://example.com/4\t[]\t[]\n"),
    ]

    behavior_rows = [
        ("I1\tU1\t01/01/2020 12:00:00 AM\t\tN1-1 N2-0 N3-0\n"),
        ("I2\tU2\t01/02/2020 12:00:00 AM\t\tN1-1 N2-0 N3-0\n"),
        ("I3\tU3\t01/03/2020 12:00:00 AM\t\tN2-1 N3-0\n"),
        ("I4\tU4\t01/04/2020 12:00:00 AM\t\tN2-0 N3-1 N4-0\n"),
        ("I5\tU5\t01/05/2020 12:00:00 AM\t\tN1-0 N4-1\n"),
    ]

    (split_path / "news.tsv").write_text(
        "".join(news_rows),
        encoding="utf-8",
    )
    (split_path / "behaviors.tsv").write_text(
        "".join(behavior_rows),
        encoding="utf-8",
    )


def test_popularity_evaluation_command_writes_report(
    tmp_path: Path,
    capsys: object,
) -> None:
    data_dir = tmp_path / "data"
    _write_popularity_fixture(data_dir)

    output = tmp_path / "reports" / "popularity_metrics.json"

    main(
        [
            "evaluate-popularity",
            "--data-dir",
            str(data_dir),
            "--output",
            str(output),
            "--k",
            "2",
            "--validation-fraction",
            "0.40",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert report["model_name"] == ("training_click_count_popularity")
    assert report["training_records"] == 3
    assert report["validation_records"] == 2
    assert report["metrics"]["k"] == 2
    assert report["metrics"]["mrr_at_k"] == 0.5
    assert "Popularity evaluation report written" in captured.out
