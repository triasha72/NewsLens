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
        "N1\tsports\tfootball\tExample title\tExample abstract\thttps://example.com/1\t[]\t[]\n",
        encoding="utf-8",
    )
    (split_path / "behaviors.tsv").write_text(
        "1\tU1\t11/15/2019 10:22:32 AM\t\tN1-1\n",
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
