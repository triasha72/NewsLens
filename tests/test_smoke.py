from __future__ import annotations

from newslens import __version__
from newslens.cli import main


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_reports_next_milestone(capsys: object) -> None:
    main()
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "implement validated MIND data ingestion" in captured.out
