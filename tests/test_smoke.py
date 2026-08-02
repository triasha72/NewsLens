from __future__ import annotations

from newslens import __version__
from newslens.cli import main


def test_package_version() -> None:
    assert __version__ == "0.2.0"


def test_cli_reports_available_commands(
    capsys: object,
) -> None:
    main([])
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert f"NewsLens {__version__} is ready." in captured.out
    assert "Use --help to view available commands." in captured.out
