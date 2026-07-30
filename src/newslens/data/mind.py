"""Loading and validation utilities for the Microsoft MIND dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NEWS_COLUMNS: tuple[str, ...] = (
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
)


class MindDataValidationError(ValueError):
    """Raised when a MIND dataset file does not follow the expected schema."""


def _validate_tsv_width(path: Path, expected_columns: int) -> None:
    row_count = 0

    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            row_count += 1
            actual_columns = len(raw_line.rstrip("\r\n").split("\t"))

            if actual_columns != expected_columns:
                raise MindDataValidationError(
                    f"{path} line {line_number} contains {actual_columns} fields; "
                    f"expected {expected_columns} tab-separated fields."
                )

    if row_count == 0:
        raise MindDataValidationError(f"{path} is empty.")


def load_news(path: str | Path) -> pd.DataFrame:
    """Load and validate a MIND news.tsv file."""

    news_path = Path(path)

    if not news_path.is_file():
        raise FileNotFoundError(f"MIND news file was not found: {news_path}")

    _validate_tsv_width(news_path, expected_columns=len(NEWS_COLUMNS))

    news = pd.read_csv(
        news_path,
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        dtype="string",
        keep_default_na=False,
    )

    for required_column in ("news_id", "category", "title"):
        empty_rows = news[required_column].str.strip().eq("")

        if empty_rows.any():
            raise MindDataValidationError(
                f"Column '{required_column}' contains empty values."
            )

    duplicate_ids = news.loc[
        news["news_id"].duplicated(keep=False),
        "news_id",
    ].unique()

    if len(duplicate_ids) > 0:
        raise MindDataValidationError(
            f"Duplicate news IDs found: {', '.join(duplicate_ids)}"
        )

    return news