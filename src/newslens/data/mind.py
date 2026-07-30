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

BEHAVIOR_COLUMNS: tuple[str, ...] = (
    "impression_id",
    "user_id",
    "timestamp",
    "history",
    "impressions",
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


def _read_tsv(path: str | Path, columns: tuple[str, ...]) -> pd.DataFrame:
    data_path = Path(path)

    if not data_path.is_file():
        raise FileNotFoundError(f"MIND dataset file was not found: {data_path}")

    _validate_tsv_width(data_path, expected_columns=len(columns))

    return pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=columns,
        dtype="string",
        keep_default_na=False,
    )


def _validate_required_fields(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    for column in required_columns:
        empty_rows = frame[column].str.strip().eq("")

        if empty_rows.any():
            raise MindDataValidationError(f"Column '{column}' contains empty values.")


def _validate_unique_column(
    frame: pd.DataFrame,
    column: str,
    description: str,
) -> None:
    duplicates = frame.loc[
        frame[column].duplicated(keep=False),
        column,
    ].unique()

    if len(duplicates) > 0:
        raise MindDataValidationError(f"Duplicate {description} found: {', '.join(duplicates)}")


def load_news(path: str | Path) -> pd.DataFrame:
    """Load and validate a MIND news.tsv file."""

    news = _read_tsv(path, NEWS_COLUMNS)

    _validate_required_fields(news, ("news_id", "category", "title"))
    _validate_unique_column(news, "news_id", "news IDs")

    return news


def _parse_impression_token(token: str) -> tuple[str, int]:
    try:
        news_id, label_text = token.rsplit("-", maxsplit=1)
    except ValueError as error:
        raise MindDataValidationError(
            f"Invalid impression token '{token}'. Expected NEWS_ID-LABEL."
        ) from error

    if not news_id:
        raise MindDataValidationError(f"Invalid impression token '{token}': news ID is empty.")

    if label_text not in {"0", "1"}:
        raise MindDataValidationError(f"Invalid impression token '{token}': label must be 0 or 1.")

    return news_id, int(label_text)


def parse_impressions(value: str) -> tuple[tuple[str, int], ...]:
    """Convert a MIND impression string into news ID and click-label pairs."""

    tokens = value.split()

    if not tokens:
        raise MindDataValidationError("An impression must contain at least one article.")

    return tuple(_parse_impression_token(token) for token in tokens)


def load_behaviors(path: str | Path) -> pd.DataFrame:
    """Load and validate a labeled MIND behaviors.tsv file."""

    behaviors = _read_tsv(path, BEHAVIOR_COLUMNS)

    _validate_required_fields(
        behaviors,
        ("impression_id", "user_id", "timestamp", "impressions"),
    )
    _validate_unique_column(behaviors, "impression_id", "impression IDs")

    try:
        behaviors["timestamp"] = pd.to_datetime(
            behaviors["timestamp"],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise MindDataValidationError(
            "Invalid MIND timestamp. Expected MM/DD/YYYY HH:MM:SS AM/PM."
        ) from error

    for row_number, value in enumerate(behaviors["impressions"], start=1):
        try:
            parse_impressions(value)
        except MindDataValidationError as error:
            raise MindDataValidationError(
                f"Invalid impressions at row {row_number}: {error}"
            ) from error

    return behaviors
