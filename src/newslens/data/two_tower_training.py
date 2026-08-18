"""Leakage-safe positive training examples for the two-tower recommender."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from .mind import (
    MindDataValidationError,
    parse_impressions,
)


class TwoTowerTrainingDataError(ValueError):
    """Raised when two-tower training examples cannot be constructed."""


@dataclass(frozen=True, slots=True)
class TwoTowerTrainingExample:
    """One history-to-click training pair."""

    impression_id: str
    history_news_ids: tuple[str, ...]
    positive_news_id: str


@dataclass(frozen=True, slots=True)
class TwoTowerTrainingBuildResult:
    """Training examples plus construction accounting."""

    examples: tuple[TwoTowerTrainingExample, ...]
    input_impressions: int
    eligible_impressions: int
    positive_click_occurrences: int
    impressions_without_positive: int
    empty_history_impressions: int
    history_without_features_impressions: int
    truncated_history_impressions: int
    positive_without_features: int
    unique_positive_articles: int
    max_history_length: int

    @property
    def usable_example_count(self) -> int:
        """Return the number of trainable history-click pairs."""

        return len(self.examples)

    @property
    def skipped_positive_occurrences(self) -> int:
        """Return clicked occurrences unavailable to the neural objective."""

        return (
            self.positive_click_occurrences
            - self.usable_example_count
        )

    def to_dict(self) -> dict[str, int]:
        """Return JSON-compatible construction accounting."""

        return {
            "input_impressions": self.input_impressions,
            "eligible_impressions": self.eligible_impressions,
            "positive_click_occurrences": self.positive_click_occurrences,
            "usable_example_count": self.usable_example_count,
            "skipped_positive_occurrences": self.skipped_positive_occurrences,
            "impressions_without_positive": self.impressions_without_positive,
            "empty_history_impressions": self.empty_history_impressions,
            "history_without_features_impressions": (
                self.history_without_features_impressions
            ),
            "truncated_history_impressions": self.truncated_history_impressions,
            "positive_without_features": self.positive_without_features,
            "unique_positive_articles": self.unique_positive_articles,
            "max_history_length": self.max_history_length,
        }


def _parse_history(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    return str(value).split()


def build_two_tower_positive_examples(
    behaviors: pd.DataFrame,
    *,
    available_news_ids: Iterable[str],
    max_history_length: int = 20,
) -> TwoTowerTrainingBuildResult:
    """Create chronological history-to-click examples.

    Each positive clicked candidate becomes one training example using only the
    history that was available at that impression time.

    The most recent ``max_history_length`` feature-supported history articles
    are retained.
    """

    if (
        isinstance(max_history_length, bool)
        or not isinstance(max_history_length, int)
        or max_history_length <= 0
    ):
        raise TwoTowerTrainingDataError(
            "max_history_length must be a positive integer."
        )

    required_columns = {
        "impression_id",
        "history",
        "impressions",
    }

    missing = required_columns.difference(
        behaviors.columns
    )

    if missing:
        formatted = ", ".join(
            sorted(missing)
        )

        raise TwoTowerTrainingDataError(
            "Missing required behavior columns: "
            f"{formatted}."
        )

    if isinstance(
        available_news_ids,
        str,
    ):
        available = set(
            available_news_ids.split()
        )
    else:
        available = {
            str(news_id)
            for news_id in available_news_ids
        }

    if not available:
        raise TwoTowerTrainingDataError(
            "At least one feature-supported article is required."
        )

    examples: list[
        TwoTowerTrainingExample
    ] = []

    eligible_impressions = 0
    positive_click_occurrences = 0
    impressions_without_positive = 0
    empty_history_impressions = 0
    history_without_features_impressions = 0
    truncated_history_impressions = 0
    positive_without_features = 0

    positive_articles: set[str] = set()

    for row in behaviors.itertuples(
        index=False
    ):
        impression_id = str(
            row.impression_id
        )

        try:
            parsed = parse_impressions(
                str(row.impressions)
            )
        except MindDataValidationError as error:
            raise TwoTowerTrainingDataError(
                "Invalid training impression "
                f"'{impression_id}': {error}"
            ) from error

        candidate_ids = [
            news_id
            for news_id, _ in parsed
        ]

        if len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise TwoTowerTrainingDataError(
                "Training impression "
                f"'{impression_id}' contains duplicate candidate IDs."
            )

        positive_ids = [
            news_id
            for news_id, label in parsed
            if label == 1
        ]

        positive_click_occurrences += len(
            positive_ids
        )

        if not positive_ids:
            impressions_without_positive += 1
            continue

        raw_history = _parse_history(
            row.history
        )

        if not raw_history:
            empty_history_impressions += 1
            continue

        usable_history = [
            news_id
            for news_id in raw_history
            if news_id in available
        ]

        if not usable_history:
            history_without_features_impressions += 1
            continue

        if len(usable_history) > max_history_length:
            truncated_history_impressions += 1

        history = tuple(
            usable_history[
                -max_history_length:
            ]
        )

        contributed = False

        for positive_news_id in positive_ids:
            if positive_news_id not in available:
                positive_without_features += 1
                continue

            examples.append(
                TwoTowerTrainingExample(
                    impression_id=impression_id,
                    history_news_ids=history,
                    positive_news_id=positive_news_id,
                )
            )

            positive_articles.add(
                positive_news_id
            )

            contributed = True

        if contributed:
            eligible_impressions += 1

    if not examples:
        raise TwoTowerTrainingDataError(
            "No usable two-tower training examples were produced."
        )

    return TwoTowerTrainingBuildResult(
        examples=tuple(examples),
        input_impressions=len(behaviors),
        eligible_impressions=eligible_impressions,
        positive_click_occurrences=positive_click_occurrences,
        impressions_without_positive=impressions_without_positive,
        empty_history_impressions=empty_history_impressions,
        history_without_features_impressions=(
            history_without_features_impressions
        ),
        truncated_history_impressions=truncated_history_impressions,
        positive_without_features=positive_without_features,
        unique_positive_articles=len(
            positive_articles
        ),
        max_history_length=max_history_length,
    )
