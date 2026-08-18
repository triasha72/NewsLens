"""Training matrix construction for the Phase-05 ranker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from newslens.ranking import (
    SECOND_STAGE_FEATURE_NAMES,
    SecondStageFeatureBuilder,
)

from .mind import parse_impressions


class RankerTrainingDataError(ValueError):
    """Raised when ranker training data cannot be constructed."""


@dataclass(frozen=True, slots=True)
class RankerTrainingBuildResult:
    """Selected training matrix and compact accounting."""

    features: np.ndarray
    labels: np.ndarray
    input_impressions: int
    eligible_impressions: int
    impressions_without_click: int
    empty_history_impressions: int
    unsupported_history_impressions: int
    truncated_history_impressions: int
    candidate_occurrences: int
    positive_rows: int
    selected_negative_rows: int
    available_negative_rows: int
    unique_positive_articles: int
    max_negatives_per_positive: int
    limited_training_run: bool

    @property
    def row_count(self) -> int:
        """Return the number of selected training rows."""

        return int(
            self.features.shape[0]
        )

    def to_dict(
        self,
    ) -> dict[str, int | bool | float]:
        """Return compact JSON-compatible accounting."""

        return {
            "input_impressions": (
                self.input_impressions
            ),
            "eligible_impressions": (
                self.eligible_impressions
            ),
            "impressions_without_click": (
                self.impressions_without_click
            ),
            "empty_history_impressions": (
                self.empty_history_impressions
            ),
            "unsupported_history_impressions": (
                self.unsupported_history_impressions
            ),
            "truncated_history_impressions": (
                self.truncated_history_impressions
            ),
            "candidate_occurrences": (
                self.candidate_occurrences
            ),
            "training_rows": (
                self.row_count
            ),
            "positive_rows": (
                self.positive_rows
            ),
            "selected_negative_rows": (
                self.selected_negative_rows
            ),
            "available_negative_rows": (
                self.available_negative_rows
            ),
            "positive_fraction": (
                self.positive_rows
                / self.row_count
            ),
            "unique_positive_articles": (
                self.unique_positive_articles
            ),
            "feature_count": len(
                SECOND_STAGE_FEATURE_NAMES
            ),
            "max_negatives_per_positive": (
                self.max_negatives_per_positive
            ),
            "limited_training_run": (
                self.limited_training_run
            ),
        }


def _parse_history(
    value: object,
) -> tuple[str, ...]:
    """Parse a MIND history field."""

    if (
        value is None
        or pd.isna(value)
    ):
        return ()

    text = str(value).strip()

    if not text:
        return ()

    return tuple(
        text.split()
    )


def build_ranker_training_matrix(
    behaviors: pd.DataFrame,
    *,
    feature_builder: SecondStageFeatureBuilder,
    max_negatives_per_positive: int = 5,
    max_impressions: int | None = None,
) -> RankerTrainingBuildResult:
    """Build deterministic clicked-vs-hard-nonclick training rows."""

    if (
        isinstance(
            max_negatives_per_positive,
            bool,
        )
        or not isinstance(
            max_negatives_per_positive,
            int,
        )
        or max_negatives_per_positive <= 0
    ):
        raise RankerTrainingDataError(
            "max_negatives_per_positive must be positive."
        )

    if (
        max_impressions is not None
        and (
            isinstance(
                max_impressions,
                bool,
            )
            or not isinstance(
                max_impressions,
                int,
            )
            or max_impressions <= 0
        )
    ):
        raise RankerTrainingDataError(
            "max_impressions must be positive when supplied."
        )

    required = {
        "impression_id",
        "history",
        "impressions",
    }

    missing = required.difference(
        behaviors.columns
    )

    if missing:
        raise RankerTrainingDataError(
            "Missing behavior columns: "
            + ", ".join(
                sorted(missing)
            )
            + "."
        )

    source = behaviors
    limited = False

    if (
        max_impressions is not None
        and max_impressions < len(source)
    ):
        source = (
            source.iloc[
                :max_impressions
            ]
            .reset_index(
                drop=True
            )
        )

        limited = True

    feature_parts: list[
        np.ndarray
    ] = []

    label_parts: list[
        np.ndarray
    ] = []

    eligible_impressions = 0
    impressions_without_click = 0
    empty_history_impressions = 0
    unsupported_history_impressions = 0
    truncated_history_impressions = 0

    candidate_occurrences = 0
    positive_rows = 0
    selected_negative_rows = 0
    available_negative_rows = 0

    positive_articles: set[str] = set()

    catalog_ids = set(
        feature_builder.catalog.news_ids
    )

    for row in source.itertuples(
        index=False
    ):
        parsed = parse_impressions(
            str(
                row.impressions
            )
        )

        candidate_ids = tuple(
            news_id
            for news_id, _
            in parsed
        )

        if (
            len(candidate_ids)
            != len(set(candidate_ids))
        ):
            raise RankerTrainingDataError(
                "Duplicate candidate IDs in impression "
                f"{row.impression_id}."
            )

        unknown_candidates = [
            news_id
            for news_id in candidate_ids
            if news_id not in catalog_ids
        ]

        if unknown_candidates:
            raise RankerTrainingDataError(
                "Training candidate missing from frozen catalog."
            )

        labels = np.asarray(
            [
                label
                for _, label
                in parsed
            ],
            dtype=np.int64,
        )

        positive_indices = (
            np.flatnonzero(
                labels == 1
            )
            .tolist()
        )

        negative_indices = (
            np.flatnonzero(
                labels == 0
            )
            .tolist()
        )

        candidate_occurrences += len(
            candidate_ids
        )

        if not positive_indices:
            impressions_without_click += 1
            continue

        raw_history = _parse_history(
            row.history
        )

        if not raw_history:
            empty_history_impressions += 1
            continue

        supported_history = [
            news_id
            for news_id in raw_history
            if news_id in catalog_ids
        ]

        if not supported_history:
            unsupported_history_impressions += 1
            continue

        if (
            len(supported_history)
            > feature_builder.max_history_length
        ):
            truncated_history_impressions += 1

        context = (
            feature_builder.build_context(
                raw_history
            )
        )

        if context is None:
            raise RankerTrainingDataError(
                "Eligible history unexpectedly produced no context."
            )

        matrix = (
            feature_builder.features_for_candidates(
                context,
                candidate_ids,
            )
        )

        ordered_negatives = sorted(
            negative_indices,
            key=lambda index: (
                -float(
                    matrix[
                        index,
                        0,
                    ]
                ),
                candidate_ids[
                    index
                ],
            ),
        )

        negative_limit = min(
            len(ordered_negatives),
            (
                max_negatives_per_positive
                * len(
                    positive_indices
                )
            ),
        )

        selected_negatives = (
            ordered_negatives[
                :negative_limit
            ]
        )

        selected_indices = (
            positive_indices
            + selected_negatives
        )

        feature_parts.append(
            matrix[
                selected_indices
            ]
        )

        label_parts.append(
            labels[
                selected_indices
            ]
        )

        eligible_impressions += 1

        positive_rows += len(
            positive_indices
        )

        selected_negative_rows += len(
            selected_negatives
        )

        available_negative_rows += len(
            negative_indices
        )

        positive_articles.update(
            candidate_ids[
                index
            ]
            for index
            in positive_indices
        )

    if not feature_parts:
        raise RankerTrainingDataError(
            "No ranker training rows were produced."
        )

    features = np.ascontiguousarray(
        np.concatenate(
            feature_parts,
            axis=0,
        ),
        dtype=np.float32,
    )

    labels = np.concatenate(
        label_parts,
        axis=0,
    ).astype(
        np.int64,
        copy=False,
    )

    if not np.isfinite(
        features
    ).all():
        raise RankerTrainingDataError(
            "Training features contain NaN or infinity."
        )

    if set(
        np.unique(
            labels
        ).tolist()
    ) != {
        0,
        1,
    }:
        raise RankerTrainingDataError(
            "Training matrix must contain both classes."
        )

    return RankerTrainingBuildResult(
        features=features,
        labels=labels,
        input_impressions=len(
            source
        ),
        eligible_impressions=(
            eligible_impressions
        ),
        impressions_without_click=(
            impressions_without_click
        ),
        empty_history_impressions=(
            empty_history_impressions
        ),
        unsupported_history_impressions=(
            unsupported_history_impressions
        ),
        truncated_history_impressions=(
            truncated_history_impressions
        ),
        candidate_occurrences=(
            candidate_occurrences
        ),
        positive_rows=(
            positive_rows
        ),
        selected_negative_rows=(
            selected_negative_rows
        ),
        available_negative_rows=(
            available_negative_rows
        ),
        unique_positive_articles=len(
            positive_articles
        ),
        max_negatives_per_positive=(
            max_negatives_per_positive
        ),
        limited_training_run=(
            limited
        ),
    )
