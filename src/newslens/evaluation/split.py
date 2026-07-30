"""Leakage-safe dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ChronologicalSplitError(ValueError):
    """Raised when a leakage-safe chronological split cannot be created."""


@dataclass(frozen=True)
class ChronologicalSplit:
    """Training and validation data separated by a strict time boundary."""

    train: pd.DataFrame
    validation: pd.DataFrame
    cutoff: pd.Timestamp
    timestamp_column: str = "timestamp"

    @property
    def is_leakage_safe(self) -> bool:
        """Return whether every training timestamp precedes validation."""

        return bool(
            self.train[self.timestamp_column].max() < self.validation[self.timestamp_column].min()
        )

    @property
    def actual_validation_fraction(self) -> float:
        """Return the fraction of records assigned to validation."""

        total_records = len(self.train) + len(self.validation)
        return len(self.validation) / total_records


def chronological_train_validation_split(
    behaviors: pd.DataFrame,
    *,
    validation_fraction: float = 0.20,
    timestamp_column: str = "timestamp",
) -> ChronologicalSplit:
    """Split behavior records using a strict chronological boundary.

    Rows with timestamps before the selected cutoff are assigned to training.
    Rows at or after the cutoff are assigned to validation. Records sharing the
    same timestamp are kept together to prevent temporal overlap.

    The selected cutoff produces the available split closest to the requested
    validation fraction without separating identical timestamps.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ChronologicalSplitError("validation_fraction must be strictly between 0 and 1.")

    if timestamp_column not in behaviors.columns:
        raise ChronologicalSplitError(f"Timestamp column '{timestamp_column}' is missing.")

    if len(behaviors) < 2:
        raise ChronologicalSplitError("At least two behavior records are required.")

    ordered = behaviors.copy(deep=True)

    try:
        ordered[timestamp_column] = pd.to_datetime(
            ordered[timestamp_column],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ChronologicalSplitError(
            f"Column '{timestamp_column}' contains invalid timestamps."
        ) from error

    if ordered[timestamp_column].isna().any():
        raise ChronologicalSplitError(f"Column '{timestamp_column}' contains missing timestamps.")

    ordered = ordered.sort_values(
        timestamp_column,
        kind="mergesort",
    ).reset_index(drop=True)

    counts_by_timestamp = ordered.groupby(
        timestamp_column,
        sort=True,
    ).size()

    if len(counts_by_timestamp) < 2:
        raise ChronologicalSplitError("At least two distinct timestamps are required.")

    target_train_records = int(len(ordered) * (1.0 - validation_fraction))
    target_train_records = min(
        max(target_train_records, 1),
        len(ordered) - 1,
    )

    rows_before_cutoff = counts_by_timestamp.cumsum().shift(fill_value=0)
    candidate_rows_before_cutoff = rows_before_cutoff.iloc[1:]

    cutoff = pd.Timestamp((candidate_rows_before_cutoff - target_train_records).abs().idxmin())

    train = ordered.loc[ordered[timestamp_column] < cutoff].reset_index(drop=True)

    validation = ordered.loc[ordered[timestamp_column] >= cutoff].reset_index(drop=True)

    if train.empty or validation.empty:
        raise ChronologicalSplitError(
            "The requested chronological split produced an empty partition."
        )

    result = ChronologicalSplit(
        train=train,
        validation=validation,
        cutoff=cutoff,
        timestamp_column=timestamp_column,
    )

    if not result.is_leakage_safe:
        raise ChronologicalSplitError("The resulting partitions contain temporal overlap.")

    return result
