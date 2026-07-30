from __future__ import annotations

import pandas as pd
import pytest

from newslens.evaluation import (
    ChronologicalSplitError,
    chronological_train_validation_split,
)


def make_behaviors(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": [str(index) for index in range(1, len(timestamps) + 1)],
            "user_id": [f"U{index}" for index in range(1, len(timestamps) + 1)],
            "timestamp": pd.to_datetime(timestamps),
            "history": [""] * len(timestamps),
            "impressions": ["N1-1 N2-0"] * len(timestamps),
        }
    )


def test_split_orders_records_and_prevents_temporal_overlap() -> None:
    behaviors = make_behaviors(
        [
            "2019-11-13 12:00:00",
            "2019-11-11 12:00:00",
            "2019-11-14 12:00:00",
            "2019-11-12 12:00:00",
        ]
    )

    result = chronological_train_validation_split(
        behaviors,
        validation_fraction=0.25,
    )

    assert len(result.train) == 3
    assert len(result.validation) == 1
    assert result.cutoff == pd.Timestamp("2019-11-14 12:00:00")
    assert result.train["timestamp"].is_monotonic_increasing
    assert result.validation["timestamp"].is_monotonic_increasing
    assert result.is_leakage_safe


def test_split_keeps_equal_boundary_timestamps_together() -> None:
    behaviors = make_behaviors(
        [
            "2019-11-11 12:00:00",
            "2019-11-12 12:00:00",
            "2019-11-13 12:00:00",
            "2019-11-13 12:00:00",
        ]
    )

    result = chronological_train_validation_split(
        behaviors,
        validation_fraction=0.25,
    )

    assert len(result.train) == 2
    assert len(result.validation) == 2
    assert result.cutoff == pd.Timestamp("2019-11-13 12:00:00")
    assert result.validation["timestamp"].nunique() == 1
    assert result.is_leakage_safe


def test_split_does_not_modify_the_input_dataframe() -> None:
    behaviors = make_behaviors(
        [
            "2019-11-13 12:00:00",
            "2019-11-11 12:00:00",
            "2019-11-12 12:00:00",
        ]
    )
    original = behaviors.copy(deep=True)

    chronological_train_validation_split(behaviors)

    pd.testing.assert_frame_equal(behaviors, original)


@pytest.mark.parametrize(
    "validation_fraction",
    [0.0, 1.0, -0.1, 1.1],
)
def test_split_rejects_invalid_validation_fraction(
    validation_fraction: float,
) -> None:
    behaviors = make_behaviors(
        [
            "2019-11-11 12:00:00",
            "2019-11-12 12:00:00",
        ]
    )

    with pytest.raises(
        ChronologicalSplitError,
        match="strictly between 0 and 1",
    ):
        chronological_train_validation_split(
            behaviors,
            validation_fraction=validation_fraction,
        )


def test_split_rejects_missing_timestamp_column() -> None:
    behaviors = pd.DataFrame(
        {
            "impression_id": ["1", "2"],
            "user_id": ["U1", "U2"],
        }
    )

    with pytest.raises(
        ChronologicalSplitError,
        match="Timestamp column 'timestamp' is missing",
    ):
        chronological_train_validation_split(behaviors)


def test_split_rejects_invalid_timestamps() -> None:
    behaviors = pd.DataFrame(
        {
            "impression_id": ["1", "2"],
            "timestamp": ["2019-11-11", "not-a-timestamp"],
        }
    )

    with pytest.raises(
        ChronologicalSplitError,
        match="contains invalid timestamps",
    ):
        chronological_train_validation_split(behaviors)


def test_split_requires_distinct_timestamps() -> None:
    behaviors = make_behaviors(
        [
            "2019-11-11 12:00:00",
            "2019-11-11 12:00:00",
            "2019-11-11 12:00:00",
        ]
    )

    with pytest.raises(
        ChronologicalSplitError,
        match="two distinct timestamps",
    ):
        chronological_train_validation_split(behaviors)


def test_split_requires_at_least_two_records() -> None:
    behaviors = make_behaviors(["2019-11-11 12:00:00"])

    with pytest.raises(
        ChronologicalSplitError,
        match="At least two behavior records",
    ):
        chronological_train_validation_split(behaviors)
