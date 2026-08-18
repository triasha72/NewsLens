"""Diversity and item-exposure metrics for Phase 06."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np


class ExposureMetricError(ValueError):
    """Raised when exposure/diversity metrics receive invalid input."""


def shannon_entropy(
    labels: Iterable[str],
) -> float:
    """Return natural-log Shannon entropy for categorical observations."""

    values = tuple(
        str(label)
        for label in labels
    )

    if not values:
        return 0.0

    counts = Counter(
        values
    )

    probabilities = np.asarray(
        [
            count / len(values)
            for count in counts.values()
        ],
        dtype=np.float64,
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities
        )
    )

    return float(
        entropy
    )


def intra_list_diversity(
    vectors: np.ndarray,
) -> float:
    """Return mean pairwise cosine distance for normalized item vectors."""

    matrix = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise ExposureMetricError(
            "vectors must be two-dimensional."
        )

    if matrix.shape[0] <= 1:
        return 0.0

    if not np.isfinite(
        matrix
    ).all():
        raise ExposureMetricError(
            "vectors must contain only finite values."
        )

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ExposureMetricError(
            "vectors must be L2-normalized."
        )

    similarities = (
        matrix
        @ matrix.T
    )

    row_indices, column_indices = (
        np.triu_indices(
            matrix.shape[0],
            k=1,
        )
    )

    distances = (
        1.0
        - similarities[
            row_indices,
            column_indices,
        ]
    )

    return float(
        np.mean(
            distances
        )
    )


def gini_coefficient(
    values: Iterable[float],
) -> float:
    """Return the Gini coefficient for non-negative values."""

    array = np.asarray(
        tuple(values),
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ExposureMetricError(
            "values must be one-dimensional."
        )

    if array.size == 0:
        raise ExposureMetricError(
            "At least one value is required."
        )

    if not np.isfinite(
        array
    ).all():
        raise ExposureMetricError(
            "values must be finite."
        )

    if np.any(
        array < 0.0
    ):
        raise ExposureMetricError(
            "Gini values must be non-negative."
        )

    total = float(
        array.sum()
    )

    if total == 0.0:
        return 0.0

    ordered = np.sort(
        array
    )

    count = ordered.size

    indices = np.arange(
        1,
        count + 1,
        dtype=np.float64,
    )

    gini = (
        (
            2.0
            * float(
                np.sum(
                    indices
                    * ordered
                )
            )
            / (
                count
                * total
            )
        )
        - (
            count + 1.0
        )
        / count
    )

    return float(
        np.clip(
            gini,
            0.0,
            1.0,
        )
    )


def top_fraction_share(
    values: Iterable[float],
    *,
    fraction: float,
) -> float:
    """Return the share captured by the highest-valued fraction."""

    if (
        isinstance(
            fraction,
            bool,
        )
        or not isinstance(
            fraction,
            (int, float),
        )
        or not 0.0
        < float(fraction)
        <= 1.0
    ):
        raise ExposureMetricError(
            "fraction must be in (0, 1]."
        )

    array = np.asarray(
        tuple(values),
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ExposureMetricError(
            "values must be one-dimensional."
        )

    if array.size == 0:
        raise ExposureMetricError(
            "At least one value is required."
        )

    if not np.isfinite(
        array
    ).all():
        raise ExposureMetricError(
            "values must be finite."
        )

    if np.any(
        array < 0.0
    ):
        raise ExposureMetricError(
            "values must be non-negative."
        )

    total = float(
        array.sum()
    )

    if total == 0.0:
        return 0.0

    selected_count = max(
        1,
        int(
            np.ceil(
                array.size
                * float(
                    fraction
                )
            )
        ),
    )

    largest = np.partition(
        array,
        array.size
        - selected_count,
    )[
        -selected_count:
    ]

    return float(
        largest.sum()
        / total
    )
