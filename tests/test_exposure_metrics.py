"""Tests for Phase-06 diversity and exposure metrics."""

import math

import numpy as np
import pytest

from newslens.evaluation.diversity_exposure import (
    ExposureMetricError,
    gini_coefficient,
    intra_list_diversity,
    shannon_entropy,
    top_fraction_share,
)


def test_identical_vectors_have_zero_diversity() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert intra_list_diversity(
        vectors
    ) == pytest.approx(
        0.0
    )


def test_orthogonal_vectors_have_unit_diversity() -> None:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    assert intra_list_diversity(
        vectors
    ) == pytest.approx(
        1.0
    )


def test_entropy_of_balanced_binary_labels() -> None:
    result = shannon_entropy(
        (
            "a",
            "a",
            "b",
            "b",
        )
    )

    assert result == pytest.approx(
        math.log(2.0)
    )


def test_equal_exposure_has_zero_gini() -> None:
    assert gini_coefficient(
        (
            1.0,
            1.0,
            1.0,
            1.0,
        )
    ) == pytest.approx(
        0.0
    )


def test_single_item_concentration_has_expected_gini() -> None:
    assert gini_coefficient(
        (
            0.0,
            0.0,
            0.0,
            10.0,
        )
    ) == pytest.approx(
        0.75
    )


def test_top_fraction_share() -> None:
    assert top_fraction_share(
        (
            10.0,
            0.0,
            0.0,
            0.0,
        ),
        fraction=0.25,
    ) == pytest.approx(
        1.0
    )


def test_invalid_fraction_rejected() -> None:
    with pytest.raises(
        ExposureMetricError
    ):
        top_fraction_share(
            (
                1.0,
                2.0,
            ),
            fraction=0.0,
        )
