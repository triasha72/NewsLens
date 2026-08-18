"""Tests for retrieval validation and catalog contracts."""

import numpy as np
import pytest

from newslens.retrieval.base import (
    RetrievalError,
    normalize_query_vector,
    validate_top_k,
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)


def test_query_normalization() -> None:
    values = normalize_query_vector(
        np.asarray(
            [3.0, 4.0],
            dtype=np.float32,
        ),
        embedding_dim=2,
    )

    assert np.isclose(
        np.linalg.norm(
            values
        ),
        1.0,
    )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_top_k(
    value,
) -> None:
    with pytest.raises(
        RetrievalError
    ):
        validate_top_k(
            value
        )


def test_query_dimension_mismatch() -> None:
    with pytest.raises(
        RetrievalError
    ):
        normalize_query_vector(
            np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            embedding_dim=3,
        )


def test_query_rejects_zero_vector() -> None:
    with pytest.raises(
        RetrievalError
    ):
        normalize_query_vector(
            np.zeros(
                2,
                dtype=np.float32,
            ),
            embedding_dim=2,
        )


def test_catalog_rejects_duplicate_ids() -> None:
    with pytest.raises(
        RetrievalError
    ):
        RetrievalCatalog(
            news_ids=(
                "a",
                "a",
            ),
            vectors=np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )


def test_catalog_rejects_non_normalized_vectors() -> None:
    with pytest.raises(
        RetrievalError
    ):
        RetrievalCatalog(
            news_ids=(
                "a",
            ),
            vectors=np.asarray(
                [
                    [2.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
