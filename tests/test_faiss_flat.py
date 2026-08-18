"""Tests for exact FAISS IndexFlatIP retrieval."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "faiss"
)

from newslens.retrieval.catalog import (
    RetrievalCatalog,
)
from newslens.retrieval.exact import (
    ExactInnerProductRetriever,
)
from newslens.retrieval.faiss_flat import (
    FaissFlatIPRetriever,
)


def _catalog() -> RetrievalCatalog:
    return RetrievalCatalog(
        news_ids=(
            "a",
            "b",
            "c",
            "d",
        ),
        vectors=np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-0.8, 0.6],
            ],
            dtype=np.float32,
        ),
    )


def test_flat_matches_exact_retrieval() -> None:
    catalog = _catalog()

    exact = ExactInnerProductRetriever(
        catalog
    )

    flat = FaissFlatIPRetriever(
        catalog
    )

    query = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    exact_hits = exact.retrieve(
        query,
        top_k=4,
    )

    flat_hits = flat.retrieve(
        query,
        top_k=4,
    )

    assert [
        hit.news_id
        for hit in flat_hits
    ] == [
        hit.news_id
        for hit in exact_hits
    ]


def test_flat_respects_exclusions() -> None:
    retriever = (
        FaissFlatIPRetriever(
            _catalog()
        )
    )

    hits = retriever.retrieve(
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=2,
        exclude_news_ids=(
            "a",
            "b",
        ),
    )

    assert [
        hit.news_id
        for hit in hits
    ] == [
        "c",
        "d",
    ]


def test_flat_handles_top_k_larger_than_catalog() -> None:
    retriever = (
        FaissFlatIPRetriever(
            _catalog()
        )
    )

    hits = retriever.retrieve(
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=100,
    )

    assert len(hits) == 4


def test_flat_returns_empty_when_everything_is_excluded() -> None:
    retriever = (
        FaissFlatIPRetriever(
            _catalog()
        )
    )

    hits = retriever.retrieve(
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=10,
        exclude_news_ids=(
            "a",
            "b",
            "c",
            "d",
        ),
    )

    assert hits == []


def test_flat_index_round_trip(
    tmp_path: Path,
) -> None:
    catalog = _catalog()

    original = (
        FaissFlatIPRetriever(
            catalog
        )
    )

    path = (
        tmp_path
        / "flat.faiss"
    )

    original.save(
        path
    )

    restored = (
        FaissFlatIPRetriever.load(
            catalog,
            path,
        )
    )

    query = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    assert original.retrieve(
        query,
        top_k=4,
    ) == restored.retrieve(
        query,
        top_k=4,
    )
