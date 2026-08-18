"""Tests for FAISS HNSW retrieval."""

import numpy as np
import pytest

pytest.importorskip(
    "faiss"
)

from newslens.retrieval.base import (
    RetrievalError,
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)
from newslens.retrieval.faiss_hnsw import (
    FaissHNSWRetriever,
)


def _catalog() -> RetrievalCatalog:
    rng = np.random.default_rng(
        42
    )

    vectors = rng.normal(
        size=(
            200,
            16,
        )
    ).astype(
        np.float32
    )

    vectors /= np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    return RetrievalCatalog(
        news_ids=tuple(
            f"N{index}"
            for index
            in range(
                200
            )
        ),
        vectors=vectors,
    )


def test_hnsw_returns_requested_count() -> None:
    catalog = _catalog()

    retriever = FaissHNSWRetriever(
        catalog,
        m=16,
        ef_construction=80,
        ef_search=64,
    )

    hits = retriever.retrieve(
        catalog.vectors[0],
        top_k=20,
    )

    assert len(hits) == 20


def test_hnsw_respects_exclusions() -> None:
    catalog = _catalog()

    retriever = FaissHNSWRetriever(
        catalog,
        m=16,
        ef_construction=80,
        ef_search=64,
    )

    exclusions = (
        "N0",
        "N1",
        "N2",
        "N3",
    )

    hits = retriever.retrieve(
        catalog.vectors[0],
        top_k=20,
        exclude_news_ids=(
            exclusions
        ),
    )

    returned = {
        hit.news_id
        for hit in hits
    }

    assert not (
        returned
        & set(exclusions)
    )


def test_hnsw_ef_search_can_change() -> None:
    retriever = FaissHNSWRetriever(
        _catalog(),
        ef_search=16,
    )

    assert (
        retriever.ef_search
        == 16
    )

    retriever.ef_search = 64

    assert (
        retriever.ef_search
        == 64
    )


def test_invalid_ef_search_rejected() -> None:
    with pytest.raises(
        RetrievalError
    ):
        FaissHNSWRetriever(
            _catalog(),
            ef_search=0,
        )
