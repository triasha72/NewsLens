"""Tests for persisted FAISS retrieval indexes."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "faiss"
)

from newslens.retrieval.catalog import (
    RetrievalCatalog,
)
from newslens.retrieval.faiss_flat import (
    FaissFlatIPRetriever,
)
from newslens.retrieval.faiss_hnsw import (
    FaissHNSWRetriever,
)
from newslens.retrieval.persistence import (
    sha256_file,
)


def _catalog() -> RetrievalCatalog:
    rng = np.random.default_rng(
        9
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


def _ids(
    hits,
) -> list[str]:
    return [
        hit.news_id
        for hit in hits
    ]


def test_flat_persistence(
    tmp_path: Path,
) -> None:
    catalog = _catalog()

    original = FaissFlatIPRetriever(
        catalog
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

    query = catalog.vectors[
        10
    ]

    assert _ids(
        original.retrieve(
            query,
            top_k=20,
        )
    ) == _ids(
        restored.retrieve(
            query,
            top_k=20,
        )
    )

    assert len(
        sha256_file(
            path
        )
    ) == 64


def test_hnsw_persistence(
    tmp_path: Path,
) -> None:
    catalog = _catalog()

    original = FaissHNSWRetriever(
        catalog,
        m=16,
        ef_construction=80,
        ef_search=64,
    )

    path = (
        tmp_path
        / "hnsw.faiss"
    )

    original.save(
        path
    )

    restored = (
        FaissHNSWRetriever.load(
            catalog,
            path,
            ef_search=64,
        )
    )

    query = catalog.vectors[
        20
    ]

    assert _ids(
        original.retrieve(
            query,
            top_k=20,
        )
    ) == _ids(
        restored.retrieve(
            query,
            top_k=20,
        )
    )
