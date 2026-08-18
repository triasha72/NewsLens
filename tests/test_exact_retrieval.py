"""Tests for the NumPy exact retrieval oracle."""

import numpy as np

from newslens.retrieval.catalog import RetrievalCatalog
from newslens.retrieval.exact import ExactInnerProductRetriever


def _catalog() -> RetrievalCatalog:
    root_two = np.sqrt(2.0)

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
                [
                    -1.0 / root_two,
                    1.0 / root_two,
                ],
            ],
            dtype=np.float32,
        ),
    )


def test_exact_retrieval_orders_inner_product() -> None:
    retriever = ExactInnerProductRetriever(
        _catalog()
    )

    hits = retriever.retrieve(
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=3,
    )

    assert [
        hit.news_id
        for hit in hits
    ] == [
        "a",
        "b",
        "c",
    ]


def test_exact_retrieval_respects_exclusions() -> None:
    retriever = ExactInnerProductRetriever(
        _catalog()
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


def test_top_k_larger_than_catalog_is_safe() -> None:
    retriever = ExactInnerProductRetriever(
        _catalog()
    )

    hits = retriever.retrieve(
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=100,
    )

    assert len(hits) == 4


def test_exact_retrieval_is_deterministic() -> None:
    retriever = ExactInnerProductRetriever(
        _catalog()
    )

    query = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    first = retriever.retrieve(
        query,
        top_k=4,
    )

    second = retriever.retrieve(
        query,
        top_k=4,
    )

    assert first == second


def test_exact_retrieval_breaks_score_ties_by_news_id() -> None:
    catalog = RetrievalCatalog(
        news_ids=(
            "z",
            "a",
            "m",
        ),
        vectors=np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    retriever = ExactInnerProductRetriever(
        catalog
    )

    hits = retriever.retrieve(
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=2,
    )

    assert [
        hit.news_id
        for hit in hits
    ] == [
        "a",
        "z",
    ]
