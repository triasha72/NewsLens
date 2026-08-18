"""Tests for the Phase-07 serving runtime."""

from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from newslens.retrieval.base import (
    RetrievalHit,
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)
from newslens.serving import (
    ServingRuntime,
    ServingRuntimeConfig,
)


class FakeNetwork:
    """Minimal deterministic history-user network."""

    config = SimpleNamespace(
        temperature=0.07
    )

    def to(
        self,
        device: str,
    ) -> "FakeNetwork":
        return self

    def eval(
        self,
    ) -> "FakeNetwork":
        return self

    def user_tower(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        del history_mask

        pooled = (
            history_embeddings.mean(
                dim=1
            )
        )

        return F.normalize(
            pooled,
            p=2,
            dim=-1,
        )


class FakeRetriever:
    """Exact NumPy retrieval for tests."""

    def __init__(
        self,
        catalog: RetrievalCatalog,
    ) -> None:
        self.catalog = catalog

    def retrieve(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        exclude_news_ids=(),
    ) -> list[RetrievalHit]:
        excluded = set(
            exclude_news_ids
        )

        scores = (
            self.catalog.vectors
            @ query_vector
        )

        rows = sorted(
            (
                (
                    news_id,
                    float(
                        scores[index]
                    ),
                )
                for index, news_id
                in enumerate(
                    self.catalog.news_ids
                )
                if news_id
                not in excluded
            ),
            key=lambda row: (
                -row[1],
                row[0],
            ),
        )[
            :top_k
        ]

        return [
            RetrievalHit(
                news_id=news_id,
                score=score,
                rank=rank,
            )
            for rank, (
                news_id,
                score,
            ) in enumerate(
                rows,
                start=1,
            )
        ]


def _runtime() -> ServingRuntime:
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.70710677, 0.70710677],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    catalog = RetrievalCatalog(
        news_ids=(
            "a",
            "b",
            "c",
            "d",
        ),
        vectors=vectors,
    )

    return ServingRuntime(
        network=FakeNetwork(),  # type: ignore[arg-type]
        catalog=catalog,
        retriever=(
            FakeRetriever(
                catalog
            )
        ),
        popularity_clicks={
            "a": 1,
            "b": 10,
            "c": 8,
            "d": 2,
        },
        config=ServingRuntimeConfig(
            artifact_version=(
                "test"
            ),
            selected_policy=(
                "mmr_lambda_0.80"
            ),
            retrieval_backend=(
                "faiss_flat"
            ),
            lambda_weight=0.80,
            retrieval_k=100,
            final_k=10,
            temperature=0.07,
            max_history_length=20,
            faiss_threads=1,
        ),
    )


def test_known_history_uses_two_tower_path() -> None:
    runtime = _runtime()

    result = runtime.recommend(
        ("a",),
        top_k=3,
    )

    assert result.fallback_used is False

    assert all(
        item.news_id != "a"
        for item
        in result.recommendations
    )

    assert all(
        item.source
        == "two_tower_faiss_mmr"
        for item
        in result.recommendations
    )


def test_unknown_history_uses_popularity_fallback() -> None:
    runtime = _runtime()

    result = runtime.recommend(
        ("UNKNOWN",),
        top_k=3,
    )

    assert result.fallback_used is True

    assert result.unknown_history_count == 1

    assert tuple(
        item.news_id
        for item
        in result.recommendations
    ) == (
        "b",
        "c",
        "d",
    )


def test_runtime_is_deterministic() -> None:
    runtime = _runtime()

    first = runtime.recommend(
        ("a",),
        top_k=3,
    )

    second = runtime.recommend(
        ("a",),
        top_k=3,
    )

    assert tuple(
        item.news_id
        for item
        in first.recommendations
    ) == tuple(
        item.news_id
        for item
        in second.recommendations
    )


def test_top_k_cannot_exceed_frozen_final_k() -> None:
    runtime = _runtime()

    try:
        runtime.recommend(
            ("a",),
            top_k=11,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "top_k > 10 must fail."
        )
