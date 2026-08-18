"""Build the frozen Phase-07 FAISS index in an isolated process."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    with np.load(
        args.catalog,
        allow_pickle=False,
    ) as payload:
        news_ids = tuple(
            str(news_id)
            for news_id
            in payload[
                "news_ids"
            ].tolist()
        )

        vectors = np.ascontiguousarray(
            np.asarray(
                payload[
                    "vectors"
                ],
                dtype=np.float32,
            )
        )

    if vectors.ndim != 2:
        raise RuntimeError(
            "Catalog vectors must be two-dimensional."
        )

    if (
        len(news_ids)
        != vectors.shape[0]
    ):
        raise RuntimeError(
            "Catalog IDs and vectors do not align."
        )

    if not news_ids:
        raise RuntimeError(
            "Catalog cannot be empty."
        )

    if not np.isfinite(
        vectors
    ).all():
        raise RuntimeError(
            "Catalog vectors contain non-finite values."
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
        rtol=1e-4,
    ):
        raise RuntimeError(
            "Catalog vectors are not L2-normalized."
        )

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    index.add(
        vectors
    )

    if (
        index.ntotal
        != len(news_ids)
    ):
        raise RuntimeError(
            "FAISS index article count mismatch."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(
            args.output
        ),
    )

    restored = faiss.read_index(
        str(
            args.output
        )
    )

    if (
        restored.ntotal
        != len(news_ids)
    ):
        raise RuntimeError(
            "Persisted FAISS index article count mismatch."
        )

    if (
        restored.d
        != vectors.shape[1]
    ):
        raise RuntimeError(
            "Persisted FAISS index dimension mismatch."
        )

    if (
        restored.metric_type
        != faiss.METRIC_INNER_PRODUCT
    ):
        raise RuntimeError(
            "Persisted index is not inner-product FAISS."
        )

    print(
        "PHASE 07 FAISS INDEX BUILD: PASSED"
    )

    print(
        "articles:",
        restored.ntotal,
    )

    print(
        "dimension:",
        restored.d,
    )


if __name__ == "__main__":
    main()
