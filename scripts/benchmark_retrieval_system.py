"""Benchmark Phase-04 exact and approximate retrieval systems."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from newslens.data import (
    load_behaviors,
)
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
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
from newslens.retrieval.faiss_hnsw import (
    FaissHNSWRetriever,
)
from newslens.retrieval.metrics import (
    retrieval_recall_at_k,
)
from newslens.retrieval.persistence import (
    sha256_file,
)
from newslens.retrieval.queries import (
    build_validation_queries,
)


def _stats(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            array.mean()
        ),
        "p50": float(
            np.percentile(
                array,
                50.0,
            )
        ),
        "p95": float(
            np.percentile(
                array,
                95.0,
            )
        ),
        "p99": float(
            np.percentile(
                array,
                99.0,
            )
        ),
    }


def _request_latency(
    retriever,
    queries,
    *,
    top_k: int,
) -> dict[str, float]:
    warmup_count = min(
        50,
        len(queries),
    )

    for query in queries[
        :warmup_count
    ]:
        retriever.retrieve(
            query.vector,
            top_k=top_k,
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

    values = []

    for query in queries:
        started = (
            time.perf_counter_ns()
        )

        retriever.retrieve(
            query.vector,
            top_k=top_k,
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        values.append(
            (
                time.perf_counter_ns()
                - started
            )
            / 1_000_000.0
        )

    return _stats(
        values
    )


def _numpy_batch_qps(
    query_matrix: np.ndarray,
    catalog: RetrievalCatalog,
    *,
    batch_size: int,
    top_k: int,
) -> float:
    started = (
        time.perf_counter_ns()
    )

    for start in range(
        0,
        query_matrix.shape[0],
        batch_size,
    ):
        batch = query_matrix[
            start:
            start + batch_size
        ]

        scores = (
            batch
            @ catalog.vectors.T
        )

        np.argpartition(
            -scores,
            kth=top_k - 1,
            axis=1,
        )[
            :,
            :top_k,
        ]

    elapsed = (
        time.perf_counter_ns()
        - started
    ) / 1_000_000_000.0

    return (
        query_matrix.shape[0]
        / elapsed
    )


def _faiss_batch_qps(
    index,
    query_matrix: np.ndarray,
    *,
    batch_size: int,
    top_k: int,
) -> float:
    started = (
        time.perf_counter_ns()
    )

    for start in range(
        0,
        query_matrix.shape[0],
        batch_size,
    ):
        batch = np.ascontiguousarray(
            query_matrix[
                start:
                start + batch_size
            ],
            dtype=np.float32,
        )

        index.search(
            batch,
            top_k,
        )

    elapsed = (
        time.perf_counter_ns()
        - started
    ) / 1_000_000_000.0

    return (
        query_matrix.shape[0]
        / elapsed
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--embedding-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--hnsw-sweep-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--persistence-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--flat-index",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--hnsw-index",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-count",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--target-recall-at-100",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--minimum-hnsw-latency-reduction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--faiss-threads",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    faiss.omp_set_num_threads(
        args.faiss_threads
    )

    embedding_report = json.loads(
        args.embedding_report.read_text()
    )

    sweep = json.loads(
        args.hnsw_sweep_report.read_text()
    )

    persistence = json.loads(
        args.persistence_report.read_text()
    )

    if (
        sha256_file(
            args.catalog
        )
        != embedding_report[
            "artifact_sha256"
        ]
    ):
        raise RuntimeError(
            "Catalog SHA mismatch."
        )

    if (
        sha256_file(
            args.checkpoint
        )
        != embedding_report[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint SHA mismatch."
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    network = TwoTowerNetwork(
        TwoTowerConfig(
            **checkpoint[
                "network_config"
            ]
        )
    )

    network.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    catalog = (
        RetrievalCatalog.load_npz(
            args.catalog
        )
    )

    ef_search = int(
        sweep[
            "selection"
        ][
            "selected_ef_search"
        ]
    )

    exact = ExactInnerProductRetriever(
        catalog
    )

    flat = FaissFlatIPRetriever.load(
        catalog,
        args.flat_index,
    )

    hnsw = FaissHNSWRetriever.load(
        catalog,
        args.hnsw_index,
        ef_search=ef_search,
    )

    behaviors = load_behaviors(
        args.data_dir
        / "MINDsmall_train"
        / "behaviors.tsv"
    )

    split = (
        chronological_train_validation_split(
            behaviors,
            validation_fraction=(
                args.validation_fraction
            ),
        )
    )

    queries = build_validation_queries(
        split.validation,
        catalog=catalog,
        network=network,
        max_history_length=int(
            checkpoint[
                "protocol"
            ][
                "max_history_length"
            ]
        ),
        query_count=(
            args.query_count
        ),
        seed=args.seed,
    )

    query_matrix = np.ascontiguousarray(
        np.stack(
            [
                query.vector
                for query
                in queries
            ],
            axis=0,
        ),
        dtype=np.float32,
    )

    print(
        "Benchmarking request latency..."
    )

    exact_latency = (
        _request_latency(
            exact,
            queries,
            top_k=args.top_k,
        )
    )

    flat_latency = (
        _request_latency(
            flat,
            queries,
            top_k=args.top_k,
        )
    )

    hnsw_latency = (
        _request_latency(
            hnsw,
            queries,
            top_k=args.top_k,
        )
    )

    print(
        "Measuring ANN quality..."
    )

    recall_10 = []
    recall_50 = []
    recall_100 = []

    for query in queries:
        reference = [
            hit.news_id
            for hit
            in flat.retrieve(
                query.vector,
                top_k=100,
                exclude_news_ids=(
                    query.exclude_news_ids
                ),
            )
        ]

        candidate = [
            hit.news_id
            for hit
            in hnsw.retrieve(
                query.vector,
                top_k=100,
                exclude_news_ids=(
                    query.exclude_news_ids
                ),
            )
        ]

        recall_10.append(
            retrieval_recall_at_k(
                reference,
                candidate,
                k=10,
            )
        )

        recall_50.append(
            retrieval_recall_at_k(
                reference,
                candidate,
                k=50,
            )
        )

        recall_100.append(
            retrieval_recall_at_k(
                reference,
                candidate,
                k=100,
            )
        )

    hnsw_recall_10 = float(
        np.mean(
            recall_10
        )
    )

    hnsw_recall_50 = float(
        np.mean(
            recall_50
        )
    )

    hnsw_recall_100 = float(
        np.mean(
            recall_100
        )
    )

    batch_sizes = (
        1,
        16,
        64,
        256,
    )

    batch_qps = {}

    for batch_size in batch_sizes:
        batch_qps[
            str(batch_size)
        ] = {
            "numpy_exact": (
                _numpy_batch_qps(
                    query_matrix,
                    catalog,
                    batch_size=(
                        batch_size
                    ),
                    top_k=args.top_k,
                )
            ),
            "faiss_flat": (
                _faiss_batch_qps(
                    flat.index,
                    query_matrix,
                    batch_size=(
                        batch_size
                    ),
                    top_k=args.top_k,
                )
            ),
            "faiss_hnsw": (
                _faiss_batch_qps(
                    hnsw.index,
                    query_matrix,
                    batch_size=(
                        batch_size
                    ),
                    top_k=args.top_k,
                )
            ),
        }

    quality_ok = (
        hnsw_recall_100
        >= args.target_recall_at_100
    )

    latency_reduction = (
        1.0
        - (
            hnsw_latency[
                "p95"
            ]
            / flat_latency[
                "p95"
            ]
        )
    )

    hnsw_materially_faster = (
        latency_reduction
        >= args.minimum_hnsw_latency_reduction
    )

    if (
        quality_ok
        and hnsw_materially_faster
    ):
        selected_index = (
            "faiss_hnsw"
        )

        reason = (
            "HNSW met the Recall@100 target and "
            "reduced p95 request latency by the "
            "predeclared minimum margin."
        )
    else:
        selected_index = (
            "faiss_flat"
        )

        reason = (
            "IndexFlatIP retained because HNSW did not "
            "simultaneously meet both the quality and "
            "material p95-latency requirements."
        )

    payload = {
        "experiment": (
            "phase04_retrieval_system_benchmark"
        ),
        "faiss_version": (
            faiss.__version__
        ),
        "faiss_threads": (
            args.faiss_threads
        ),
        "article_count": (
            catalog.article_count
        ),
        "embedding_dim": (
            catalog.embedding_dim
        ),
        "query_count": (
            len(queries)
        ),
        "top_k": args.top_k,
        "quality": {
            "hnsw_recall_at_10": (
                hnsw_recall_10
            ),
            "hnsw_recall_at_50": (
                hnsw_recall_50
            ),
            "hnsw_recall_at_100": (
                hnsw_recall_100
            ),
            "target_recall_at_100": (
                args.target_recall_at_100
            ),
        },
        "single_query_latency_ms": {
            "numpy_exact": (
                exact_latency
            ),
            "faiss_flat": (
                flat_latency
            ),
            "faiss_hnsw": (
                hnsw_latency
            ),
        },
        "batch_backend_qps": (
            batch_qps
        ),
        "artifacts": {
            "catalog_vector_bytes": int(
                catalog.vectors.nbytes
            ),
            "flat_index_bytes": int(
                args.flat_index
                .stat()
                .st_size
            ),
            "hnsw_index_bytes": int(
                args.hnsw_index
                .stat()
                .st_size
            ),
        },
        "persistence_timing_ms": (
            persistence[
                "timing_ms"
            ]
        ),
        "hnsw_configuration": {
            "m": sweep["m"],
            "ef_construction": (
                sweep[
                    "ef_construction"
                ]
            ),
            "ef_search": (
                ef_search
            ),
        },
        "selection": {
            "selected_index": (
                selected_index
            ),
            "quality_ok": (
                quality_ok
            ),
            "hnsw_materially_faster": (
                hnsw_materially_faster
            ),
            "p95_latency_reduction_fraction": (
                latency_reduction
            ),
            "minimum_latency_reduction_fraction": (
                args.minimum_hnsw_latency_reduction
            ),
            "reason": reason,
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
