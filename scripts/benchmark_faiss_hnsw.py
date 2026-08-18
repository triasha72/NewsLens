"""Benchmark HNSW quality and latency against exact FAISS retrieval."""

from __future__ import annotations

import argparse
import hashlib
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
from newslens.retrieval.faiss_flat import (
    FaissFlatIPRetriever,
)
from newslens.retrieval.faiss_hnsw import (
    FaissHNSWRetriever,
)
from newslens.retrieval.metrics import (
    retrieval_recall_at_k,
)
from newslens.retrieval.queries import (
    build_validation_queries,
)


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _latency_stats(
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
        "--training-report",
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
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-count",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--m",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--ef-search-values",
        default="16,32,64,128",
    )

    parser.add_argument(
        "--target-recall-at-100",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--cutoff",
        default="2019-11-13T20:36:26",
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

    training_report = json.loads(
        args.training_report.read_text()
    )

    embedding_report = json.loads(
        args.embedding_report.read_text()
    )

    if (
        _sha256(args.checkpoint)
        != training_report[
            "checkpoint"
        ]["sha256"]
    ):
        raise RuntimeError(
            "Checkpoint SHA mismatch."
        )

    if (
        _sha256(args.catalog)
        != embedding_report[
            "artifact_sha256"
        ]
    ):
        raise RuntimeError(
            "Embedding artifact SHA mismatch."
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

    if (
        split.cutoff.isoformat()
        != args.cutoff
    ):
        raise RuntimeError(
            "Chronological cutoff mismatch."
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

    flat = FaissFlatIPRetriever(
        catalog
    )

    print(
        "Building exact FAISS reference rankings..."
    )

    reference_rankings = []

    for query in queries:
        reference_rankings.append(
            [
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
        )

    print(
        "Building HNSW index..."
    )

    build_started = (
        time.perf_counter_ns()
    )

    hnsw = FaissHNSWRetriever(
        catalog,
        m=args.m,
        ef_construction=(
            args.ef_construction
        ),
        ef_search=16,
    )

    build_time_ms = (
        time.perf_counter_ns()
        - build_started
    ) / 1_000_000.0

    ef_values = tuple(
        int(value.strip())
        for value
        in args.ef_search_values.split(
            ","
        )
        if value.strip()
    )

    results = []

    for ef_search in ef_values:
        hnsw.ef_search = (
            ef_search
        )

        for query in queries[
            :min(
                50,
                len(queries),
            )
        ]:
            hnsw.retrieve(
                query.vector,
                top_k=100,
                exclude_news_ids=(
                    query.exclude_news_ids
                ),
            )

        recall_10 = []
        recall_50 = []
        recall_100 = []
        latency_ms = []

        short_rankings = 0
        exclusion_violations = 0

        for query, reference_ids in zip(
            queries,
            reference_rankings,
            strict=True,
        ):
            started = (
                time.perf_counter_ns()
            )

            hits = hnsw.retrieve(
                query.vector,
                top_k=100,
                exclude_news_ids=(
                    query.exclude_news_ids
                ),
            )

            latency_ms.append(
                (
                    time.perf_counter_ns()
                    - started
                )
                / 1_000_000.0
            )

            if len(hits) != 100:
                short_rankings += 1

            excluded = set(
                query.exclude_news_ids
            )

            if any(
                hit.news_id
                in excluded
                for hit in hits
            ):
                exclusion_violations += 1

            candidate_ids = [
                hit.news_id
                for hit in hits
            ]

            recall_10.append(
                retrieval_recall_at_k(
                    reference_ids,
                    candidate_ids,
                    k=10,
                )
            )

            recall_50.append(
                retrieval_recall_at_k(
                    reference_ids,
                    candidate_ids,
                    k=50,
                )
            )

            recall_100.append(
                retrieval_recall_at_k(
                    reference_ids,
                    candidate_ids,
                    k=100,
                )
            )

        recall_at_100 = float(
            np.mean(
                recall_100
            )
        )

        quality_gate = (
            recall_at_100
            >= args.target_recall_at_100
            and short_rankings == 0
            and exclusion_violations == 0
        )

        results.append(
            {
                "ef_search": (
                    ef_search
                ),
                "recall_at_10": float(
                    np.mean(
                        recall_10
                    )
                ),
                "recall_at_50": float(
                    np.mean(
                        recall_50
                    )
                ),
                "recall_at_100": (
                    recall_at_100
                ),
                "latency_ms": (
                    _latency_stats(
                        latency_ms
                    )
                ),
                "short_rankings": (
                    short_rankings
                ),
                "exclusion_violations": (
                    exclusion_violations
                ),
                "quality_gate_passed": (
                    quality_gate
                ),
            }
        )

    eligible = [
        result
        for result in results
        if result[
            "quality_gate_passed"
        ]
    ]

    if eligible:
        selected = min(
            eligible,
            key=lambda result: (
                result[
                    "latency_ms"
                ]["p95"]
            ),
        )

        target_met = True

        reason = (
            "Lowest-p95 HNSW configuration meeting "
            "Recall@100 >= target."
        )
    else:
        selected = max(
            results,
            key=lambda result: (
                result[
                    "recall_at_100"
                ],
                -result[
                    "latency_ms"
                ]["p95"],
            ),
        )

        target_met = False

        reason = (
            "No HNSW configuration met the quality target; "
            "highest-recall configuration retained only for "
            "final systems comparison."
        )

    payload = {
        "experiment": (
            "phase04_hnsw_sweep"
        ),
        "dataset": (
            "MINDsmall_train"
        ),
        "official_dev_used": False,
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
        "m": args.m,
        "ef_construction": (
            args.ef_construction
        ),
        "target_recall_at_100": (
            args.target_recall_at_100
        ),
        "index_build_time_ms": (
            build_time_ms
        ),
        "results": (
            results
        ),
        "selection": {
            "selected_ef_search": (
                selected[
                    "ef_search"
                ]
            ),
            "quality_target_met": (
                target_met
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
