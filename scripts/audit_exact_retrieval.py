"""Audit the frozen exact vector-retrieval implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

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


def _percentile(
    values: list[float],
    percentile: float,
) -> float:
    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            percentile,
        )
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
        "--output",
        type=Path,
        required=True,
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
        "--query-count",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    report = json.loads(
        args.training_report.read_text()
    )

    expected_sha = (
        report[
            "checkpoint"
        ]["sha256"]
    )

    actual_sha = _sha256(
        args.checkpoint
    )

    if actual_sha != expected_sha:
        raise RuntimeError(
            "Checkpoint SHA mismatch."
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    catalog = (
        RetrievalCatalog.load_npz(
            args.catalog
        )
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

    data_path = (
        args.data_dir
        / "MINDsmall_train"
        / "behaviors.tsv"
    )

    behaviors = (
        load_behaviors(
            data_path
        )
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

    queries = (
        build_validation_queries(
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
    )

    retriever = (
        ExactInnerProductRetriever(
            catalog
        )
    )

    deterministic_mismatches = 0
    exclusion_violations = 0
    short_rankings = 0
    score_order_violations = 0
    latencies_ms: list[
        float
    ] = []

    for query in queries:
        started = (
            time.perf_counter_ns()
        )

        first = retriever.retrieve(
            query.vector,
            top_k=args.top_k,
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        elapsed_ms = (
            time.perf_counter_ns()
            - started
        ) / 1_000_000.0

        latencies_ms.append(
            elapsed_ms
        )

        second = retriever.retrieve(
            query.vector,
            top_k=args.top_k,
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        if first != second:
            deterministic_mismatches += 1

        excluded = set(
            query.exclude_news_ids
        )

        if any(
            hit.news_id
            in excluded
            for hit in first
        ):
            exclusion_violations += 1

        if len(first) != args.top_k:
            short_rankings += 1

        if any(
            first[index].score
            < first[index + 1].score
            for index
            in range(
                len(first) - 1
            )
        ):
            score_order_violations += 1

    payload = {
        "experiment": (
            "phase04_exact_retrieval_audit"
        ),
        "query_count": len(
            queries
        ),
        "top_k": args.top_k,
        "article_count": (
            catalog.article_count
        ),
        "embedding_dim": (
            catalog.embedding_dim
        ),
        "deterministic_mismatches": (
            deterministic_mismatches
        ),
        "exclusion_violations": (
            exclusion_violations
        ),
        "short_rankings": (
            short_rankings
        ),
        "score_order_violations": (
            score_order_violations
        ),
        "latency_ms": {
            "mean": float(
                np.mean(
                    latencies_ms
                )
            ),
            "p50": _percentile(
                latencies_ms,
                50.0,
            ),
            "p95": _percentile(
                latencies_ms,
                95.0,
            ),
            "p99": _percentile(
                latencies_ms,
                99.0,
            ),
        },
        "passed": (
            deterministic_mismatches == 0
            and exclusion_violations == 0
            and short_rankings == 0
            and score_order_violations == 0
        ),
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

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
