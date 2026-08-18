"""Evaluate exact FAISS retrieval against the frozen NumPy oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
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
        default=1000,
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

    args = parser.parse_args()

    training_report = json.loads(
        args.training_report.read_text()
    )

    embedding_report = json.loads(
        args.embedding_report.read_text()
    )

    if training_report[
        "protocol"
    ]["limited_training_run"]:
        raise RuntimeError(
            "Refusing to evaluate a limited Phase-03 checkpoint."
        )

    expected_checkpoint_sha = (
        training_report[
            "checkpoint"
        ]["sha256"]
    )

    actual_checkpoint_sha = (
        _sha256(
            args.checkpoint
        )
    )

    if (
        actual_checkpoint_sha
        != expected_checkpoint_sha
    ):
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch."
        )

    expected_catalog_sha = (
        embedding_report[
            "artifact_sha256"
        ]
    )

    actual_catalog_sha = (
        _sha256(
            args.catalog
        )
    )

    if (
        actual_catalog_sha
        != expected_catalog_sha
    ):
        raise RuntimeError(
            "Frozen embedding catalog SHA-256 mismatch."
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint[
            "protocol"
        ][
            "cutoff_timestamp"
        ]
        != args.cutoff
    ):
        raise RuntimeError(
            "Checkpoint cutoff mismatch."
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

    if (
        catalog.article_count
        != embedding_report[
            "article_count"
        ]
    ):
        raise RuntimeError(
            "Embedding catalog article-count mismatch."
        )

    if (
        catalog.embedding_dim
        != embedding_report[
            "embedding_dim"
        ]
    ):
        raise RuntimeError(
            "Embedding catalog dimension mismatch."
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

    if len(split.validation) != 31_393:
        raise RuntimeError(
            "Unexpected chronological validation size."
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

    exact = ExactInnerProductRetriever(
        catalog
    )

    flat = FaissFlatIPRetriever(
        catalog
    )

    cutoffs = (
        10,
        50,
        100,
    )

    recalls: dict[
        int,
        list[float],
    ] = {
        cutoff: []
        for cutoff in cutoffs
    }

    exact_order_matches = {
        cutoff: 0
        for cutoff in cutoffs
    }

    exclusion_violations = 0
    short_rankings = 0

    for query in queries:
        exact_hits = exact.retrieve(
            query.vector,
            top_k=100,
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        flat_hits = flat.retrieve(
            query.vector,
            top_k=100,
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        if len(flat_hits) != 100:
            short_rankings += 1

        excluded = set(
            query.exclude_news_ids
        )

        if any(
            hit.news_id
            in excluded
            for hit in flat_hits
        ):
            exclusion_violations += 1

        exact_ids = [
            hit.news_id
            for hit in exact_hits
        ]

        flat_ids = [
            hit.news_id
            for hit in flat_hits
        ]

        for cutoff in cutoffs:
            recalls[
                cutoff
            ].append(
                retrieval_recall_at_k(
                    exact_ids,
                    flat_ids,
                    k=cutoff,
                )
            )

            if (
                exact_ids[
                    :cutoff
                ]
                == flat_ids[
                    :cutoff
                ]
            ):
                exact_order_matches[
                    cutoff
                ] += 1

    recall_summary = {
        f"recall_at_{cutoff}": float(
            np.mean(
                recalls[
                    cutoff
                ]
            )
        )
        for cutoff in cutoffs
    }

    order_summary = {
        (
            f"exact_order_match_fraction_at_{cutoff}"
        ): (
            exact_order_matches[
                cutoff
            ]
            / len(queries)
        )
        for cutoff in cutoffs
    }

    parity_passed = (
        all(
            value >= 0.999999
            for value
            in recall_summary.values()
        )
        and exclusion_violations == 0
        and short_rankings == 0
    )

    payload = {
        "experiment": (
            "phase04_faiss_flat_parity"
        ),
        "dataset": (
            "MINDsmall_train"
        ),
        "official_dev_used": False,
        "cutoff_timestamp": (
            args.cutoff
        ),
        "faiss_version": (
            faiss.__version__
        ),
        "checkpoint_sha256": (
            actual_checkpoint_sha
        ),
        "embedding_artifact_sha256": (
            actual_catalog_sha
        ),
        "query_count": len(
            queries
        ),
        "article_count": (
            catalog.article_count
        ),
        "embedding_dim": (
            catalog.embedding_dim
        ),
        "metric": (
            "inner_product"
        ),
        **recall_summary,
        **order_summary,
        "exclusion_violations": (
            exclusion_violations
        ),
        "short_rankings": (
            short_rankings
        ),
        "parity_passed": (
            parity_passed
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

    print()
    print("=" * 78)
    print("PHASE 04B — FAISS INDEXFLATIP PARITY")
    print("=" * 78)

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
