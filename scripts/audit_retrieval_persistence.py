"""Audit FAISS persistence and filtered-serving behavior."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
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
from newslens.retrieval.persistence import (
    sha256_file,
    write_json,
)
from newslens.retrieval.queries import (
    build_validation_queries,
)


def _ids(
    hits,
) -> list[str]:
    return [
        hit.news_id
        for hit in hits
    ]


def _timed(
    function,
):
    started = (
        time.perf_counter_ns()
    )

    value = function()

    milliseconds = (
        time.perf_counter_ns()
        - started
    ) / 1_000_000.0

    return (
        value,
        milliseconds,
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
        "--flat-index-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--hnsw-index-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--query-count",
        type=int,
        default=256,
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

    faiss.omp_set_num_threads(
        1
    )

    embedding_report = json.loads(
        args.embedding_report.read_text()
    )

    sweep = json.loads(
        args.hnsw_sweep_report.read_text()
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
            "Embedding artifact SHA mismatch."
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

    ef_search = int(
        sweep[
            "selection"
        ][
            "selected_ef_search"
        ]
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

    flat, flat_build_ms = _timed(
        lambda: FaissFlatIPRetriever(
            catalog
        )
    )

    hnsw, hnsw_build_ms = _timed(
        lambda: FaissHNSWRetriever(
            catalog,
            m=int(
                sweep["m"]
            ),
            ef_construction=int(
                sweep[
                    "ef_construction"
                ]
            ),
            ef_search=(
                ef_search
            ),
        )
    )

    _, flat_write_ms = _timed(
        lambda: flat.save(
            args.flat_index_output
        )
    )

    _, hnsw_write_ms = _timed(
        lambda: hnsw.save(
            args.hnsw_index_output
        )
    )

    flat_restored, flat_reload_ms = (
        _timed(
            lambda: (
                FaissFlatIPRetriever.load(
                    catalog,
                    args.flat_index_output,
                )
            )
        )
    )

    hnsw_restored, hnsw_reload_ms = (
        _timed(
            lambda: (
                FaissHNSWRetriever.load(
                    catalog,
                    args.hnsw_index_output,
                    ef_search=(
                        ef_search
                    ),
                )
            )
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

    flat_reload_mismatches = 0
    hnsw_reload_mismatches = 0

    flat_exclusion_violations = 0
    hnsw_exclusion_violations = 0

    for query in queries:
        flat_before = flat.retrieve(
            query.vector,
            top_k=100,
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        flat_after = (
            flat_restored.retrieve(
                query.vector,
                top_k=100,
                exclude_news_ids=(
                    query.exclude_news_ids
                ),
            )
        )

        hnsw_before = hnsw.retrieve(
            query.vector,
            top_k=100,
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        hnsw_after = (
            hnsw_restored.retrieve(
                query.vector,
                top_k=100,
                exclude_news_ids=(
                    query.exclude_news_ids
                ),
            )
        )

        if (
            _ids(flat_before)
            != _ids(flat_after)
        ):
            flat_reload_mismatches += 1

        if (
            _ids(hnsw_before)
            != _ids(hnsw_after)
        ):
            hnsw_reload_mismatches += 1

        excluded = set(
            query.exclude_news_ids
        )

        if any(
            hit.news_id
            in excluded
            for hit
            in flat_after
        ):
            flat_exclusion_violations += 1

        if any(
            hit.news_id
            in excluded
            for hit
            in hnsw_after
        ):
            hnsw_exclusion_violations += 1

    metadata = {
        "article_count": (
            catalog.article_count
        ),
        "embedding_dim": (
            catalog.embedding_dim
        ),
        "flat_index_sha256": (
            sha256_file(
                args.flat_index_output
            )
        ),
        "hnsw_index_sha256": (
            sha256_file(
                args.hnsw_index_output
            )
        ),
        "flat_index_bytes": (
            args.flat_index_output
            .stat()
            .st_size
        ),
        "hnsw_index_bytes": (
            args.hnsw_index_output
            .stat()
            .st_size
        ),
        "hnsw": {
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
    }

    write_json(
        args.metadata_output,
        metadata,
    )

    passed = (
        flat_reload_mismatches == 0
        and hnsw_reload_mismatches == 0
        and flat_exclusion_violations == 0
        and hnsw_exclusion_violations == 0
    )

    payload = {
        "experiment": (
            "phase04_retrieval_persistence"
        ),
        **metadata,
        "query_count": (
            len(queries)
        ),
        "timing_ms": {
            "flat_build": (
                flat_build_ms
            ),
            "flat_write": (
                flat_write_ms
            ),
            "flat_reload": (
                flat_reload_ms
            ),
            "hnsw_build": (
                hnsw_build_ms
            ),
            "hnsw_write": (
                hnsw_write_ms
            ),
            "hnsw_reload": (
                hnsw_reload_ms
            ),
        },
        "flat_reload_mismatches": (
            flat_reload_mismatches
        ),
        "hnsw_reload_mismatches": (
            hnsw_reload_mismatches
        ),
        "flat_exclusion_violations": (
            flat_exclusion_violations
        ),
        "hnsw_exclusion_violations": (
            hnsw_exclusion_violations
        ),
        "passed": passed,
    }

    write_json(
        args.report_output,
        payload,
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
