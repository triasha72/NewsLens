"""Freeze the final Phase-04 retrieval-system selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--exact-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--flat-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--hnsw-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--persistence-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--benchmark-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    embedding = json.loads(
        args.embedding_report.read_text()
    )

    exact = json.loads(
        args.exact_report.read_text()
    )

    flat = json.loads(
        args.flat_report.read_text()
    )

    hnsw = json.loads(
        args.hnsw_report.read_text()
    )

    persistence = json.loads(
        args.persistence_report.read_text()
    )

    benchmark = json.loads(
        args.benchmark_report.read_text()
    )

    if not exact["passed"]:
        raise RuntimeError(
            "Exact retrieval audit failed."
        )

    if not flat[
        "parity_passed"
    ]:
        raise RuntimeError(
            "IndexFlatIP parity failed."
        )

    if not persistence[
        "passed"
    ]:
        raise RuntimeError(
            "Persistence audit failed."
        )

    selected = (
        benchmark[
            "selection"
        ][
            "selected_index"
        ]
    )

    payload = {
        "phase": (
            "phase04"
        ),
        "status": (
            "frozen"
        ),
        "selected_index": (
            selected
        ),
        "embedding": {
            "article_count": (
                embedding[
                    "article_count"
                ]
            ),
            "embedding_dim": (
                embedding[
                    "embedding_dim"
                ]
            ),
            "artifact_sha256": (
                embedding[
                    "artifact_sha256"
                ]
            ),
            "checkpoint_sha256": (
                embedding[
                    "checkpoint_sha256"
                ]
            ),
        },
        "exact_oracle": {
            "audit_passed": (
                exact[
                    "passed"
                ]
            ),
            "p95_ms": (
                exact[
                    "latency_ms"
                ][
                    "p95"
                ]
            ),
        },
        "faiss_flat": {
            "parity_passed": (
                flat[
                    "parity_passed"
                ]
            ),
            "recall_at_100": (
                flat[
                    "recall_at_100"
                ]
            ),
        },
        "hnsw": {
            "m": (
                hnsw["m"]
            ),
            "ef_construction": (
                hnsw[
                    "ef_construction"
                ]
            ),
            "selected_ef_search": (
                hnsw[
                    "selection"
                ][
                    "selected_ef_search"
                ]
            ),
            "quality_target_met_in_sweep": (
                hnsw[
                    "selection"
                ][
                    "quality_target_met"
                ]
            ),
            "recall_at_100": (
                benchmark[
                    "quality"
                ][
                    "hnsw_recall_at_100"
                ]
            ),
        },
        "single_query_latency_ms": (
            benchmark[
                "single_query_latency_ms"
            ]
        ),
        "batch_backend_qps": (
            benchmark[
                "batch_backend_qps"
            ]
        ),
        "artifacts": (
            benchmark[
                "artifacts"
            ]
        ),
        "selection": (
            benchmark[
                "selection"
            ]
        ),
    }

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.json_output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    flat_latency = (
        benchmark[
            "single_query_latency_ms"
        ][
            "faiss_flat"
        ]
    )

    hnsw_latency = (
        benchmark[
            "single_query_latency_ms"
        ][
            "faiss_hnsw"
        ]
    )

    markdown = f"""## Phase 04F: Final retrieval selection

Phase 04 freezes the candidate-retrieval subsystem built on the frozen
Phase-03 hard-negative two-tower representation.

### Frozen representation

- article count: {embedding["article_count"]:,}
- embedding dimension: {embedding["embedding_dim"]}
- similarity: inner product over L2-normalized vectors
- embedding SHA-256:
  `{embedding["artifact_sha256"]}`
- two-tower checkpoint SHA-256:
  `{embedding["checkpoint_sha256"]}`

### Exact FAISS validation

IndexFlatIP Recall@100 against the NumPy exact oracle:

- {flat["recall_at_100"]:.6f}

Exact-parity gate:

- {flat["parity_passed"]}

### HNSW candidate

- M: {hnsw["m"]}
- efConstruction: {hnsw["ef_construction"]}
- selected efSearch: {hnsw["selection"]["selected_ef_search"]}
- HNSW Recall@100: {benchmark["quality"]["hnsw_recall_at_100"]:.6f}

### Request latency

IndexFlatIP:

- p50: {flat_latency["p50"]:.6f} ms
- p95: {flat_latency["p95"]:.6f} ms
- p99: {flat_latency["p99"]:.6f} ms

HNSW:

- p50: {hnsw_latency["p50"]:.6f} ms
- p95: {hnsw_latency["p95"]:.6f} ms
- p99: {hnsw_latency["p99"]:.6f} ms

### Selected serving index

**{selected}**

{benchmark["selection"]["reason"]}

The selection followed the predeclared rule:

- HNSW Recall@100 must be at least 0.99; and
- HNSW p95 request latency must be at least 20% lower than IndexFlatIP.

Otherwise exact IndexFlatIP is retained at the current catalog scale.

No Phase-03 model parameters were modified during Phase 04.

Phase 04 is frozen after this result.
"""

    args.markdown_output.write_text(
        markdown
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
