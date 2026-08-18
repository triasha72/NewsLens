"""Benchmark optimized MMR against the frozen correctness-first reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch

from newslens.data import load_behaviors
from newslens.evaluation.split import chronological_train_validation_split
from newslens.models.two_tower import TwoTowerConfig, TwoTowerNetwork
from newslens.reranking import (
    MMRConfig,
    maximal_marginal_relevance,
    maximal_marginal_relevance_vectorized,
)
from newslens.retrieval.catalog import RetrievalCatalog
from newslens.retrieval.faiss_flat import FaissFlatIPRetriever
from newslens.retrieval.queries import build_validation_queries


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """One frozen FAISS candidate pool used by both MMR implementations."""

    impression_id: str
    news_ids: tuple[str, ...]
    relevance_scores: np.ndarray
    vectors: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _latency_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        array.size == 0
        or not np.isfinite(array).all()
    ):
        raise RuntimeError(
            "Latency observations must be finite and nonempty."
        )

    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _workload_sha256(
    pools: tuple[CandidatePool, ...],
) -> str:
    digest = hashlib.sha256()

    for pool in pools:
        digest.update(
            pool.impression_id.encode("utf-8")
        )
        digest.update(b"\0")

        for news_id in pool.news_ids:
            digest.update(
                news_id.encode("utf-8")
            )
            digest.update(b"\0")

        digest.update(
            np.ascontiguousarray(
                pool.relevance_scores,
                dtype=np.float64,
            ).tobytes()
        )

    return digest.hexdigest()


def _run_reference(
    pool: CandidatePool,
    *,
    config: MMRConfig,
    top_k: int,
) -> tuple[str, ...]:
    return maximal_marginal_relevance(
        pool.news_ids,
        pool.relevance_scores,
        pool.vectors,
        top_k=top_k,
        config=config,
    )


def _run_optimized(
    pool: CandidatePool,
    *,
    config: MMRConfig,
    top_k: int,
) -> tuple[str, ...]:
    return maximal_marginal_relevance_vectorized(
        pool.news_ids,
        pool.relevance_scores,
        pool.vectors,
        top_k=top_k,
        config=config,
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
        "--phase05-audit-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--phase06-benchmark-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    phase05_audit = json.loads(
        args.phase05_audit_report.read_text()
    )

    phase06 = json.loads(
        args.phase06_benchmark_report.read_text()
    )

    if phase06["integrity"]["passed"] is not True:
        raise RuntimeError(
            "Frozen Phase-06E benchmark integrity did not pass."
        )

    protocol = phase06["protocol"]

    if (
        protocol["quality_claim_for_global_candidates"]
        != "none"
    ):
        raise RuntimeError(
            "Unexpected Phase-06 global-quality claim."
        )

    if protocol["retrieval_backend"] != "faiss_flat":
        raise RuntimeError(
            "Unexpected Phase-06 retrieval backend."
        )

    query_count = int(
        protocol["query_count"]
    )
    seed = int(
        protocol["seed"]
    )
    faiss_threads = int(
        protocol["faiss_threads"]
    )
    retrieval_k = int(
        protocol["retrieval_k"]
    )
    final_k = int(
        protocol["final_k"]
    )
    selected_lambda = float(
        protocol["selected_lambda"]
    )
    frozen_temperature = float(
        protocol["temperature"]
    )
    warmup_count = int(
        protocol["warmup_count"]
    )

    if query_count != 512:
        raise RuntimeError(
            "Frozen Phase-06 query count is not 512."
        )

    if retrieval_k != 100 or final_k != 10:
        raise RuntimeError(
            "Frozen retrieval/final depth differs from 100/10."
        )

    if selected_lambda != 0.80:
        raise RuntimeError(
            "Frozen Phase-06 MMR lambda is not 0.80."
        )

    faiss.omp_set_num_threads(
        faiss_threads
    )

    checkpoint_sha = _sha256(
        args.checkpoint
    )
    catalog_sha = _sha256(
        args.catalog
    )

    if (
        checkpoint_sha
        != phase06["frozen_inputs"][
            "phase03_checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Phase-03 checkpoint SHA mismatch."
        )

    if (
        catalog_sha
        != phase06["frozen_inputs"][
            "phase04_catalog_sha256"
        ]
    ):
        raise RuntimeError(
            "Phase-04 catalog SHA mismatch."
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    network = TwoTowerNetwork(
        TwoTowerConfig(
            **checkpoint["network_config"]
        )
    )

    network.load_state_dict(
        checkpoint["model_state_dict"]
    )
    network.to("cpu")
    network.eval()

    temperature = float(
        network.config.temperature
    )

    if not np.isclose(
        temperature,
        frozen_temperature,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Two-tower temperature differs from frozen Phase-06 value."
        )

    catalog = RetrievalCatalog.load_npz(
        args.catalog
    )

    retriever = FaissFlatIPRetriever(
        catalog
    )

    behaviors = load_behaviors(
        args.data_dir
        / "MINDsmall_train"
        / "behaviors.tsv"
    )

    phase03 = chronological_train_validation_split(
        behaviors,
        validation_fraction=0.20,
    )

    phase07 = chronological_train_validation_split(
        phase03.validation,
        validation_fraction=0.20,
    )

    if (
        phase03.cutoff.isoformat()
        != phase05_audit["phase03"]["cutoff"]
    ):
        raise RuntimeError(
            "Phase-03 cutoff differs from frozen audit."
        )

    if (
        phase07.cutoff.isoformat()
        != phase05_audit["phase05"]["cutoff"]
    ):
        raise RuntimeError(
            "Phase-07 benchmark split differs from frozen Phase-06 split."
        )

    queries = build_validation_queries(
        phase07.validation,
        catalog=catalog,
        network=network,
        max_history_length=int(
            checkpoint["protocol"]["max_history_length"]
        ),
        query_count=query_count,
        seed=seed,
    )

    if len(queries) != query_count:
        raise RuntimeError(
            "Did not reconstruct all frozen benchmark queries."
        )

    pools: list[CandidatePool] = []

    short_candidate_pools = 0
    duplicate_candidate_violations = 0
    exclusion_violations = 0

    for query in queries:
        hits = retriever.retrieve(
            query.vector,
            top_k=retrieval_k,
            exclude_news_ids=query.exclude_news_ids,
        )

        news_ids = tuple(
            hit.news_id
            for hit in hits
        )

        if len(news_ids) < retrieval_k:
            short_candidate_pools += 1

        if len(news_ids) != len(set(news_ids)):
            duplicate_candidate_violations += 1

        excluded = set(
            query.exclude_news_ids
        )

        if any(
            news_id in excluded
            for news_id in news_ids
        ):
            exclusion_violations += 1

        positions = [
            catalog.id_to_position[
                news_id
            ]
            for news_id in news_ids
        ]

        vectors = np.ascontiguousarray(
            catalog.vectors[
                positions
            ],
            dtype=np.float32,
        )

        relevance_scores = np.asarray(
            [
                float(hit.score)
                / temperature
                for hit in hits
            ],
            dtype=np.float64,
        )

        pools.append(
            CandidatePool(
                impression_id=(
                    query.impression_id
                ),
                news_ids=news_ids,
                relevance_scores=(
                    relevance_scores
                ),
                vectors=vectors,
            )
        )

    frozen_occurrences = int(
        phase06["accounting"][
            "candidate_pool_occurrences"
        ]
    )

    frozen_unique = int(
        phase06["accounting"][
            "candidate_pool_unique_articles"
        ]
    )

    candidate_occurrences = sum(
        len(pool.news_ids)
        for pool in pools
    )

    unique_articles = len(
        {
            news_id
            for pool in pools
            for news_id in pool.news_ids
        }
    )

    workload_matches_phase06_accounting = (
        candidate_occurrences
        == frozen_occurrences
        and unique_articles
        == frozen_unique
        and short_candidate_pools == 0
        and duplicate_candidate_violations == 0
        and exclusion_violations == 0
    )

    if not workload_matches_phase06_accounting:
        raise RuntimeError(
            "Phase-07 workload does not reproduce "
            "the frozen Phase-06 candidate accounting."
        )

    frozen_pools = tuple(
        pools
    )

    workload_sha = _workload_sha256(
        frozen_pools
    )

    config = MMRConfig(
        lambda_weight=selected_lambda
    )

    effective_warmup = min(
        warmup_count,
        len(frozen_pools),
    )

    for pool in frozen_pools[
        :effective_warmup
    ]:
        reference = _run_reference(
            pool,
            config=config,
            top_k=final_k,
        )

        optimized = _run_optimized(
            pool,
            config=config,
            top_k=final_k,
        )

        if optimized != reference:
            raise RuntimeError(
                "Parity failed during warm-up."
            )

    reference_ms: list[float] = []
    optimized_ms: list[float] = []

    parity_mismatches: list[
        dict[str, object]
    ] = []

    for index, pool in enumerate(
        frozen_pools
    ):
        if index % 2 == 0:
            started = time.perf_counter_ns()

            reference = _run_reference(
                pool,
                config=config,
                top_k=final_k,
            )

            reference_ms.append(
                (
                    time.perf_counter_ns()
                    - started
                )
                / 1_000_000.0
            )

            started = time.perf_counter_ns()

            optimized = _run_optimized(
                pool,
                config=config,
                top_k=final_k,
            )

            optimized_ms.append(
                (
                    time.perf_counter_ns()
                    - started
                )
                / 1_000_000.0
            )

        else:
            started = time.perf_counter_ns()

            optimized = _run_optimized(
                pool,
                config=config,
                top_k=final_k,
            )

            optimized_ms.append(
                (
                    time.perf_counter_ns()
                    - started
                )
                / 1_000_000.0
            )

            started = time.perf_counter_ns()

            reference = _run_reference(
                pool,
                config=config,
                top_k=final_k,
            )

            reference_ms.append(
                (
                    time.perf_counter_ns()
                    - started
                )
                / 1_000_000.0
            )

        if optimized != reference:
            parity_mismatches.append(
                {
                    "query_index": index,
                    "impression_id": (
                        pool.impression_id
                    ),
                    "reference": list(
                        reference
                    ),
                    "optimized": list(
                        optimized
                    ),
                }
            )

    reference_stats = _latency_stats(
        reference_ms
    )

    optimized_stats = _latency_stats(
        optimized_ms
    )

    frozen_reference_p95 = float(
        phase06["latency_ms"][
            "mmr_rerank_top100_to_top10"
        ][
            "p95"
        ]
    )

    optimized_p95 = float(
        optimized_stats["p95"]
    )

    speedup_vs_frozen = (
        frozen_reference_p95
        / optimized_p95
    )

    speedup_vs_measured = (
        float(
            reference_stats["p95"]
        )
        / optimized_p95
    )

    parity_count = (
        len(frozen_pools)
        - len(parity_mismatches)
    )

    exact_parity_passed = (
        len(parity_mismatches) == 0
        and parity_count == query_count
    )

    speedup_gate_passed = (
        speedup_vs_frozen >= 20.0
    )

    latency_gate_passed = (
        optimized_p95 <= 10.0
    )

    promotion_gate_passed = (
        exact_parity_passed
        and speedup_gate_passed
        and latency_gate_passed
    )

    payload = {
        "experiment": (
            "phase07_mmr_vectorization_benchmark"
        ),
        "protocol": {
            "benchmark_source": (
                "phase06_faiss_mmr_serving_benchmark"
            ),
            "query_count": query_count,
            "seed": seed,
            "faiss_threads": (
                faiss_threads
            ),
            "retrieval_k": retrieval_k,
            "final_k": final_k,
            "lambda": selected_lambda,
            "temperature": temperature,
            "warmup_count": (
                effective_warmup
            ),
            "execution_order": (
                "alternating_reference_optimized"
            ),
            "timing_scope": (
                "mmr_top100_to_top10_only"
            ),
        },
        "frozen_inputs": {
            "phase03_checkpoint_sha256": (
                checkpoint_sha
            ),
            "phase04_catalog_sha256": (
                catalog_sha
            ),
            "workload_sha256": (
                workload_sha
            ),
            "phase06_reference_p95_ms": (
                frozen_reference_p95
            ),
        },
        "workload": {
            "candidate_pool_occurrences": (
                candidate_occurrences
            ),
            "candidate_pool_unique_articles": (
                unique_articles
            ),
            "short_candidate_pools": (
                short_candidate_pools
            ),
            "duplicate_candidate_violations": (
                duplicate_candidate_violations
            ),
            "history_exclusion_violations": (
                exclusion_violations
            ),
            "matches_phase06_accounting": (
                workload_matches_phase06_accounting
            ),
        },
        "parity": {
            "parity_count": (
                parity_count
            ),
            "parity_fraction": (
                parity_count
                / query_count
            ),
            "mismatch_count": (
                len(parity_mismatches)
            ),
            "mismatches": (
                parity_mismatches
            ),
            "passed": (
                exact_parity_passed
            ),
        },
        "latency_ms": {
            "reference": (
                reference_stats
            ),
            "optimized": (
                optimized_stats
            ),
            "speedup_vs_frozen_phase06_p95": (
                speedup_vs_frozen
            ),
            "speedup_vs_measured_reference_p95": (
                speedup_vs_measured
            ),
        },
        "promotion_gate": {
            "required_parity_count": (
                query_count
            ),
            "required_minimum_speedup": (
                20.0
            ),
            "maximum_optimized_p95_ms": (
                10.0
            ),
            "exact_parity_passed": (
                exact_parity_passed
            ),
            "speedup_gate_passed": (
                speedup_gate_passed
            ),
            "latency_gate_passed": (
                latency_gate_passed
            ),
            "passed": (
                promotion_gate_passed
            ),
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

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
