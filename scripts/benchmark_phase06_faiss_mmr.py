"""Benchmark frozen Phase-06 FAISS top-100 -> MMR top-10 behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from statistics import fmean

import faiss
import numpy as np
import pandas as pd
import torch

from newslens.data import load_behaviors, load_news
from newslens.evaluation.diversity_exposure import (
    gini_coefficient,
    intra_list_diversity,
    shannon_entropy,
    top_fraction_share,
)
from newslens.evaluation.split import chronological_train_validation_split
from newslens.models.two_tower import TwoTowerConfig, TwoTowerNetwork
from newslens.reranking import MMRConfig, maximal_marginal_relevance
from newslens.retrieval.catalog import RetrievalCatalog
from newslens.retrieval.faiss_flat import FaissFlatIPRetriever
from newslens.retrieval.queries import build_validation_queries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _latency_stats(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        array.size == 0
        or not np.isfinite(array).all()
    ):
        raise RuntimeError(
            "Latency measurements must be finite and nonempty."
        )

    return {
        "mean": float(array.mean()),
        "p50": float(
            np.percentile(array, 50)
        ),
        "p95": float(
            np.percentile(array, 95)
        ),
        "p99": float(
            np.percentile(array, 99)
        ),
    }


def _metadata_maps(
    news: pd.DataFrame,
) -> tuple[
    dict[str, str],
    dict[str, str],
]:
    categories: dict[str, str] = {}
    subcategories: dict[str, str] = {}

    for row in news.itertuples(
        index=False
    ):
        news_id = str(row.news_id)

        category = (
            str(row.category).strip()
            or "__missing__"
        )

        subcategory = (
            str(row.subcategory).strip()
            or "__missing__"
        )

        categories[news_id] = category
        subcategories[news_id] = subcategory

    return categories, subcategories


def _mmr_from_hits(
    hits,
    *,
    catalog: RetrievalCatalog,
    lambda_weight: float,
    temperature: float,
    final_k: int,
) -> tuple[str, ...]:
    news_ids = tuple(
        hit.news_id
        for hit in hits
    )

    if not news_ids:
        return ()

    positions = [
        catalog.id_to_position[
            news_id
        ]
        for news_id in news_ids
    ]

    vectors = catalog.vectors[
        positions
    ]

    scores = np.asarray(
        [
            float(hit.score)
            / temperature
            for hit in hits
        ],
        dtype=np.float64,
    )

    return maximal_marginal_relevance(
        news_ids,
        scores,
        vectors,
        top_k=final_k,
        config=MMRConfig(
            lambda_weight=lambda_weight
        ),
    )


def _summary(
    rankings: list[
        tuple[str, ...]
    ],
    *,
    catalog: RetrievalCatalog,
    categories: dict[str, str],
    subcategories: dict[str, str],
) -> dict[str, object]:
    ild: list[float] = []
    unique_categories: list[float] = []
    unique_subcategories: list[float] = []
    category_entropy: list[float] = []
    subcategory_entropy: list[float] = []

    article_exposure: Counter[str] = (
        Counter()
    )

    category_exposure: Counter[str] = (
        Counter()
    )

    subcategory_exposure: Counter[str] = (
        Counter()
    )

    for ranking in rankings:
        if not ranking:
            continue

        positions = [
            catalog.id_to_position[
                news_id
            ]
            for news_id in ranking
        ]

        vectors = catalog.vectors[
            positions
        ]

        cats = [
            categories[news_id]
            for news_id in ranking
        ]

        subcats = [
            subcategories[news_id]
            for news_id in ranking
        ]

        ild.append(
            intra_list_diversity(
                vectors
            )
        )

        unique_categories.append(
            float(
                len(
                    set(cats)
                )
            )
        )

        unique_subcategories.append(
            float(
                len(
                    set(subcats)
                )
            )
        )

        category_entropy.append(
            shannon_entropy(
                cats
            )
        )

        subcategory_entropy.append(
            shannon_entropy(
                subcats
            )
        )

        for (
            news_id,
            category,
            subcategory,
        ) in zip(
            ranking,
            cats,
            subcats,
            strict=True,
        ):
            article_exposure[
                news_id
            ] += 1

            category_exposure[
                category
            ] += 1

            subcategory_exposure[
                subcategory
            ] += 1

    exposure_vector = np.asarray(
        [
            article_exposure.get(
                news_id,
                0,
            )
            for news_id
            in catalog.news_ids
        ],
        dtype=np.float64,
    )

    unique_exposed = int(
        np.count_nonzero(
            exposure_vector
        )
    )

    return {
        "diversity": {
            "mean_intra_list_diversity": (
                float(
                    fmean(ild)
                )
            ),
            "mean_unique_categories": (
                float(
                    fmean(
                        unique_categories
                    )
                )
            ),
            "mean_unique_subcategories": (
                float(
                    fmean(
                        unique_subcategories
                    )
                )
            ),
            "mean_category_entropy": (
                float(
                    fmean(
                        category_entropy
                    )
                )
            ),
            "mean_subcategory_entropy": (
                float(
                    fmean(
                        subcategory_entropy
                    )
                )
            ),
        },
        "exposure": {
            "unique_exposed_articles": (
                unique_exposed
            ),
            "catalog_coverage": float(
                unique_exposed
                / catalog.article_count
            ),
            "exposure_gini": (
                gini_coefficient(
                    exposure_vector
                )
            ),
            "top_1_percent_exposure_share": (
                top_fraction_share(
                    exposure_vector,
                    fraction=0.01,
                )
            ),
            "top_10_percent_exposure_share": (
                top_fraction_share(
                    exposure_vector,
                    fraction=0.10,
                )
            ),
            "category_exposure_entropy": (
                shannon_entropy(
                    label
                    for label, count
                    in category_exposure.items()
                    for _ in range(
                        count
                    )
                )
            ),
            "subcategory_exposure_entropy": (
                shannon_entropy(
                    label
                    for label, count
                    in subcategory_exposure.items()
                    for _ in range(
                        count
                    )
                )
            ),
        },
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
        "--phase05-audit-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--phase05-final-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--policy-selection-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--phase03-validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--ranker-validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--query-count",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--final-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--warmup-count",
        type=int,
        default=50,
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

    if (
        args.retrieval_k
        < args.final_k
    ):
        raise RuntimeError(
            "retrieval-k must be >= final-k."
        )

    faiss.omp_set_num_threads(
        args.faiss_threads
    )

    embedding_report = json.loads(
        args.embedding_report.read_text()
    )

    phase05_audit = json.loads(
        args.phase05_audit_report.read_text()
    )

    phase05_final = json.loads(
        args.phase05_final_report.read_text()
    )

    selection = json.loads(
        args.policy_selection_report.read_text()
    )

    checkpoint_sha = _sha256(
        args.checkpoint
    )

    catalog_sha = _sha256(
        args.catalog
    )

    if (
        checkpoint_sha
        != embedding_report[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Phase-03 checkpoint SHA mismatch."
        )

    if (
        catalog_sha
        != embedding_report[
            "artifact_sha256"
        ]
    ):
        raise RuntimeError(
            "Phase-04 catalog SHA mismatch."
        )

    if (
        phase05_final[
            "selected_retrieval_backend"
        ]
        != "faiss_flat"
    ):
        raise RuntimeError(
            "Frozen retrieval backend is not faiss_flat."
        )

    selected_lambda = float(
        selection[
            "selection"
        ][
            "selected_lambda"
        ]
    )

    if selected_lambda != 0.80:
        raise RuntimeError(
            "Frozen Phase-06D policy is not lambda=0.80."
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

    network.to(
        "cpu"
    )

    network.eval()

    catalog = (
        RetrievalCatalog.load_npz(
            args.catalog
        )
    )

    retriever = (
        FaissFlatIPRetriever(
            catalog
        )
    )

    root = (
        args.data_dir
        / "MINDsmall_train"
    )

    news = load_news(
        root
        / "news.tsv"
    )

    behaviors = load_behaviors(
        root
        / "behaviors.tsv"
    )

    phase03 = (
        chronological_train_validation_split(
            behaviors,
            validation_fraction=(
                args.phase03_validation_fraction
            ),
        )
    )

    phase06 = (
        chronological_train_validation_split(
            phase03.validation,
            validation_fraction=(
                args.ranker_validation_fraction
            ),
        )
    )

    if (
        phase03.cutoff.isoformat()
        != phase05_audit[
            "phase03"
        ][
            "cutoff"
        ]
    ):
        raise RuntimeError(
            "Phase-03 cutoff differs from frozen audit."
        )

    if (
        phase06.cutoff.isoformat()
        != phase05_audit[
            "phase05"
        ][
            "cutoff"
        ]
    ):
        raise RuntimeError(
            "Phase-06 cutoff differs from frozen audit."
        )

    queries = (
        build_validation_queries(
            phase06.validation,
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

    if (
        len(queries)
        != args.query_count
    ):
        raise RuntimeError(
            "Did not produce requested query count."
        )

    categories, subcategories = (
        _metadata_maps(
            news
        )
    )

    temperature = float(
        network.config.temperature
    )

    warmup_count = min(
        args.warmup_count,
        len(queries),
    )

    for query in queries[
        :warmup_count
    ]:
        hits = retriever.retrieve(
            query.vector,
            top_k=(
                args.retrieval_k
            ),
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        _mmr_from_hits(
            hits,
            catalog=catalog,
            lambda_weight=(
                selected_lambda
            ),
            temperature=(
                temperature
            ),
            final_k=args.final_k,
        )

    retrieval_ms: list[float] = []
    rerank_ms: list[float] = []

    baseline_rankings: list[
        tuple[str, ...]
    ] = []

    mmr_rankings: list[
        tuple[str, ...]
    ] = []

    candidate_pools: list[
        tuple[str, ...]
    ] = []

    duplicate_violations = 0
    exclusion_violations = 0
    short_candidate_pools = 0

    for query in queries:
        started = (
            time.perf_counter_ns()
        )

        hits = retriever.retrieve(
            query.vector,
            top_k=(
                args.retrieval_k
            ),
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        retrieval_ms.append(
            (
                time.perf_counter_ns()
                - started
            )
            / 1_000_000.0
        )

        candidate_ids = tuple(
            hit.news_id
            for hit in hits
        )

        candidate_pools.append(
            candidate_ids
        )

        if (
            len(candidate_ids)
            < args.retrieval_k
        ):
            short_candidate_pools += 1

        if (
            len(candidate_ids)
            != len(
                set(candidate_ids)
            )
        ):
            duplicate_violations += 1

        excluded = set(
            query.exclude_news_ids
        )

        if any(
            news_id in excluded
            for news_id
            in candidate_ids
        ):
            exclusion_violations += 1

        baseline_rankings.append(
            candidate_ids[
                :args.final_k
            ]
        )

        started = (
            time.perf_counter_ns()
        )

        mmr_rankings.append(
            _mmr_from_hits(
                hits,
                catalog=catalog,
                lambda_weight=(
                    selected_lambda
                ),
                temperature=(
                    temperature
                ),
                final_k=(
                    args.final_k
                ),
            )
        )

        rerank_ms.append(
            (
                time.perf_counter_ns()
                - started
            )
            / 1_000_000.0
        )

    baseline_post_ms: list[
        float
    ] = []

    mmr_post_ms: list[
        float
    ] = []

    for query in queries:
        started = (
            time.perf_counter_ns()
        )

        hits = retriever.retrieve(
            query.vector,
            top_k=(
                args.retrieval_k
            ),
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        tuple(
            hit.news_id
            for hit
            in hits[
                :args.final_k
            ]
        )

        baseline_post_ms.append(
            (
                time.perf_counter_ns()
                - started
            )
            / 1_000_000.0
        )

    for query in queries:
        started = (
            time.perf_counter_ns()
        )

        hits = retriever.retrieve(
            query.vector,
            top_k=(
                args.retrieval_k
            ),
            exclude_news_ids=(
                query.exclude_news_ids
            ),
        )

        _mmr_from_hits(
            hits,
            catalog=catalog,
            lambda_weight=(
                selected_lambda
            ),
            temperature=(
                temperature
            ),
            final_k=args.final_k,
        )

        mmr_post_ms.append(
            (
                time.perf_counter_ns()
                - started
            )
            / 1_000_000.0
        )

    baseline = _summary(
        baseline_rankings,
        catalog=catalog,
        categories=categories,
        subcategories=(
            subcategories
        ),
    )

    mmr = _summary(
        mmr_rankings,
        catalog=catalog,
        categories=categories,
        subcategories=(
            subcategories
        ),
    )

    changed = sum(
        baseline_ranking
        != mmr_ranking
        for (
            baseline_ranking,
            mmr_ranking,
        ) in zip(
            baseline_rankings,
            mmr_rankings,
            strict=True,
        )
    )

    overlap = [
        (
            len(
                set(
                    baseline_ranking
                )
                & set(
                    mmr_ranking
                )
            )
            / args.final_k
        )
        for (
            baseline_ranking,
            mmr_ranking,
        )
        in zip(
            baseline_rankings,
            mmr_rankings,
            strict=True,
        )
    ]

    retrieval_stats = (
        _latency_stats(
            retrieval_ms
        )
    )

    rerank_stats = (
        _latency_stats(
            rerank_ms
        )
    )

    baseline_post = (
        _latency_stats(
            baseline_post_ms
        )
    )

    mmr_post = (
        _latency_stats(
            mmr_post_ms
        )
    )

    integrity_passed = (
        duplicate_violations == 0
        and exclusion_violations == 0
        and all(
            np.isfinite(value)
            for stats in (
                retrieval_stats,
                rerank_stats,
                baseline_post,
                mmr_post,
            )
            for value in (
                stats.values()
            )
        )
    )

    candidate_unique = len(
        {
            news_id
            for candidate_pool
            in candidate_pools
            for news_id
            in candidate_pool
        }
    )

    payload = {
        "experiment": (
            "phase06_faiss_mmr_serving_benchmark"
        ),
        "protocol": {
            "dataset": (
                "MINDsmall_train"
            ),
            "benchmark_status": (
                "systems_and_diversity_only"
            ),
            "quality_claim_for_global_candidates": (
                "none"
            ),
            "latency_scope": (
                "post_user_embedding"
            ),
            "official_dev_used": False,
            "query_count": (
                len(queries)
            ),
            "seed": args.seed,
            "faiss_threads": (
                args.faiss_threads
            ),
            "retrieval_backend": (
                "faiss_flat"
            ),
            "retrieval_k": (
                args.retrieval_k
            ),
            "final_k": (
                args.final_k
            ),
            "selected_lambda": (
                selected_lambda
            ),
            "temperature": (
                temperature
            ),
            "warmup_count": (
                warmup_count
            ),
        },
        "frozen_inputs": {
            "phase03_checkpoint_sha256": (
                checkpoint_sha
            ),
            "phase04_catalog_sha256": (
                catalog_sha
            ),
            "phase06_selected_policy": (
                selection[
                    "selection"
                ][
                    "selected_policy"
                ]
            ),
        },
        "accounting": {
            "candidate_pool_unique_articles": (
                candidate_unique
            ),
            "candidate_pool_occurrences": (
                sum(
                    map(
                        len,
                        candidate_pools,
                    )
                )
            ),
            "short_candidate_pools": (
                short_candidate_pools
            ),
            "duplicate_candidate_violations": (
                duplicate_violations
            ),
            "exclusion_violations": (
                exclusion_violations
            ),
            "changed_top10_count": (
                changed
            ),
            "changed_top10_fraction": (
                changed
                / len(queries)
            ),
            "mean_top10_set_overlap_fraction": (
                float(
                    np.mean(
                        overlap
                    )
                )
            ),
        },
        "latency_ms": {
            "retrieval_top100": (
                retrieval_stats
            ),
            "mmr_rerank_top100_to_top10": (
                rerank_stats
            ),
            "baseline_post_embedding_end_to_end": (
                baseline_post
            ),
            "mmr_post_embedding_end_to_end": (
                mmr_post
            ),
            "p95_mmr_absolute_overhead": (
                mmr_post[
                    "p95"
                ]
                - baseline_post[
                    "p95"
                ]
            ),
            "p95_mmr_over_baseline_ratio": (
                mmr_post[
                    "p95"
                ]
                / baseline_post[
                    "p95"
                ]
            ),
        },
        "baseline_relevance_top10": (
            baseline
        ),
        "selected_mmr_top10": (
            mmr
        ),
        "deltas_selected_minus_baseline": {
            "mean_intra_list_diversity": (
                mmr[
                    "diversity"
                ][
                    "mean_intra_list_diversity"
                ]
                - baseline[
                    "diversity"
                ][
                    "mean_intra_list_diversity"
                ]
            ),
            "mean_unique_categories": (
                mmr[
                    "diversity"
                ][
                    "mean_unique_categories"
                ]
                - baseline[
                    "diversity"
                ][
                    "mean_unique_categories"
                ]
            ),
            "mean_unique_subcategories": (
                mmr[
                    "diversity"
                ][
                    "mean_unique_subcategories"
                ]
                - baseline[
                    "diversity"
                ][
                    "mean_unique_subcategories"
                ]
            ),
            "unique_exposed_articles": (
                mmr[
                    "exposure"
                ][
                    "unique_exposed_articles"
                ]
                - baseline[
                    "exposure"
                ][
                    "unique_exposed_articles"
                ]
            ),
            "catalog_coverage": (
                mmr[
                    "exposure"
                ][
                    "catalog_coverage"
                ]
                - baseline[
                    "exposure"
                ][
                    "catalog_coverage"
                ]
            ),
            "exposure_gini": (
                mmr[
                    "exposure"
                ][
                    "exposure_gini"
                ]
                - baseline[
                    "exposure"
                ][
                    "exposure_gini"
                ]
            ),
            "top_1_percent_exposure_share": (
                mmr[
                    "exposure"
                ][
                    "top_1_percent_exposure_share"
                ]
                - baseline[
                    "exposure"
                ][
                    "top_1_percent_exposure_share"
                ]
            ),
        },
        "integrity": {
            "passed": (
                integrity_passed
            ),
            "no_duplicate_candidates": (
                duplicate_violations
                == 0
            ),
            "no_history_exclusion_violations": (
                exclusion_violations
                == 0
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

    if not integrity_passed:
        raise RuntimeError(
            "Phase-06E benchmark integrity failed."
        )


if __name__ == "__main__":
    main()
