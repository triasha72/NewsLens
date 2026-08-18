"""Audit diversity and item exposure for the selected NewsLens baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd
import torch

from newslens.data import (
    load_behaviors,
    load_news,
    parse_impressions,
)
from newslens.evaluation.diversity_exposure import (
    gini_coefficient,
    intra_list_diversity,
    shannon_entropy,
    top_fraction_share,
)
from newslens.evaluation.evaluator import (
    RankingExample,
    evaluate_rankings,
)
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.models.popularity import (
    PopularityRecommender,
)
from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
)
from newslens.reranking import (
    MMRConfig,
    maximal_marginal_relevance,
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
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


def _parse_history(
    value: object,
) -> tuple[str, ...]:
    if (
        value is None
        or pd.isna(value)
    ):
        return ()

    text = str(
        value
    ).strip()

    if not text:
        return ()

    return tuple(
        text.split()
    )


def _metadata_maps(
    news: pd.DataFrame,
) -> tuple[
    dict[str, str],
    dict[str, str],
]:
    category_by_id: dict[
        str,
        str
    ] = {}

    subcategory_by_id: dict[
        str,
        str
    ] = {}

    for row in news.itertuples(
        index=False
    ):
        news_id = str(
            row.news_id
        )

        category = str(
            row.category
        ).strip()

        subcategory = str(
            row.subcategory
        ).strip()

        category_by_id[
            news_id
        ] = (
            category
            if category
            else "__missing__"
        )

        subcategory_by_id[
            news_id
        ] = (
            subcategory
            if subcategory
            else "__missing__"
        )

    return (
        category_by_id,
        subcategory_by_id,
    )


def _build_popularity_groups(
    catalog: RetrievalCatalog,
    popularity: PopularityRecommender,
) -> dict[str, str]:
    ordered = sorted(
        catalog.news_ids,
        key=lambda news_id: (
            -popularity.statistics(
                news_id
            ).exposures,
            news_id,
        ),
    )

    count = len(
        ordered
    )

    head_end = int(
        np.ceil(
            count
            * 0.20
        )
    )

    mid_end = int(
        np.ceil(
            count
            * 0.50
        )
    )

    groups: dict[
        str,
        str
    ] = {}

    for index, news_id in enumerate(
        ordered
    ):
        if index < head_end:
            group = "head"
        elif index < mid_end:
            group = "mid"
        else:
            group = "tail"

        groups[
            news_id
        ] = group

    return groups


def _shares(
    counts: Counter[str],
) -> dict[str, float]:
    total = sum(
        counts.values()
    )

    if total <= 0:
        return {
            "head": 0.0,
            "mid": 0.0,
            "tail": 0.0,
        }

    return {
        group: (
            counts.get(
                group,
                0,
            )
            / total
        )
        for group in (
            "head",
            "mid",
            "tail",
        )
    }


def _mean_or_zero(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return float(
        fmean(
            values
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
        "--k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    embedding_report = json.loads(
        args.embedding_report.read_text()
    )

    phase05_audit = json.loads(
        args.phase05_audit_report.read_text()
    )

    phase05_final = json.loads(
        args.phase05_final_report.read_text()
    )

    if (
        phase05_final[
            "selected_model"
        ]
        != "phase03_two_tower_popularity"
    ):
        raise RuntimeError(
            "Phase 06 must start from the frozen Phase-05 selected model."
        )

    if (
        phase05_final[
            "learned_ranker_promoted"
        ]
        is not False
    ):
        raise RuntimeError(
            "Rejected Phase-05 ranker must not enter Phase 06."
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
            "Phase-04 embedding catalog SHA mismatch."
        )

    root = (
        args.data_dir
        / "MINDsmall_train"
    )

    print(
        "Loading MINDsmall_train..."
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
            "Phase-03 split differs from frozen Phase-05 audit."
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
            "Phase-06 benchmark split differs from frozen Phase-05 split."
        )

    if (
        len(
            phase06.validation
        )
        != phase05_audit[
            "phase05"
        ][
            "ranker_validation_impressions"
        ]
    ):
        raise RuntimeError(
            "Phase-06 benchmark size differs from frozen Phase-05 benchmark."
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

    network.eval()

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
            "Catalog article count differs from frozen embedding report."
        )

    popularity = (
        PopularityRecommender()
        .fit(
            phase03.train
        )
    )

    category_by_id, subcategory_by_id = (
        _metadata_maps(
            news
        )
    )

    popularity_group = (
        _build_popularity_groups(
            catalog,
            popularity,
        )
    )

    id_to_position = (
        catalog.id_to_position
    )

    max_history_length = int(
        checkpoint[
            "protocol"
        ][
            "max_history_length"
        ]
    )

    ranking_examples: list[
        RankingExample
    ] = []

    ild_values: list[
        float
    ] = []

    unique_category_values: list[
        float
    ] = []

    unique_subcategory_values: list[
        float
    ] = []

    category_entropy_values: list[
        float
    ] = []

    subcategory_entropy_values: list[
        float
    ] = []

    article_exposure = Counter()
    category_exposure = Counter()
    subcategory_exposure = Counter()

    recommendation_group_counts: Counter[
        str
    ] = Counter()

    opportunity_group_counts: Counter[
        str
    ] = Counter()

    two_tower_ranked = 0
    popularity_fallbacks = 0
    truncated_histories = 0

    lambda_one_parity_mismatches = 0
    candidate_occurrences = 0
    recommendation_occurrences = 0

    print(
        "Auditing selected Phase-06 baseline..."
    )

    network.to(
        "cpu"
    )

    with torch.no_grad():
        for row in phase06.validation.itertuples(
            index=False
        ):
            impression_id = str(
                row.impression_id
            )

            history_ids = _parse_history(
                row.history
            )

            parsed = parse_impressions(
                str(
                    row.impressions
                )
            )

            candidate_ids = tuple(
                news_id
                for news_id, _
                in parsed
            )

            if (
                len(candidate_ids)
                != len(
                    set(candidate_ids)
                )
            ):
                raise RuntimeError(
                    "Duplicate candidate IDs in benchmark impression."
                )

            missing_candidates = [
                news_id
                for news_id in candidate_ids
                if news_id
                not in id_to_position
            ]

            if missing_candidates:
                raise RuntimeError(
                    "Benchmark candidate missing from frozen catalog."
                )

            relevant_items = frozenset(
                news_id
                for news_id, label
                in parsed
                if label == 1
            )

            candidate_occurrences += len(
                candidate_ids
            )

            for news_id in candidate_ids:
                opportunity_group_counts[
                    popularity_group[
                        news_id
                    ]
                ] += 1

            usable_history = tuple(
                news_id
                for news_id in history_ids
                if news_id
                in id_to_position
            )

            if usable_history:
                if (
                    len(usable_history)
                    > max_history_length
                ):
                    truncated_histories += 1

                encoded_history = (
                    usable_history[
                        -max_history_length:
                    ]
                )

                history_positions = [
                    id_to_position[
                        news_id
                    ]
                    for news_id
                    in encoded_history
                ]

                history_embeddings = (
                    torch.from_numpy(
                        catalog.vectors[
                            history_positions
                        ]
                    )
                    .unsqueeze(0)
                )

                history_mask = torch.ones(
                    (
                        1,
                        len(
                            history_positions
                        ),
                    ),
                    dtype=torch.bool,
                )

                user_embedding = (
                    network.user_tower(
                        history_embeddings,
                        history_mask,
                    )
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                candidate_positions = [
                    id_to_position[
                        news_id
                    ]
                    for news_id
                    in candidate_ids
                ]

                candidate_vectors = (
                    catalog.vectors[
                        candidate_positions
                    ]
                )

                relevance_scores = (
                    candidate_vectors
                    @ user_embedding
                )

                ranked_items = tuple(
                    news_id
                    for news_id, _
                    in sorted(
                        zip(
                            candidate_ids,
                            relevance_scores.tolist(),
                            strict=True,
                        ),
                        key=lambda item: (
                            -float(
                                item[1]
                            ),
                            item[0],
                        ),
                    )[
                        :args.k
                    ]
                )

                parity_items = (
                    maximal_marginal_relevance(
                        candidate_ids,
                        relevance_scores,
                        candidate_vectors,
                        top_k=args.k,
                        config=MMRConfig(
                            lambda_weight=1.0
                        ),
                    )
                )

                if (
                    parity_items
                    != ranked_items
                ):
                    lambda_one_parity_mismatches += 1

                two_tower_ranked += 1

            else:
                ranked_items = tuple(
                    popularity.rank_candidates(
                        candidate_ids,
                        top_k=args.k,
                        exclude_news_ids=(
                            history_ids
                        ),
                    )
                )

                popularity_fallbacks += 1

            ranking_examples.append(
                RankingExample(
                    impression_id=(
                        impression_id
                    ),
                    ranked_items=(
                        ranked_items
                    ),
                    relevant_items=(
                        relevant_items
                    ),
                )
            )

            recommendation_occurrences += len(
                ranked_items
            )

            if ranked_items:
                ranking_positions = [
                    id_to_position[
                        news_id
                    ]
                    for news_id
                    in ranked_items
                ]

                ranking_vectors = (
                    catalog.vectors[
                        ranking_positions
                    ]
                )

                ild_values.append(
                    intra_list_diversity(
                        ranking_vectors
                    )
                )

                categories = [
                    category_by_id[
                        news_id
                    ]
                    for news_id
                    in ranked_items
                ]

                subcategories = [
                    subcategory_by_id[
                        news_id
                    ]
                    for news_id
                    in ranked_items
                ]

                unique_category_values.append(
                    float(
                        len(
                            set(
                                categories
                            )
                        )
                    )
                )

                unique_subcategory_values.append(
                    float(
                        len(
                            set(
                                subcategories
                            )
                        )
                    )
                )

                category_entropy_values.append(
                    shannon_entropy(
                        categories
                    )
                )

                subcategory_entropy_values.append(
                    shannon_entropy(
                        subcategories
                    )
                )

                for (
                    news_id,
                    category,
                    subcategory,
                ) in zip(
                    ranked_items,
                    categories,
                    subcategories,
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

                    recommendation_group_counts[
                        popularity_group[
                            news_id
                        ]
                    ] += 1

    metrics = evaluate_rankings(
        ranking_examples,
        catalog.news_ids,
        k=args.k,
    )

    frozen_baseline = (
        phase05_final[
            "evaluation"
        ][
            "baseline"
        ]
    )

    baseline_checks = {
        "ndcg_at_k": (
            metrics.ndcg_at_k
        ),
        "mrr_at_k": (
            metrics.mrr_at_k
        ),
        "recall_at_k": (
            metrics.recall_at_k
        ),
        "hit_rate_at_k": (
            metrics.hit_rate_at_k
        ),
        "catalog_coverage_at_k": (
            metrics.catalog_coverage_at_k
        ),
    }

    baseline_matches_phase05 = all(
        np.isclose(
            value,
            float(
                frozen_baseline[
                    name
                ]
            ),
            atol=1e-12,
            rtol=0.0,
        )
        for name, value
        in baseline_checks.items()
    )

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

    recommendation_shares = _shares(
        recommendation_group_counts
    )

    opportunity_shares = _shares(
        opportunity_group_counts
    )

    group_gap = {
        group: (
            recommendation_shares[
                group
            ]
            - opportunity_shares[
                group
            ]
        )
        for group in (
            "head",
            "mid",
            "tail",
        )
    }

    group_catalog_counts = Counter(
        popularity_group.values()
    )

    diversity = {
        "mean_intra_list_diversity": (
            _mean_or_zero(
                ild_values
            )
        ),
        "mean_unique_categories": (
            _mean_or_zero(
                unique_category_values
            )
        ),
        "mean_unique_subcategories": (
            _mean_or_zero(
                unique_subcategory_values
            )
        ),
        "mean_category_entropy": (
            _mean_or_zero(
                category_entropy_values
            )
        ),
        "mean_subcategory_entropy": (
            _mean_or_zero(
                subcategory_entropy_values
            )
        ),
    }

    exposure = {
        "unique_exposed_articles": int(
            np.count_nonzero(
                exposure_vector
            )
        ),
        "catalog_coverage_at_k": (
            metrics.catalog_coverage_at_k
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
                item
                for item, count
                in category_exposure.items()
                for _ in range(
                    count
                )
            )
        ),
        "subcategory_exposure_entropy": (
            shannon_entropy(
                item
                for item, count
                in subcategory_exposure.items()
                for _ in range(
                    count
                )
            )
        ),
    }

    all_values = (
        list(
            diversity.values()
        )
        + [
            value
            for value
            in exposure.values()
            if isinstance(
                value,
                (int, float),
            )
        ]
        + list(
            recommendation_shares.values()
        )
        + list(
            opportunity_shares.values()
        )
        + list(
            group_gap.values()
        )
    )

    finite = all(
        np.isfinite(
            float(
                value
            )
        )
        for value in all_values
    )

    passed = (
        baseline_matches_phase05
        and lambda_one_parity_mismatches == 0
        and finite
        and len(
            ranking_examples
        )
        == len(
            phase06.validation
        )
    )

    payload = {
        "experiment": (
            "phase06_selected_baseline_audit"
        ),
        "protocol": {
            "dataset": (
                "MINDsmall_train"
            ),
            "benchmark_status": (
                "previously_observed_development_benchmark"
            ),
            "official_dev_used": False,
            "phase03_cutoff": (
                phase03.cutoff.isoformat()
            ),
            "phase06_cutoff": (
                phase06.cutoff.isoformat()
            ),
            "benchmark_impressions": len(
                phase06.validation
            ),
            "k": args.k,
        },
        "frozen_inputs": {
            "selected_model": (
                phase05_final[
                    "selected_model"
                ]
            ),
            "selected_retrieval_backend": (
                phase05_final[
                    "selected_retrieval_backend"
                ]
            ),
            "phase03_checkpoint_sha256": (
                checkpoint_sha
            ),
            "phase04_catalog_sha256": (
                catalog_sha
            ),
        },
        "accounting": {
            "candidate_occurrences": (
                candidate_occurrences
            ),
            "recommendation_occurrences": (
                recommendation_occurrences
            ),
            "two_tower_ranked_impressions": (
                two_tower_ranked
            ),
            "popularity_fallback_impressions": (
                popularity_fallbacks
            ),
            "truncated_history_impressions": (
                truncated_histories
            ),
            "lambda_one_parity_mismatches": (
                lambda_one_parity_mismatches
            ),
        },
        "relevance": (
            metrics.to_dict()
        ),
        "diversity": (
            diversity
        ),
        "exposure": (
            exposure
        ),
        "popularity_groups": {
            "definition": {
                "head": (
                    "top 20% by Phase-03 training candidate exposures"
                ),
                "mid": (
                    "next 30% by Phase-03 training candidate exposures"
                ),
                "tail": (
                    "bottom 50% by Phase-03 training candidate exposures"
                ),
            },
            "catalog_article_counts": {
                group: (
                    group_catalog_counts[
                        group
                    ]
                )
                for group in (
                    "head",
                    "mid",
                    "tail",
                )
            },
            "recommendation_exposure_share": (
                recommendation_shares
            ),
            "candidate_opportunity_share": (
                opportunity_shares
            ),
            "exposure_minus_opportunity_gap": (
                group_gap
            ),
            "long_tail_recommendation_share": (
                recommendation_shares[
                    "tail"
                ]
            ),
        },
        "integrity": {
            "baseline_matches_phase05": (
                baseline_matches_phase05
            ),
            "all_reported_values_finite": (
                finite
            ),
            "passed": (
                passed
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

    print()
    print("=" * 78)
    print("PHASE 06B — SELECTED BASELINE AUDIT")
    print("=" * 78)

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
