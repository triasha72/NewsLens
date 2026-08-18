"""Evaluate the preregistered Phase-06 MMR sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
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
from newslens.evaluation.comparison import (
    paired_bootstrap_ranking_comparison,
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
from newslens.retrieval.catalog import RetrievalCatalog

FROZEN_LAMBDAS: tuple[float, ...] = (
    1.00,
    0.95,
    0.90,
    0.85,
    0.80,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _parse_history(value: object) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()

    text = str(value).strip()

    if not text:
        return ()

    return tuple(text.split())


def _metadata_maps(
    news: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    categories: dict[str, str] = {}
    subcategories: dict[str, str] = {}

    for row in news.itertuples(index=False):
        news_id = str(row.news_id)
        category = str(row.category).strip()
        subcategory = str(row.subcategory).strip()

        categories[news_id] = (
            category if category else "__missing__"
        )
        subcategories[news_id] = (
            subcategory if subcategory else "__missing__"
        )

    return categories, subcategories


def _build_popularity_groups(
    catalog: RetrievalCatalog,
    popularity: PopularityRecommender,
) -> dict[str, str]:
    ordered = sorted(
        catalog.news_ids,
        key=lambda news_id: (
            -popularity.statistics(news_id).exposures,
            news_id,
        ),
    )

    count = len(ordered)
    head_end = int(np.ceil(count * 0.20))
    mid_end = int(np.ceil(count * 0.50))

    groups: dict[str, str] = {}

    for index, news_id in enumerate(ordered):
        if index < head_end:
            group = "head"
        elif index < mid_end:
            group = "mid"
        else:
            group = "tail"

        groups[news_id] = group

    return groups


def _shares(
    counts: Counter[str],
) -> dict[str, float]:
    total = sum(counts.values())

    if total == 0:
        return {
            "head": 0.0,
            "mid": 0.0,
            "tail": 0.0,
        }

    return {
        group: counts.get(group, 0) / total
        for group in ("head", "mid", "tail")
    }


@dataclass
class PolicyAccumulator:
    """Collect rankings and Phase-06 diagnostics."""

    examples: list[RankingExample] = field(
        default_factory=list
    )
    ild: list[float] = field(default_factory=list)
    unique_categories: list[float] = field(
        default_factory=list
    )
    unique_subcategories: list[float] = field(
        default_factory=list
    )
    category_entropy: list[float] = field(
        default_factory=list
    )
    subcategory_entropy: list[float] = field(
        default_factory=list
    )
    article_exposure: Counter[str] = field(
        default_factory=Counter
    )
    category_exposure: Counter[str] = field(
        default_factory=Counter
    )
    subcategory_exposure: Counter[str] = field(
        default_factory=Counter
    )
    popularity_group_exposure: Counter[str] = field(
        default_factory=Counter
    )
    changed_rankings: int = 0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(fmean(values))


def _record_ranking(
    accumulator: PolicyAccumulator,
    *,
    impression_id: str,
    ranked_items: tuple[str, ...],
    relevant_items: frozenset[str],
    catalog: RetrievalCatalog,
    category_by_id: dict[str, str],
    subcategory_by_id: dict[str, str],
    popularity_group: dict[str, str],
) -> None:
    accumulator.examples.append(
        RankingExample(
            impression_id=impression_id,
            ranked_items=ranked_items,
            relevant_items=relevant_items,
        )
    )

    if not ranked_items:
        return

    positions = [
        catalog.id_to_position[news_id]
        for news_id in ranked_items
    ]

    vectors = catalog.vectors[positions]

    accumulator.ild.append(
        intra_list_diversity(vectors)
    )

    categories = [
        category_by_id[news_id]
        for news_id in ranked_items
    ]

    subcategories = [
        subcategory_by_id[news_id]
        for news_id in ranked_items
    ]

    accumulator.unique_categories.append(
        float(len(set(categories)))
    )

    accumulator.unique_subcategories.append(
        float(len(set(subcategories)))
    )

    accumulator.category_entropy.append(
        shannon_entropy(categories)
    )

    accumulator.subcategory_entropy.append(
        shannon_entropy(subcategories)
    )

    for news_id, category, subcategory in zip(
        ranked_items,
        categories,
        subcategories,
        strict=True,
    ):
        accumulator.article_exposure[news_id] += 1
        accumulator.category_exposure[category] += 1
        accumulator.subcategory_exposure[subcategory] += 1
        accumulator.popularity_group_exposure[
            popularity_group[news_id]
        ] += 1


def _policy_summary(
    accumulator: PolicyAccumulator,
    *,
    catalog: RetrievalCatalog,
    opportunity_group_counts: Counter[str],
    k: int,
) -> dict[str, object]:
    relevance = evaluate_rankings(
        accumulator.examples,
        catalog.news_ids,
        k=k,
    )

    exposure_vector = np.asarray(
        [
            accumulator.article_exposure.get(
                news_id,
                0,
            )
            for news_id in catalog.news_ids
        ],
        dtype=np.float64,
    )

    recommendation_shares = _shares(
        accumulator.popularity_group_exposure
    )

    opportunity_shares = _shares(
        opportunity_group_counts
    )

    gaps = {
        group: (
            recommendation_shares[group]
            - opportunity_shares[group]
        )
        for group in ("head", "mid", "tail")
    }

    category_labels = (
        category
        for category, count
        in accumulator.category_exposure.items()
        for _ in range(count)
    )

    subcategory_labels = (
        subcategory
        for subcategory, count
        in accumulator.subcategory_exposure.items()
        for _ in range(count)
    )

    return {
        "relevance": relevance.to_dict(),
        "diversity": {
            "mean_intra_list_diversity": _mean(
                accumulator.ild
            ),
            "mean_unique_categories": _mean(
                accumulator.unique_categories
            ),
            "mean_unique_subcategories": _mean(
                accumulator.unique_subcategories
            ),
            "mean_category_entropy": _mean(
                accumulator.category_entropy
            ),
            "mean_subcategory_entropy": _mean(
                accumulator.subcategory_entropy
            ),
        },
        "exposure": {
            "unique_exposed_articles": int(
                np.count_nonzero(exposure_vector)
            ),
            "catalog_coverage_at_k": (
                relevance.catalog_coverage_at_k
            ),
            "exposure_gini": gini_coefficient(
                exposure_vector
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
                shannon_entropy(category_labels)
            ),
            "subcategory_exposure_entropy": (
                shannon_entropy(subcategory_labels)
            ),
        },
        "popularity_groups": {
            "recommendation_exposure_share": (
                recommendation_shares
            ),
            "candidate_opportunity_share": (
                opportunity_shares
            ),
            "exposure_minus_opportunity_gap": gaps,
            "long_tail_recommendation_share": (
                recommendation_shares["tail"]
            ),
        },
        "changed_rankings_vs_lambda_1": (
            accumulator.changed_rankings
        ),
        "changed_ranking_fraction_vs_lambda_1": (
            accumulator.changed_rankings
            / len(accumulator.examples)
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
        "--baseline-report",
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
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
    baseline_report = json.loads(
        args.baseline_report.read_text()
    )

    if baseline_report["integrity"]["passed"] is not True:
        raise RuntimeError(
            "Phase-06 baseline audit did not pass."
        )

    if (
        phase05_final["selected_model"]
        != "phase03_two_tower_popularity"
    ):
        raise RuntimeError(
            "Unexpected Phase-05 selected model."
        )

    checkpoint_sha = _sha256(args.checkpoint)
    catalog_sha = _sha256(args.catalog)

    if (
        checkpoint_sha
        != embedding_report["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "Phase-03 checkpoint SHA mismatch."
        )

    if (
        catalog_sha
        != embedding_report["artifact_sha256"]
    ):
        raise RuntimeError(
            "Phase-04 catalog SHA mismatch."
        )

    root = args.data_dir / "MINDsmall_train"

    news = load_news(root / "news.tsv")
    behaviors = load_behaviors(
        root / "behaviors.tsv"
    )

    phase03 = chronological_train_validation_split(
        behaviors,
        validation_fraction=(
            args.phase03_validation_fraction
        ),
    )

    phase06 = chronological_train_validation_split(
        phase03.validation,
        validation_fraction=(
            args.ranker_validation_fraction
        ),
    )

    if (
        phase03.cutoff.isoformat()
        != phase05_audit["phase03"]["cutoff"]
    ):
        raise RuntimeError(
            "Phase-03 split differs from frozen audit."
        )

    if (
        phase06.cutoff.isoformat()
        != phase05_audit["phase05"]["cutoff"]
    ):
        raise RuntimeError(
            "Phase-06 split differs from frozen audit."
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

    catalog = RetrievalCatalog.load_npz(
        args.catalog
    )

    popularity = PopularityRecommender().fit(
        phase03.train
    )

    category_by_id, subcategory_by_id = (
        _metadata_maps(news)
    )

    popularity_group = _build_popularity_groups(
        catalog,
        popularity,
    )

    opportunity_group_counts: Counter[str] = (
        Counter()
    )

    accumulators = {
        lambda_weight: PolicyAccumulator()
        for lambda_weight in FROZEN_LAMBDAS
    }

    id_to_position = catalog.id_to_position

    max_history_length = int(
        checkpoint["protocol"][
            "max_history_length"
        ]
    )

    baseline_rankings: dict[
        str,
        tuple[str, ...],
    ] = {}

    two_tower_impressions = 0
    fallback_impressions = 0

    with torch.no_grad():
        for row in phase06.validation.itertuples(
            index=False
        ):
            impression_id = str(
                row.impression_id
            )

            parsed = parse_impressions(
                str(row.impressions)
            )

            candidate_ids = tuple(
                news_id
                for news_id, _
                in parsed
            )

            if (
                len(candidate_ids)
                != len(set(candidate_ids))
            ):
                raise RuntimeError(
                    "Duplicate candidate IDs."
                )

            if any(
                news_id not in id_to_position
                for news_id in candidate_ids
            ):
                raise RuntimeError(
                    "Candidate missing from frozen catalog."
                )

            relevant_items = frozenset(
                news_id
                for news_id, label
                in parsed
                if label == 1
            )

            for news_id in candidate_ids:
                opportunity_group_counts[
                    popularity_group[news_id]
                ] += 1

            history_ids = _parse_history(
                row.history
            )

            usable_history = tuple(
                news_id
                for news_id in history_ids
                if news_id in id_to_position
            )

            rankings: dict[
                float,
                tuple[str, ...],
            ] = {}

            if not usable_history:
                fallback = tuple(
                    popularity.rank_candidates(
                        candidate_ids,
                        top_k=args.k,
                        exclude_news_ids=(
                            history_ids
                        ),
                    )
                )

                for lambda_weight in FROZEN_LAMBDAS:
                    rankings[lambda_weight] = fallback

                fallback_impressions += 1

            else:
                encoded_history = usable_history[
                    -max_history_length:
                ]

                history_positions = [
                    id_to_position[news_id]
                    for news_id in encoded_history
                ]

                history_tensor = (
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
                        len(history_positions),
                    ),
                    dtype=torch.bool,
                )

                user_embedding = network.user_tower(
                    history_tensor,
                    history_mask,
                )

                candidate_positions = [
                    id_to_position[news_id]
                    for news_id in candidate_ids
                ]

                candidate_vectors = (
                    catalog.vectors[
                        candidate_positions
                    ]
                )

                candidate_tensor = (
                    torch.from_numpy(
                        candidate_vectors
                    )
                )

                relevance_scores = (
                    network.score_candidates(
                        user_embedding,
                        candidate_tensor,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64,
                        copy=False,
                    )
                )

                for lambda_weight in FROZEN_LAMBDAS:
                    rankings[lambda_weight] = (
                        maximal_marginal_relevance(
                            candidate_ids,
                            relevance_scores,
                            candidate_vectors,
                            top_k=args.k,
                            config=MMRConfig(
                                lambda_weight=(
                                    lambda_weight
                                )
                            ),
                        )
                    )

                two_tower_impressions += 1

            baseline_ranking = rankings[1.0]

            baseline_rankings[
                impression_id
            ] = baseline_ranking

            for lambda_weight in FROZEN_LAMBDAS:
                ranking = rankings[
                    lambda_weight
                ]

                accumulator = accumulators[
                    lambda_weight
                ]

                if (
                    ranking
                    != baseline_ranking
                ):
                    accumulator.changed_rankings += 1

                _record_ranking(
                    accumulator,
                    impression_id=impression_id,
                    ranked_items=ranking,
                    relevant_items=(
                        relevant_items
                    ),
                    catalog=catalog,
                    category_by_id=(
                        category_by_id
                    ),
                    subcategory_by_id=(
                        subcategory_by_id
                    ),
                    popularity_group=(
                        popularity_group
                    ),
                )

    summaries = {
        f"{lambda_weight:.2f}": (
            _policy_summary(
                accumulators[
                    lambda_weight
                ],
                catalog=catalog,
                opportunity_group_counts=(
                    opportunity_group_counts
                ),
                k=args.k,
            )
        )
        for lambda_weight
        in FROZEN_LAMBDAS
    }

    baseline_examples = (
        accumulators[1.0].examples
    )

    comparisons: dict[
        str,
        object,
    ] = {}

    for lambda_weight in FROZEN_LAMBDAS[1:]:
        comparisons[
            f"{lambda_weight:.2f}"
        ] = (
            paired_bootstrap_ranking_comparison(
                baseline_examples,
                accumulators[
                    lambda_weight
                ].examples,
                baseline_model_name=(
                    "mmr_lambda_1.00"
                ),
                candidate_model_name=(
                    f"mmr_lambda_{lambda_weight:.2f}"
                ),
                k=args.k,
                bootstrap_samples=(
                    args.bootstrap_samples
                ),
                confidence_level=(
                    args.confidence_level
                ),
                random_seed=args.seed,
            ).to_dict()
        )

    baseline = summaries["1.00"]

    audited_relevance = (
        baseline_report["relevance"]
    )
    audited_diversity = (
        baseline_report["diversity"]
    )
    audited_exposure = (
        baseline_report["exposure"]
    )

    relevance_names = (
        "ndcg_at_k",
        "mrr_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "catalog_coverage_at_k",
    )

    relevance_parity = all(
        np.isclose(
            float(
                baseline["relevance"][name]
            ),
            float(
                audited_relevance[name]
            ),
            atol=1e-12,
            rtol=0.0,
        )
        for name in relevance_names
    )

    diversity_parity = all(
        np.isclose(
            float(
                baseline["diversity"][name]
            ),
            float(
                audited_diversity[name]
            ),
            atol=1e-12,
            rtol=0.0,
        )
        for name in audited_diversity
    )

    exposure_parity = all(
        np.isclose(
            float(
                baseline["exposure"][name]
            ),
            float(
                audited_exposure[name]
            ),
            atol=1e-12,
            rtol=0.0,
        )
        for name in audited_exposure
    )

    baseline_parity_passed = (
        relevance_parity
        and diversity_parity
        and exposure_parity
        and summaries[
            "1.00"
        ][
            "changed_rankings_vs_lambda_1"
        ]
        == 0
    )

    payload = {
        "experiment": (
            "phase06_preregistered_mmr_sweep"
        ),
        "protocol": {
            "dataset": "MINDsmall_train",
            "benchmark_status": (
                "previously_observed_development_benchmark"
            ),
            "official_dev_used": False,
            "benchmark_impressions": len(
                phase06.validation
            ),
            "k": args.k,
            "lambdas": list(
                FROZEN_LAMBDAS
            ),
            "bootstrap_samples": (
                args.bootstrap_samples
            ),
            "confidence_level": (
                args.confidence_level
            ),
            "seed": args.seed,
            "relevance_score": (
                "frozen_two_tower_inner_product_"
                "divided_by_temperature"
            ),
            "temperature": float(
                network.config.temperature
            ),
            "selection_performed": False,
        },
        "accounting": {
            "two_tower_impressions": (
                two_tower_impressions
            ),
            "popularity_fallback_impressions": (
                fallback_impressions
            ),
        },
        "baseline_parity": {
            "relevance_matches_phase06b": (
                relevance_parity
            ),
            "diversity_matches_phase06b": (
                diversity_parity
            ),
            "exposure_matches_phase06b": (
                exposure_parity
            ),
            "passed": (
                baseline_parity_passed
            ),
        },
        "policies": summaries,
        "paired_comparisons_vs_lambda_1": (
            comparisons
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

    if not baseline_parity_passed:
        raise RuntimeError(
            "Lambda=1.00 does not reproduce "
            "the frozen Phase-06B baseline."
        )


if __name__ == "__main__":
    main()
