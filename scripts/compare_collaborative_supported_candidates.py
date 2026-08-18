"""Matched-candidate diagnostic for BPR versus Content+Fallback.

This experiment is intentionally diagnostic rather than deployable.
It conditions on clicked items being representable by BPR, then gives
BPR and Content+Fallback exactly the same BPR-supported candidate set.

The purpose is to distinguish representation/coverage failure from
ranking-signal failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newslens.data import load_behaviors, load_news
from newslens.data.recsys_training import load_bpr_triples
from newslens.evaluation.collaborative import (
    _load_validation_impressions,
)
from newslens.evaluation.comparison import (
    paired_bootstrap_ranking_comparison,
)
from newslens.evaluation.content import (
    _parse_history,
    _prepare_catalog,
    _training_vocabulary_news_ids,
)
from newslens.evaluation.evaluator import (
    RankingExample,
    evaluate_rankings,
)
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.models import (
    ContentBasedRecommender,
    ContentPopularityFallbackRecommender,
    PopularityRecommender,
)
from newslens.models.collaborative import (
    CollaborativeRecommender,
)


def _evaluate_group(
    *,
    name: str,
    bpr_examples: list[RankingExample],
    fallback_examples: list[RankingExample],
    catalog: frozenset[str],
    k: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    total_validation_impressions: int,
    original_candidate_occurrences: int,
    supported_candidate_occurrences: int,
) -> dict[str, object]:
    if len(bpr_examples) != len(fallback_examples):
        raise RuntimeError(
            "BPR and fallback example counts must match."
        )

    if not bpr_examples:
        return {
            "name": name,
            "impressions": 0,
            "fraction_of_validation": 0.0,
            "bpr_metrics": None,
            "fallback_metrics": None,
            "paired_comparison": None,
        }

    bpr_metrics = evaluate_rankings(
        bpr_examples,
        catalog,
        k=k,
    )

    fallback_metrics = evaluate_rankings(
        fallback_examples,
        catalog,
        k=k,
    )

    comparison = paired_bootstrap_ranking_comparison(
        fallback_examples,
        bpr_examples,
        baseline_model_name=(
            "tfidf_content_with_popularity_fallback"
        ),
        candidate_model_name=(
            f"bpr_shared_candidate_{name}"
        ),
        k=k,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        random_seed=seed,
    )

    return {
        "name": name,
        "impressions": len(bpr_examples),
        "fraction_of_validation": (
            len(bpr_examples)
            / total_validation_impressions
        ),
        "original_candidate_occurrences": (
            original_candidate_occurrences
        ),
        "supported_candidate_occurrences": (
            supported_candidate_occurrences
        ),
        "supported_candidate_fraction": (
            supported_candidate_occurrences
            / original_candidate_occurrences
            if original_candidate_occurrences
            else 0.0
        ),
        "bpr_metrics": bpr_metrics.to_dict(),
        "fallback_metrics": fallback_metrics.to_dict(),
        "paired_comparison": comparison.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cutoff",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--max-negatives-per-positive",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1_000,
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

    split_path = args.data_dir / "MINDsmall_train"

    print("Loading MIND-small...")

    news = load_news(
        split_path / "news.tsv"
    )

    behaviors = load_behaviors(
        split_path / "behaviors.tsv"
    )

    print("Creating chronological split...")

    split = chronological_train_validation_split(
        behaviors,
        validation_fraction=0.20,
    )

    actual_cutoff = split.cutoff.isoformat()

    if actual_cutoff != args.cutoff:
        raise RuntimeError(
            "Cutoff mismatch: "
            f"expected {args.cutoff}, "
            f"got {actual_cutoff}."
        )

    catalog = _prepare_catalog(news)

    history_by_impression = {
        str(row.impression_id): _parse_history(
            row.history
        )
        for row in split.validation.itertuples(
            index=False
        )
    }

    print("Training Content + Fallback baseline...")

    popularity_model = PopularityRecommender().fit(
        split.train
    )

    vocabulary_news_ids = (
        _training_vocabulary_news_ids(
            split.train,
            catalog,
        )
    )

    content_model = ContentBasedRecommender(
        max_features=args.max_features,
    ).fit(
        news,
        vocabulary_news_ids=(
            vocabulary_news_ids
        ),
    )

    fallback_model = (
        ContentPopularityFallbackRecommender(
            content_model,
            popularity_model,
        )
    )

    print("Loading BPR triples...")

    triples = load_bpr_triples(
        args.database,
        cutoff_timestamp=args.cutoff,
        max_negatives_per_positive=(
            args.max_negatives_per_positive
        ),
        seed=args.seed,
    )

    print(
        "Training BPR:",
        {
            "triples": len(triples),
            "embedding_dim": (
                args.embedding_dim
            ),
            "epochs": args.epochs,
            "negatives": (
                args.max_negatives_per_positive
            ),
        },
    )

    bpr_model = CollaborativeRecommender(
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    ).fit(
        triples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    validation = _load_validation_impressions(
        args.database,
        cutoff_timestamp=args.cutoff,
    )

    if len(validation) != len(split.validation):
        raise RuntimeError(
            "Validation count mismatch: "
            f"warehouse={len(validation)}, "
            f"raw={len(split.validation)}."
        )

    all_supported_bpr: list[RankingExample] = []
    all_supported_fallback: list[
        RankingExample
    ] = []

    topk_supported_bpr: list[
        RankingExample
    ] = []
    topk_supported_fallback: list[
        RankingExample
    ] = []

    all_original_candidates = 0
    all_supported_candidates = 0

    topk_original_candidates = 0
    topk_supported_candidates_count = 0

    skipped_unknown_user = 0
    skipped_unknown_clicked_item = 0
    skipped_no_supported_candidates = 0

    print(
        "Building matched candidate comparisons..."
    )

    for impression in validation:
        if (
            impression.user_id
            not in bpr_model.user_to_index
        ):
            skipped_unknown_user += 1
            continue

        relevant_items = (
            impression.relevant_items
        )

        if not relevant_items:
            continue

        if any(
            news_id not in bpr_model.item_to_index
            for news_id in relevant_items
        ):
            skipped_unknown_clicked_item += 1
            continue

        supported_candidates = tuple(
            news_id
            for news_id
            in impression.candidate_news_ids
            if news_id in bpr_model.item_to_index
        )

        if not supported_candidates:
            skipped_no_supported_candidates += 1
            continue

        if not relevant_items.issubset(
            supported_candidates
        ):
            raise RuntimeError(
                "Relevant items should all be present "
                "in the supported candidate set."
            )

        history = history_by_impression.get(
            impression.impression_id
        )

        if history is None:
            raise RuntimeError(
                "Missing raw validation history for "
                f"impression "
                f"{impression.impression_id}."
            )

        bpr_recommendations = (
            bpr_model.recommend_for_user(
                impression.user_id,
                candidate_news_ids=(
                    supported_candidates
                ),
                top_k=args.k,
            )
        )

        fallback_recommendations = (
            fallback_model.recommend(
                history,
                candidate_news_ids=(
                    supported_candidates
                ),
                top_k=args.k,
            )
        )

        bpr_example = RankingExample(
            impression_id=(
                impression.impression_id
            ),
            ranked_items=tuple(
                recommendation.news_id
                for recommendation
                in bpr_recommendations
            ),
            relevant_items=relevant_items,
        )

        fallback_example = RankingExample(
            impression_id=(
                impression.impression_id
            ),
            ranked_items=tuple(
                recommendation.news_id
                for recommendation
                in fallback_recommendations
            ),
            relevant_items=relevant_items,
        )

        all_supported_bpr.append(
            bpr_example
        )
        all_supported_fallback.append(
            fallback_example
        )

        all_original_candidates += len(
            impression.candidate_news_ids
        )

        all_supported_candidates += len(
            supported_candidates
        )

        required_scoreable = min(
            args.k,
            len(
                impression.candidate_news_ids
            ),
        )

        if (
            len(supported_candidates)
            >= required_scoreable
        ):
            topk_supported_bpr.append(
                bpr_example
            )
            topk_supported_fallback.append(
                fallback_example
            )

            topk_original_candidates += len(
                impression.candidate_news_ids
            )

            topk_supported_candidates_count += (
                len(supported_candidates)
            )

    total_validation = len(validation)

    groups = {
        "all_clicked_known_shared_candidates": (
            _evaluate_group(
                name=(
                    "all_clicked_known_"
                    "shared_candidates"
                ),
                bpr_examples=all_supported_bpr,
                fallback_examples=(
                    all_supported_fallback
                ),
                catalog=catalog,
                k=args.k,
                bootstrap_samples=(
                    args.bootstrap_samples
                ),
                confidence_level=(
                    args.confidence_level
                ),
                seed=args.seed,
                total_validation_impressions=(
                    total_validation
                ),
                original_candidate_occurrences=(
                    all_original_candidates
                ),
                supported_candidate_occurrences=(
                    all_supported_candidates
                ),
            )
        ),
        (
            "all_clicked_known_shared_"
            "topk_scoreable"
        ): (
            _evaluate_group(
                name=(
                    "all_clicked_known_"
                    "shared_topk_scoreable"
                ),
                bpr_examples=(
                    topk_supported_bpr
                ),
                fallback_examples=(
                    topk_supported_fallback
                ),
                catalog=catalog,
                k=args.k,
                bootstrap_samples=(
                    args.bootstrap_samples
                ),
                confidence_level=(
                    args.confidence_level
                ),
                seed=args.seed,
                total_validation_impressions=(
                    total_validation
                ),
                original_candidate_occurrences=(
                    topk_original_candidates
                ),
                supported_candidate_occurrences=(
                    topk_supported_candidates_count
                ),
            )
        ),
    }

    payload = {
        "experiment": (
            "phase01_bpr_matched_candidate_"
            "diagnostic"
        ),
        "purpose": (
            "Compare BPR and Content+Fallback "
            "on exactly the same BPR-supported "
            "candidate universe."
        ),
        "interpretation_boundary": {
            "oracle_diagnostic": True,
            "deployable_gate": False,
            "reason": (
                "Eligibility conditions on clicked "
                "items being representable by BPR."
            ),
        },
        "protocol": {
            "cutoff_timestamp": (
                args.cutoff
            ),
            "validation_impressions": (
                total_validation
            ),
            "k": args.k,
            "bootstrap_samples": (
                args.bootstrap_samples
            ),
            "confidence_level": (
                args.confidence_level
            ),
            "seed": args.seed,
        },
        "selected_bpr": {
            "embedding_dim": (
                args.embedding_dim
            ),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "max_negatives_per_positive": (
                args.max_negatives_per_positive
            ),
            "training_triples": len(
                triples
            ),
            "known_users": len(
                bpr_model.user_to_index
            ),
            "known_items": len(
                bpr_model.item_to_index
            ),
        },
        "selection_accounting": {
            "skipped_unknown_user": (
                skipped_unknown_user
            ),
            "skipped_unknown_clicked_item": (
                skipped_unknown_clicked_item
            ),
            "skipped_no_supported_candidates": (
                skipped_no_supported_candidates
            ),
        },
        "groups": groups,
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
        "Matched candidate diagnostic complete."
    )

    for name, report in groups.items():
        bpr = report["bpr_metrics"]
        fallback = report[
            "fallback_metrics"
        ]
        comparison = report[
            "paired_comparison"
        ]

        print()
        print(name)
        print(
            "  impressions:",
            report["impressions"],
        )
        print(
            "  supported candidate fraction:",
            round(
                report[
                    "supported_candidate_fraction"
                ],
                4,
            ),
        )

        if (
            bpr is not None
            and fallback is not None
            and comparison is not None
        ):
            ndcg = comparison[
                "metrics"
            ]["ndcg_at_k"]

            print(
                "  BPR NDCG:",
                round(
                    bpr["ndcg_at_k"],
                    6,
                ),
            )
            print(
                "  Fallback NDCG:",
                round(
                    fallback[
                        "ndcg_at_k"
                    ],
                    6,
                ),
            )
            print(
                "  BPR-Fallback delta:",
                round(
                    ndcg[
                        "point_difference"
                    ],
                    6,
                ),
            )
            print(
                "  95% CI:",
                (
                    ndcg["lower_bound"],
                    ndcg["upper_bound"],
                ),
            )

    print()
    print(
        f"Wrote report to {args.output}"
    )


if __name__ == "__main__":
    main()
