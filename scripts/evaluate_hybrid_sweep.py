"""Chronological weight sweep for support-aware hybrid recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newslens.data import (
    MindDataValidationError,
    load_behaviors,
    load_news,
    parse_impressions,
)
from newslens.data.recsys_training import load_bpr_triples
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
from newslens.evaluation.fallback import (
    _build_ranking_examples as _build_fallback_examples,
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
from newslens.models.hybrid import HybridRecommender


def _build_hybrid_examples(
    validation_behaviors,
    model: HybridRecommender,
    collaborative_model: CollaborativeRecommender,
    catalog: frozenset[str],
    *,
    k: int,
) -> tuple[list[RankingExample], dict[str, int | float]]:
    examples: list[RankingExample] = []

    candidate_occurrences = 0
    supported_candidate_occurrences = 0
    impressions_with_bpr_support = 0
    popularity_fallback_impressions = 0
    empty_ranking_impressions = 0
    hybrid_topk_occurrences = 0
    content_only_topk_occurrences = 0
    popularity_topk_occurrences = 0

    for row in validation_behaviors.itertuples(index=False):
        impression_id = str(row.impression_id)
        user_id = str(row.user_id)
        history_ids = _parse_history(row.history)

        try:
            parsed = parse_impressions(
                str(row.impressions)
            )
        except MindDataValidationError as error:
            raise RuntimeError(
                "Invalid validation impression "
                f"'{impression_id}': {error}"
            ) from error

        candidate_ids = [
            news_id
            for news_id, _ in parsed
        ]

        if len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise RuntimeError(
                "Duplicate candidate IDs in "
                f"impression '{impression_id}'."
            )

        unknown_candidates = (
            set(candidate_ids) - catalog
        )

        if unknown_candidates:
            raise RuntimeError(
                "Candidate IDs missing from catalog "
                f"in impression '{impression_id}'."
            )

        relevant_items = frozenset(
            news_id
            for news_id, label in parsed
            if label == 1
        )

        candidate_occurrences += len(
            candidate_ids
        )

        if (
            user_id
            in collaborative_model.user_to_index
        ):
            supported_count = sum(
                news_id
                in collaborative_model.item_to_index
                for news_id in candidate_ids
            )
        else:
            supported_count = 0

        supported_candidate_occurrences += (
            supported_count
        )

        if supported_count > 0:
            impressions_with_bpr_support += 1

        recommendations = (
            model.recommend_for_user(
                user_id,
                history_ids,
                candidate_news_ids=(
                    candidate_ids
                ),
                top_k=k,
            )
        )

        if not recommendations:
            empty_ranking_impressions += 1

        sources = {
            result.source
            for result in recommendations
        }

        if sources == {"popularity"}:
            popularity_fallback_impressions += 1

        hybrid_topk_occurrences += sum(
            result.source == "hybrid"
            for result in recommendations
        )

        content_only_topk_occurrences += sum(
            result.source == "content"
            for result in recommendations
        )

        popularity_topk_occurrences += sum(
            result.source == "popularity"
            for result in recommendations
        )

        examples.append(
            RankingExample(
                impression_id=impression_id,
                ranked_items=tuple(
                    result.news_id
                    for result in recommendations
                ),
                relevant_items=relevant_items,
            )
        )

    return examples, {
        "candidate_occurrences": (
            candidate_occurrences
        ),
        "supported_candidate_occurrences": (
            supported_candidate_occurrences
        ),
        "supported_candidate_fraction": (
            supported_candidate_occurrences
            / candidate_occurrences
            if candidate_occurrences
            else 0.0
        ),
        "impressions_with_bpr_support": (
            impressions_with_bpr_support
        ),
        "popularity_fallback_impressions": (
            popularity_fallback_impressions
        ),
        "empty_ranking_impressions": (
            empty_ranking_impressions
        ),
        "hybrid_topk_occurrences": (
            hybrid_topk_occurrences
        ),
        "content_only_topk_occurrences": (
            content_only_topk_occurrences
        ),
        "popularity_topk_occurrences": (
            popularity_topk_occurrences
        ),
    }


def _assert_zero_weight_parity(
    fallback_examples: list[RankingExample],
    hybrid_examples: list[RankingExample],
) -> None:
    """Require alpha=0 to reproduce Content+Fallback exactly."""

    if len(fallback_examples) != len(
        hybrid_examples
    ):
        raise RuntimeError(
            "Zero-weight hybrid and fallback "
            "example counts differ."
        )

    for baseline, candidate in zip(
        fallback_examples,
        hybrid_examples,
        strict=True,
    ):
        if (
            baseline.impression_id
            != candidate.impression_id
        ):
            raise RuntimeError(
                "Zero-weight parity failed: "
                "impression IDs differ."
            )

        if (
            baseline.relevant_items
            != candidate.relevant_items
        ):
            raise RuntimeError(
                "Zero-weight parity failed: "
                "relevance sets differ for "
                f"{baseline.impression_id}."
            )

        if (
            baseline.ranked_items
            != candidate.ranked_items
        ):
            raise RuntimeError(
                "Zero-weight parity failed: "
                "rankings differ for "
                f"{baseline.impression_id}."
            )


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
        "--weights",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.05,
            0.10,
            0.20,
            0.30,
        ],
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

    weights = sorted(
        set(args.weights)
    )

    if 0.0 not in weights:
        raise RuntimeError(
            "The sweep must include weight 0.0 "
            "for baseline parity."
        )

    if any(
        weight < 0.0 or weight > 1.0
        for weight in weights
    ):
        raise RuntimeError(
            "All weights must lie in [0, 1]."
        )

    split_path = (
        args.data_dir
        / "MINDsmall_train"
    )

    print("Loading MIND-small...")

    news = load_news(
        split_path / "news.tsv"
    )

    behaviors = load_behaviors(
        split_path / "behaviors.tsv"
    )

    print(
        "Creating chronological split..."
    )

    split = (
        chronological_train_validation_split(
            behaviors,
            validation_fraction=0.20,
        )
    )

    actual_cutoff = split.cutoff.isoformat()

    if actual_cutoff != args.cutoff:
        raise RuntimeError(
            "Cutoff mismatch: "
            f"expected {args.cutoff}, "
            f"got {actual_cutoff}."
        )

    if len(split.validation) != 31_393:
        raise RuntimeError(
            "Unexpected validation count: "
            f"{len(split.validation)}."
        )

    catalog = _prepare_catalog(news)

    print(
        "Training popularity baseline..."
    )

    popularity_model = (
        PopularityRecommender().fit(
            split.train
        )
    )

    print(
        "Training TF-IDF content model..."
    )

    vocabulary_news_ids = (
        _training_vocabulary_news_ids(
            split.train,
            catalog,
        )
    )

    content_model = (
        ContentBasedRecommender(
            max_features=args.max_features,
        ).fit(
            news,
            vocabulary_news_ids=(
                vocabulary_news_ids
            ),
        )
    )

    fallback_model = (
        ContentPopularityFallbackRecommender(
            content_model,
            popularity_model,
        )
    )

    print(
        "Building Content+Fallback baseline..."
    )

    fallback_result = (
        _build_fallback_examples(
            split.validation,
            fallback_model,
            content_model,
            catalog,
            k=args.k,
        )
    )

    baseline_metrics = evaluate_rankings(
        fallback_result.examples,
        catalog,
        k=args.k,
    )

    print(
        "Loading selected BPR training data..."
    )

    triples = load_bpr_triples(
        args.database,
        cutoff_timestamp=args.cutoff,
        max_negatives_per_positive=(
            args.max_negatives_per_positive
        ),
        seed=args.seed,
    )

    print(
        "Training selected BPR:",
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

    collaborative_model = (
        CollaborativeRecommender(
            embedding_dim=(
                args.embedding_dim
            ),
            seed=args.seed,
        ).fit(
            triples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=(
                args.learning_rate
            ),
            weight_decay=(
                args.weight_decay
            ),
        )
    )

    runs: list[dict[str, object]] = []

    for weight in weights:
        print()
        print(
            "Evaluating collaborative weight:",
            weight,
        )

        hybrid_model = HybridRecommender(
            content_model,
            collaborative_model,
            popularity_model,
            collaborative_weight=weight,
        )

        examples, accounting = (
            _build_hybrid_examples(
                split.validation,
                hybrid_model,
                collaborative_model,
                catalog,
                k=args.k,
            )
        )

        metrics = evaluate_rankings(
            examples,
            catalog,
            k=args.k,
        )

        if weight == 0.0:
            _assert_zero_weight_parity(
                fallback_result.examples,
                examples,
            )

            print(
                "  alpha=0 parity with "
                "Content+Fallback: PASSED"
            )

            comparison = None
        else:
            comparison = (
                paired_bootstrap_ranking_comparison(
                    fallback_result.examples,
                    examples,
                    baseline_model_name=(
                        "tfidf_content_with_"
                        "popularity_fallback"
                    ),
                    candidate_model_name=(
                        "support_aware_hybrid_"
                        f"alpha_{weight:g}"
                    ),
                    k=args.k,
                    bootstrap_samples=(
                        args.bootstrap_samples
                    ),
                    confidence_level=(
                        args.confidence_level
                    ),
                    random_seed=args.seed,
                )
            )

        run = {
            "collaborative_weight": (
                weight
            ),
            "metrics": metrics.to_dict(),
            "accounting": accounting,
            "paired_comparison_vs_fallback": (
                comparison.to_dict()
                if comparison is not None
                else None
            ),
        }

        runs.append(run)

        print(
            "  NDCG@10:",
            round(
                metrics.ndcg_at_k,
                6,
            ),
        )
        print(
            "  MRR@10:",
            round(
                metrics.mrr_at_k,
                6,
            ),
        )
        print(
            "  Recall@10:",
            round(
                metrics.recall_at_k,
                6,
            ),
        )
        print(
            "  Hit Rate@10:",
            round(
                metrics.hit_rate_at_k,
                6,
            ),
        )

    best = max(
        runs,
        key=lambda run: (
            run["metrics"]["ndcg_at_k"],
            -float(
                run[
                    "collaborative_weight"
                ]
            ),
        ),
    )

    baseline_ndcg = (
        baseline_metrics.ndcg_at_k
    )

    best_ndcg = float(
        best["metrics"]["ndcg_at_k"]
    )

    payload = {
        "experiment": (
            "phase02_support_aware_"
            "hybrid_weight_sweep"
        ),
        "protocol": {
            "cutoff_timestamp": (
                args.cutoff
            ),
            "training_impressions": (
                len(split.train)
            ),
            "validation_impressions": (
                len(split.validation)
            ),
            "official_dev_used": False,
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
            "batch_size": (
                args.batch_size
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "max_negatives_per_positive": (
                args.max_negatives_per_positive
            ),
            "training_triples": (
                len(triples)
            ),
            "known_users": len(
                collaborative_model
                .user_to_index
            ),
            "known_items": len(
                collaborative_model
                .item_to_index
            ),
        },
        "baseline": {
            "model_name": (
                "tfidf_content_with_"
                "popularity_fallback"
            ),
            "metrics": (
                baseline_metrics.to_dict()
            ),
        },
        "runs": runs,
        "selection": {
            "criterion": (
                "maximum internal-validation "
                "NDCG@10; smaller alpha wins "
                "exact ties"
            ),
            "best_weight": (
                best[
                    "collaborative_weight"
                ]
            ),
            "best_ndcg_at_k": (
                best_ndcg
            ),
            "baseline_ndcg_at_k": (
                baseline_ndcg
            ),
            "ndcg_difference": (
                best_ndcg
                - baseline_ndcg
            ),
            "nonzero_weight_selected": (
                float(
                    best[
                        "collaborative_weight"
                    ]
                )
                > 0.0
            ),
        },
        "interpretation_boundary": {
            "weight_selection_uses_internal_validation": True,
            "paired_intervals_are_post_selection_exploratory": True,
            "official_dev_remains_final_holdout": True,
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
    print("=" * 84)
    print("HYBRID SWEEP SUMMARY")
    print("=" * 84)

    print(
        f"{'alpha':>8}"
        f"{'NDCG':>12}"
        f"{'MRR':>12}"
        f"{'Recall':>12}"
        f"{'Hit':>12}"
        f"{'Coverage':>12}"
        f"{'Empty':>10}"
    )

    for run in runs:
        metrics = run["metrics"]

        print(
            f"{run['collaborative_weight']:>8.2f}"
            f"{metrics['ndcg_at_k']:>12.6f}"
            f"{metrics['mrr_at_k']:>12.6f}"
            f"{metrics['recall_at_k']:>12.6f}"
            f"{metrics['hit_rate_at_k']:>12.6f}"
            f"{metrics['catalog_coverage_at_k']:>12.6f}"
            f"{metrics['empty_ranking_impressions']:>10}"
        )

    print()
    print(
        "Selected alpha:",
        best["collaborative_weight"],
    )
    print(
        "NDCG difference vs fallback:",
        round(
            best_ndcg
            - baseline_ndcg,
            8,
        ),
    )
    print()
    print(
        f"Wrote report to {args.output}"
    )


if __name__ == "__main__":
    main()
