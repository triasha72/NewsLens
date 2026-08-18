"""Evaluate a serving-time support gate for the Phase 02 hybrid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newslens.data import (
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
        "--alpha",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--minimum-supported-candidates",
        type=int,
        default=10,
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

    if args.minimum_supported_candidates <= 0:
        raise ValueError(
            "minimum-supported-candidates must be positive."
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

    if len(split.validation) != 31_393:
        raise RuntimeError(
            "Unexpected validation count: "
            f"{len(split.validation)}."
        )

    catalog = _prepare_catalog(news)

    print("Training popularity model...")

    popularity_model = (
        PopularityRecommender().fit(
            split.train
        )
    )

    print("Training content model...")

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

    print("Building baseline rankings...")

    fallback_result = _build_fallback_examples(
        split.validation,
        fallback_model,
        content_model,
        catalog,
        k=args.k,
    )

    fallback_by_id = {
        example.impression_id: example
        for example in fallback_result.examples
    }

    print("Loading selected BPR training data...")

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
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "negatives": (
                args.max_negatives_per_positive
            ),
        },
    )

    collaborative_model = (
        CollaborativeRecommender(
            embedding_dim=args.embedding_dim,
            seed=args.seed,
        ).fit(
            triples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    )

    hybrid_model = HybridRecommender(
        content_model,
        collaborative_model,
        popularity_model,
        collaborative_weight=args.alpha,
    )

    ungated_examples: list[
        RankingExample
    ] = []

    gated_examples: list[
        RankingExample
    ] = []

    gate_open_impressions = 0
    gate_closed_impressions = 0

    gate_open_candidate_occurrences = 0
    gate_open_supported_occurrences = 0

    ungated_changed_rankings = 0
    gated_changed_rankings = 0

    print("Building gated and ungated rankings...")

    for row in split.validation.itertuples(
        index=False
    ):
        impression_id = str(
            row.impression_id
        )

        user_id = str(
            row.user_id
        )

        history_ids = _parse_history(
            row.history
        )

        parsed = parse_impressions(
            str(row.impressions)
        )

        candidate_ids = [
            news_id
            for news_id, _ in parsed
        ]

        baseline_example = (
            fallback_by_id[
                impression_id
            ]
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

        recommendations = (
            hybrid_model.recommend_for_user(
                user_id,
                history_ids,
                candidate_news_ids=(
                    candidate_ids
                ),
                top_k=args.k,
            )
        )

        ungated_example = RankingExample(
            impression_id=impression_id,
            ranked_items=tuple(
                recommendation.news_id
                for recommendation
                in recommendations
            ),
            relevant_items=(
                baseline_example.relevant_items
            ),
        )

        ungated_examples.append(
            ungated_example
        )

        if (
            ungated_example.ranked_items
            != baseline_example.ranked_items
        ):
            ungated_changed_rankings += 1

        gate_open = (
            supported_count
            >= args.minimum_supported_candidates
        )

        if gate_open:
            gate_open_impressions += 1

            gate_open_candidate_occurrences += (
                len(candidate_ids)
            )

            gate_open_supported_occurrences += (
                supported_count
            )

            gated_example = (
                ungated_example
            )
        else:
            gate_closed_impressions += 1

            gated_example = (
                baseline_example
            )

        gated_examples.append(
            gated_example
        )

        if (
            gated_example.ranked_items
            != baseline_example.ranked_items
        ):
            gated_changed_rankings += 1

    if gate_open_impressions + gate_closed_impressions != len(
        split.validation
    ):
        raise RuntimeError(
            "Gate accounting does not cover validation."
        )

    baseline_metrics = evaluate_rankings(
        fallback_result.examples,
        catalog,
        k=args.k,
    )

    ungated_metrics = evaluate_rankings(
        ungated_examples,
        catalog,
        k=args.k,
    )

    gated_metrics = evaluate_rankings(
        gated_examples,
        catalog,
        k=args.k,
    )

    print(
        "Running gated vs fallback bootstrap..."
    )

    gated_vs_fallback = (
        paired_bootstrap_ranking_comparison(
            fallback_result.examples,
            gated_examples,
            baseline_model_name=(
                "content_fallback"
            ),
            candidate_model_name=(
                "support_gated_hybrid"
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

    print(
        "Running ungated vs fallback bootstrap..."
    )

    ungated_vs_fallback = (
        paired_bootstrap_ranking_comparison(
            fallback_result.examples,
            ungated_examples,
            baseline_model_name=(
                "content_fallback"
            ),
            candidate_model_name=(
                "ungated_hybrid"
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

    print(
        "Running gated vs ungated bootstrap..."
    )

    gated_vs_ungated = (
        paired_bootstrap_ranking_comparison(
            ungated_examples,
            gated_examples,
            baseline_model_name=(
                "ungated_hybrid"
            ),
            candidate_model_name=(
                "support_gated_hybrid"
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

    gated_comparison = (
        gated_vs_fallback.to_dict()
    )

    ndcg_result = gated_comparison[
        "metrics"
    ]["ndcg_at_k"]

    no_metric_significantly_worse = all(
        metric["upper_bound"] >= 0.0
        for metric
        in gated_comparison[
            "metrics"
        ].values()
    )

    promotion_candidate = (
        ndcg_result["lower_bound"] > 0.0
        and no_metric_significantly_worse
    )

    payload = {
        "experiment": (
            "phase02_support_gated_hybrid"
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
            "alpha": args.alpha,
            "minimum_supported_candidates": (
                args.minimum_supported_candidates
            ),
            "bootstrap_samples": (
                args.bootstrap_samples
            ),
            "confidence_level": (
                args.confidence_level
            ),
            "seed": args.seed,
        },
        "gate_accounting": {
            "gate_open_impressions": (
                gate_open_impressions
            ),
            "gate_closed_impressions": (
                gate_closed_impressions
            ),
            "gate_open_fraction": (
                gate_open_impressions
                / len(split.validation)
            ),
            "gate_open_candidate_occurrences": (
                gate_open_candidate_occurrences
            ),
            "gate_open_supported_occurrences": (
                gate_open_supported_occurrences
            ),
            "ungated_changed_rankings": (
                ungated_changed_rankings
            ),
            "gated_changed_rankings": (
                gated_changed_rankings
            ),
        },
        "baseline_metrics": (
            baseline_metrics.to_dict()
        ),
        "ungated_metrics": (
            ungated_metrics.to_dict()
        ),
        "gated_metrics": (
            gated_metrics.to_dict()
        ),
        "comparisons": {
            "gated_vs_fallback": (
                gated_comparison
            ),
            "ungated_vs_fallback": (
                ungated_vs_fallback.to_dict()
            ),
            "gated_vs_ungated": (
                gated_vs_ungated.to_dict()
            ),
        },
        "decision": {
            "criterion": (
                "positive NDCG confidence interval "
                "versus Content+Fallback and no "
                "primary metric significantly worse"
            ),
            "promotion_candidate": (
                promotion_candidate
            ),
        },
        "interpretation_boundary": {
            "gate_was_motivated_by_internal_validation_segments": True,
            "follow_up_is_still_internal_validation": True,
            "official_dev_remains_untouched": True,
            "result_should_be_treated_as_architecture_selection_not_final_generalization": True,
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
    print("=" * 88)
    print("SUPPORT-GATED HYBRID RESULT")
    print("=" * 88)

    print(
        "Gate open:",
        gate_open_impressions,
        "/",
        len(split.validation),
        f"({gate_open_impressions / len(split.validation):.2%})",
    )

    print()
    print(
        f"{'Model':<24}"
        f"{'NDCG':>12}"
        f"{'MRR':>12}"
        f"{'Recall':>12}"
        f"{'Hit':>12}"
    )

    for name, metrics in (
        (
            "Content+Fallback",
            baseline_metrics,
        ),
        (
            "Ungated hybrid",
            ungated_metrics,
        ),
        (
            "Gated hybrid",
            gated_metrics,
        ),
    ):
        print(
            f"{name:<24}"
            f"{metrics.ndcg_at_k:>12.6f}"
            f"{metrics.mrr_at_k:>12.6f}"
            f"{metrics.recall_at_k:>12.6f}"
            f"{metrics.hit_rate_at_k:>12.6f}"
        )

    print()
    print(
        "Gated hybrid minus fallback:"
    )

    for metric_name, result in (
        gated_comparison[
            "metrics"
        ].items()
    ):
        print(
            f"  {metric_name:<15}"
            f"delta={result['point_difference']:+.6f} "
            f"CI=[{result['lower_bound']:+.6f}, "
            f"{result['upper_bound']:+.6f}] "
            f"excludes_zero={result['excludes_zero']}"
        )

    print()
    print(
        "Promotion candidate:",
        promotion_candidate,
    )

    print()
    print(
        f"Wrote report to {args.output}"
    )


if __name__ == "__main__":
    main()
