"""Deployable support-segment diagnostics for the hybrid recommender."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def _support_bin(
    fraction: float,
) -> str:
    if fraction == 0.0:
        return "support_0"
    if fraction < 0.25:
        return "support_0_25"
    if fraction < 0.50:
        return "support_25_50"
    if fraction < 0.75:
        return "support_50_75"
    if fraction < 1.0:
        return "support_75_100"
    return "support_100"


def _history_bin(
    length: int,
) -> str:
    if length == 0:
        return "history_0"
    if length <= 5:
        return "history_1_5"
    if length <= 20:
        return "history_6_20"
    if length <= 50:
        return "history_21_50"
    return "history_51_plus"


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

    split = chronological_train_validation_split(
        behaviors,
        validation_fraction=0.20,
    )

    if split.cutoff.isoformat() != args.cutoff:
        raise RuntimeError(
            "Chronological cutoff mismatch."
        )

    catalog = _prepare_catalog(news)

    print("Training popularity...")

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

    fallback_result = (
        _build_fallback_examples(
            split.validation,
            fallback_model,
            content_model,
            catalog,
            k=args.k,
        )
    )

    fallback_by_id = {
        example.impression_id: example
        for example in fallback_result.examples
    }

    print("Training selected BPR...")

    triples = load_bpr_triples(
        args.database,
        cutoff_timestamp=args.cutoff,
        max_negatives_per_positive=(
            args.max_negatives_per_positive
        ),
        seed=args.seed,
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

    grouped_hybrid: dict[
        str,
        list[RankingExample],
    ] = defaultdict(list)

    grouped_fallback: dict[
        str,
        list[RankingExample],
    ] = defaultdict(list)

    support_fractions: list[float] = []

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

        relevant_items = frozenset(
            news_id
            for news_id, label in parsed
            if label == 1
        )

        known_user = (
            user_id
            in collaborative_model.user_to_index
        )

        if known_user:
            supported_count = sum(
                news_id
                in collaborative_model.item_to_index
                for news_id in candidate_ids
            )
        else:
            supported_count = 0

        support_fraction = (
            supported_count
            / len(candidate_ids)
            if candidate_ids
            else 0.0
        )

        support_fractions.append(
            support_fraction
        )

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

        hybrid_example = RankingExample(
            impression_id=impression_id,
            ranked_items=tuple(
                result.news_id
                for result in recommendations
            ),
            relevant_items=relevant_items,
        )

        fallback_example = (
            fallback_by_id[
                impression_id
            ]
        )

        groups = [
            "all_validation",
            (
                "known_user"
                if known_user
                else "unknown_user"
            ),
            _support_bin(
                support_fraction
            ),
            _history_bin(
                len(history_ids)
            ),
        ]

        if (
            known_user
            and supported_count > 0
        ):
            groups.append(
                "deployable_bpr_support"
            )

        if supported_count >= args.k:
            groups.append(
                "at_least_k_supported"
            )

        for group in groups:
            grouped_hybrid[group].append(
                hybrid_example
            )

            grouped_fallback[group].append(
                fallback_example
            )

    reports: dict[
        str,
        dict[str, object],
    ] = {}

    for group in sorted(
        grouped_hybrid
    ):
        hybrid_examples = (
            grouped_hybrid[group]
        )

        fallback_examples = (
            grouped_fallback[group]
        )

        print(
            "Evaluating",
            group,
            len(hybrid_examples),
        )

        hybrid_metrics = (
            evaluate_rankings(
                hybrid_examples,
                catalog,
                k=args.k,
            )
        )

        fallback_metrics = (
            evaluate_rankings(
                fallback_examples,
                catalog,
                k=args.k,
            )
        )

        if len(hybrid_examples) >= 2:
            comparison = (
                paired_bootstrap_ranking_comparison(
                    fallback_examples,
                    hybrid_examples,
                    baseline_model_name=(
                        "content_fallback"
                    ),
                    candidate_model_name=(
                        f"hybrid_alpha_{args.alpha:g}"
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
        else:
            comparison = None

        reports[group] = {
            "impressions": (
                len(hybrid_examples)
            ),
            "fraction_of_validation": (
                len(hybrid_examples)
                / len(split.validation)
            ),
            "hybrid_metrics": (
                hybrid_metrics.to_dict()
            ),
            "fallback_metrics": (
                fallback_metrics.to_dict()
            ),
            "paired_comparison": (
                comparison
            ),
        }

    payload = {
        "experiment": (
            "phase02_hybrid_support_diagnostics"
        ),
        "protocol": {
            "cutoff_timestamp": (
                args.cutoff
            ),
            "validation_impressions": (
                len(split.validation)
            ),
            "alpha": args.alpha,
            "k": args.k,
            "bootstrap_samples": (
                args.bootstrap_samples
            ),
            "confidence_level": (
                args.confidence_level
            ),
            "seed": args.seed,
            "official_dev_used": False,
        },
        "segments": reports,
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
        f"Wrote {args.output}"
    )


if __name__ == "__main__":
    main()
