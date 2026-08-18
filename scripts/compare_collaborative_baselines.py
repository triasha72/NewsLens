from __future__ import annotations

import argparse
import json
from pathlib import Path

from newslens.data import load_behaviors, load_news
from newslens.evaluation.collaborative import (
    evaluate_collaborative_model,
)
from newslens.evaluation.comparison import (
    paired_bootstrap_ranking_comparison,
)
from newslens.evaluation.content import (
    _build_ranking_examples as _build_content_examples,
)
from newslens.evaluation.content import (
    _prepare_catalog,
    _training_vocabulary_news_ids,
)
from newslens.evaluation.fallback import (
    _build_ranking_examples as _build_fallback_examples,
)
from newslens.evaluation.popularity import (
    _build_ranking_examples as _build_popularity_examples,
)
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.models import (
    ContentBasedRecommender,
    ContentPopularityFallbackRecommender,
    PopularityRecommender,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--max-negatives-per-positive",
        type=int,
        default=3,
    )
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_path = args.data_dir / "MINDsmall_train"

    print("Loading MIND-small...")
    news = load_news(split_path / "news.tsv")
    behaviors = load_behaviors(split_path / "behaviors.tsv")

    print("Creating chronological split...")
    split = chronological_train_validation_split(
        behaviors,
        validation_fraction=0.20,
    )

    actual_cutoff = split.cutoff.isoformat()

    if actual_cutoff != args.cutoff:
        raise RuntimeError(
            "Cutoff mismatch: "
            f"expected {args.cutoff}, got {actual_cutoff}."
        )

    print(
        "Split:",
        {
            "train": len(split.train),
            "validation": len(split.validation),
            "cutoff": actual_cutoff,
        },
    )

    catalog = _prepare_catalog(news)

    print("Training popularity baseline...")
    popularity_model = PopularityRecommender().fit(split.train)

    popularity_examples, _, _, _ = _build_popularity_examples(
        split.validation,
        popularity_model,
        catalog,
        k=args.k,
    )

    print("Training TF-IDF content baseline...")
    vocabulary_news_ids = _training_vocabulary_news_ids(
        split.train,
        catalog,
    )

    content_model = ContentBasedRecommender(
        max_features=args.max_features,
    ).fit(
        news,
        vocabulary_news_ids=vocabulary_news_ids,
    )

    content_result = _build_content_examples(
        split.validation,
        content_model,
        catalog,
        k=args.k,
    )

    print("Building content + popularity fallback rankings...")
    fallback_model = ContentPopularityFallbackRecommender(
        content_model,
        popularity_model,
    )

    fallback_result = _build_fallback_examples(
        split.validation,
        fallback_model,
        content_model,
        catalog,
        k=args.k,
    )

    print("Training selected BPR configuration...")
    bpr_report, bpr_examples = evaluate_collaborative_model(
        args.database,
        cutoff_timestamp=args.cutoff,
        k=args.k,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        max_negatives_per_positive=(
            args.max_negatives_per_positive
        ),
        seed=args.seed,
    )

    baselines = {
        "popularity": (
            "training_click_count_popularity",
            popularity_examples,
        ),
        "content": (
            "tfidf_history_content",
            content_result.examples,
        ),
        "content_fallback": (
            "tfidf_content_with_popularity_fallback",
            fallback_result.examples,
        ),
    }

    comparisons = {}

    for key, (baseline_name, examples) in baselines.items():
        print(f"Bootstrap comparison: BPR minus {baseline_name}")

        comparison = paired_bootstrap_ranking_comparison(
            examples,
            bpr_examples,
            baseline_model_name=baseline_name,
            candidate_model_name=(
                "bpr_matrix_factorization_neg3"
            ),
            k=args.k,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            random_seed=args.seed,
        )

        comparisons[key] = comparison.to_dict()

    payload = {
        "experiment": "phase01_collaborative_baseline_comparison",
        "protocol": {
            "cutoff_timestamp": args.cutoff,
            "validation_impressions": len(split.validation),
            "k": args.k,
            "bootstrap_samples": args.bootstrap_samples,
            "confidence_level": args.confidence_level,
            "random_seed": args.seed,
            "difference_direction": "BPR minus baseline",
        },
        "selected_bpr": bpr_report.to_dict(),
        "comparisons": comparisons,
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
    print(json.dumps(payload, indent=2, sort_keys=True))
    print()
    print(f"Wrote comparison report to {args.output}")


if __name__ == "__main__":
    main()
