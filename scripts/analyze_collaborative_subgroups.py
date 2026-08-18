"""Support-aware subgroup analysis for BPR collaborative filtering."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from newslens.data import load_behaviors, load_news
from newslens.data.recsys_training import load_bpr_triples
from newslens.evaluation.collaborative import (
    ValidationImpression,
    _load_validation_impressions,
)
from newslens.evaluation.comparison import (
    paired_bootstrap_ranking_comparison,
)
from newslens.evaluation.content import (
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
from newslens.models.collaborative import CollaborativeRecommender


@dataclass(frozen=True, slots=True)
class DiagnosticRecord:
    ranking: RankingExample
    known_user: bool
    candidate_count: int
    known_candidate_count: int
    relevant_count: int
    known_relevant_count: int

    @property
    def has_scoreable_candidate(self) -> bool:
        return self.known_candidate_count > 0

    @property
    def all_relevant_known(self) -> bool:
        return (
            self.relevant_count > 0
            and self.known_relevant_count == self.relevant_count
        )

    def has_topk_scoreable_candidates(self, k: int) -> bool:
        required = min(k, self.candidate_count)
        return self.known_candidate_count >= required

    @property
    def all_candidates_known(self) -> bool:
        return self.known_candidate_count == self.candidate_count


def _build_bpr_record(
    impression: ValidationImpression,
    model: CollaborativeRecommender,
    *,
    k: int,
) -> DiagnosticRecord:
    candidate_ids = impression.candidate_news_ids

    known_candidate_count = sum(
        news_id in model.item_to_index
        for news_id in candidate_ids
    )

    known_relevant_count = sum(
        news_id in model.item_to_index
        for news_id in impression.relevant_items
    )

    recommendations = model.recommend_for_user(
        impression.user_id,
        candidate_news_ids=candidate_ids,
        top_k=k,
    )

    ranking = RankingExample(
        impression_id=impression.impression_id,
        ranked_items=tuple(
            recommendation.news_id
            for recommendation in recommendations
        ),
        relevant_items=impression.relevant_items,
    )

    return DiagnosticRecord(
        ranking=ranking,
        known_user=impression.user_id in model.user_to_index,
        candidate_count=len(candidate_ids),
        known_candidate_count=known_candidate_count,
        relevant_count=len(impression.relevant_items),
        known_relevant_count=known_relevant_count,
    )


def _evaluate_group(
    name: str,
    records: list[DiagnosticRecord],
    fallback_by_id: dict[str, RankingExample],
    catalog: frozenset[str],
    *,
    total_impressions: int,
    k: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    count = len(records)

    result: dict[str, object] = {
        "name": name,
        "impressions": count,
        "fraction_of_validation": (
            count / total_impressions
            if total_impressions
            else 0.0
        ),
    }

    if not records:
        result["bpr_metrics"] = None
        result["fallback_metrics"] = None
        result["paired_comparison"] = None
        return result

    bpr_examples = [
        record.ranking
        for record in records
    ]

    fallback_examples = [
        fallback_by_id[record.ranking.impression_id]
        for record in records
    ]

    result["bpr_metrics"] = evaluate_rankings(
        bpr_examples,
        catalog,
        k=k,
    ).to_dict()

    result["fallback_metrics"] = evaluate_rankings(
        fallback_examples,
        catalog,
        k=k,
    ).to_dict()

    if count < 2:
        result["paired_comparison"] = None
        return result

    result["paired_comparison"] = (
        paired_bootstrap_ranking_comparison(
            fallback_examples,
            bpr_examples,
            baseline_model_name=(
                "tfidf_content_with_popularity_fallback"
            ),
            candidate_model_name=(
                f"bpr_matrix_factorization_{name}"
            ),
            k=k,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            random_seed=seed,
        ).to_dict()
    )

    return result


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
    news = load_news(split_path / "news.tsv")
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

    print("Training Content + Fallback baseline...")

    popularity_model = PopularityRecommender().fit(
        split.train
    )

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

    fallback_by_id = {
        example.impression_id: example
        for example in fallback_result.examples
    }

    print("Loading BPR training triples...")

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

    model = CollaborativeRecommender(
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    ).fit(
        triples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Loading validation impressions...")

    validation = _load_validation_impressions(
        args.database,
        cutoff_timestamp=args.cutoff,
    )

    if len(validation) != len(split.validation):
        raise RuntimeError(
            "Raw-data and warehouse validation counts differ: "
            f"{len(split.validation)} vs {len(validation)}."
        )

    print("Building support-aware BPR records...")

    records = [
        _build_bpr_record(
            impression,
            model,
            k=args.k,
        )
        for impression in validation
    ]

    groups = {
        "all_validation": records,
        "known_user": [
            record
            for record in records
            if record.known_user
        ],
        "known_user_with_scoreable_candidate": [
            record
            for record in records
            if (
                record.known_user
                and record.has_scoreable_candidate
            )
        ],
        "known_user_all_clicked_items_known": [
            record
            for record in records
            if (
                record.known_user
                and record.all_relevant_known
            )
        ],
        (
            "known_user_all_clicked_known_"
            "and_topk_scoreable"
        ): [
            record
            for record in records
            if (
                record.known_user
                and record.all_relevant_known
                and record.has_topk_scoreable_candidates(
                    args.k
                )
            )
        ],
        "known_user_all_candidates_known": [
            record
            for record in records
            if (
                record.known_user
                and record.all_candidates_known
            )
        ],
    }

    group_reports = {}

    for name, group_records in groups.items():
        print(
            f"Evaluating subgroup "
            f"{name}: {len(group_records)} impressions"
        )

        group_reports[name] = _evaluate_group(
            name,
            group_records,
            fallback_by_id,
            catalog,
            total_impressions=len(records),
            k=args.k,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )

    total_candidate_occurrences = sum(
        record.candidate_count
        for record in records
    )

    known_candidate_occurrences = sum(
        record.known_candidate_count
        for record in records
    )

    total_relevant_occurrences = sum(
        record.relevant_count
        for record in records
    )

    known_relevant_occurrences = sum(
        record.known_relevant_count
        for record in records
    )

    payload = {
        "experiment": (
            "phase01_bpr_support_aware_subgroup_analysis"
        ),
        "protocol": {
            "cutoff_timestamp": args.cutoff,
            "validation_impressions": len(records),
            "k": args.k,
            "bootstrap_samples": args.bootstrap_samples,
            "confidence_level": args.confidence_level,
            "seed": args.seed,
        },
        "selected_bpr": {
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_negatives_per_positive": (
                args.max_negatives_per_positive
            ),
            "training_triples": len(triples),
            "known_users": len(model.user_to_index),
            "known_items": len(model.item_to_index),
        },
        "support": {
            "catalog_size": len(catalog),
            "known_item_catalog_fraction": (
                len(model.item_to_index) / len(catalog)
            ),
            "candidate_occurrences": (
                total_candidate_occurrences
            ),
            "known_candidate_occurrences": (
                known_candidate_occurrences
            ),
            "known_candidate_fraction": (
                known_candidate_occurrences
                / total_candidate_occurrences
            ),
            "relevant_occurrences": (
                total_relevant_occurrences
            ),
            "known_relevant_occurrences": (
                known_relevant_occurrences
            ),
            "known_relevant_fraction": (
                known_relevant_occurrences
                / total_relevant_occurrences
            ),
        },
        "subgroups": group_reports,
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
        f"Wrote subgroup report to {args.output}"
    )


if __name__ == "__main__":
    main()
