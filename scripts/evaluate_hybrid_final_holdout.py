"""One-time final MIND-small dev evaluation for the frozen Phase 02 hybrid."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd

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
from newslens.models import (
    ContentBasedRecommender,
    ContentPopularityFallbackRecommender,
    PopularityRecommender,
)
from newslens.models.collaborative import (
    CollaborativeRecommender,
)
from newslens.models.hybrid import HybridRecommender

# ---------------------------------------------------------------------------
# Frozen before official MIND-small dev evaluation.
# Do not change these values based on holdout results.
# ---------------------------------------------------------------------------

EXPECTED_DEV_ZIP_SHA256 = (
    "d6ce515dcaa6b6d47ddf0a326eebc8a31b84735ae410285c9882ca2a06eec669"
)

COLLABORATIVE_WEIGHT = 0.20
MINIMUM_SUPPORTED_CANDIDATES = 10

K = 10

EMBEDDING_DIM = 64
EPOCHS = 10
BATCH_SIZE = 2048
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-6
MAX_NEGATIVES_PER_POSITIVE = 3

MAX_FEATURES = 50_000

BOOTSTRAP_SAMPLES = 1_000
CONFIDENCE_LEVEL = 0.95
SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _prepare_combined_catalog(
    train_news: pd.DataFrame,
    dev_news: pd.DataFrame,
) -> pd.DataFrame:
    """Use train metadata for known IDs and dev metadata for dev-only IDs."""

    train_ids = set(
        train_news["news_id"].astype(str)
    )

    dev_only = dev_news.loc[
        ~dev_news["news_id"].astype(str).isin(
            train_ids
        )
    ].copy()

    combined = pd.concat(
        [
            train_news,
            dev_only,
        ],
        ignore_index=True,
    )

    if combined["news_id"].duplicated().any():
        raise RuntimeError(
            "Combined train/dev article catalog "
            "contains duplicate news IDs."
        )

    return combined


def _full_training_bpr_cutoff(
    database: Path,
    *,
    expected_behavior_count: int,
):
    """Return an exclusive timestamp immediately after all train events."""

    with duckdb.connect(
        str(database),
        read_only=True,
    ) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS behavior_count,
                MAX(event_timestamp) AS max_timestamp
            FROM behavior_events
            """
        ).fetchone()

    if row is None:
        raise RuntimeError(
            "Training warehouse returned no summary."
        )

    behavior_count = int(row[0])
    max_timestamp = row[1]

    if behavior_count != expected_behavior_count:
        raise RuntimeError(
            "Raw training behaviors and warehouse "
            "behavior counts differ: "
            f"{expected_behavior_count} vs "
            f"{behavior_count}."
        )

    if max_timestamp is None:
        raise RuntimeError(
            "Training warehouse has no maximum timestamp."
        )

    return (
        max_timestamp + timedelta(microseconds=1),
        behavior_count,
        max_timestamp,
    )


def main() -> None:
    repo_root = Path.cwd()

    local_root = (
        Path.home()
        / "newslens-local-data"
    )

    train_dir = (
        local_root
        / "MINDsmall_train"
    )

    dev_dir = (
        local_root
        / "MINDsmall_dev"
    )

    dev_zip = (
        local_root
        / "MINDsmall_dev.zip"
    )

    database = (
        repo_root
        / "warehouses"
        / "mindsmall_train.duckdb"
    )

    output = (
        repo_root
        / "reports"
        / "hybrid_final_mindsmall_dev.json"
    )

    # ------------------------------------------------------------------
    # Verify frozen holdout artifact.
    # ------------------------------------------------------------------

    print("Verifying official holdout artifact...")

    if not dev_zip.is_file():
        raise FileNotFoundError(
            f"Missing holdout ZIP: {dev_zip}"
        )

    actual_dev_sha256 = _sha256(
        dev_zip
    )

    if (
        actual_dev_sha256
        != EXPECTED_DEV_ZIP_SHA256
    ):
        raise RuntimeError(
            "MINDsmall_dev ZIP checksum mismatch. "
            f"Expected {EXPECTED_DEV_ZIP_SHA256}, "
            f"got {actual_dev_sha256}."
        )

    print(
        "  SHA-256:",
        actual_dev_sha256,
    )

    # ------------------------------------------------------------------
    # Load raw train and frozen dev sets.
    # ------------------------------------------------------------------

    print("Loading full MIND-small training set...")

    train_news = load_news(
        train_dir / "news.tsv"
    )

    train_behaviors = load_behaviors(
        train_dir / "behaviors.tsv"
    )

    print("Loading official MIND-small dev holdout...")

    dev_news = load_news(
        dev_dir / "news.tsv"
    )

    dev_behaviors = load_behaviors(
        dev_dir / "behaviors.tsv"
    )

    print(
        "  training behaviors:",
        len(train_behaviors),
    )

    print(
        "  holdout behaviors:",
        len(dev_behaviors),
    )

    # ------------------------------------------------------------------
    # Article information boundary.
    #
    # Vocabulary / IDF is learned only from train-referenced articles.
    # Dev-only article text may be transformed by the frozen vectorizer.
    # ------------------------------------------------------------------

    print("Preparing train/dev article catalog...")

    train_catalog = _prepare_catalog(
        train_news
    )

    vocabulary_news_ids = (
        _training_vocabulary_news_ids(
            train_behaviors,
            train_catalog,
        )
    )

    combined_news = (
        _prepare_combined_catalog(
            train_news,
            dev_news,
        )
    )

    combined_catalog = (
        _prepare_catalog(
            combined_news
        )
    )

    train_ids = set(
        train_news["news_id"].astype(str)
    )

    dev_ids = set(
        dev_news["news_id"].astype(str)
    )

    dev_only_article_count = len(
        dev_ids - train_ids
    )

    print(
        "  training articles:",
        len(train_news),
    )

    print(
        "  holdout articles:",
        len(dev_news),
    )

    print(
        "  dev-only articles:",
        dev_only_article_count,
    )

    # ------------------------------------------------------------------
    # Fit frozen Content + Popularity baseline on full train.
    # ------------------------------------------------------------------

    print("Training popularity on full train...")

    popularity_model = (
        PopularityRecommender().fit(
            train_behaviors
        )
    )

    print(
        "Training TF-IDF vocabulary/IDF "
        "from train-referenced articles..."
    )

    content_model = (
        ContentBasedRecommender(
            max_features=MAX_FEATURES,
        ).fit(
            combined_news,
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
        "Building Content+Fallback "
        "holdout rankings..."
    )

    fallback_result = (
        _build_fallback_examples(
            dev_behaviors,
            fallback_model,
            content_model,
            combined_catalog,
            k=K,
        )
    )

    fallback_by_id = {
        example.impression_id: example
        for example
        in fallback_result.examples
    }

    # ------------------------------------------------------------------
    # Fit frozen BPR configuration on the complete train warehouse.
    # ------------------------------------------------------------------

    print(
        "Preparing full-training BPR boundary..."
    )

    (
        full_training_cutoff,
        warehouse_behavior_count,
        warehouse_max_timestamp,
    ) = _full_training_bpr_cutoff(
        database,
        expected_behavior_count=(
            len(train_behaviors)
        ),
    )

    print(
        "  warehouse behaviors:",
        warehouse_behavior_count,
    )

    print(
        "  warehouse max timestamp:",
        warehouse_max_timestamp,
    )

    print(
        "  exclusive BPR cutoff:",
        full_training_cutoff,
    )

    print(
        "Loading full-training BPR triples..."
    )

    triples = load_bpr_triples(
        database,
        cutoff_timestamp=(
            full_training_cutoff
        ),
        max_negatives_per_positive=(
            MAX_NEGATIVES_PER_POSITIVE
        ),
        seed=SEED,
    )

    print(
        "Training frozen BPR:",
        {
            "triples": len(triples),
            "embedding_dim": (
                EMBEDDING_DIM
            ),
            "epochs": EPOCHS,
            "negatives": (
                MAX_NEGATIVES_PER_POSITIVE
            ),
        },
    )

    collaborative_model = (
        CollaborativeRecommender(
            embedding_dim=(
                EMBEDDING_DIM
            ),
            seed=SEED,
        ).fit(
            triples,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    )

    # ------------------------------------------------------------------
    # Frozen support-gated hybrid.
    # ------------------------------------------------------------------

    hybrid_model = HybridRecommender(
        content_model,
        collaborative_model,
        popularity_model,
        collaborative_weight=(
            COLLABORATIVE_WEIGHT
        ),
        minimum_supported_candidates=(
            MINIMUM_SUPPORTED_CANDIDATES
        ),
    )

    print(
        "Evaluating frozen support-gated hybrid "
        "on official dev..."
    )

    hybrid_examples: list[
        RankingExample
    ] = []

    gate_open_impressions = 0
    gate_closed_impressions = 0
    changed_rankings = 0

    candidate_occurrences = 0
    supported_candidate_occurrences = 0

    hybrid_topk_occurrences = 0
    content_topk_occurrences = 0
    popularity_topk_occurrences = 0

    for row in dev_behaviors.itertuples(
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

        candidate_occurrences += len(
            candidate_ids
        )

        baseline_example = (
            fallback_by_id[
                impression_id
            ]
        )

        supported_count = (
            hybrid_model
            .supported_candidate_count(
                user_id,
                candidate_ids,
            )
        )

        supported_candidate_occurrences += (
            supported_count
        )

        gate_open = (
            hybrid_model
            .collaborative_gate_open(
                user_id,
                candidate_ids,
            )
        )

        if gate_open:
            gate_open_impressions += 1
        else:
            gate_closed_impressions += 1

        recommendations = (
            hybrid_model
            .recommend_for_user(
                user_id,
                history_ids,
                candidate_news_ids=(
                    candidate_ids
                ),
                top_k=K,
            )
        )

        ranked_items = tuple(
            recommendation.news_id
            for recommendation
            in recommendations
        )

        example = RankingExample(
            impression_id=(
                impression_id
            ),
            ranked_items=(
                ranked_items
            ),
            relevant_items=(
                baseline_example
                .relevant_items
            ),
        )

        hybrid_examples.append(
            example
        )

        if (
            ranked_items
            != baseline_example.ranked_items
        ):
            changed_rankings += 1

        hybrid_topk_occurrences += sum(
            recommendation.source
            == "hybrid"
            for recommendation
            in recommendations
        )

        content_topk_occurrences += sum(
            recommendation.source
            == "content"
            for recommendation
            in recommendations
        )

        popularity_topk_occurrences += sum(
            recommendation.source
            == "popularity"
            for recommendation
            in recommendations
        )

    if (
        gate_open_impressions
        + gate_closed_impressions
        != len(dev_behaviors)
    ):
        raise RuntimeError(
            "Hybrid gate accounting does "
            "not cover all dev impressions."
        )

    # ------------------------------------------------------------------
    # Final metrics.
    # ------------------------------------------------------------------

    baseline_metrics = (
        evaluate_rankings(
            fallback_result.examples,
            combined_catalog,
            k=K,
        )
    )

    hybrid_metrics = (
        evaluate_rankings(
            hybrid_examples,
            combined_catalog,
            k=K,
        )
    )

    print(
        "Running predefined paired bootstrap..."
    )

    comparison = (
        paired_bootstrap_ranking_comparison(
            fallback_result.examples,
            hybrid_examples,
            baseline_model_name=(
                "content_popularity_fallback"
            ),
            candidate_model_name=(
                "frozen_support_gated_hybrid"
            ),
            k=K,
            bootstrap_samples=(
                BOOTSTRAP_SAMPLES
            ),
            confidence_level=(
                CONFIDENCE_LEVEL
            ),
            random_seed=SEED,
        )
    )

    comparison_dict = (
        comparison.to_dict()
    )

    ndcg = comparison_dict[
        "metrics"
    ]["ndcg_at_k"]

    holdout_direction_positive = (
        ndcg["point_difference"]
        > 0.0
    )

    holdout_interval_positive = (
        ndcg["lower_bound"]
        > 0.0
    )

    payload = {
        "experiment": (
            "phase02_final_mindsmall_dev_holdout"
        ),
        "evaluation_status": {
            "architecture_frozen_before_holdout": True,
            "pre_holdout_commit": "4d0b493",
            "official_dev_evaluation_number": 1,
            "hyperparameter_tuning_on_dev": False,
            "post_holdout_tuning_permitted": False,
        },
        "data": {
            "training_source": (
                "MINDsmall_train"
            ),
            "evaluation_source": (
                "MINDsmall_dev"
            ),
            "dev_zip_sha256": (
                actual_dev_sha256
            ),
            "training_behaviors": (
                len(train_behaviors)
            ),
            "holdout_behaviors": (
                len(dev_behaviors)
            ),
            "training_news": (
                len(train_news)
            ),
            "holdout_news": (
                len(dev_news)
            ),
            "dev_only_articles": (
                dev_only_article_count
            ),
            "combined_catalog_size": (
                len(combined_catalog)
            ),
        },
        "information_boundary": {
            "popularity_fit_source": (
                "MINDsmall_train behaviors only"
            ),
            "bpr_fit_source": (
                "MINDsmall_train warehouse only"
            ),
            "tfidf_fit_source": (
                "train-referenced article text only"
            ),
            "dev_article_text_usage": (
                "transformed using frozen train-fitted TF-IDF vectorizer"
            ),
            "dev_behavior_usage": (
                "evaluation only"
            ),
        },
        "frozen_configuration": {
            "k": K,
            "collaborative_weight": (
                COLLABORATIVE_WEIGHT
            ),
            "minimum_supported_candidates": (
                MINIMUM_SUPPORTED_CANDIDATES
            ),
            "embedding_dim": (
                EMBEDDING_DIM
            ),
            "epochs": EPOCHS,
            "batch_size": (
                BATCH_SIZE
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "max_negatives_per_positive": (
                MAX_NEGATIVES_PER_POSITIVE
            ),
            "max_features": (
                MAX_FEATURES
            ),
            "bootstrap_samples": (
                BOOTSTRAP_SAMPLES
            ),
            "confidence_level": (
                CONFIDENCE_LEVEL
            ),
            "seed": SEED,
        },
        "training_accounting": {
            "warehouse_behavior_count": (
                warehouse_behavior_count
            ),
            "warehouse_max_timestamp": (
                warehouse_max_timestamp.isoformat()
            ),
            "bpr_exclusive_cutoff": (
                full_training_cutoff.isoformat()
            ),
            "bpr_training_triples": (
                len(triples)
            ),
            "bpr_known_users": (
                len(
                    collaborative_model
                    .user_to_index
                )
            ),
            "bpr_known_items": (
                len(
                    collaborative_model
                    .item_to_index
                )
            ),
            "tfidf_vocabulary_articles": (
                content_model
                .vocabulary_article_count
            ),
            "tfidf_indexed_articles": (
                content_model
                .indexed_article_count
            ),
            "tfidf_vocabulary_size": (
                content_model
                .vocabulary_size
            ),
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
                / len(dev_behaviors)
            ),
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
            "changed_rankings": (
                changed_rankings
            ),
            "hybrid_topk_occurrences": (
                hybrid_topk_occurrences
            ),
            "content_topk_occurrences": (
                content_topk_occurrences
            ),
            "popularity_topk_occurrences": (
                popularity_topk_occurrences
            ),
        },
        "baseline_metrics": (
            baseline_metrics.to_dict()
        ),
        "hybrid_metrics": (
            hybrid_metrics.to_dict()
        ),
        "paired_comparison": (
            comparison_dict
        ),
        "holdout_conclusion": {
            "ndcg_direction_matches_internal_validation": (
                holdout_direction_positive
            ),
            "ndcg_95pct_interval_entirely_positive": (
                holdout_interval_positive
            ),
            "no_post_holdout_tuning": True,
        },
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 88)
    print(
        "FINAL MIND-SMALL DEV HOLDOUT"
    )
    print("=" * 88)

    print(
        f"{'Model':<28}"
        f"{'NDCG':>12}"
        f"{'MRR':>12}"
        f"{'Recall':>12}"
        f"{'Hit':>12}"
    )

    print(
        f"{'Content+Fallback':<28}"
        f"{baseline_metrics.ndcg_at_k:>12.6f}"
        f"{baseline_metrics.mrr_at_k:>12.6f}"
        f"{baseline_metrics.recall_at_k:>12.6f}"
        f"{baseline_metrics.hit_rate_at_k:>12.6f}"
    )

    print(
        f"{'Frozen gated hybrid':<28}"
        f"{hybrid_metrics.ndcg_at_k:>12.6f}"
        f"{hybrid_metrics.mrr_at_k:>12.6f}"
        f"{hybrid_metrics.recall_at_k:>12.6f}"
        f"{hybrid_metrics.hit_rate_at_k:>12.6f}"
    )

    print()
    print(
        "Frozen hybrid minus baseline:"
    )

    for metric_name, result in (
        comparison_dict[
            "metrics"
        ].items()
    ):
        print(
            f"  {metric_name:<15}"
            f"delta="
            f"{result['point_difference']:+.6f} "
            f"CI=["
            f"{result['lower_bound']:+.6f}, "
            f"{result['upper_bound']:+.6f}] "
            f"excludes_zero="
            f"{result['excludes_zero']}"
        )

    print()
    print(
        "Gate open:",
        gate_open_impressions,
        "/",
        len(dev_behaviors),
        (
            f" ({gate_open_impressions / len(dev_behaviors):.2%})"
        ),
    )

    print()
    print(
        "NDCG direction matches internal validation:",
        holdout_direction_positive,
    )

    print(
        "NDCG 95% interval entirely positive:",
        holdout_interval_positive,
    )

    print()
    print(
        f"Wrote final holdout report to {output}"
    )


if __name__ == "__main__":
    main()
