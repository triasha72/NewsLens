"""Chronological ranking evaluation of the trained Phase 03 two-tower model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from newslens.data import (
    load_behaviors,
    load_news,
    parse_impressions,
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
from newslens.evaluation.fallback import (
    _build_ranking_examples as _build_fallback_examples,
)
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.features import (
    ArticleTextFeatureEncoder,
)
from newslens.models import (
    ContentBasedRecommender,
    ContentPopularityFallbackRecommender,
    PopularityRecommender,
)
from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _fraction(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _precompute_article_embeddings(
    network: TwoTowerNetwork,
    values: np.ndarray,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Encode the complete article catalog exactly once."""

    if batch_size <= 0:
        raise ValueError(
            "embedding batch size must be positive."
        )

    network.eval()

    encoded_batches: list[
        torch.Tensor
    ] = []

    with torch.no_grad():
        for start in range(
            0,
            values.shape[0],
            batch_size,
        ):
            batch = torch.from_numpy(
                values[
                    start:
                    start + batch_size
                ]
            ).to(
                dtype=torch.float32
            )

            encoded = (
                network.encode_articles(
                    batch
                )
                .detach()
                .cpu()
            )

            encoded_batches.append(
                encoded
            )

    return torch.cat(
        encoded_batches,
        dim=0,
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
        "--training-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cutoff",
        default=(
            "2019-11-13T20:36:26"
        ),
    )

    parser.add_argument(
        "--validation-fraction",
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

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=2048,
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Verify the training artifact before evaluating it.
    # ------------------------------------------------------------------

    print("Loading frozen training report...")

    training_report = json.loads(
        args.training_report.read_text()
    )

    if training_report[
        "protocol"
    ]["limited_training_run"]:
        raise RuntimeError(
            "Refusing to evaluate a limited smoke-training checkpoint."
        )

    if (
        training_report[
            "protocol"
        ]["cutoff_timestamp"]
        != args.cutoff
    ):
        raise RuntimeError(
            "Training-report cutoff does not match evaluation cutoff."
        )

    expected_checkpoint_sha = (
        training_report[
            "checkpoint"
        ]["sha256"]
    )

    actual_checkpoint_sha = (
        _sha256(
            args.checkpoint
        )
    )

    if (
        actual_checkpoint_sha
        != expected_checkpoint_sha
    ):
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha}, "
            f"got {actual_checkpoint_sha}."
        )

    print(
        "  checkpoint SHA-256:",
        actual_checkpoint_sha,
    )

    # ------------------------------------------------------------------
    # Load chronological train/validation data.
    # ------------------------------------------------------------------

    split_path = (
        args.data_dir
        / "MINDsmall_train"
    )

    print("Loading MIND-small train...")

    news = load_news(
        split_path
        / "news.tsv"
    )

    behaviors = load_behaviors(
        split_path
        / "behaviors.tsv"
    )

    split = (
        chronological_train_validation_split(
            behaviors,
            validation_fraction=(
                args.validation_fraction
            ),
        )
    )

    actual_cutoff = (
        split.cutoff.isoformat()
    )

    if actual_cutoff != args.cutoff:
        raise RuntimeError(
            "Chronological cutoff mismatch: "
            f"expected {args.cutoff}, "
            f"got {actual_cutoff}."
        )

    if len(split.validation) != 31_393:
        raise RuntimeError(
            "Unexpected validation count: "
            f"{len(split.validation)}."
        )

    catalog = _prepare_catalog(
        news
    )

    fitting_news_ids = (
        _training_vocabulary_news_ids(
            split.train,
            catalog,
        )
    )

    # ------------------------------------------------------------------
    # Reconstruct the frozen article representation.
    # ------------------------------------------------------------------

    print(
        "Reconstructing train-only article features..."
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint_protocol = (
        checkpoint["protocol"]
    )

    checkpoint_cutoff = (
        checkpoint_protocol[
            "cutoff_timestamp"
        ]
    )

    if checkpoint_cutoff != args.cutoff:
        raise RuntimeError(
            "Checkpoint cutoff does not match evaluation cutoff."
        )

    max_history_length = int(
        checkpoint_protocol[
            "max_history_length"
        ]
    )

    if max_history_length <= 0:
        raise RuntimeError(
            "Checkpoint maximum history length is invalid."
        )

    feature_config = (
        checkpoint[
            "feature_encoder"
        ]
    )

    encoder = (
        ArticleTextFeatureEncoder(
            max_features=int(
                feature_config[
                    "max_features"
                ]
            ),
            svd_components=int(
                feature_config[
                    "svd_components"
                ]
            ),
            seed=int(
                feature_config[
                    "seed"
                ]
            ),
        )
    )

    feature_batch = (
        encoder.fit_transform(
            news,
            fitting_news_ids=(
                fitting_news_ids
            ),
        )
    )

    if (
        feature_batch.nonzero_article_count
        != feature_batch.article_count
    ):
        raise RuntimeError(
            "Evaluation article feature coverage is not complete."
        )

    # ------------------------------------------------------------------
    # Restore the trained two-tower network.
    # ------------------------------------------------------------------

    network_config = (
        TwoTowerConfig(
            **checkpoint[
                "network_config"
            ]
        )
    )

    network = TwoTowerNetwork(
        network_config
    )

    network.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    network.eval()

    print(
        "Precomputing learned article embeddings..."
    )

    article_embeddings = (
        _precompute_article_embeddings(
            network,
            feature_batch.values,
            batch_size=(
                args.embedding_batch_size
            ),
        )
    )

    news_to_index = {
        news_id: index
        for index, news_id
        in enumerate(
            feature_batch.news_ids
        )
    }

    if (
        article_embeddings.shape[0]
        != len(news_to_index)
    ):
        raise RuntimeError(
            "Article embedding count does not match article index."
        )

    # ------------------------------------------------------------------
    # Rebuild the fixed Content + popularity baseline.
    # ------------------------------------------------------------------

    print(
        "Building Content+Fallback baseline..."
    )

    popularity_model = (
        PopularityRecommender()
        .fit(
            split.train
        )
    )

    content_model = (
        ContentBasedRecommender(
            max_features=int(
                feature_config[
                    "max_features"
                ]
            ),
        ).fit(
            news,
            vocabulary_news_ids=(
                fitting_news_ids
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
        for example
        in fallback_result.examples
    }

    # ------------------------------------------------------------------
    # Build raw two-tower and production-style fallback rankings.
    # ------------------------------------------------------------------

    print(
        "Evaluating trained two-tower rankings..."
    )

    raw_examples: list[
        RankingExample
    ] = []

    routed_examples: list[
        RankingExample
    ] = []

    candidate_occurrences = 0

    raw_ranked_impressions = 0
    raw_empty_impressions = 0

    empty_history_impressions = 0
    unusable_history_impressions = 0

    truncated_history_impressions = 0

    popularity_fallback_impressions = 0
    changed_vs_baseline = 0

    with torch.no_grad():
        for row in split.validation.itertuples(
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

            candidate_ids = [
                news_id
                for news_id, _ in parsed
            ]

            if len(candidate_ids) != len(
                set(candidate_ids)
            ):
                raise RuntimeError(
                    "Validation impression "
                    f"{impression_id} contains duplicate candidates."
                )

            unknown_candidates = [
                news_id
                for news_id
                in candidate_ids
                if news_id
                not in news_to_index
            ]

            if unknown_candidates:
                raise RuntimeError(
                    "Two-tower features are missing "
                    "validation candidates."
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

            if not history_ids:
                empty_history_impressions += 1

            usable_history = [
                news_id
                for news_id
                in history_ids
                if news_id
                in news_to_index
            ]

            if (
                history_ids
                and not usable_history
            ):
                unusable_history_impressions += 1

            if (
                len(usable_history)
                > max_history_length
            ):
                truncated_history_impressions += 1

            usable_history = (
                usable_history[
                    -max_history_length:
                ]
            )

            if usable_history:
                history_indices = torch.tensor(
                    [
                        news_to_index[
                            news_id
                        ]
                        for news_id
                        in usable_history
                    ],
                    dtype=torch.long,
                )

                history_embedding_batch = (
                    article_embeddings[
                        history_indices
                    ]
                    .unsqueeze(0)
                )

                history_mask = torch.ones(
                    (
                        1,
                        len(
                            usable_history
                        ),
                    ),
                    dtype=torch.bool,
                )

                user_embedding = (
                    network.user_tower(
                        history_embedding_batch,
                        history_mask,
                    )
                )

                candidate_indices = torch.tensor(
                    [
                        news_to_index[
                            news_id
                        ]
                        for news_id
                        in candidate_ids
                    ],
                    dtype=torch.long,
                )

                candidate_embedding_batch = (
                    article_embeddings[
                        candidate_indices
                    ]
                )

                scores = (
                    network.score_candidates(
                        user_embedding,
                        candidate_embedding_batch,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )

                ranked_pairs = sorted(
                    zip(
                        candidate_ids,
                        scores,
                        strict=True,
                    ),
                    key=lambda item: (
                        -float(
                            item[1]
                        ),
                        item[0],
                    ),
                )[
                    : args.k
                ]

                raw_ranked_items = tuple(
                    news_id
                    for news_id, _
                    in ranked_pairs
                )

                raw_ranked_impressions += 1
            else:
                raw_ranked_items = ()
                raw_empty_impressions += 1

            raw_example = RankingExample(
                impression_id=(
                    impression_id
                ),
                ranked_items=(
                    raw_ranked_items
                ),
                relevant_items=(
                    relevant_items
                ),
            )

            raw_examples.append(
                raw_example
            )

            if raw_ranked_items:
                routed_items = (
                    raw_ranked_items
                )
            else:
                popularity_fallback_impressions += 1

                routed_items = tuple(
                    popularity_model
                    .rank_candidates(
                        candidate_ids,
                        top_k=args.k,
                        exclude_news_ids=(
                            history_ids
                        ),
                    )
                )

            routed_example = RankingExample(
                impression_id=(
                    impression_id
                ),
                ranked_items=(
                    routed_items
                ),
                relevant_items=(
                    relevant_items
                ),
            )

            routed_examples.append(
                routed_example
            )

            baseline_example = (
                fallback_by_id[
                    impression_id
                ]
            )

            if (
                routed_items
                != baseline_example.ranked_items
            ):
                changed_vs_baseline += 1

    if len(raw_examples) != len(
        split.validation
    ):
        raise RuntimeError(
            "Raw two-tower evaluation did not cover validation."
        )

    if len(routed_examples) != len(
        split.validation
    ):
        raise RuntimeError(
            "Routed two-tower evaluation did not cover validation."
        )

    # ------------------------------------------------------------------
    # Metrics and paired inference.
    # ------------------------------------------------------------------

    baseline_metrics = (
        evaluate_rankings(
            fallback_result.examples,
            catalog,
            k=args.k,
        )
    )

    raw_metrics = evaluate_rankings(
        raw_examples,
        catalog,
        k=args.k,
    )

    routed_metrics = (
        evaluate_rankings(
            routed_examples,
            catalog,
            k=args.k,
        )
    )

    print(
        "Running paired bootstrap comparisons..."
    )

    raw_vs_baseline = (
        paired_bootstrap_ranking_comparison(
            fallback_result.examples,
            raw_examples,
            baseline_model_name=(
                "content_popularity_fallback"
            ),
            candidate_model_name=(
                "raw_two_tower"
            ),
            k=args.k,
            bootstrap_samples=(
                args.bootstrap_samples
            ),
            confidence_level=(
                args.confidence_level
            ),
            random_seed=(
                args.seed
            ),
        )
    )

    routed_vs_baseline = (
        paired_bootstrap_ranking_comparison(
            fallback_result.examples,
            routed_examples,
            baseline_model_name=(
                "content_popularity_fallback"
            ),
            candidate_model_name=(
                "two_tower_popularity_fallback"
            ),
            k=args.k,
            bootstrap_samples=(
                args.bootstrap_samples
            ),
            confidence_level=(
                args.confidence_level
            ),
            random_seed=(
                args.seed
            ),
        )
    )

    routed_vs_raw = (
        paired_bootstrap_ranking_comparison(
            raw_examples,
            routed_examples,
            baseline_model_name=(
                "raw_two_tower"
            ),
            candidate_model_name=(
                "two_tower_popularity_fallback"
            ),
            k=args.k,
            bootstrap_samples=(
                args.bootstrap_samples
            ),
            confidence_level=(
                args.confidence_level
            ),
            random_seed=(
                args.seed
            ),
        )
    )

    payload = {
        "experiment": (
            "phase03_two_tower_chronological_evaluation"
        ),
        "protocol": {
            "dataset": (
                "MINDsmall_train"
            ),
            "cutoff_timestamp": (
                actual_cutoff
            ),
            "training_impressions": (
                len(split.train)
            ),
            "validation_impressions": (
                len(split.validation)
            ),
            "official_dev_used": False,
            "previously_exposed_dev_used": False,
            "k": args.k,
            "bootstrap_samples": (
                args.bootstrap_samples
            ),
            "confidence_level": (
                args.confidence_level
            ),
            "seed": args.seed,
            "max_history_length": (
                max_history_length
            ),
        },
        "checkpoint": {
            "sha256": (
                actual_checkpoint_sha
            ),
            "network_config": (
                checkpoint[
                    "network_config"
                ]
            ),
            "training_config": (
                checkpoint[
                    "training_config"
                ]
            ),
        },
        "feature_reconstruction": {
            "fit_article_count": (
                encoder.fit_article_count
            ),
            "indexed_article_count": (
                feature_batch.article_count
            ),
            "nonzero_article_count": (
                feature_batch
                .nonzero_article_count
            ),
            "tfidf_vocabulary_size": (
                encoder.vocabulary_size
            ),
            "svd_components": (
                encoder.output_dim
            ),
        },
        "accounting": {
            "candidate_occurrences": (
                candidate_occurrences
            ),
            "raw_ranked_impressions": (
                raw_ranked_impressions
            ),
            "raw_empty_impressions": (
                raw_empty_impressions
            ),
            "empty_history_impressions": (
                empty_history_impressions
            ),
            "unusable_history_impressions": (
                unusable_history_impressions
            ),
            "truncated_history_impressions": (
                truncated_history_impressions
            ),
            "popularity_fallback_impressions": (
                popularity_fallback_impressions
            ),
            "changed_vs_content_fallback": (
                changed_vs_baseline
            ),
            "raw_ranked_fraction": (
                _fraction(
                    raw_ranked_impressions,
                    len(
                        split.validation
                    ),
                )
            ),
        },
        "metrics": {
            "content_popularity_fallback": (
                baseline_metrics.to_dict()
            ),
            "raw_two_tower": (
                raw_metrics.to_dict()
            ),
            "two_tower_popularity_fallback": (
                routed_metrics.to_dict()
            ),
        },
        "comparisons": {
            "raw_two_tower_vs_content_fallback": (
                raw_vs_baseline.to_dict()
            ),
            "two_tower_fallback_vs_content_fallback": (
                routed_vs_baseline.to_dict()
            ),
            "two_tower_fallback_vs_raw_two_tower": (
                routed_vs_raw.to_dict()
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
    print("=" * 92)
    print(
        "PHASE 03D TWO-TOWER EVALUATION"
    )
    print("=" * 92)

    print(
        f"{'Model':<34}"
        f"{'NDCG':>12}"
        f"{'MRR':>12}"
        f"{'Recall':>12}"
        f"{'Hit':>12}"
        f"{'Empty':>10}"
    )

    for (
        name,
        metrics,
    ) in (
        (
            "Content+Fallback",
            baseline_metrics,
        ),
        (
            "Raw Two-Tower",
            raw_metrics,
        ),
        (
            "Two-Tower+Popularity",
            routed_metrics,
        ),
    ):
        print(
            f"{name:<34}"
            f"{metrics.ndcg_at_k:>12.6f}"
            f"{metrics.mrr_at_k:>12.6f}"
            f"{metrics.recall_at_k:>12.6f}"
            f"{metrics.hit_rate_at_k:>12.6f}"
            f"{metrics.empty_ranking_impressions:>10}"
        )

    print()
    print(
        "Two-Tower+Popularity minus Content+Fallback:"
    )

    routed_comparison = (
        routed_vs_baseline.to_dict()
    )

    for metric_name, result in (
        routed_comparison[
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
        "Raw ranked impressions:",
        raw_ranked_impressions,
        "/",
        len(split.validation),
        (
            f" ({_fraction(
                raw_ranked_impressions,
                len(split.validation),
            ):.2%})"
        ),
    )

    print(
        "Popularity fallbacks:",
        popularity_fallback_impressions,
    )

    print(
        "History truncations:",
        truncated_history_impressions,
    )

    print()
    print(
        f"Wrote report to {args.output}"
    )


if __name__ == "__main__":
    main()
