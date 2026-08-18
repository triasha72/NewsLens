"""Direct paired comparison of Phase 03 two-tower checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from evaluate_two_tower import (
    _precompute_article_embeddings,
    _sha256,
)

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
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.features import (
    ArticleTextFeatureEncoder,
)
from newslens.models import (
    PopularityRecommender,
)
from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
)


def _load_artifact(
    checkpoint_path: Path,
    report_path: Path,
    *,
    cutoff: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    str,
]:
    report = json.loads(
        report_path.read_text()
    )

    protocol = report["protocol"]

    if protocol["limited_training_run"]:
        raise RuntimeError(
            "Refusing to compare a limited training run."
        )

    if protocol["cutoff_timestamp"] != cutoff:
        raise RuntimeError(
            "Training-report cutoff mismatch."
        )

    expected_sha = report[
        "checkpoint"
    ]["sha256"]

    actual_sha = _sha256(
        checkpoint_path
    )

    if actual_sha != expected_sha:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch: "
            f"{checkpoint_path}."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint["protocol"][
            "cutoff_timestamp"
        ]
        != cutoff
    ):
        raise RuntimeError(
            "Checkpoint cutoff mismatch."
        )

    return (
        report,
        checkpoint,
        actual_sha,
    )


def _restore_network(
    checkpoint: dict[str, object],
) -> TwoTowerNetwork:
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

    return network


def _build_routed_examples(
    validation,
    *,
    network: TwoTowerNetwork,
    article_embeddings: torch.Tensor,
    news_to_index: dict[str, int],
    popularity_model: PopularityRecommender,
    max_history_length: int,
    k: int,
) -> tuple[
    tuple[RankingExample, ...],
    dict[str, int],
]:
    examples: list[
        RankingExample
    ] = []

    raw_ranked = 0
    popularity_fallbacks = 0
    truncated_histories = 0

    with torch.no_grad():
        for row in validation.itertuples(
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
                for news_id, _
                in parsed
            ]

            relevant_items = frozenset(
                news_id
                for news_id, label
                in parsed
                if label == 1
            )

            if any(
                news_id
                not in news_to_index
                for news_id in candidate_ids
            ):
                raise RuntimeError(
                    "Validation candidate is missing features."
                )

            usable_history = [
                news_id
                for news_id in history_ids
                if news_id
                in news_to_index
            ]

            if (
                len(usable_history)
                > max_history_length
            ):
                truncated_histories += 1

            usable_history = (
                usable_history[
                    -max_history_length:
                ]
            )

            if usable_history:
                history_indices = (
                    torch.tensor(
                        [
                            news_to_index[
                                news_id
                            ]
                            for news_id
                            in usable_history
                        ],
                        dtype=torch.long,
                    )
                )

                history_embeddings = (
                    article_embeddings[
                        history_indices
                    ].unsqueeze(0)
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
                        history_embeddings,
                        history_mask,
                    )
                )

                candidate_indices = (
                    torch.tensor(
                        [
                            news_to_index[
                                news_id
                            ]
                            for news_id
                            in candidate_ids
                        ],
                        dtype=torch.long,
                    )
                )

                candidate_embeddings = (
                    article_embeddings[
                        candidate_indices
                    ]
                )

                scores = (
                    network.score_candidates(
                        user_embedding,
                        candidate_embeddings,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )

                ranked_items = tuple(
                    news_id
                    for news_id, _
                    in sorted(
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
                    )[:k]
                )

                raw_ranked += 1

            else:
                ranked_items = tuple(
                    popularity_model
                    .rank_candidates(
                        candidate_ids,
                        top_k=k,
                        exclude_news_ids=(
                            history_ids
                        ),
                    )
                )

                popularity_fallbacks += 1

            examples.append(
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

    return (
        tuple(examples),
        {
            "raw_ranked_impressions": (
                raw_ranked
            ),
            "popularity_fallback_impressions": (
                popularity_fallbacks
            ),
            "truncated_history_impressions": (
                truncated_histories
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--v01-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--v01-training-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--v02-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--v02-training-report",
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
        default="2019-11-13T20:36:26",
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

    print(
        "Loading frozen v0.1 and v0.2 artifacts..."
    )

    (
        _,
        v01_checkpoint,
        v01_sha,
    ) = _load_artifact(
        args.v01_checkpoint,
        args.v01_training_report,
        cutoff=args.cutoff,
    )

    (
        _,
        v02_checkpoint,
        v02_sha,
    ) = _load_artifact(
        args.v02_checkpoint,
        args.v02_training_report,
        cutoff=args.cutoff,
    )

    if (
        v01_checkpoint[
            "network_config"
        ]
        != v02_checkpoint[
            "network_config"
        ]
    ):
        raise RuntimeError(
            "Network configuration differs between versions."
        )

    if (
        v01_checkpoint[
            "feature_encoder"
        ]
        != v02_checkpoint[
            "feature_encoder"
        ]
    ):
        raise RuntimeError(
            "Feature configuration differs between versions."
        )

    v01_history_length = int(
        v01_checkpoint[
            "protocol"
        ]["max_history_length"]
    )

    v02_history_length = int(
        v02_checkpoint[
            "protocol"
        ]["max_history_length"]
    )

    if (
        v01_history_length
        != v02_history_length
    ):
        raise RuntimeError(
            "History-length protocol differs between versions."
        )

    print(
        "Loading chronological MIND-small split..."
    )

    split_path = (
        args.data_dir
        / "MINDsmall_train"
    )

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

    if (
        split.cutoff.isoformat()
        != args.cutoff
    ):
        raise RuntimeError(
            "Chronological cutoff mismatch."
        )

    if len(split.validation) != 31_393:
        raise RuntimeError(
            "Unexpected validation size."
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

    feature_config = (
        v01_checkpoint[
            "feature_encoder"
        ]
    )

    print(
        "Reconstructing shared train-only article features..."
    )

    encoder = ArticleTextFeatureEncoder(
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
            "Article feature coverage is incomplete."
        )

    news_to_index = {
        news_id: index
        for index, news_id
        in enumerate(
            feature_batch.news_ids
        )
    }

    print(
        "Restoring v0.1 network..."
    )

    v01_network = _restore_network(
        v01_checkpoint
    )

    v01_embeddings = (
        _precompute_article_embeddings(
            v01_network,
            feature_batch.values,
            batch_size=(
                args.embedding_batch_size
            ),
        )
    )

    print(
        "Restoring v0.2 network..."
    )

    v02_network = _restore_network(
        v02_checkpoint
    )

    v02_embeddings = (
        _precompute_article_embeddings(
            v02_network,
            feature_batch.values,
            batch_size=(
                args.embedding_batch_size
            ),
        )
    )

    popularity_model = (
        PopularityRecommender()
        .fit(
            split.train
        )
    )

    print(
        "Building v0.1 rankings..."
    )

    (
        v01_examples,
        v01_accounting,
    ) = _build_routed_examples(
        split.validation,
        network=v01_network,
        article_embeddings=(
            v01_embeddings
        ),
        news_to_index=(
            news_to_index
        ),
        popularity_model=(
            popularity_model
        ),
        max_history_length=(
            v01_history_length
        ),
        k=args.k,
    )

    print(
        "Building v0.2 rankings..."
    )

    (
        v02_examples,
        v02_accounting,
    ) = _build_routed_examples(
        split.validation,
        network=v02_network,
        article_embeddings=(
            v02_embeddings
        ),
        news_to_index=(
            news_to_index
        ),
        popularity_model=(
            popularity_model
        ),
        max_history_length=(
            v02_history_length
        ),
        k=args.k,
    )

    v01_metrics = evaluate_rankings(
        v01_examples,
        catalog,
        k=args.k,
    )

    v02_metrics = evaluate_rankings(
        v02_examples,
        catalog,
        k=args.k,
    )

    changed_rankings = sum(
        first.ranked_items
        != second.ranked_items
        for first, second
        in zip(
            v01_examples,
            v02_examples,
            strict=True,
        )
    )

    print(
        "Running direct paired bootstrap..."
    )

    comparison = (
        paired_bootstrap_ranking_comparison(
            v01_examples,
            v02_examples,
            baseline_model_name=(
                "two_tower_v0_1"
            ),
            candidate_model_name=(
                "hard_negative_two_tower_v0_2"
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
            "phase03_direct_v02_vs_v01"
        ),
        "protocol": {
            "dataset": (
                "MINDsmall_train"
            ),
            "cutoff_timestamp": (
                args.cutoff
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
        },
        "checkpoints": {
            "v0_1_sha256": v01_sha,
            "v0_2_sha256": v02_sha,
        },
        "configuration_parity": {
            "network_config_equal": True,
            "feature_config_equal": True,
            "max_history_length_equal": True,
        },
        "accounting": {
            "v0_1": v01_accounting,
            "v0_2": v02_accounting,
            "changed_rankings": (
                changed_rankings
            ),
        },
        "metrics": {
            "v0_1": (
                v01_metrics.to_dict()
            ),
            "v0_2": (
                v02_metrics.to_dict()
            ),
        },
        "comparison": (
            comparison.to_dict()
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

    print()
    print("=" * 90)
    print(
        "PHASE 03F DIRECT V0.2 VS V0.1"
    )
    print("=" * 90)

    print(
        f"{'Model':<24}"
        f"{'NDCG':>12}"
        f"{'MRR':>12}"
        f"{'Recall':>12}"
        f"{'Hit':>12}"
    )

    for name, metrics in (
        (
            "v0.1",
            v01_metrics,
        ),
        (
            "v0.2 hard-negative",
            v02_metrics,
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

    result = comparison.to_dict()

    for metric, values in (
        result[
            "metrics"
        ].items()
    ):
        print(
            f"{metric:<15}"
            f"delta="
            f"{values['point_difference']:+.6f} "
            f"CI=["
            f"{values['lower_bound']:+.6f}, "
            f"{values['upper_bound']:+.6f}] "
            f"excludes_zero="
            f"{values['excludes_zero']}"
        )

    print()
    print(
        "Changed rankings:",
        changed_rankings,
        "/",
        len(split.validation),
    )

    print()
    print(
        "Report:",
        args.output,
    )


if __name__ == "__main__":
    main()
