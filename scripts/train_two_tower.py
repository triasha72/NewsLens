"""Train the Phase 03 native PyTorch two-tower recommender."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from newslens.data import (
    load_behaviors,
    load_news,
)
from newslens.data.two_tower_training import (
    build_two_tower_positive_examples,
)
from newslens.evaluation.content import (
    _prepare_catalog,
    _training_vocabulary_news_ids,
)
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.features import (
    ArticleTextFeatureEncoder,
)
from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
)
from newslens.training.two_tower import (
    TwoTowerTrainingConfig,
    train_two_tower,
)


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
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
        "--max-features",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--svd-components",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-history-length",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "mps",
            "cuda",
            "auto",
        ],
        default="cpu",
    )

    parser.add_argument(
        "--max-training-examples",
        type=int,
        default=0,
        help=(
            "Optional deterministic prefix limit. "
            "Use only for smoke tests; 0 uses all examples."
        ),
    )

    args = parser.parse_args()

    split_path = (
        args.data_dir
        / "MINDsmall_train"
    )

    print(
        "Loading MIND-small train..."
    )

    news = load_news(
        split_path
        / "news.tsv"
    )

    behaviors = load_behaviors(
        split_path
        / "behaviors.tsv"
    )

    print(
        "Creating chronological split..."
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

    catalog = _prepare_catalog(
        news
    )

    fitting_news_ids = (
        _training_vocabulary_news_ids(
            split.train,
            catalog,
        )
    )

    print(
        "Fitting train-only article features..."
    )

    encoder = (
        ArticleTextFeatureEncoder(
            max_features=(
                args.max_features
            ),
            svd_components=(
                args.svd_components
            ),
            seed=args.seed,
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

    article_features = (
        feature_batch.as_mapping(
            include_zero=False
        )
    )

    print(
        "Building chronological "
        "history-to-click examples..."
    )

    build_result = (
        build_two_tower_positive_examples(
            split.train,
            available_news_ids=(
                article_features
            ),
            max_history_length=(
                args.max_history_length
            ),
        )
    )

    examples = (
        build_result.examples
    )

    if args.max_training_examples < 0:
        raise RuntimeError(
            "max-training-examples cannot be negative."
        )

    if args.max_training_examples > 0:
        examples = examples[
            : args.max_training_examples
        ]

    if len(examples) < 2:
        raise RuntimeError(
            "At least two examples are required for training."
        )

    print(
        "Training examples:",
        len(examples),
        "/",
        build_result.usable_example_count,
    )

    network_config = (
        TwoTowerConfig(
            input_dim=(
                args.svd_components
            ),
            hidden_dim=(
                args.hidden_dim
            ),
            embedding_dim=(
                args.embedding_dim
            ),
            dropout=args.dropout,
            temperature=(
                args.temperature
            ),
        )
    )

    torch.manual_seed(
        args.seed
    )

    network = TwoTowerNetwork(
        network_config
    )

    training_config = (
        TwoTowerTrainingConfig(
            epochs=args.epochs,
            batch_size=(
                args.batch_size
            ),
            learning_rate=(
                args.learning_rate
            ),
            weight_decay=(
                args.weight_decay
            ),
            gradient_clip_norm=(
                args.gradient_clip_norm
            ),
            seed=args.seed,
            device=args.device,
        )
    )

    print(
        "Training two-tower network..."
    )

    training_result = train_two_tower(
        network,
        examples,
        article_features,
        config=training_config,
    )

    args.checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                network.state_dict()
            ),
            "network_config": (
                asdict(
                    network_config
                )
            ),
            "training_config": (
                asdict(
                    training_config
                )
            ),
            "feature_encoder": {
                "max_features": (
                    args.max_features
                ),
                "svd_components": (
                    args.svd_components
                ),
                "seed": args.seed,
            },
            "protocol": {
                "cutoff_timestamp": (
                    actual_cutoff
                ),
                "max_history_length": (
                    args.max_history_length
                ),
            },
        },
        args.checkpoint,
    )

    checkpoint_sha256 = (
        _sha256(
            args.checkpoint
        )
    )

    payload = {
        "experiment": (
            "phase03_two_tower_training"
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
            "seed": args.seed,
            "max_training_examples": (
                args.max_training_examples
            ),
            "limited_training_run": (
                args.max_training_examples
                > 0
            ),
        },
        "features": {
            "fit_article_count": (
                encoder.fit_article_count
            ),
            "indexed_article_count": (
                feature_batch.article_count
            ),
            "nonzero_article_count": (
                feature_batch.nonzero_article_count
            ),
            "tfidf_vocabulary_size": (
                encoder.vocabulary_size
            ),
            "svd_components": (
                args.svd_components
            ),
            "svd_explained_variance_ratio_sum": (
                encoder
                .explained_variance_ratio_sum
            ),
        },
        "training_data": (
            build_result.to_dict()
        ),
        "network_config": (
            asdict(
                network_config
            )
        ),
        "training_config": (
            asdict(
                training_config
            )
        ),
        "training_result": (
            training_result.to_dict()
        ),
        "checkpoint": {
            "path": str(
                args.checkpoint
            ),
            "sha256": (
                checkpoint_sha256
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
    print("=" * 84)
    print(
        "PHASE 03C TWO-TOWER TRAINING"
    )
    print("=" * 84)

    print(
        "Device:",
        training_result.device,
    )

    print(
        "Usable examples:",
        build_result.usable_example_count,
    )

    print(
        "Examples used this run:",
        len(examples),
    )

    print()

    print(
        f"{'Epoch':>8}"
        f"{'Loss':>14}"
        f"{'Top1':>14}"
        f"{'Batches':>12}"
        f"{'Seen':>12}"
        f"{'Skipped':>12}"
    )

    for epoch in (
        training_result.epochs
    ):
        print(
            f"{epoch.epoch:>8}"
            f"{epoch.average_loss:>14.6f}"
            f"{epoch.in_batch_top1_accuracy:>14.4%}"
            f"{epoch.batches:>12}"
            f"{epoch.examples_seen:>12}"
            f"{epoch.skipped_no_negative_examples:>12}"
        )

    print()
    print(
        "Checkpoint SHA-256:",
        checkpoint_sha256,
    )

    print(
        "Report:",
        args.output,
    )


if __name__ == "__main__":
    main()
