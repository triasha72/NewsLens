"""Phase 03E two-tower training with same-impression hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from newslens.data import (
    load_behaviors,
    load_news,
    parse_impressions,
)
from newslens.evaluation.content import (
    _parse_history,
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


@dataclass(frozen=True, slots=True)
class HardNegativeExample:
    """One history-click pair plus same-impression unclicked candidates."""

    impression_id: str
    history_news_ids: tuple[str, ...]
    positive_news_id: str
    negative_news_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HardNegativeBuildResult:
    """Hard-negative construction result and accounting."""

    examples: tuple[HardNegativeExample, ...]
    input_impressions: int
    eligible_impressions: int
    positive_click_occurrences: int
    usable_example_count: int
    empty_history_impressions: int
    history_without_features_impressions: int
    positive_without_features: int
    truncated_history_impressions: int
    examples_with_hard_negatives: int
    examples_without_hard_negatives: int
    attached_hard_negative_occurrences: int
    unique_positive_articles: int
    unique_hard_negative_articles: int

    def to_dict(self) -> dict[str, int]:
        """Return compact accounting without serializing training examples."""

        return {
            "input_impressions": self.input_impressions,
            "eligible_impressions": self.eligible_impressions,
            "positive_click_occurrences": self.positive_click_occurrences,
            "usable_example_count": self.usable_example_count,
            "skipped_positive_occurrences": (
                self.positive_click_occurrences
                - self.usable_example_count
            ),
            "empty_history_impressions": self.empty_history_impressions,
            "history_without_features_impressions": (
                self.history_without_features_impressions
            ),
            "positive_without_features": self.positive_without_features,
            "truncated_history_impressions": (
                self.truncated_history_impressions
            ),
            "examples_with_hard_negatives": (
                self.examples_with_hard_negatives
            ),
            "examples_without_hard_negatives": (
                self.examples_without_hard_negatives
            ),
            "attached_hard_negative_occurrences": (
                self.attached_hard_negative_occurrences
            ),
            "unique_positive_articles": self.unique_positive_articles,
            "unique_hard_negative_articles": (
                self.unique_hard_negative_articles
            ),
        }


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Optimization accounting for one hard-negative epoch."""

    epoch: int
    average_loss: float
    top1_accuracy: float
    batches: int
    examples_seen: int
    skipped_batches: int
    hard_negative_occurrences_seen: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _build_examples(
    behaviors,
    *,
    available_news_ids: set[str],
    max_history_length: int,
) -> HardNegativeBuildResult:
    examples: list[HardNegativeExample] = []

    eligible_impressions = 0
    positive_click_occurrences = 0
    empty_history_impressions = 0
    history_without_features_impressions = 0
    positive_without_features = 0
    truncated_history_impressions = 0

    examples_with_hard_negatives = 0
    examples_without_hard_negatives = 0
    attached_hard_negative_occurrences = 0

    unique_positive_articles: set[str] = set()
    unique_hard_negative_articles: set[str] = set()

    for row in behaviors.itertuples(index=False):
        impression_id = str(row.impression_id)

        parsed = parse_impressions(
            str(row.impressions)
        )

        candidate_ids = [
            news_id
            for news_id, _ in parsed
        ]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise RuntimeError(
                "Duplicate candidates in training "
                f"impression {impression_id}."
            )

        positive_ids = [
            news_id
            for news_id, label in parsed
            if label == 1
        ]

        negative_ids = tuple(
            news_id
            for news_id, label in parsed
            if (
                label == 0
                and news_id in available_news_ids
            )
        )

        positive_click_occurrences += len(
            positive_ids
        )

        if not positive_ids:
            continue

        history_ids = _parse_history(
            row.history
        )

        if not history_ids:
            empty_history_impressions += 1
            continue

        usable_history = [
            news_id
            for news_id in history_ids
            if news_id in available_news_ids
        ]

        if not usable_history:
            history_without_features_impressions += 1
            continue

        if len(usable_history) > max_history_length:
            truncated_history_impressions += 1

        history = tuple(
            usable_history[
                -max_history_length:
            ]
        )

        impression_contributed = False

        for positive_news_id in positive_ids:
            if positive_news_id not in available_news_ids:
                positive_without_features += 1
                continue

            examples.append(
                HardNegativeExample(
                    impression_id=impression_id,
                    history_news_ids=history,
                    positive_news_id=positive_news_id,
                    negative_news_ids=negative_ids,
                )
            )

            unique_positive_articles.add(
                positive_news_id
            )

            unique_hard_negative_articles.update(
                negative_ids
            )

            attached_hard_negative_occurrences += len(
                negative_ids
            )

            if negative_ids:
                examples_with_hard_negatives += 1
            else:
                examples_without_hard_negatives += 1

            impression_contributed = True

        if impression_contributed:
            eligible_impressions += 1

    if not examples:
        raise RuntimeError(
            "No hard-negative training examples were produced."
        )

    return HardNegativeBuildResult(
        examples=tuple(examples),
        input_impressions=len(behaviors),
        eligible_impressions=eligible_impressions,
        positive_click_occurrences=positive_click_occurrences,
        usable_example_count=len(examples),
        empty_history_impressions=empty_history_impressions,
        history_without_features_impressions=(
            history_without_features_impressions
        ),
        positive_without_features=positive_without_features,
        truncated_history_impressions=(
            truncated_history_impressions
        ),
        examples_with_hard_negatives=(
            examples_with_hard_negatives
        ),
        examples_without_hard_negatives=(
            examples_without_hard_negatives
        ),
        attached_hard_negative_occurrences=(
            attached_hard_negative_occurrences
        ),
        unique_positive_articles=len(
            unique_positive_articles
        ),
        unique_hard_negative_articles=len(
            unique_hard_negative_articles
        ),
    )


def _materialize_batch(
    examples: list[HardNegativeExample],
    article_features: dict[str, np.ndarray],
    *,
    feature_dim: int,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    batch_size = len(examples)

    maximum_history = max(
        len(example.history_news_ids)
        for example in examples
    )

    maximum_hard_negatives = max(
        len(example.negative_news_ids)
        for example in examples
    )

    history_values = np.zeros(
        (
            batch_size,
            maximum_history,
            feature_dim,
        ),
        dtype=np.float32,
    )

    history_mask = np.zeros(
        (
            batch_size,
            maximum_history,
        ),
        dtype=np.bool_,
    )

    for row_index, example in enumerate(examples):
        for history_index, news_id in enumerate(
            example.history_news_ids
        ):
            history_values[
                row_index,
                history_index,
            ] = article_features[news_id]

            history_mask[
                row_index,
                history_index,
            ] = True

    unique_positive_ids: list[str] = []
    positive_to_target: dict[str, int] = {}

    targets = np.empty(
        batch_size,
        dtype=np.int64,
    )

    for row_index, example in enumerate(examples):
        target = positive_to_target.get(
            example.positive_news_id
        )

        if target is None:
            target = len(unique_positive_ids)

            positive_to_target[
                example.positive_news_id
            ] = target

            unique_positive_ids.append(
                example.positive_news_id
            )

        targets[row_index] = target

    positive_values = np.stack(
        [
            article_features[news_id]
            for news_id in unique_positive_ids
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    # Distinct articles clicked in the same impression are known positives
    # for that user context and must not be used as in-batch negatives.
    in_batch_valid_mask = np.ones(
        (
            batch_size,
            len(unique_positive_ids),
        ),
        dtype=np.bool_,
    )

    impression_positive_ids: dict[
        str,
        set[str],
    ] = {}

    for example in examples:
        impression_positive_ids.setdefault(
            example.impression_id,
            set(),
        ).add(
            example.positive_news_id
        )

    for row_index, example in enumerate(
        examples
    ):
        for positive_news_id in (
            impression_positive_ids[
                example.impression_id
            ]
        ):
            if (
                positive_news_id
                == example.positive_news_id
            ):
                continue

            column_index = (
                positive_to_target.get(
                    positive_news_id
                )
            )

            if column_index is not None:
                in_batch_valid_mask[
                    row_index,
                    column_index,
                ] = False

        # The row's own clicked item must always remain a valid target.
        in_batch_valid_mask[
            row_index,
            targets[row_index],
        ] = True

    hard_negative_values = np.zeros(
        (
            batch_size,
            maximum_hard_negatives,
            feature_dim,
        ),
        dtype=np.float32,
    )

    hard_negative_mask = np.zeros(
        (
            batch_size,
            maximum_hard_negatives,
        ),
        dtype=np.bool_,
    )

    for row_index, example in enumerate(examples):
        for negative_index, news_id in enumerate(
            example.negative_news_ids
        ):
            hard_negative_values[
                row_index,
                negative_index,
            ] = article_features[news_id]

            hard_negative_mask[
                row_index,
                negative_index,
            ] = True

    return (
        torch.from_numpy(history_values),
        torch.from_numpy(history_mask),
        torch.from_numpy(positive_values),
        torch.from_numpy(targets),
        torch.from_numpy(in_batch_valid_mask),
        torch.from_numpy(hard_negative_values),
        torch.from_numpy(hard_negative_mask),
    )


def _train(
    network: TwoTowerNetwork,
    examples: tuple[HardNegativeExample, ...],
    article_features: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    seed: int,
) -> tuple[EpochResult, ...]:
    torch.manual_seed(seed)

    network.to("cpu")

    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    example_count = len(examples)

    epoch_results: list[EpochResult] = []

    for epoch_index in range(epochs):
        network.train()

        rng = np.random.default_rng(
            seed + epoch_index
        )

        order = rng.permutation(
            example_count
        )

        total_loss = 0.0
        total_correct = 0
        examples_seen = 0
        batch_count = 0
        skipped_batches = 0
        hard_negative_occurrences_seen = 0

        for start in range(
            0,
            example_count,
            batch_size,
        ):
            indices = order[
                start:
                start + batch_size
            ]

            batch_examples = [
                examples[int(index)]
                for index in indices
            ]

            (
                history_tensor,
                history_mask,
                positive_tensor,
                targets,
                in_batch_valid_mask,
                hard_negative_tensor,
                hard_negative_mask,
            ) = _materialize_batch(
                batch_examples,
                article_features,
                feature_dim=network.config.input_dim,
            )

            unique_positive_count = (
                positive_tensor.shape[0]
            )

            hard_count = int(
                hard_negative_mask.sum().item()
            )

            if (
                unique_positive_count < 2
                and hard_count == 0
            ):
                skipped_batches += 1
                continue

            optimizer.zero_grad(
                set_to_none=True
            )

            user_embeddings = (
                network.encode_users(
                    history_tensor,
                    history_mask,
                )
            )

            positive_embeddings = (
                network.encode_articles(
                    positive_tensor
                )
            )

            in_batch_logits = (
                user_embeddings
                @ positive_embeddings.T
            ) / network.config.temperature

            in_batch_logits = (
                in_batch_logits.masked_fill(
                    ~in_batch_valid_mask,
                    torch.finfo(
                        in_batch_logits.dtype
                    ).min,
                )
            )

            if hard_negative_tensor.shape[1] > 0:
                batch_dimension = (
                    hard_negative_tensor.shape[0]
                )

                hard_dimension = (
                    hard_negative_tensor.shape[1]
                )

                flattened_hard_negatives = (
                    hard_negative_tensor.reshape(
                        batch_dimension
                        * hard_dimension,
                        network.config.input_dim,
                    )
                )

                hard_embeddings = (
                    network.encode_articles(
                        flattened_hard_negatives
                    )
                    .reshape(
                        batch_dimension,
                        hard_dimension,
                        network.config.embedding_dim,
                    )
                )

                hard_logits = torch.einsum(
                    "be,bhe->bh",
                    user_embeddings,
                    hard_embeddings,
                ) / network.config.temperature

                hard_logits = (
                    hard_logits.masked_fill(
                        ~hard_negative_mask,
                        torch.finfo(
                            hard_logits.dtype
                        ).min,
                    )
                )

                logits = torch.cat(
                    [
                        in_batch_logits,
                        hard_logits,
                    ],
                    dim=1,
                )
            else:
                logits = in_batch_logits

            loss = F.cross_entropy(
                logits,
                targets,
            )

            if not bool(
                torch.isfinite(loss).item()
            ):
                raise RuntimeError(
                    "Non-finite hard-negative training loss."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                network.parameters(),
                max_norm=gradient_clip_norm,
            )

            optimizer.step()

            current_batch_size = len(
                batch_examples
            )

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * current_batch_size
            )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += int(
                (
                    predictions
                    == targets
                )
                .sum()
                .detach()
                .cpu()
            )

            examples_seen += (
                current_batch_size
            )

            hard_negative_occurrences_seen += (
                hard_count
            )

            batch_count += 1

        if examples_seen == 0:
            raise RuntimeError(
                "No hard-negative batches were trainable."
            )

        epoch_results.append(
            EpochResult(
                epoch=epoch_index + 1,
                average_loss=(
                    total_loss
                    / examples_seen
                ),
                top1_accuracy=(
                    total_correct
                    / examples_seen
                ),
                batches=batch_count,
                examples_seen=examples_seen,
                skipped_batches=skipped_batches,
                hard_negative_occurrences_seen=(
                    hard_negative_occurrences_seen
                ),
            )
        )

    network.eval()

    return tuple(epoch_results)


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
        default="2019-11-13T20:36:26",
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
        "--max-training-examples",
        type=int,
        default=0,
    )

    args = parser.parse_args()

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

    print("Creating chronological split...")

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
        "Fitting frozen train-only "
        "TF-IDF + SVD features..."
    )

    encoder = ArticleTextFeatureEncoder(
        max_features=args.max_features,
        svd_components=args.svd_components,
        seed=args.seed,
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
        "Building same-impression "
        "hard-negative examples..."
    )

    build_result = _build_examples(
        split.train,
        available_news_ids=set(
            article_features
        ),
        max_history_length=(
            args.max_history_length
        ),
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

    print(
        "Training examples:",
        len(examples),
        "/",
        build_result.usable_example_count,
    )

    print(
        "Attached hard-negative occurrences:",
        sum(
            len(example.negative_news_ids)
            for example in examples
        ),
    )

    network_config = TwoTowerConfig(
        input_dim=args.svd_components,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        temperature=args.temperature,
    )

    torch.manual_seed(
        args.seed
    )

    network = TwoTowerNetwork(
        network_config
    )

    print(
        "Training hard-negative two-tower..."
    )

    epoch_results = _train(
        network,
        examples,
        article_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=(
            args.gradient_clip_norm
        ),
        seed=args.seed,
    )

    checkpoint_payload = {
        "model_state_dict": (
            network.state_dict()
        ),
        "network_config": (
            asdict(
                network_config
            )
        ),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "gradient_clip_norm": (
                args.gradient_clip_norm
            ),
            "seed": args.seed,
            "device": "cpu",
            "objective": (
                "in_batch_plus_all_same_impression_unclicked"
            ),
        },
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
            "same_impression_hard_negatives": True,
            "hard_negative_sampling": "all",
        },
    }

    args.checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint_payload,
        args.checkpoint,
    )

    checkpoint_sha256 = (
        _sha256(
            args.checkpoint
        )
    )

    payload = {
        "experiment": (
            "phase03_two_tower_same_impression_hard_negatives"
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
            "limited_training_run": (
                args.max_training_examples
                > 0
            ),
            "max_training_examples": (
                args.max_training_examples
            ),
            "seed": args.seed,
            "objective": (
                "in_batch_plus_all_same_impression_unclicked"
            ),
            "hard_negative_sampling": "all",
        },
        "features": {
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
                args.svd_components
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
            checkpoint_payload[
                "training_config"
            ]
        ),
        "training_result": {
            "device": "cpu",
            "training_examples": (
                len(examples)
            ),
            "feature_dim": (
                args.svd_components
            ),
            "epochs": [
                result.to_dict()
                for result in epoch_results
            ],
        },
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
    print("=" * 88)
    print(
        "PHASE 03E HARD-NEGATIVE TRAINING"
    )
    print("=" * 88)

    print(
        f"{'Epoch':>8}"
        f"{'Loss':>14}"
        f"{'Top1':>14}"
        f"{'Batches':>12}"
        f"{'Seen':>12}"
        f"{'Hard Negs':>14}"
    )

    for result in epoch_results:
        print(
            f"{result.epoch:>8}"
            f"{result.average_loss:>14.6f}"
            f"{result.top1_accuracy:>14.4%}"
            f"{result.batches:>12}"
            f"{result.examples_seen:>12}"
            f"{result.hard_negative_occurrences_seen:>14}"
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
