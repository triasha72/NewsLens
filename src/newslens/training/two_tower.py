"""Deterministic training loop for the native PyTorch two-tower model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from newslens.data.two_tower_training import (
    TwoTowerTrainingExample,
)
from newslens.models.two_tower import (
    TwoTowerNetwork,
)


class TwoTowerTrainingError(ValueError):
    """Raised when two-tower optimization cannot be completed."""


@dataclass(frozen=True, slots=True)
class TwoTowerTrainingConfig:
    """Fixed optimization configuration."""

    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs <= 0
        ):
            raise TwoTowerTrainingError(
                "epochs must be a positive integer."
            )

        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 2
        ):
            raise TwoTowerTrainingError(
                "batch_size must be an integer of at least 2."
            )

        if self.learning_rate <= 0.0:
            raise TwoTowerTrainingError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0.0:
            raise TwoTowerTrainingError(
                "weight_decay cannot be negative."
            )

        if self.gradient_clip_norm <= 0.0:
            raise TwoTowerTrainingError(
                "gradient_clip_norm must be positive."
            )

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise TwoTowerTrainingError(
                "seed must be a non-negative integer."
            )

        if self.device not in {
            "cpu",
            "mps",
            "cuda",
            "auto",
        }:
            raise TwoTowerTrainingError(
                "device must be one of "
                "'cpu', 'mps', 'cuda', or 'auto'."
            )


@dataclass(frozen=True, slots=True)
class TwoTowerEpochMetrics:
    """Optimization statistics for one epoch."""

    epoch: int
    average_loss: float
    in_batch_top1_accuracy: float
    batches: int
    examples_seen: int
    skipped_no_negative_examples: int

    def to_dict(
        self,
    ) -> dict[str, int | float]:
        return {
            "epoch": self.epoch,
            "average_loss": self.average_loss,
            "in_batch_top1_accuracy": (
                self.in_batch_top1_accuracy
            ),
            "batches": self.batches,
            "examples_seen": self.examples_seen,
            "skipped_no_negative_examples": (
                self.skipped_no_negative_examples
            ),
        }


@dataclass(frozen=True, slots=True)
class TwoTowerTrainingResult:
    """Complete neural optimization result."""

    device: str
    training_examples: int
    feature_dim: int
    epochs: tuple[
        TwoTowerEpochMetrics,
        ...
    ]

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "device": self.device,
            "training_examples": (
                self.training_examples
            ),
            "feature_dim": self.feature_dim,
            "epochs": [
                epoch.to_dict()
                for epoch in self.epochs
            ],
        }


def _resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        if (
            torch.backends.mps.is_available()
        ):
            return torch.device(
                "mps"
            )

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    if (
        requested == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise TwoTowerTrainingError(
            "MPS was requested but is not available."
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise TwoTowerTrainingError(
            "CUDA was requested but is not available."
        )

    return torch.device(
        requested
    )


def _prepare_feature_mapping(
    article_features: Mapping[
        str,
        np.ndarray,
    ],
    *,
    expected_dim: int,
) -> dict[str, np.ndarray]:
    prepared: dict[
        str,
        np.ndarray,
    ] = {}

    for news_id, values in article_features.items():
        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if array.ndim != 1:
            raise TwoTowerTrainingError(
                "Every article feature must be one-dimensional."
            )

        if array.shape[0] != expected_dim:
            raise TwoTowerTrainingError(
                f"Article '{news_id}' has dimension "
                f"{array.shape[0]}; expected {expected_dim}."
            )

        prepared[str(news_id)] = array

    if not prepared:
        raise TwoTowerTrainingError(
            "At least one article feature vector is required."
        )

    return prepared


def _materialize_batch(
    examples: Sequence[
        TwoTowerTrainingExample
    ],
    article_features: Mapping[
        str,
        np.ndarray,
    ],
    *,
    feature_dim: int,
    device: torch.device,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    batch_size = len(
        examples
    )

    maximum_history = max(
        len(example.history_news_ids)
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

    unique_positive_ids: list[str] = []
    positive_to_target: dict[
        str,
        int,
    ] = {}

    targets = np.empty(
        batch_size,
        dtype=np.int64,
    )

    for row_index, example in enumerate(
        examples
    ):
        for history_index, news_id in enumerate(
            example.history_news_ids
        ):
            try:
                values = article_features[
                    news_id
                ]
            except KeyError as error:
                raise TwoTowerTrainingError(
                    "History article is missing "
                    f"features: {news_id}."
                ) from error

            history_values[
                row_index,
                history_index,
            ] = values

            history_mask[
                row_index,
                history_index,
            ] = True

        positive_news_id = (
            example.positive_news_id
        )

        if positive_news_id not in article_features:
            raise TwoTowerTrainingError(
                "Positive article is missing "
                f"features: {positive_news_id}."
            )

        target = positive_to_target.get(
            positive_news_id
        )

        if target is None:
            target = len(
                unique_positive_ids
            )

            positive_to_target[
                positive_news_id
            ] = target

            unique_positive_ids.append(
                positive_news_id
            )

        targets[row_index] = target

    positive_values = np.stack(
        [
            article_features[
                news_id
            ]
            for news_id in unique_positive_ids
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    return (
        torch.from_numpy(
            history_values
        ).to(
            device
        ),
        torch.from_numpy(
            history_mask
        ).to(
            device
        ),
        torch.from_numpy(
            positive_values
        ).to(
            device
        ),
        torch.from_numpy(
            targets
        ).to(
            device
        ),
    )


def train_two_tower(
    network: TwoTowerNetwork,
    examples: Sequence[
        TwoTowerTrainingExample
    ],
    article_features: Mapping[
        str,
        np.ndarray,
    ],
    *,
    config: TwoTowerTrainingConfig,
) -> TwoTowerTrainingResult:
    """Optimize the two-tower network with unique in-batch article negatives.

    Positive article IDs are deduplicated inside each batch. Multiple users
    clicking the same article therefore share the same target class instead
    of incorrectly treating duplicate copies of that article as negatives.
    """

    if not examples:
        raise TwoTowerTrainingError(
            "At least one training example is required."
        )

    device = _resolve_device(
        config.device
    )

    features = _prepare_feature_mapping(
        article_features,
        expected_dim=(
            network.config.input_dim
        ),
    )

    torch.manual_seed(
        config.seed
    )

    network.to(
        device
    )

    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epoch_results: list[
        TwoTowerEpochMetrics
    ] = []

    example_count = len(
        examples
    )

    for epoch_index in range(
        config.epochs
    ):
        network.train()

        rng = np.random.default_rng(
            config.seed
            + epoch_index
        )

        order = rng.permutation(
            example_count
        )

        total_loss = 0.0
        total_correct = 0
        examples_seen = 0
        skipped_no_negative_examples = 0
        batch_count = 0

        for start in range(
            0,
            example_count,
            config.batch_size,
        ):
            selected_indices = order[
                start:
                start
                + config.batch_size
            ]

            batch_examples = [
                examples[
                    int(index)
                ]
                for index
                in selected_indices
            ]

            (
                history_tensor,
                history_mask,
                positive_tensor,
                targets,
            ) = _materialize_batch(
                batch_examples,
                features,
                feature_dim=(
                    network.config.input_dim
                ),
                device=device,
            )

            if positive_tensor.shape[0] < 2:
                skipped_no_negative_examples += len(
                    batch_examples
                )
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

            article_embeddings = (
                network.encode_articles(
                    positive_tensor
                )
            )

            logits = (
                user_embeddings
                @ article_embeddings.T
            ) / network.config.temperature

            loss = F.cross_entropy(
                logits,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                network.parameters(),
                max_norm=(
                    config.gradient_clip_norm
                ),
            )

            optimizer.step()

            batch_size = len(
                batch_examples
            )

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * batch_size
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

            examples_seen += batch_size
            batch_count += 1

        if examples_seen == 0:
            raise TwoTowerTrainingError(
                "No batch contained at least two "
                "unique positive articles."
            )

        average_loss = (
            total_loss
            / examples_seen
        )

        top1_accuracy = (
            total_correct
            / examples_seen
        )

        epoch_results.append(
            TwoTowerEpochMetrics(
                epoch=(
                    epoch_index + 1
                ),
                average_loss=(
                    average_loss
                ),
                in_batch_top1_accuracy=(
                    top1_accuracy
                ),
                batches=batch_count,
                examples_seen=examples_seen,
                skipped_no_negative_examples=(
                    skipped_no_negative_examples
                ),
            )
        )

    network.eval()

    return TwoTowerTrainingResult(
        device=str(device),
        training_examples=(
            example_count
        ),
        feature_dim=(
            network.config.input_dim
        ),
        epochs=tuple(
            epoch_results
        ),
    )
