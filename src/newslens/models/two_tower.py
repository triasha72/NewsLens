"""Native PyTorch content-aware two-tower recommendation model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import Recommendation


class TwoTowerModelError(ValueError):
    """Raised when two-tower configuration or input is invalid."""


@dataclass(frozen=True, slots=True)
class TwoTowerConfig:
    """Configuration for the native PyTorch two-tower network."""

    input_dim: int
    hidden_dim: int = 128
    embedding_dim: int = 64
    dropout: float = 0.10
    temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise TwoTowerModelError(
                "input_dim must be positive."
            )

        if self.hidden_dim <= 0:
            raise TwoTowerModelError(
                "hidden_dim must be positive."
            )

        if self.embedding_dim <= 0:
            raise TwoTowerModelError(
                "embedding_dim must be positive."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise TwoTowerModelError(
                "dropout must be in [0, 1)."
            )

        if self.temperature <= 0.0:
            raise TwoTowerModelError(
                "temperature must be positive."
            )


class ArticleTower(nn.Module):
    """Map dense article content features into retrieval embeddings."""

    def __init__(
        self,
        config: TwoTowerConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.network = nn.Sequential(
            nn.Linear(
                config.input_dim,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_dim,
                config.embedding_dim,
            ),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        if features.shape[-1] != self.config.input_dim:
            raise TwoTowerModelError(
                "Article feature dimension does not "
                f"match input_dim={self.config.input_dim}."
            )

        embeddings = self.network(
            features
        )

        return F.normalize(
            embeddings,
            p=2,
            dim=-1,
        )


class HistoryUserTower(nn.Module):
    """Build a user embedding from encoded history articles."""

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise TwoTowerModelError(
                "embedding_dim must be positive."
            )

        self.embedding_dim = (
            embedding_dim
        )

        self.projection = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.GELU(),
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        history_embeddings: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        if history_embeddings.ndim != 3:
            raise TwoTowerModelError(
                "history_embeddings must have shape "
                "[batch, history, embedding]."
            )

        if (
            history_embeddings.shape[-1]
            != self.embedding_dim
        ):
            raise TwoTowerModelError(
                "History embedding dimension mismatch."
            )

        if history_mask.ndim != 2:
            raise TwoTowerModelError(
                "history_mask must have shape "
                "[batch, history]."
            )

        if (
            history_mask.shape
            != history_embeddings.shape[:2]
        ):
            raise TwoTowerModelError(
                "history_mask shape does not match "
                "history embeddings."
            )

        mask = history_mask.to(
            dtype=history_embeddings.dtype
        ).unsqueeze(-1)

        counts = mask.sum(
            dim=1
        )

        if bool(
            (counts.squeeze(-1) == 0)
            .any()
            .item()
        ):
            raise TwoTowerModelError(
                "Every user example must contain at "
                "least one usable history article."
            )

        pooled = (
            history_embeddings
            * mask
        ).sum(
            dim=1
        ) / counts

        projected = (
            pooled
            + self.projection(
                pooled
            )
        )

        return F.normalize(
            projected,
            p=2,
            dim=-1,
        )


class TwoTowerNetwork(nn.Module):
    """Trainable article and history-user towers."""

    def __init__(
        self,
        config: TwoTowerConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.article_tower = (
            ArticleTower(
                config
            )
        )

        self.user_tower = (
            HistoryUserTower(
                config.embedding_dim
            )
        )

    def encode_articles(
        self,
        article_features: Tensor,
    ) -> Tensor:
        return self.article_tower(
            article_features
        )

    def encode_users(
        self,
        history_features: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        if history_features.ndim != 3:
            raise TwoTowerModelError(
                "history_features must have shape "
                "[batch, history, feature]."
            )

        batch_size, history_size, feature_dim = (
            history_features.shape
        )

        if feature_dim != self.config.input_dim:
            raise TwoTowerModelError(
                "History article feature dimension "
                "does not match the article tower."
            )

        flattened = history_features.reshape(
            batch_size * history_size,
            feature_dim,
        )

        encoded = self.encode_articles(
            flattened
        ).reshape(
            batch_size,
            history_size,
            self.config.embedding_dim,
        )

        return self.user_tower(
            encoded,
            history_mask,
        )

    def in_batch_logits(
        self,
        history_features: Tensor,
        history_mask: Tensor,
        positive_article_features: Tensor,
    ) -> Tensor:
        """Return user-to-positive logits with batch items as negatives."""

        user_embeddings = (
            self.encode_users(
                history_features,
                history_mask,
            )
        )

        article_embeddings = (
            self.encode_articles(
                positive_article_features
            )
        )

        if (
            user_embeddings.shape[0]
            != article_embeddings.shape[0]
        ):
            raise TwoTowerModelError(
                "User and positive article batch "
                "sizes must match."
            )

        return (
            user_embeddings
            @ article_embeddings.T
        ) / self.config.temperature

    def score_candidates(
        self,
        user_embedding: Tensor,
        candidate_embeddings: Tensor,
    ) -> Tensor:
        if user_embedding.ndim != 2:
            raise TwoTowerModelError(
                "user_embedding must have shape "
                "[batch, embedding]."
            )

        if candidate_embeddings.ndim != 2:
            raise TwoTowerModelError(
                "candidate_embeddings must have shape "
                "[candidate, embedding]."
            )

        if user_embedding.shape[0] != 1:
            raise TwoTowerModelError(
                "Inference scoring currently expects "
                "one user at a time."
            )

        return (
            user_embedding
            @ candidate_embeddings.T
        ).squeeze(0) / self.config.temperature


class TwoTowerRecommender:
    """Rank supplied candidates using history-derived two-tower embeddings."""

    def __init__(
        self,
        network: TwoTowerNetwork,
        article_features: Mapping[
            str,
            Tensor | Iterable[float],
        ],
        *,
        device: str = "cpu",
    ) -> None:
        self.network = network
        self.device = torch.device(
            device
        )

        prepared: dict[
            str,
            Tensor,
        ] = {}

        for news_id, values in article_features.items():
            tensor = torch.as_tensor(
                values,
                dtype=torch.float32,
            )

            if tensor.ndim != 1:
                raise TwoTowerModelError(
                    "Each article feature vector must "
                    "be one-dimensional."
                )

            if (
                tensor.shape[0]
                != network.config.input_dim
            ):
                raise TwoTowerModelError(
                    f"Article '{news_id}' has feature "
                    "dimension "
                    f"{tensor.shape[0]}; expected "
                    f"{network.config.input_dim}."
                )

            prepared[str(news_id)] = tensor

        if not prepared:
            raise TwoTowerModelError(
                "At least one article feature vector "
                "is required."
            )

        self.article_features = prepared

        self.network.to(
            self.device
        )

        self.network.eval()

    def recommend(
        self,
        history_news_ids: Iterable[str],
        *,
        candidate_news_ids: Iterable[str],
        top_k: int = 10,
    ) -> list[Recommendation]:
        """Rank candidate articles from the user's readable history."""

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise TwoTowerModelError(
                "top_k must be a positive integer."
            )

        if isinstance(
            history_news_ids,
            str,
        ):
            history_ids = (
                history_news_ids.split()
            )
        else:
            history_ids = list(
                history_news_ids
            )

        if isinstance(
            candidate_news_ids,
            str,
        ):
            candidate_ids = (
                candidate_news_ids.split()
            )
        else:
            candidate_ids = list(
                candidate_news_ids
            )

        known_history = [
            news_id
            for news_id in history_ids
            if news_id
            in self.article_features
        ]

        if not known_history:
            return []

        known_candidates = sorted(
            {
                news_id
                for news_id in candidate_ids
                if news_id
                in self.article_features
            }
        )

        if not known_candidates:
            return []

        history_tensor = torch.stack(
            [
                self.article_features[
                    news_id
                ]
                for news_id
                in known_history
            ]
        ).unsqueeze(0).to(
            self.device
        )

        history_mask = torch.ones(
            (
                1,
                len(known_history),
            ),
            dtype=torch.bool,
            device=self.device,
        )

        candidate_tensor = torch.stack(
            [
                self.article_features[
                    news_id
                ]
                for news_id
                in known_candidates
            ]
        ).to(
            self.device
        )

        with torch.no_grad():
            user_embedding = (
                self.network.encode_users(
                    history_tensor,
                    history_mask,
                )
            )

            candidate_embeddings = (
                self.network.encode_articles(
                    candidate_tensor
                )
            )

            scores = (
                self.network.score_candidates(
                    user_embedding,
                    candidate_embeddings,
                )
                .detach()
                .cpu()
                .tolist()
            )

        ranked = sorted(
            zip(
                known_candidates,
                scores,
                strict=True,
            ),
            key=lambda row: (
                -float(row[1]),
                row[0],
            ),
        )[:top_k]

        return [
            Recommendation(
                news_id=news_id,
                score=float(score),
                source="two_tower",
            )
            for news_id, score
            in ranked
        ]
