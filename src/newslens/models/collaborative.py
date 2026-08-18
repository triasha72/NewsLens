"""Pairwise matrix-factorization recommender for implicit feedback."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Collaborative filtering requires PyTorch. Install the NewsLens recsys extras."
    ) from exc

from .base import Recommendation


@dataclass(frozen=True, slots=True)
class InteractionTriple:
    user_id: str
    positive_news_id: str
    negative_news_id: str


class BPRMatrixFactorization(nn.Module):
    """Implicit-feedback matrix factorization trained with BPR loss."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_embedding(users) * self.item_embedding(items)).sum(dim=-1)

    def bpr_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        pos_score = self.score(users, positives)
        neg_score = self.score(users, negatives)
        return -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()


class CollaborativeRecommender:
    """Train and serve BPR matrix factorization with deterministic indexing."""

    def __init__(self, embedding_dim: int = 64, seed: int = 42) -> None:
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.user_to_index: dict[str, int] = {}
        self.item_to_index: dict[str, int] = {}
        self.model: BPRMatrixFactorization | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def fit(
        self,
        triples: Sequence[InteractionTriple],
        *,
        epochs: int = 10,
        batch_size: int = 2048,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-6,
    ) -> CollaborativeRecommender:
        if not triples:
            raise ValueError("At least one interaction triple is required.")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        users = sorted({row.user_id for row in triples})
        items = sorted(
            {row.positive_news_id for row in triples}
            | {row.negative_news_id for row in triples}
        )
        self.user_to_index = {value: i for i, value in enumerate(users)}
        self.item_to_index = {value: i for i, value in enumerate(items)}

        encoded = [
            (
                self.user_to_index[row.user_id],
                self.item_to_index[row.positive_news_id],
                self.item_to_index[row.negative_news_id],
            )
            for row in triples
        ]

        model = BPRMatrixFactorization(
            len(self.user_to_index),
            len(self.item_to_index),
            self.embedding_dim,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        order = np.arange(len(encoded))
        for _ in range(epochs):
            np.random.shuffle(order)
            for start in range(0, len(order), batch_size):
                batch_idx = order[start : start + batch_size]
                batch = [encoded[int(i)] for i in batch_idx]
                u = torch.tensor([x[0] for x in batch], dtype=torch.long)
                p = torch.tensor([x[1] for x in batch], dtype=torch.long)
                n = torch.tensor([x[2] for x in batch], dtype=torch.long)

                optimizer.zero_grad()
                loss = model.bpr_loss(u, p, n)
                loss.backward()
                optimizer.step()

        self.model = model.eval()
        return self

    def recommend_for_user(
        self,
        user_id: str,
        *,
        candidate_news_ids: Iterable[str],
        top_k: int = 10,
    ) -> list[Recommendation]:
        if self.model is None:
            raise RuntimeError("Fit the collaborative model first.")
        if user_id not in self.user_to_index:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        candidate_ids = sorted(
            {
                item
                for item in candidate_news_ids
                if item in self.item_to_index
            }
        )
        if not candidate_ids:
            return []

        user_index = self.user_to_index[user_id]
        item_indices = [self.item_to_index[item] for item in candidate_ids]
        users = torch.full((len(item_indices),), user_index, dtype=torch.long)
        items = torch.tensor(item_indices, dtype=torch.long)

        with torch.inference_mode():
            scores = self.model.score(users, items).cpu().numpy()

        ranked = sorted(
            zip(candidate_ids, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0]),
        )[:top_k]

        return [
            Recommendation(news_id=item, score=float(score), source="collaborative")
            for item, score in ranked
        ]
