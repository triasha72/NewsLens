"""Leakage-aware second-stage ranking features and model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from newslens.models.popularity import PopularityRecommender
from newslens.models.two_tower import TwoTowerNetwork
from newslens.retrieval.catalog import RetrievalCatalog


class SecondStageRankingError(ValueError):
    """Raised when ranker data or configuration is invalid."""


SECOND_STAGE_FEATURE_NAMES: tuple[str, ...] = (
    "two_tower_score",
    "mean_history_similarity",
    "max_history_similarity",
    "log1p_popularity_clicks",
    "log1p_popularity_exposures",
    "popularity_ctr",
    "category_match_any",
    "subcategory_match_any",
    "category_history_fraction",
    "subcategory_history_fraction",
    "usable_history_length",
)


@dataclass(frozen=True, slots=True)
class SecondStageRankerConfig:
    """Frozen Phase-05 gradient-boosting configuration."""

    learning_rate: float = 0.05
    max_iter: int = 200
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 50
    l2_regularization: float = 0.001
    seed: int = 42

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SecondStageContext:
    """User/history information required to score candidates."""

    user_vector: np.ndarray
    history_vectors: np.ndarray
    mean_history_vector: np.ndarray
    usable_history_news_ids: tuple[str, ...]
    source_usable_history_count: int
    category_counts: dict[str, int]
    subcategory_counts: dict[str, int]


class SecondStageFeatureBuilder:
    """Construct fixed candidate-ranking features."""

    def __init__(
        self,
        *,
        catalog: RetrievalCatalog,
        network: TwoTowerNetwork,
        news: pd.DataFrame,
        popularity_model: PopularityRecommender,
        max_history_length: int = 20,
    ) -> None:
        if (
            isinstance(max_history_length, bool)
            or not isinstance(max_history_length, int)
            or max_history_length <= 0
        ):
            raise SecondStageRankingError(
                "max_history_length must be a positive integer."
            )

        required = {
            "news_id",
            "category",
            "subcategory",
        }

        missing = required.difference(
            news.columns
        )

        if missing:
            raise SecondStageRankingError(
                "Missing article metadata columns: "
                + ", ".join(sorted(missing))
                + "."
            )

        if not popularity_model.is_fitted:
            raise SecondStageRankingError(
                "popularity_model must be fitted."
            )

        self.catalog = catalog
        self.network = network.to("cpu")
        self.network.eval()
        self.popularity_model = popularity_model
        self.max_history_length = max_history_length
        self._id_to_position = catalog.id_to_position

        prepared = news[
            [
                "news_id",
                "category",
                "subcategory",
            ]
        ].copy()

        for column in prepared.columns:
            prepared[column] = (
                prepared[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        self._category_by_id = dict(
            zip(
                prepared["news_id"],
                prepared["category"],
                strict=True,
            )
        )

        self._subcategory_by_id = dict(
            zip(
                prepared["news_id"],
                prepared["subcategory"],
                strict=True,
            )
        )

        missing_metadata = (
            set(catalog.news_ids)
            - set(self._category_by_id)
        )

        if missing_metadata:
            preview = ", ".join(
                sorted(missing_metadata)[:3]
            )

            raise SecondStageRankingError(
                "Catalog articles are missing metadata: "
                f"{preview}."
            )

    def build_context(
        self,
        history_news_ids: Iterable[str],
    ) -> SecondStageContext | None:
        """Build one user context from supported history articles."""

        if isinstance(history_news_ids, str):
            raw_history = tuple(
                history_news_ids.split()
            )
        else:
            raw_history = tuple(
                str(news_id)
                for news_id in history_news_ids
            )

        usable_full = tuple(
            news_id
            for news_id in raw_history
            if news_id in self._id_to_position
        )

        if not usable_full:
            return None

        usable = usable_full[
            -self.max_history_length:
        ]

        positions = [
            self._id_to_position[news_id]
            for news_id in usable
        ]

        history_vectors = np.ascontiguousarray(
            self.catalog.vectors[
                positions
            ],
            dtype=np.float32,
        )

        history_tensor = (
            torch.from_numpy(
                history_vectors
            )
            .unsqueeze(0)
        )

        history_mask = torch.ones(
            (
                1,
                len(usable),
            ),
            dtype=torch.bool,
        )

        with torch.no_grad():
            user_vector = (
                self.network.user_tower(
                    history_tensor,
                    history_mask,
                )
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

        mean_vector = history_vectors.mean(
            axis=0
        )

        mean_norm = float(
            np.linalg.norm(
                mean_vector
            )
        )

        if mean_norm <= 0.0:
            raise SecondStageRankingError(
                "History mean embedding is zero."
            )

        mean_vector = (
            mean_vector
            / mean_norm
        ).astype(
            np.float32,
            copy=False,
        )

        categories = [
            self._category_by_id[
                news_id
            ]
            for news_id in usable
            if self._category_by_id[
                news_id
            ]
        ]

        subcategories = [
            self._subcategory_by_id[
                news_id
            ]
            for news_id in usable
            if self._subcategory_by_id[
                news_id
            ]
        ]

        return SecondStageContext(
            user_vector=user_vector,
            history_vectors=history_vectors,
            mean_history_vector=(
                mean_vector
            ),
            usable_history_news_ids=(
                usable
            ),
            source_usable_history_count=(
                len(usable_full)
            ),
            category_counts=dict(
                Counter(categories)
            ),
            subcategory_counts=dict(
                Counter(subcategories)
            ),
        )

    def features_for_candidates(
        self,
        context: SecondStageContext,
        candidate_news_ids: Iterable[str],
    ) -> np.ndarray:
        """Return fixed feature rows aligned with candidate IDs."""

        candidate_ids = tuple(
            str(news_id)
            for news_id in candidate_news_ids
        )

        if not candidate_ids:
            raise SecondStageRankingError(
                "At least one candidate is required."
            )

        if (
            len(candidate_ids)
            != len(set(candidate_ids))
        ):
            raise SecondStageRankingError(
                "Candidate IDs must be unique."
            )

        unknown = [
            news_id
            for news_id in candidate_ids
            if news_id
            not in self._id_to_position
        ]

        if unknown:
            raise SecondStageRankingError(
                "Candidate embeddings are missing: "
                + ", ".join(
                    sorted(unknown)[:3]
                )
                + "."
            )

        positions = [
            self._id_to_position[
                news_id
            ]
            for news_id in candidate_ids
        ]

        candidate_vectors = (
            self.catalog.vectors[
                positions
            ]
        )

        two_tower_score = (
            candidate_vectors
            @ context.user_vector
        )

        mean_similarity = (
            candidate_vectors
            @ context.mean_history_vector
        )

        history_similarity = (
            candidate_vectors
            @ context.history_vectors.T
        )

        max_similarity = (
            history_similarity.max(
                axis=1
            )
        )

        popularity_clicks: list[float] = []
        popularity_exposures: list[float] = []
        popularity_ctr: list[float] = []

        category_match: list[float] = []
        subcategory_match: list[float] = []
        category_fraction: list[float] = []
        subcategory_fraction: list[float] = []

        history_length = len(
            context.usable_history_news_ids
        )

        for news_id in candidate_ids:
            stats = (
                self.popularity_model.statistics(
                    news_id
                )
            )

            popularity_clicks.append(
                float(
                    np.log1p(
                        stats.clicks
                    )
                )
            )

            popularity_exposures.append(
                float(
                    np.log1p(
                        stats.exposures
                    )
                )
            )

            popularity_ctr.append(
                float(
                    stats.click_through_rate
                )
            )

            category = (
                self._category_by_id[
                    news_id
                ]
            )

            subcategory = (
                self._subcategory_by_id[
                    news_id
                ]
            )

            category_count = (
                context.category_counts.get(
                    category,
                    0,
                )
                if category
                else 0
            )

            subcategory_count = (
                context.subcategory_counts.get(
                    subcategory,
                    0,
                )
                if subcategory
                else 0
            )

            category_match.append(
                float(
                    category_count > 0
                )
            )

            subcategory_match.append(
                float(
                    subcategory_count > 0
                )
            )

            category_fraction.append(
                category_count
                / history_length
            )

            subcategory_fraction.append(
                subcategory_count
                / history_length
            )

        matrix = np.column_stack(
            [
                two_tower_score,
                mean_similarity,
                max_similarity,
                popularity_clicks,
                popularity_exposures,
                popularity_ctr,
                category_match,
                subcategory_match,
                category_fraction,
                subcategory_fraction,
                np.full(
                    len(candidate_ids),
                    history_length,
                    dtype=np.float32,
                ),
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        if (
            matrix.shape[1]
            != len(
                SECOND_STAGE_FEATURE_NAMES
            )
        ):
            raise SecondStageRankingError(
                "Unexpected second-stage feature dimension."
            )

        if not np.isfinite(
            matrix
        ).all():
            raise SecondStageRankingError(
                "Second-stage features contain NaN or infinity."
            )

        return np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )


class SecondStageRanker:
    """Fixed HistGradientBoosting click-ranking model."""

    def __init__(
        self,
        config: SecondStageRankerConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            or SecondStageRankerConfig()
        )

        self._model: (
            HistGradientBoostingClassifier
            | None
        ) = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _require_fitted(
        self,
    ) -> HistGradientBoostingClassifier:
        if self._model is None:
            raise SecondStageRankingError(
                "Fit the ranker before scoring."
            )

        return self._model

    @staticmethod
    def _validate_features(
        features: np.ndarray,
    ) -> np.ndarray:
        matrix = np.asarray(
            features,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise SecondStageRankingError(
                "features must be two-dimensional."
            )

        if (
            matrix.shape[1]
            != len(
                SECOND_STAGE_FEATURE_NAMES
            )
        ):
            raise SecondStageRankingError(
                "feature dimension mismatch."
            )

        if not np.isfinite(
            matrix
        ).all():
            raise SecondStageRankingError(
                "features contain NaN or infinity."
            )

        return np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> SecondStageRanker:
        """Fit the fixed gradient-boosting model."""

        matrix = self._validate_features(
            features
        )

        target = np.asarray(
            labels,
            dtype=np.int64,
        )

        if target.ndim != 1:
            raise SecondStageRankingError(
                "labels must be one-dimensional."
            )

        if len(target) != len(matrix):
            raise SecondStageRankingError(
                "feature and label rows must match."
            )

        if set(
            np.unique(target).tolist()
        ) != {
            0,
            1,
        }:
            raise SecondStageRankingError(
                "Training labels must contain both 0 and 1."
            )

        self._model = (
            HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=(
                    self.config.learning_rate
                ),
                max_iter=(
                    self.config.max_iter
                ),
                max_leaf_nodes=(
                    self.config.max_leaf_nodes
                ),
                min_samples_leaf=(
                    self.config.min_samples_leaf
                ),
                l2_regularization=(
                    self.config.l2_regularization
                ),
                early_stopping=False,
                random_state=(
                    self.config.seed
                ),
            )
        )

        self._model.fit(
            matrix,
            target,
        )

        return self

    def score(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Return learned click-ranking scores."""

        matrix = self._validate_features(
            features
        )

        model = self._require_fitted()

        scores = model.predict_proba(
            matrix
        )[:, 1]

        return np.asarray(
            scores,
            dtype=np.float64,
        )

    def rank(
        self,
        candidate_news_ids: Iterable[str],
        features: np.ndarray,
        *,
        top_k: int,
    ) -> list[str]:
        """Rank candidate IDs by learned score."""

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise SecondStageRankingError(
                "top_k must be a positive integer."
            )

        candidate_ids = tuple(
            str(news_id)
            for news_id in candidate_news_ids
        )

        matrix = self._validate_features(
            features
        )

        if len(candidate_ids) != len(
            matrix
        ):
            raise SecondStageRankingError(
                "Candidate and feature rows must match."
            )

        scores = self.score(
            matrix
        )

        ordered = sorted(
            zip(
                candidate_ids,
                scores.tolist(),
                strict=True,
            ),
            key=lambda row: (
                -float(
                    row[1]
                ),
                row[0],
            ),
        )

        return [
            news_id
            for news_id, _
            in ordered[
                :top_k
            ]
        ]

    def save(
        self,
        path: Path,
        *,
        metadata: dict[str, Any],
    ) -> None:
        """Persist the fitted ranker and frozen metadata."""

        model = self._require_fitted()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "model": model,
                "config": (
                    self.config.to_dict()
                ),
                "feature_names": (
                    SECOND_STAGE_FEATURE_NAMES
                ),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> tuple[
        SecondStageRanker,
        dict[str, Any],
    ]:
        """Restore one persisted ranker artifact."""

        payload = joblib.load(
            path
        )

        if tuple(
            payload[
                "feature_names"
            ]
        ) != (
            SECOND_STAGE_FEATURE_NAMES
        ):
            raise SecondStageRankingError(
                "Persisted feature contract mismatch."
            )

        instance = cls(
            SecondStageRankerConfig(
                **payload[
                    "config"
                ]
            )
        )

        instance._model = (
            payload[
                "model"
            ]
        )

        return (
            instance,
            dict(
                payload[
                    "metadata"
                ]
            ),
        )
