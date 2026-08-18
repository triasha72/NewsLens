"""Frozen two-tower -> FAISS -> vectorized-MMR serving runtime."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
import torch

from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
)
from newslens.reranking import (
    MMRConfig,
    maximal_marginal_relevance_vectorized,
)
from newslens.retrieval.base import RetrievalHit
from newslens.retrieval.catalog import RetrievalCatalog
from newslens.retrieval.faiss_flat import (
    FaissFlatIPRetriever,
)

from .bundle import (
    ServingBundleError,
    load_json,
    verify_file_sha256,
)
from .types import (
    ServingRecommendation,
    ServingResult,
    ServingRuntimeConfig,
    ServingTimings,
)


class RetrievalBackend(Protocol):
    """Minimal serving-time retrieval interface."""

    def retrieve(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        exclude_news_ids: Iterable[str] = (),
    ) -> list[RetrievalHit]:
        """Return retrieval hits."""


class ServingRuntime:
    """Production-oriented frozen NewsLens recommendation runtime."""

    def __init__(
        self,
        *,
        network: TwoTowerNetwork,
        catalog: RetrievalCatalog,
        retriever: RetrievalBackend,
        popularity_clicks: dict[str, int],
        config: ServingRuntimeConfig,
    ) -> None:
        self.network = network
        self.catalog = catalog
        self.retriever = retriever
        self.config = config

        self._id_to_position = (
            catalog.id_to_position
        )

        self._popularity_clicks = {
            str(news_id): int(clicks)
            for news_id, clicks
            in popularity_clicks.items()
        }

        self._popularity_order = tuple(
            sorted(
                catalog.news_ids,
                key=lambda news_id: (
                    -self._popularity_clicks.get(
                        news_id,
                        0,
                    ),
                    news_id,
                ),
            )
        )

        self.network.to("cpu")
        self.network.eval()

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: Path,
    ) -> ServingRuntime:
        """Load and verify a frozen Phase-07 serving bundle."""

        bundle_dir = (
            Path(bundle_dir)
            .expanduser()
            .resolve()
        )

        manifest = load_json(
            bundle_dir
            / "manifest.json"
        )

        if (
            manifest.get("schema_version")
            != "1.0.0"
        ):
            raise ServingBundleError(
                "Unsupported serving bundle schema."
            )

        files = manifest["files"]

        checkpoint_path = (
            bundle_dir
            / files["checkpoint"]["name"]
        )

        catalog_path = (
            bundle_dir
            / files["catalog"]["name"]
        )

        index_path = (
            bundle_dir
            / files["faiss_index"]["name"]
        )

        popularity_path = (
            bundle_dir
            / files["popularity"]["name"]
        )

        verify_file_sha256(
            checkpoint_path,
            files["checkpoint"]["sha256"],
        )

        verify_file_sha256(
            catalog_path,
            files["catalog"]["sha256"],
        )

        verify_file_sha256(
            index_path,
            files["faiss_index"]["sha256"],
        )

        verify_file_sha256(
            popularity_path,
            files["popularity"]["sha256"],
        )

        config = ServingRuntimeConfig(
            **manifest["config"]
        )

        try:
            import faiss
        except ImportError as error:
            raise ServingBundleError(
                "FAISS is required for Phase-07 serving."
            ) from error

        faiss.omp_set_num_threads(
            config.faiss_threads
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

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

        if not np.isclose(
            float(
                network.config.temperature
            ),
            config.temperature,
            atol=0.0,
            rtol=0.0,
        ):
            raise ServingBundleError(
                "Bundle temperature differs from checkpoint."
            )

        catalog = (
            RetrievalCatalog.load_npz(
                catalog_path
            )
        )

        retriever = (
            FaissFlatIPRetriever.load(
                catalog,
                index_path,
            )
        )

        popularity = load_json(
            popularity_path
        )

        popularity_clicks = {
            str(news_id): int(clicks)
            for news_id, clicks
            in popularity["clicks"].items()
        }

        return cls(
            network=network,
            catalog=catalog,
            retriever=retriever,
            popularity_clicks=(
                popularity_clicks
            ),
            config=config,
        )

    def _fallback(
        self,
        *,
        history_ids: tuple[str, ...],
        top_k: int,
        unknown_history_count: int,
        total_started: float,
    ) -> ServingResult:
        """Return frozen training-popularity fallback."""

        excluded = set(
            history_ids
        )

        selected = [
            news_id
            for news_id
            in self._popularity_order
            if news_id not in excluded
        ][
            :top_k
        ]

        recommendations = tuple(
            ServingRecommendation(
                rank=rank,
                news_id=news_id,
                score=float(
                    self._popularity_clicks.get(
                        news_id,
                        0,
                    )
                ),
                source=(
                    "popularity_fallback"
                ),
            )
            for rank, news_id
            in enumerate(
                selected,
                start=1,
            )
        )

        total_ms = (
            perf_counter()
            - total_started
        ) * 1_000.0

        return ServingResult(
            recommendations=(
                recommendations
            ),
            fallback_used=True,
            unknown_history_count=(
                unknown_history_count
            ),
            timings=ServingTimings(
                user_embedding_ms=0.0,
                retrieval_ms=0.0,
                rerank_ms=0.0,
                total_ms=total_ms,
            ),
        )

    def recommend(
        self,
        history_news_ids: Iterable[str],
        *,
        top_k: int | None = None,
    ) -> ServingResult:
        """Return frozen global recommendations for one history."""

        total_started = (
            perf_counter()
        )

        effective_top_k = (
            self.config.final_k
            if top_k is None
            else top_k
        )

        if (
            isinstance(
                effective_top_k,
                bool,
            )
            or not isinstance(
                effective_top_k,
                int,
            )
            or effective_top_k <= 0
            or effective_top_k
            > self.config.final_k
        ):
            raise ValueError(
                "top_k must be between 1 and "
                f"{self.config.final_k}."
            )

        if isinstance(
            history_news_ids,
            str,
        ):
            raw_history = tuple(
                history_news_ids.split()
            )
        else:
            raw_history = tuple(
                str(news_id).strip()
                for news_id
                in history_news_ids
                if str(news_id).strip()
            )

        usable_history = tuple(
            news_id
            for news_id in raw_history
            if news_id
            in self._id_to_position
        )

        unknown_history_count = sum(
            news_id
            not in self._id_to_position
            for news_id in raw_history
        )

        if not usable_history:
            return self._fallback(
                history_ids=raw_history,
                top_k=effective_top_k,
                unknown_history_count=(
                    unknown_history_count
                ),
                total_started=(
                    total_started
                ),
            )

        encoded_history = usable_history[
            -self.config.max_history_length:
        ]

        positions = [
            self._id_to_position[
                news_id
            ]
            for news_id
            in encoded_history
        ]

        user_started = (
            perf_counter()
        )

        history_embeddings = (
            torch.from_numpy(
                self.catalog.vectors[
                    positions
                ]
            )
            .unsqueeze(0)
        )

        history_mask = torch.ones(
            (
                1,
                len(positions),
            ),
            dtype=torch.bool,
        )

        with torch.no_grad():
            user_vector = (
                self.network.user_tower(
                    history_embeddings,
                    history_mask,
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

        user_embedding_ms = (
            perf_counter()
            - user_started
        ) * 1_000.0

        retrieval_started = (
            perf_counter()
        )

        hits = self.retriever.retrieve(
            user_vector,
            top_k=(
                self.config.retrieval_k
            ),
            exclude_news_ids=(
                raw_history
            ),
        )

        retrieval_ms = (
            perf_counter()
            - retrieval_started
        ) * 1_000.0

        if not hits:
            return self._fallback(
                history_ids=raw_history,
                top_k=effective_top_k,
                unknown_history_count=(
                    unknown_history_count
                ),
                total_started=(
                    total_started
                ),
            )

        candidate_ids = tuple(
            hit.news_id
            for hit in hits
        )

        candidate_positions = [
            self._id_to_position[
                news_id
            ]
            for news_id
            in candidate_ids
        ]

        candidate_vectors = (
            self.catalog.vectors[
                candidate_positions
            ]
        )

        relevance_scores = np.asarray(
            [
                float(hit.score)
                / self.config.temperature
                for hit in hits
            ],
            dtype=np.float64,
        )

        rerank_started = (
            perf_counter()
        )

        selected_ids = (
            maximal_marginal_relevance_vectorized(
                candidate_ids,
                relevance_scores,
                candidate_vectors,
                top_k=effective_top_k,
                config=MMRConfig(
                    lambda_weight=(
                        self.config.lambda_weight
                    )
                ),
            )
        )

        rerank_ms = (
            perf_counter()
            - rerank_started
        ) * 1_000.0

        score_by_id = {
            hit.news_id: float(
                hit.score
            )
            for hit in hits
        }

        recommendations = tuple(
            ServingRecommendation(
                rank=rank,
                news_id=news_id,
                score=score_by_id[
                    news_id
                ],
                source=(
                    "two_tower_faiss_mmr"
                ),
            )
            for rank, news_id
            in enumerate(
                selected_ids,
                start=1,
            )
        )

        total_ms = (
            perf_counter()
            - total_started
        ) * 1_000.0

        return ServingResult(
            recommendations=(
                recommendations
            ),
            fallback_used=False,
            unknown_history_count=(
                unknown_history_count
            ),
            timings=ServingTimings(
                user_embedding_ms=(
                    user_embedding_ms
                ),
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                total_ms=total_ms,
            ),
        )
