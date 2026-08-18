"""Core types for the frozen Phase-07 serving runtime."""

from __future__ import annotations

from dataclasses import dataclass


class ServingConfigurationError(ValueError):
    """Raised when the frozen serving configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ServingRuntimeConfig:
    """Frozen production recommendation configuration."""

    artifact_version: str
    selected_policy: str
    retrieval_backend: str
    lambda_weight: float
    retrieval_k: int
    final_k: int
    temperature: float
    max_history_length: int
    faiss_threads: int = 1

    def __post_init__(self) -> None:
        if self.selected_policy != "mmr_lambda_0.80":
            raise ServingConfigurationError(
                "Phase-07 serving requires mmr_lambda_0.80."
            )

        if self.retrieval_backend != "faiss_flat":
            raise ServingConfigurationError(
                "Phase-07 serving requires faiss_flat."
            )

        if self.lambda_weight != 0.80:
            raise ServingConfigurationError(
                "Phase-07 serving lambda must remain 0.80."
            )

        if self.retrieval_k != 100:
            raise ServingConfigurationError(
                "Phase-07 retrieval depth must remain 100."
            )

        if self.final_k != 10:
            raise ServingConfigurationError(
                "Phase-07 final depth must remain 10."
            )

        if self.temperature <= 0.0:
            raise ServingConfigurationError(
                "temperature must be positive."
            )

        if self.max_history_length <= 0:
            raise ServingConfigurationError(
                "max_history_length must be positive."
            )

        if self.faiss_threads <= 0:
            raise ServingConfigurationError(
                "faiss_threads must be positive."
            )


@dataclass(frozen=True, slots=True)
class ServingTimings:
    """Per-request model-stage timings."""

    user_embedding_ms: float
    retrieval_ms: float
    rerank_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class ServingRecommendation:
    """One production recommendation."""

    rank: int
    news_id: str
    score: float
    source: str


@dataclass(frozen=True, slots=True)
class ServingResult:
    """One complete production recommendation result."""

    recommendations: tuple[ServingRecommendation, ...]
    fallback_used: bool
    unknown_history_count: int
    timings: ServingTimings
