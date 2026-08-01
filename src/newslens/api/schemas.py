"""Request and response schemas for the NewsLens HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class HealthResponse(BaseModel):
    """Liveness information for the HTTP service."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"]
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    """Metadata about the model exposed by the service."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    model_ready: bool
    artifact_version: str | None
    ranking_cutoff: int


class RecommendationRequest(BaseModel):
    """Candidate-ranking request for the recommendation model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    history_news_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=500,
    )
    candidate_news_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )

    @field_validator(
        "history_news_ids",
        "candidate_news_ids",
    )
    @classmethod
    def normalize_news_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Trim IDs and reject empty identifiers."""

        normalized = tuple(news_id.strip() for news_id in values)

        if any(not news_id for news_id in normalized):
            raise ValueError("News identifiers cannot be empty.")

        return normalized

    @field_validator("candidate_news_ids")
    @classmethod
    def reject_duplicate_candidates(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Require candidates to be unique."""

        if len(values) != len(set(values)):
            raise ValueError("Candidate news identifiers must be unique.")

        return values


class RecommendationItem(BaseModel):
    """One ranked recommendation returned by NewsLens."""

    model_config = ConfigDict(frozen=True)

    news_id: str
    score: float
    source: Literal["content", "popularity"]


class RecommendationResponse(BaseModel):
    """Ranked recommendations and serving metadata."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    artifact_version: str
    requested_top_k: int
    returned_count: int
    recommendations: tuple[RecommendationItem, ...]
