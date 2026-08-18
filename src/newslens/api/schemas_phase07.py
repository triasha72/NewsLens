"""Phase-07 production API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class ProductionHealthResponse(BaseModel):
    model_config = ConfigDict(
        frozen=True
    )

    status: Literal["ok"]
    service: str
    version: str


class ProductionReadinessResponse(BaseModel):
    model_config = ConfigDict(
        frozen=True
    )

    status: Literal["ready"]
    model_ready: Literal[True]
    artifact_version: str


class ProductionModelInfoResponse(BaseModel):
    model_config = ConfigDict(
        frozen=True
    )

    model_ready: bool
    artifact_version: str | None
    selected_policy: str | None
    retrieval_backend: str | None
    retrieval_k: int | None
    final_k: int | None


class ProductionRecommendationRequest(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    history_news_ids: tuple[
        str,
        ...
    ] = Field(
        default_factory=tuple,
        max_length=500,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=10,
    )

    @field_validator(
        "history_news_ids"
    )
    @classmethod
    def normalize_history(
        cls,
        values: tuple[
            str,
            ...
        ],
    ) -> tuple[str, ...]:
        normalized = tuple(
            value.strip()
            for value in values
        )

        if any(
            not value
            for value in normalized
        ):
            raise ValueError(
                "History identifiers cannot be empty."
            )

        return normalized


class ProductionTimingResponse(BaseModel):
    model_config = ConfigDict(
        frozen=True
    )

    user_embedding_ms: float = Field(
        ge=0.0
    )
    retrieval_ms: float = Field(
        ge=0.0
    )
    rerank_ms: float = Field(
        ge=0.0
    )
    total_ms: float = Field(
        ge=0.0
    )


class ProductionRecommendationItem(BaseModel):
    model_config = ConfigDict(
        frozen=True
    )

    rank: int = Field(
        ge=1
    )

    news_id: str
    score: float

    source: Literal[
        "two_tower_faiss_mmr",
        "popularity_fallback",
    ]


class ProductionRecommendationResponse(BaseModel):
    model_config = ConfigDict(
        frozen=True
    )

    request_id: str
    artifact_version: str
    selected_policy: str
    retrieval_backend: str
    requested_top_k: int
    returned_count: int
    fallback_used: bool
    unknown_history_count: int
    timings: ProductionTimingResponse
    recommendations: tuple[
        ProductionRecommendationItem,
        ...
    ]
