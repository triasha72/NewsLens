"""Response schemas for the NewsLens HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Liveness information for the HTTP service."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"]
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    """Metadata about the recommendation model exposed by the service."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    model_ready: bool
    artifact_version: str | None
    ranking_cutoff: int
