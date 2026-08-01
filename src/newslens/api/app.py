"""FastAPI application factory for NewsLens."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)

from newslens import __version__
from newslens.artifacts import LoadedArtifact, load_artifact
from newslens.models import (
    ContentPopularityFallbackRecommender,
)

from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from .settings import ApiSettings

SERVICE_NAME = "newslens"
MODEL_NAME = "tfidf_content_with_popularity_fallback"
RANKING_CUTOFF = 10


def _resolve_artifact_path(
    artifact_path: str | Path | None,
) -> Path | None:
    """Resolve an explicit artifact path or environment setting."""

    if artifact_path is not None:
        return Path(artifact_path).expanduser()

    return ApiSettings.from_environment().artifact_path


def _loaded_artifact(
    application: FastAPI,
) -> LoadedArtifact | None:
    """Return the artifact currently held by the application."""

    return getattr(
        application.state,
        "loaded_artifact",
        None,
    )


def _require_loaded_artifact(
    application: FastAPI,
) -> LoadedArtifact:
    """Return the loaded artifact or reject inference."""

    loaded = _loaded_artifact(application)

    if loaded is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not ready.",
        )

    return loaded


def create_app(
    *,
    artifact_path: str | Path | None = None,
) -> FastAPI:
    """Create an isolated NewsLens ASGI application."""

    configured_artifact_path = _resolve_artifact_path(artifact_path)

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        loaded_artifact: LoadedArtifact | None = None

        if configured_artifact_path is not None:
            loaded_artifact = load_artifact(configured_artifact_path)

            if not isinstance(
                loaded_artifact.model,
                ContentPopularityFallbackRecommender,
            ):
                raise RuntimeError(
                    "The configured artifact does not contain a NewsLens fallback recommender."
                )

        application.state.loaded_artifact = loaded_artifact

        try:
            yield
        finally:
            application.state.loaded_artifact = None

    application = FastAPI(
        title="NewsLens API",
        summary=("Leakage-aware news search and recommendation service."),
        version=__version__,
        lifespan=lifespan,
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["service"],
        summary="Check service liveness",
    )
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=SERVICE_NAME,
            version=__version__,
        )

    @application.get(
        "/model-info",
        response_model=ModelInfoResponse,
        tags=["service"],
        summary="Inspect configured model metadata",
    )
    def model_info(
        request: Request,
    ) -> ModelInfoResponse:
        loaded = _loaded_artifact(request.app)

        if loaded is None:
            return ModelInfoResponse(
                model_name=MODEL_NAME,
                model_ready=False,
                artifact_version=None,
                ranking_cutoff=RANKING_CUTOFF,
            )

        return ModelInfoResponse(
            model_name=loaded.metadata.model_name,
            model_ready=True,
            artifact_version=(loaded.metadata.artifact_version),
            ranking_cutoff=(loaded.metadata.ranking_cutoff),
        )

    @application.post(
        "/recommend",
        response_model=RecommendationResponse,
        tags=["recommendation"],
        summary="Rank candidate news articles",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": ("The recommendation model is not ready.")
            }
        },
    )
    def recommend(
        payload: RecommendationRequest,
        request: Request,
    ) -> RecommendationResponse:
        loaded = _require_loaded_artifact(request.app)
        model = cast(
            ContentPopularityFallbackRecommender,
            loaded.model,
        )

        requested_top_k = (
            payload.top_k if payload.top_k is not None else loaded.metadata.ranking_cutoff
        )

        recommendations = model.recommend(
            payload.history_news_ids,
            candidate_news_ids=(payload.candidate_news_ids),
            top_k=requested_top_k,
        )

        items = tuple(
            RecommendationItem(
                news_id=recommendation.news_id,
                score=recommendation.score,
                source=recommendation.source.value,
            )
            for recommendation in recommendations
        )

        return RecommendationResponse(
            model_name=loaded.metadata.model_name,
            artifact_version=(loaded.metadata.artifact_version),
            requested_top_k=requested_top_k,
            returned_count=len(items),
            recommendations=items,
        )

    return application


app = create_app()
