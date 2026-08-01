"""FastAPI application factory for NewsLens."""

from fastapi import FastAPI

from newslens import __version__

from .schemas import HealthResponse, ModelInfoResponse

SERVICE_NAME = "newslens"
MODEL_NAME = "tfidf_content_with_popularity_fallback"
RANKING_CUTOFF = 10


def create_app() -> FastAPI:
    """Create an isolated NewsLens ASGI application."""

    application = FastAPI(
        title="NewsLens API",
        summary="Leakage-aware news search and recommendation service.",
        version=__version__,
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
    def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(
            model_name=MODEL_NAME,
            model_ready=False,
            artifact_version=None,
            ranking_cutoff=RANKING_CUTOFF,
        )

    return application


app = create_app()
