"""Production FastAPI application for Phase-07 serving."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)

from newslens import __version__
from newslens.serving import ServingRuntime

from .observability import (
    get_request_id,
    install_request_observability,
)
from .schemas_phase07 import (
    ProductionHealthResponse,
    ProductionModelInfoResponse,
    ProductionReadinessResponse,
    ProductionRecommendationItem,
    ProductionRecommendationRequest,
    ProductionRecommendationResponse,
    ProductionTimingResponse,
)
from .settings_phase07 import (
    ProductionApiSettings,
)

SERVICE_NAME = "newslens-production"


def _require_runtime(
    application: FastAPI,
) -> ServingRuntime:
    runtime = getattr(
        application.state,
        "serving_runtime",
        None,
    )

    if runtime is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Production recommendation runtime "
                "is not ready."
            ),
        )

    return runtime


def create_production_app(
    *,
    bundle_path: str | Path | None = None,
    runtime: ServingRuntime | None = None,
) -> FastAPI:
    """Create the Phase-07 production API."""

    configured_path = (
        Path(bundle_path).expanduser()
        if bundle_path is not None
        else (
            ProductionApiSettings
            .from_environment()
            .serving_bundle_path
        )
    )

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        loaded_runtime = runtime

        if (
            loaded_runtime is None
            and configured_path is not None
        ):
            loaded_runtime = (
                ServingRuntime.from_bundle(
                    configured_path
                )
            )

        application.state.serving_runtime = (
            loaded_runtime
        )

        try:
            yield
        finally:
            application.state.serving_runtime = None

    application = FastAPI(
        title="NewsLens Production API",
        version=__version__,
        lifespan=lifespan,
    )

    install_request_observability(
        application
    )

    @application.get(
        "/health",
        response_model=ProductionHealthResponse,
    )
    def health() -> ProductionHealthResponse:
        return ProductionHealthResponse(
            status="ok",
            service=SERVICE_NAME,
            version=__version__,
        )

    @application.get(
        "/ready",
        response_model=ProductionReadinessResponse,
    )
    def ready(
        request: Request,
    ) -> ProductionReadinessResponse:
        active_runtime = _require_runtime(
            request.app
        )

        return ProductionReadinessResponse(
            status="ready",
            model_ready=True,
            artifact_version=(
                active_runtime
                .config
                .artifact_version
            ),
        )

    @application.get(
        "/model-info",
        response_model=ProductionModelInfoResponse,
    )
    def model_info(
        request: Request,
    ) -> ProductionModelInfoResponse:
        active_runtime = getattr(
            request.app.state,
            "serving_runtime",
            None,
        )

        if active_runtime is None:
            return ProductionModelInfoResponse(
                model_ready=False,
                artifact_version=None,
                selected_policy=None,
                retrieval_backend=None,
                retrieval_k=None,
                final_k=None,
            )

        return ProductionModelInfoResponse(
            model_ready=True,
            artifact_version=(
                active_runtime
                .config
                .artifact_version
            ),
            selected_policy=(
                active_runtime
                .config
                .selected_policy
            ),
            retrieval_backend=(
                active_runtime
                .config
                .retrieval_backend
            ),
            retrieval_k=(
                active_runtime
                .config
                .retrieval_k
            ),
            final_k=(
                active_runtime
                .config
                .final_k
            ),
        )

    @application.post(
        "/v1/recommend",
        response_model=(
            ProductionRecommendationResponse
        ),
    )
    def recommend(
        payload: ProductionRecommendationRequest,
        request: Request,
    ) -> ProductionRecommendationResponse:
        active_runtime = _require_runtime(
            request.app
        )

        result = active_runtime.recommend(
            payload.history_news_ids,
            top_k=payload.top_k,
        )

        return ProductionRecommendationResponse(
            request_id=get_request_id(
                request
            ),
            artifact_version=(
                active_runtime
                .config
                .artifact_version
            ),
            selected_policy=(
                active_runtime
                .config
                .selected_policy
            ),
            retrieval_backend=(
                active_runtime
                .config
                .retrieval_backend
            ),
            requested_top_k=payload.top_k,
            returned_count=len(
                result.recommendations
            ),
            fallback_used=(
                result.fallback_used
            ),
            unknown_history_count=(
                result.unknown_history_count
            ),
            timings=ProductionTimingResponse(
                user_embedding_ms=(
                    result.timings
                    .user_embedding_ms
                ),
                retrieval_ms=(
                    result.timings
                    .retrieval_ms
                ),
                rerank_ms=(
                    result.timings
                    .rerank_ms
                ),
                total_ms=(
                    result.timings
                    .total_ms
                ),
            ),
            recommendations=tuple(
                ProductionRecommendationItem(
                    rank=item.rank,
                    news_id=item.news_id,
                    score=item.score,
                    source=item.source,
                )
                for item
                in result.recommendations
            ),
        )

    return application


app = create_production_app()
