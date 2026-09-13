"""FastAPI application factory for NewsLens."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import cast

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import RedirectResponse

from newslens import __version__
from newslens.artifacts import LoadedArtifact, load_artifact
from newslens.models import (
    ContentPopularityFallbackRecommender,
)
from newslens.realtime import (
    ArticleRepository,
    FreshnessRanker,
    PostgresArticleRepository,
    create_demo_repository,
    understand_query,
)

from .observability import (
    LOGGER,
    get_request_id,
    install_request_observability,
)
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    QueryIntentResponse,
    ReadinessResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    SearchReadinessResponse,
    SearchResponse,
    SearchResultItem,
)
from .settings import ApiSettings

SERVICE_NAME = "newslens"
# TODO(newslens-v0.4): serve RecommendationModel-compatible artifacts generically.
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
    """Return the artifact held by the application."""

    return getattr(
        application.state,
        "loaded_artifact",
        None,
    )


def _require_loaded_artifact(
    application: FastAPI,
) -> LoadedArtifact:
    """Return the loaded artifact or reject model traffic."""

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
    realtime_repository: ArticleRepository | None = None,
) -> FastAPI:
    """Create an isolated NewsLens ASGI application."""

    settings = ApiSettings.from_environment()
    configured_artifact_path = (
        Path(artifact_path).expanduser()
        if artifact_path is not None
        else settings.artifact_path
    )
    configured_realtime_repository = realtime_repository
    candidate_limit = settings.realtime_candidate_limit

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        loaded_artifact: LoadedArtifact | None = None
        live_repository = configured_realtime_repository
        owns_live_repository = False

        if configured_artifact_path is not None:
            loaded_artifact = load_artifact(configured_artifact_path)

            if not isinstance(
                loaded_artifact.model,
                ContentPopularityFallbackRecommender,
            ):
                raise RuntimeError(
                    "The configured artifact does not contain a NewsLens fallback recommender."
                )

        if live_repository is None and settings.realtime_database_url is not None:
            live_repository = PostgresArticleRepository(settings.realtime_database_url)
            owns_live_repository = True
        elif live_repository is None and settings.demo_mode:
            live_repository = create_demo_repository()

        application.state.loaded_artifact = loaded_artifact
        application.state.realtime_repository = live_repository

        try:
            yield
        finally:
            application.state.loaded_artifact = None
            application.state.realtime_repository = None

            if owns_live_repository and live_repository is not None:
                live_repository.close()

    application = FastAPI(
        title="NewsLens API",
        summary=("Leakage-aware news search and recommendation service."),
        version=__version__,
        lifespan=lifespan,
    )

    install_request_observability(application)
    ranker = FreshnessRanker()

    @application.get("/", include_in_schema=False)
    def demo_entrypoint() -> RedirectResponse:
        """Send a visitor at the service root to the interactive API page."""

        return RedirectResponse(url="/docs")

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
        "/ready",
        response_model=ReadinessResponse,
        tags=["service"],
        summary="Check model-serving readiness",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": ("The recommendation model is not ready.")
            }
        },
    )
    def readiness(
        request: Request,
    ) -> ReadinessResponse:
        loaded = _require_loaded_artifact(request.app)

        return ReadinessResponse(
            status="ready",
            model_ready=True,
            artifact_version=(loaded.metadata.artifact_version),
        )

    @application.get(
        "/realtime/ready",
        response_model=SearchReadinessResponse,
        tags=["service"],
        summary="Check streamed-article search readiness",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": "The real-time article store is not ready."
            }
        },
    )
    def realtime_readiness(request: Request) -> SearchReadinessResponse:
        repository = cast(
            ArticleRepository | None,
            getattr(request.app.state, "realtime_repository", None),
        )

        if repository is None or not repository.ping():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Real-time article store is not ready.",
            )

        return SearchReadinessResponse(status="ready", realtime_store_ready=True)

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

        inference_started_at = perf_counter()

        recommendations = model.recommend(
            payload.history_news_ids,
            candidate_news_ids=(payload.candidate_news_ids),
            top_k=requested_top_k,
        )

        inference_ms = (perf_counter() - inference_started_at) * 1_000

        items = tuple(
            RecommendationItem(
                news_id=recommendation.news_id,
                score=recommendation.score,
                source=recommendation.source.value,
            )
            for recommendation in recommendations
        )

        request_id = get_request_id(request)
        routing_sources = ",".join(sorted({item.source for item in items}))

        if not routing_sources:
            routing_sources = "none"

        LOGGER.info(
            "recommendation_completed request_id=%s "
            "artifact_version=%s history_count=%d "
            "candidate_count=%d top_k=%d "
            "returned_count=%d source=%s "
            "inference_ms=%.3f",
            request_id,
            loaded.metadata.artifact_version,
            len(payload.history_news_ids),
            len(payload.candidate_news_ids),
            requested_top_k,
            len(items),
            routing_sources,
            inference_ms,
        )

        return RecommendationResponse(
            request_id=request_id,
            model_name=loaded.metadata.model_name,
            artifact_version=(loaded.metadata.artifact_version),
            requested_top_k=requested_top_k,
            returned_count=len(items),
            inference_ms=inference_ms,
            recommendations=items,
        )

    @application.get(
        "/search",
        response_model=SearchResponse,
        tags=["search"],
        summary="Search newly ingested articles with freshness-aware ranking",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": "The real-time article store is not ready."
            }
        },
    )
    def search(
        request: Request,
        q: str = Query(min_length=1, max_length=500),
        top_k: int = Query(default=10, ge=1, le=100),
    ) -> SearchResponse:
        repository = cast(
            ArticleRepository | None,
            getattr(request.app.state, "realtime_repository", None),
        )

        if repository is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Real-time article store is not ready.",
            )

        intent = understand_query(q)
        search_started_at = perf_counter()
        candidates = repository.list_candidates(intent, limit=candidate_limit)
        ranked = ranker.rank(intent, candidates, top_k=top_k)
        search_ms = (perf_counter() - search_started_at) * 1_000
        request_id = get_request_id(request)

        LOGGER.info(
            "search_completed request_id=%s candidate_count=%d returned_count=%d "
            "freshness_intent=%s search_ms=%.3f",
            request_id,
            len(candidates),
            len(ranked),
            intent.prefers_freshness,
            search_ms,
        )

        return SearchResponse(
            request_id=request_id,
            intent=QueryIntentResponse(
                normalized_query=intent.normalized_query,
                category=intent.category,
                entity=intent.entity,
                prefers_freshness=intent.prefers_freshness,
            ),
            candidate_count=len(candidates),
            returned_count=len(ranked),
            search_ms=search_ms,
            results=tuple(
                SearchResultItem(
                    article_id=item.document.article_id,
                    title=item.document.title,
                    category=item.document.category,
                    published_at=item.document.published_at.isoformat(),
                    score=item.score,
                    relevance_score=item.relevance_score,
                    freshness_score=item.freshness_score,
                    popularity_score=item.popularity_score,
                    index_freshness_ms=item.index_freshness_ms,
                )
                for item in ranked
            ),
        )

    return application


app = create_app()
