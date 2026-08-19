"""Low-cardinality Prometheus metrics for Phase-07 serving."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from newslens.serving import ServingResult


class ServingMetrics:
    """Per-application serving metrics registry."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        self.http_requests = Counter(
            "newslens_http_requests_total",
            "NewsLens HTTP requests.",
            (
                "method",
                "route",
                "status",
            ),
            registry=self.registry,
        )

        self.http_duration = Histogram(
            "newslens_http_request_duration_seconds",
            "NewsLens HTTP request duration.",
            (
                "method",
                "route",
            ),
            buckets=(
                0.001,
                0.0025,
                0.005,
                0.010,
                0.025,
                0.050,
                0.100,
                0.250,
                0.500,
                1.0,
                2.5,
                5.0,
            ),
            registry=self.registry,
        )

        self.stage_duration = Histogram(
            "newslens_stage_duration_seconds",
            "Recommendation-model stage duration.",
            ("stage",),
            buckets=(
                0.0001,
                0.00025,
                0.0005,
                0.001,
                0.0025,
                0.005,
                0.010,
                0.025,
                0.050,
                0.100,
                0.250,
                0.500,
            ),
            registry=self.registry,
        )

        self.recommendations = Counter(
            "newslens_recommendations_total",
            "Recommendations returned by source.",
            ("source",),
            registry=self.registry,
        )

        self.fallbacks = Counter(
            "newslens_popularity_fallback_total",
            "Requests routed to popularity fallback.",
            registry=self.registry,
        )

        self.unknown_history = Counter(
            "newslens_unknown_history_items_total",
            "Unknown history identifiers observed.",
            registry=self.registry,
        )

        self.model_ready = Gauge(
            "newslens_model_ready",
            "Whether the Phase-07 model is ready.",
            registry=self.registry,
        )

    def set_ready(
        self,
        ready: bool,
    ) -> None:
        self.model_ready.set(
            1.0 if ready else 0.0
        )

    def observe_http(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        self.http_requests.labels(
            method,
            route,
            str(status_code),
        ).inc()

        self.http_duration.labels(
            method,
            route,
        ).observe(
            duration_seconds
        )

    def observe_result(
        self,
        result: ServingResult,
    ) -> None:
        timings = result.timings

        for stage, milliseconds in (
            (
                "user_embedding",
                timings.user_embedding_ms,
            ),
            (
                "retrieval",
                timings.retrieval_ms,
            ),
            (
                "rerank",
                timings.rerank_ms,
            ),
            (
                "total",
                timings.total_ms,
            ),
        ):
            self.stage_duration.labels(
                stage
            ).observe(
                milliseconds / 1_000.0
            )

        for recommendation in (
            result.recommendations
        ):
            self.recommendations.labels(
                recommendation.source
            ).inc()

        if result.fallback_used:
            self.fallbacks.inc()

        if (
            result.unknown_history_count
            > 0
        ):
            self.unknown_history.inc(
                result.unknown_history_count
            )

    def render(
        self,
    ) -> tuple[bytes, str]:
        """Render Prometheus exposition format."""

        return (
            generate_latest(
                self.registry
            ),
            CONTENT_TYPE_LATEST,
        )
