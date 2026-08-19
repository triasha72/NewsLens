"""Tests for Phase-07 Prometheus metrics."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from newslens.api.production import (
    create_production_app,
)
from newslens.serving import (
    ServingRecommendation,
    ServingResult,
    ServingTimings,
)


class FakeMetricsRuntime:
    config = SimpleNamespace(
        artifact_version="phase07-test",
        selected_policy="mmr_lambda_0.80",
        retrieval_backend="faiss_flat",
        retrieval_k=100,
        final_k=10,
    )

    def recommend(
        self,
        history_news_ids,
        *,
        top_k=None,
    ) -> ServingResult:
        del history_news_ids

        effective_k = (
            10
            if top_k is None
            else top_k
        )

        return ServingResult(
            recommendations=tuple(
                ServingRecommendation(
                    rank=index,
                    news_id=f"N{index}",
                    score=1.0,
                    source=(
                        "two_tower_faiss_mmr"
                    ),
                )
                for index
                in range(
                    1,
                    effective_k + 1,
                )
            ),
            fallback_used=False,
            unknown_history_count=0,
            timings=ServingTimings(
                user_embedding_ms=1.0,
                retrieval_ms=0.5,
                rerank_ms=4.0,
                total_ms=5.5,
            ),
        )


def test_metrics_endpoint() -> None:
    app = create_production_app(
        runtime=FakeMetricsRuntime(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/recommend",
            json={
                "history_news_ids": [
                    "N100",
                ],
                "top_k": 2,
            },
        )

        metrics = client.get(
            "/metrics"
        )

    assert response.status_code == 200
    assert metrics.status_code == 200

    text = metrics.text

    assert (
        "newslens_http_requests_total"
        in text
    )

    assert (
        "newslens_stage_duration_seconds"
        in text
    )

    assert (
        "newslens_recommendations_total"
        in text
    )

    assert (
        "newslens_model_ready"
        in text
    )

    assert 'news_id="' not in text
    assert 'request_id="' not in text
