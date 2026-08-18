"""Tests for the Phase-07 production API."""

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


class FakeRuntime:
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
            10 if top_k is None else top_k
        )

        return ServingResult(
            recommendations=tuple(
                ServingRecommendation(
                    rank=index,
                    news_id=f"N{index}",
                    score=1.0 / index,
                    source="two_tower_faiss_mmr",
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


def test_health_without_model() -> None:
    app = create_production_app()

    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200


def test_ready_requires_model() -> None:
    app = create_production_app()

    with TestClient(app) as client:
        response = client.get(
            "/ready"
        )

    assert response.status_code == 503


def test_model_info_and_recommendation() -> None:
    app = create_production_app(
        runtime=FakeRuntime(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        info = client.get(
            "/model-info"
        )

        response = client.post(
            "/v1/recommend",
            headers={
                "X-Request-ID": "phase07-test"
            },
            json={
                "history_news_ids": [
                    "N100",
                    "N101",
                ],
                "top_k": 3,
            },
        )

    assert info.status_code == 200

    assert (
        info.json()["selected_policy"]
        == "mmr_lambda_0.80"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["request_id"] == "phase07-test"
    assert body["returned_count"] == 3
    assert body["fallback_used"] is False


def test_top_k_above_ten_is_rejected() -> None:
    app = create_production_app(
        runtime=FakeRuntime(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/recommend",
            json={
                "history_news_ids": [],
                "top_k": 11,
            },
        )

    assert response.status_code == 422
