"""Tests for the artifact-backed NewsLens HTTP API."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from newslens import __version__
from newslens.api import create_app
from newslens.artifacts import (
    ArtifactNotFoundError,
    export_fallback_artifact,
)
from newslens.realtime import InMemoryArticleRepository, SearchDocument

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)


def save_api_artifact(tmp_path: Path) -> Path:
    """Create a small production-shaped artifact for API tests."""

    news = pd.DataFrame(
        {
            "news_id": ["N1", "N2", "N3", "N4"],
            "title": [
                "Mars mission discovers water",
                "Mars rover searches for water",
                "Football championship begins",
                "Football team wins championship",
            ],
            "abstract": [
                "Spacecraft explores the planet.",
                "A rover begins planetary exploration.",
                "Players prepare for the match.",
                "The coach celebrates a victory.",
            ],
            "category": [
                "science",
                "science",
                "sports",
                "sports",
            ],
            "subcategory": [
                "space",
                "space",
                "football",
                "football",
            ],
        }
    )

    behaviors = pd.DataFrame(
        {
            "impression_id": ["I1", "I2", "I3"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-02 00:00:00",
                    "2020-01-03 00:00:00",
                ]
            ),
            "history": ["", "N1", "N3"],
            "impressions": [
                "N1-1 N2-0 N3-0",
                "N2-1 N3-0 N4-0",
                "N4-1 N1-0 N2-0",
            ],
        }
    )

    destination = tmp_path / "newslens-test-0.3.0"

    export_fallback_artifact(
        news,
        behaviors,
        destination,
        artifact_version="0.3.0",
        ranking_cutoff=2,
        max_features=100,
    )

    return destination


def test_health_endpoint_reports_liveness() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "newslens",
        "version": __version__,
    }


def test_model_info_is_honest_without_artifact() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": ("tfidf_content_with_popularity_fallback"),
        "model_ready": False,
        "artifact_version": None,
        "ranking_cutoff": 10,
    }


def test_model_info_reports_loaded_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = save_api_artifact(tmp_path)

    with TestClient(create_app(artifact_path=artifact_path)) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": ("tfidf_content_with_popularity_fallback"),
        "model_ready": True,
        "artifact_version": "0.3.0",
        "ranking_cutoff": 2,
    }


def test_environment_variable_configures_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = save_api_artifact(tmp_path)

    monkeypatch.setenv(
        "NEWSLENS_ARTIFACT_PATH",
        str(artifact_path),
    )

    with TestClient(create_app()) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_ready"] is True
    assert response.json()["artifact_version"] == "0.3.0"


def test_missing_configured_artifact_fails_startup(
    tmp_path: Path,
) -> None:
    application = create_app(artifact_path=tmp_path / "missing")

    with (
        pytest.raises(
            ArtifactNotFoundError,
            match="not found",
        ),
        TestClient(application),
    ):
        pass


def test_openapi_schema_lists_service_endpoints() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["version"] == __version__
    assert set(response.json()["paths"]) == {
        "/health",
        "/ready",
        "/realtime/ready",
        "/model-info",
        "/recommend",
        "/search",
    }


def test_realtime_readiness_is_honest_without_store() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/realtime/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Real-time article store is not ready."}


def test_search_requires_realtime_store() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/search", params={"q": "latest Apple earnings"})

    assert response.status_code == 503


def test_demo_mode_exposes_only_synthetic_search_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSLENS_DEMO_MODE", "true")

    with TestClient(create_app()) as client:
        readiness = client.get("/realtime/ready")
        response = client.get("/search", params={"q": "latest AI chip", "top_k": 2})

    assert readiness.status_code == 200
    assert response.status_code == 200
    assert response.json()["results"][0]["article_id"] == "demo-ai-chip"


def test_demo_mode_rejects_a_live_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSLENS_DEMO_MODE", "true")
    monkeypatch.setenv("NEWSLENS_REALTIME_DATABASE_URL", "postgresql://example.test/newslens")

    with pytest.raises(ValueError, match="cannot be combined"):
        create_app()


def test_search_returns_intent_and_freshness_diagnostics() -> None:
    now = datetime.now(UTC)
    repository = InMemoryArticleRepository(
        (
            SearchDocument(
                article_id="N-new",
                title="Apple reports latest earnings",
                body="Quarterly market results.",
                category="business",
                published_at=now - timedelta(hours=1),
                produced_at=now - timedelta(seconds=1),
                indexed_at=now - timedelta(milliseconds=800),
                popularity=10,
            ),
            SearchDocument(
                article_id="N-old",
                title="Apple reports earnings",
                body="An older quarterly market report.",
                category="business",
                published_at=now - timedelta(days=5),
                produced_at=now - timedelta(days=5, seconds=1),
                indexed_at=now - timedelta(days=5),
                popularity=100,
            ),
        )
    )

    with TestClient(create_app(realtime_repository=repository)) as client:
        readiness = client.get("/realtime/ready")
        response = client.get(
            "/search",
            params={"q": "latest Apple earnings", "top_k": 2},
        )

    assert readiness.status_code == 200
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == {
        "normalized_query": "latest Apple earnings",
        "category": "business",
        "entity": "Apple",
        "prefers_freshness": True,
    }
    assert body["results"][0]["article_id"] == "N-new"
    assert body["results"][0]["index_freshness_ms"] == pytest.approx(200.0)


def test_unknown_route_returns_not_found() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/missing")

    assert response.status_code == 404


def test_recommend_requires_loaded_artifact() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/recommend",
            json={
                "history_news_ids": ["N1"],
                "candidate_news_ids": [
                    "N2",
                    "N3",
                ],
                "top_k": 2,
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Recommendation model is not ready."}


def test_recommend_returns_content_rankings(
    tmp_path: Path,
) -> None:
    artifact_path = save_api_artifact(tmp_path)

    with TestClient(create_app(artifact_path=artifact_path)) as client:
        response = client.post(
            "/recommend",
            json={
                "history_news_ids": ["N1"],
                "candidate_news_ids": [
                    "N2",
                    "N3",
                    "N4",
                ],
                "top_k": 2,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["model_name"] == ("tfidf_content_with_popularity_fallback")
    assert body["artifact_version"] == "0.3.0"
    assert body["requested_top_k"] == 2
    assert body["returned_count"] == 2
    assert body["recommendations"][0]["news_id"] == "N2"
    assert {item["source"] for item in body["recommendations"]} == {"content"}


def test_recommend_uses_popularity_for_cold_start(
    tmp_path: Path,
) -> None:
    artifact_path = save_api_artifact(tmp_path)

    with TestClient(create_app(artifact_path=artifact_path)) as client:
        response = client.post(
            "/recommend",
            json={
                "history_news_ids": [],
                "candidate_news_ids": [
                    "N1",
                    "N2",
                    "N3",
                    "N4",
                ],
                "top_k": 2,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["returned_count"] == 2
    assert {item["source"] for item in body["recommendations"]} == {"popularity"}


def test_recommend_uses_artifact_default_cutoff(
    tmp_path: Path,
) -> None:
    artifact_path = save_api_artifact(tmp_path)

    with TestClient(create_app(artifact_path=artifact_path)) as client:
        response = client.post(
            "/recommend",
            json={
                "history_news_ids": ["N1"],
                "candidate_news_ids": [
                    "N2",
                    "N3",
                    "N4",
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["requested_top_k"] == 2
    assert response.json()["returned_count"] == 2


def test_recommend_rejects_duplicate_candidates(
    tmp_path: Path,
) -> None:
    artifact_path = save_api_artifact(tmp_path)

    with TestClient(create_app(artifact_path=artifact_path)) as client:
        response = client.post(
            "/recommend",
            json={
                "history_news_ids": ["N1"],
                "candidate_news_ids": [
                    "N2",
                    "N2",
                ],
                "top_k": 2,
            },
        )

    assert response.status_code == 422


def test_recommend_rejects_empty_candidates(
    tmp_path: Path,
) -> None:
    artifact_path = save_api_artifact(tmp_path)

    with TestClient(create_app(artifact_path=artifact_path)) as client:
        response = client.post(
            "/recommend",
            json={
                "history_news_ids": ["N1"],
                "candidate_news_ids": [],
                "top_k": 2,
            },
        )

    assert response.status_code == 422


def test_recommend_rejects_invalid_top_k(
    tmp_path: Path,
) -> None:
    artifact_path = save_api_artifact(tmp_path)

    with TestClient(create_app(artifact_path=artifact_path)) as client:
        response = client.post(
            "/recommend",
            json={
                "history_news_ids": ["N1"],
                "candidate_news_ids": [
                    "N2",
                    "N3",
                ],
                "top_k": 0,
            },
        )

    assert response.status_code == 422
