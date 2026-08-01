"""Tests for the NewsLens HTTP API foundation."""

from fastapi.testclient import TestClient

from newslens import __version__
from newslens.api import create_app


def test_health_endpoint_reports_liveness() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "newslens",
        "version": __version__,
    }


def test_model_info_is_honest_before_artifact_loading() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "tfidf_content_with_popularity_fallback",
        "model_ready": False,
        "artifact_version": None,
        "ranking_cutoff": 10,
    }


def test_openapi_schema_lists_service_endpoints() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["version"] == __version__
    assert set(response.json()["paths"]) == {"/health", "/model-info"}


def test_unknown_route_returns_not_found() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/missing")

    assert response.status_code == 404
