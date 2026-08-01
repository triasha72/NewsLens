"""Tests for production fallback-model artifact export."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from newslens.artifacts import (
    ArtifactExportError,
    export_fallback_artifact,
    load_artifact,
)
from newslens.models import ContentPopularityFallbackRecommender

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)


def make_news() -> pd.DataFrame:
    return pd.DataFrame(
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


def make_behaviors() -> pd.DataFrame:
    return pd.DataFrame(
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


def test_export_writes_loadable_full_training_artifact(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact"
    created_at = datetime(
        2026,
        8,
        1,
        12,
        0,
        tzinfo=UTC,
    )

    result = export_fallback_artifact(
        make_news(),
        make_behaviors(),
        destination,
        artifact_version="0.3.0",
        ranking_cutoff=2,
        max_features=100,
        created_at_utc=created_at,
    )
    loaded = load_artifact(destination)

    assert result.path == destination.resolve()
    assert result.metadata.training_records == 3
    assert result.metadata.training_cutoff.isoformat() == "2020-01-03T00:00:00"
    assert result.metadata.created_at_utc == created_at
    assert result.metadata.indexed_article_count == 4
    assert result.metadata.vocabulary_article_count == 4
    assert result.metadata.ranking_cutoff == 2
    assert result.metadata.tfidf.max_features == 100
    assert loaded.metadata == result.metadata
    assert isinstance(
        loaded.model,
        ContentPopularityFallbackRecommender,
    )


def test_loaded_export_produces_recommendations(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact"
    export_fallback_artifact(
        make_news(),
        make_behaviors(),
        destination,
        max_features=100,
    )

    loaded = load_artifact(destination)
    recommendations = loaded.model.recommend(
        ["N1"],
        candidate_news_ids=["N2", "N3", "N4"],
        top_k=2,
    )

    assert len(recommendations) == 2
    assert recommendations[0].news_id == "N2"


def test_export_rejects_training_reference_missing_from_catalog(
    tmp_path: Path,
) -> None:
    behaviors = make_behaviors()
    behaviors.loc[0, "history"] = "N999"

    with pytest.raises(
        ArtifactExportError,
        match="missing from the catalog",
    ):
        export_fallback_artifact(
            make_news(),
            behaviors,
            tmp_path / "artifact",
        )


def test_export_rejects_duplicate_candidates(
    tmp_path: Path,
) -> None:
    behaviors = make_behaviors()
    behaviors.loc[0, "impressions"] = "N1-1 N1-0"

    with pytest.raises(
        ArtifactExportError,
        match="duplicate candidate IDs",
    ):
        export_fallback_artifact(
            make_news(),
            behaviors,
            tmp_path / "artifact",
        )


def test_export_rejects_empty_behavior_data(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ArtifactExportError,
        match="behavior record",
    ):
        export_fallback_artifact(
            make_news(),
            make_behaviors().iloc[0:0],
            tmp_path / "artifact",
        )


def test_export_rejects_invalid_timestamp(
    tmp_path: Path,
) -> None:
    behaviors = make_behaviors()
    behaviors["timestamp"] = behaviors["timestamp"].astype("object")
    behaviors.loc[0, "timestamp"] = "not-a-timestamp"

    with pytest.raises(
        ArtifactExportError,
        match="timestamps are invalid",
    ):
        export_fallback_artifact(
            make_news(),
            behaviors,
            tmp_path / "artifact",
        )
