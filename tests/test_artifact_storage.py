"""Tests for atomic NewsLens artifact storage and verification."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from newslens.artifacts import (
    ArtifactAlreadyExistsError,
    ArtifactFileRecord,
    ArtifactIntegrityError,
    ArtifactManifest,
    ArtifactMetadata,
    ArtifactNotFoundError,
    ArtifactStorageError,
    load_artifact,
    save_artifact,
)
from newslens.models import (
    ContentBasedRecommender,
    ContentPopularityFallbackRecommender,
    PopularityRecommender,
)


def make_model() -> ContentPopularityFallbackRecommender:
    """Return a small fitted fallback model suitable for serialization."""

    news = pd.DataFrame(
        {
            "news_id": ["N1", "N2", "N3"],
            "title": [
                "Space mission launches a new rocket",
                "Stock market closes higher",
                "Astronauts prepare for an orbital launch",
            ],
            "abstract": [
                "A rocket begins its journey into space.",
                "Technology and finance shares gained today.",
                "The space crew prepares for launch.",
            ],
            "category": ["science", "finance", "science"],
            "subcategory": ["space", "markets", "space"],
        }
    )
    behaviors = pd.DataFrame(
        {
            "impressions": [
                "N1-1 N2-0 N3-0",
                "N3-1 N2-0",
            ]
        }
    )

    content = ContentBasedRecommender().fit(news)
    popularity = PopularityRecommender().fit(behaviors)

    return ContentPopularityFallbackRecommender(content, popularity)


def make_metadata() -> ArtifactMetadata:
    """Return metadata for the small test artifact."""

    return ArtifactMetadata(
        artifact_version="0.3.0",
        package_version="0.2.0",
        model_name="tfidf_content_with_popularity_fallback",
        created_at_utc=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        training_cutoff=datetime.fromisoformat("2019-11-13T20:36:26"),
        training_records=2,
        indexed_article_count=3,
        vocabulary_article_count=3,
        vocabulary_size=31,
    )


def save_test_artifact(tmp_path: Path) -> Path:
    destination = tmp_path / "newslens-0.3.0"
    save_artifact(
        destination,
        model=make_model(),
        metadata=make_metadata(),
    )
    return destination


def checksum(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_save_artifact_writes_complete_directory(tmp_path: Path) -> None:
    destination = save_test_artifact(tmp_path)

    assert sorted(path.name for path in destination.iterdir()) == [
        "manifest.json",
        "metadata.json",
        "model.joblib",
    ]

    manifest = ArtifactManifest.model_validate_json(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest.artifact_version == "0.3.0"
    assert manifest.file("metadata.json").sha256 == checksum(destination / "metadata.json")
    assert manifest.file("model.joblib").sha256 == checksum(destination / "model.joblib")


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_model_round_trip_preserves_recommendations(tmp_path: Path) -> None:
    model = make_model()
    destination = tmp_path / "artifact"
    expected = model.recommend(
        ["N1"],
        candidate_news_ids=["N2", "N3"],
        top_k=2,
    )

    save_artifact(destination, model=model, metadata=make_metadata())
    loaded = load_artifact(destination)

    assert loaded.metadata == make_metadata()
    assert isinstance(
        loaded.model,
        ContentPopularityFallbackRecommender,
    )
    assert (
        loaded.model.recommend(
            ["N1"],
            candidate_news_ids=["N2", "N3"],
            top_k=2,
        )
        == expected
    )


def test_save_refuses_to_overwrite_existing_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(
        ArtifactAlreadyExistsError,
        match="already exists",
    ):
        save_artifact(
            destination,
            model=make_model(),
            metadata=make_metadata(),
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_serialization_failure_removes_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dump(model: object, path: Path) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(
        "newslens.artifacts.storage.joblib.dump",
        fail_dump,
    )

    with pytest.raises(ArtifactStorageError, match="Could not save"):
        save_artifact(
            tmp_path / "artifact",
            model=make_model(),
            metadata=make_metadata(),
        )

    assert list(tmp_path.iterdir()) == []


def test_load_rejects_missing_artifact_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArtifactNotFoundError, match="not found"):
        load_artifact(tmp_path / "missing")


def test_load_rejects_missing_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()

    with pytest.raises(
        ArtifactIntegrityError,
        match="manifest is missing",
    ):
        load_artifact(destination)


def test_load_rejects_missing_model_file(tmp_path: Path) -> None:
    destination = save_test_artifact(tmp_path)
    (destination / "model.joblib").unlink()

    with pytest.raises(ArtifactIntegrityError, match="model.joblib"):
        load_artifact(destination)


def test_load_rejects_changed_model_size(tmp_path: Path) -> None:
    destination = save_test_artifact(tmp_path)
    model_path = destination / "model.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"changed")

    with pytest.raises(ArtifactIntegrityError, match="size mismatch"):
        load_artifact(destination)


def test_load_rejects_changed_model_checksum(tmp_path: Path) -> None:
    destination = save_test_artifact(tmp_path)
    model_path = destination / "model.joblib"
    contents = bytearray(model_path.read_bytes())
    contents[-1] ^= 1
    model_path.write_bytes(contents)

    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        load_artifact(destination)


def test_load_rejects_changed_metadata(tmp_path: Path) -> None:
    destination = save_test_artifact(tmp_path)
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="size mismatch"):
        load_artifact(destination)


def test_load_rejects_invalid_manifest(tmp_path: Path) -> None:
    destination = save_test_artifact(tmp_path)
    (destination / "manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactIntegrityError,
        match="manifest is invalid",
    ):
        load_artifact(destination)


def test_load_rejects_metadata_manifest_version_mismatch(
    tmp_path: Path,
) -> None:
    destination = save_test_artifact(tmp_path)
    metadata_path = destination / "metadata.json"
    manifest_path = destination / "manifest.json"

    metadata = ArtifactMetadata.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    ).model_copy(update={"artifact_version": "0.3.1"})
    metadata_path.write_text(
        f"{metadata.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )

    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    records = tuple(
        ArtifactFileRecord(
            file_name=record.file_name,
            size_bytes=(destination / record.file_name).stat().st_size,
            sha256=checksum(destination / record.file_name),
        )
        for record in manifest.files
    )
    updated_manifest = ArtifactManifest(
        artifact_version=manifest.artifact_version,
        files=records,
    )
    manifest_path.write_text(
        f"{updated_manifest.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactIntegrityError,
        match="version mismatch",
    ):
        load_artifact(destination)
