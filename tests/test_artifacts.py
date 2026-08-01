"""Tests for versioned NewsLens model-artifact contracts."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from newslens.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactFileRecord,
    ArtifactManifest,
    ArtifactMetadata,
    TfidfArtifactParameters,
)

CHECKSUM_A = "a" * 64
CHECKSUM_B = "b" * 64


def make_metadata(**updates: object) -> ArtifactMetadata:
    """Return valid metadata with selected field overrides."""

    payload: dict[str, object] = {
        "artifact_version": "0.3.0",
        "package_version": "0.2.0",
        "model_name": "tfidf_content_with_popularity_fallback",
        "created_at_utc": datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        "training_cutoff": datetime.fromisoformat("2019-11-13T20:36:26"),
        "training_records": 125_572,
        "indexed_article_count": 51_282,
        "vocabulary_article_count": 47_367,
        "vocabulary_size": 50_000,
    }
    payload.update(updates)
    return ArtifactMetadata.model_validate(payload)


def make_manifest(*records: ArtifactFileRecord) -> ArtifactManifest:
    """Return a valid manifest, using default required records when omitted."""

    selected_records = records or (
        ArtifactFileRecord(
            file_name="metadata.json",
            size_bytes=1_024,
            sha256=CHECKSUM_A,
        ),
        ArtifactFileRecord(
            file_name="model.joblib",
            size_bytes=2_048,
            sha256=CHECKSUM_B,
        ),
    )

    return ArtifactManifest(
        artifact_version="0.3.0",
        files=selected_records,
    )


def test_metadata_records_selected_model_contract() -> None:
    metadata = make_metadata()

    assert metadata.schema_version == ARTIFACT_SCHEMA_VERSION
    assert metadata.model_name == "tfidf_content_with_popularity_fallback"
    assert metadata.training_records == 125_572
    assert metadata.ranking_cutoff == 10
    assert metadata.tfidf.ngram_range == (1, 2)


def test_metadata_json_round_trip_is_lossless() -> None:
    metadata = make_metadata()

    restored = ArtifactMetadata.model_validate_json(metadata.model_dump_json())

    assert restored == metadata


def test_metadata_is_frozen() -> None:
    metadata = make_metadata()

    with pytest.raises(ValidationError, match="frozen"):
        metadata.ranking_cutoff = 20


@pytest.mark.parametrize("version", ["1", "v1.0.0", "1.0", "1.0.x", " 1.0.0"])
def test_metadata_rejects_invalid_semantic_versions(version: str) -> None:
    with pytest.raises(ValidationError):
        make_metadata(artifact_version=version)


def test_metadata_rejects_unsupported_schema_version() -> None:
    payload = make_metadata().model_dump()
    payload["schema_version"] = "2.0"

    with pytest.raises(ValidationError, match="1.0"):
        ArtifactMetadata.model_validate(payload)


def test_metadata_requires_utc_creation_timestamp() -> None:
    non_utc = datetime(
        2026,
        8,
        1,
        12,
        0,
        tzinfo=timezone(timedelta(hours=-4)),
    )

    with pytest.raises(ValidationError, match="UTC offset"):
        make_metadata(created_at_utc=non_utc)


def test_metadata_rejects_more_vocabulary_articles_than_indexed_articles() -> None:
    with pytest.raises(ValidationError, match="vocabulary_article_count"):
        make_metadata(
            indexed_article_count=100,
            vocabulary_article_count=101,
        )


def test_metadata_rejects_vocabulary_larger_than_configured_limit() -> None:
    with pytest.raises(ValidationError, match="vocabulary_size"):
        make_metadata(vocabulary_size=50_001)


def test_tfidf_parameters_reject_reversed_ngram_range() -> None:
    with pytest.raises(ValidationError, match="lower bound"):
        TfidfArtifactParameters(ngram_range=(2, 1))


def test_manifest_records_required_files() -> None:
    manifest = make_manifest()

    assert manifest.schema_version == ARTIFACT_SCHEMA_VERSION
    assert manifest.file("model.joblib").size_bytes == 2_048


def test_manifest_json_round_trip_is_lossless() -> None:
    manifest = make_manifest()

    restored = ArtifactManifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest


def test_manifest_rejects_duplicate_file_names() -> None:
    metadata_record = ArtifactFileRecord(
        file_name="metadata.json",
        size_bytes=1,
        sha256=CHECKSUM_A,
    )

    with pytest.raises(ValidationError, match="unique"):
        make_manifest(
            metadata_record,
            metadata_record,
            ArtifactFileRecord(
                file_name="model.joblib",
                size_bytes=1,
                sha256=CHECKSUM_B,
            ),
        )


def test_manifest_rejects_missing_required_file() -> None:
    with pytest.raises(ValidationError, match="model.joblib"):
        ArtifactManifest(
            artifact_version="0.3.0",
            files=(
                ArtifactFileRecord(
                    file_name="metadata.json",
                    size_bytes=1,
                    sha256=CHECKSUM_A,
                ),
                ArtifactFileRecord(
                    file_name="notes.txt",
                    size_bytes=1,
                    sha256=CHECKSUM_B,
                ),
            ),
        )


@pytest.mark.parametrize(
    "file_name",
    ["", "../model.joblib", r"..\model.joblib", "parts/model.joblib", "."],
)
def test_file_record_rejects_unsafe_names(file_name: str) -> None:
    with pytest.raises(ValidationError, match="file_name"):
        ArtifactFileRecord(
            file_name=file_name,
            size_bytes=1,
            sha256=CHECKSUM_A,
        )


@pytest.mark.parametrize("checksum", ["", "a" * 63, "A" * 64, "g" * 64])
def test_file_record_rejects_invalid_sha256(checksum: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactFileRecord(
            file_name="metadata.json",
            size_bytes=1,
            sha256=checksum,
        )


def test_manifest_lookup_rejects_unknown_file() -> None:
    with pytest.raises(KeyError, match="missing.bin"):
        make_manifest().file("missing.bin")
