"""Atomic storage and integrity verification for NewsLens artifacts."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
from pydantic import ValidationError

from .manifest import ArtifactFileRecord, ArtifactManifest
from .metadata import ArtifactMetadata

METADATA_FILE_NAME = "metadata.json"
MODEL_FILE_NAME = "model.joblib"
MANIFEST_FILE_NAME = "manifest.json"
CHECKSUM_CHUNK_SIZE = 1024 * 1024


class ArtifactStorageError(RuntimeError):
    """Base exception for artifact storage failures."""


class ArtifactAlreadyExistsError(ArtifactStorageError):
    """Raised when a save target already exists."""


class ArtifactNotFoundError(ArtifactStorageError):
    """Raised when an artifact directory cannot be found."""


class ArtifactIntegrityError(ArtifactStorageError):
    """Raised when an artifact is incomplete, invalid, or corrupted."""


@dataclass(frozen=True)
class LoadedArtifact:
    """A validated metadata contract and its deserialized model."""

    path: Path
    model: object
    metadata: ArtifactMetadata
    manifest: ArtifactManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(CHECKSUM_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def _file_record(path: Path) -> ArtifactFileRecord:
    return ArtifactFileRecord(
        file_name=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _write_json(path: Path, payload: str) -> None:
    path.write_text(f"{payload}\n", encoding="utf-8")


def save_artifact(
    destination: str | Path,
    *,
    model: object,
    metadata: ArtifactMetadata,
) -> ArtifactManifest:
    """Save one artifact using a same-filesystem atomic directory rename.

    Existing destinations are never overwritten. The model is written to a
    temporary sibling directory, followed by metadata and a checksum manifest.
    The completed directory becomes visible only after every write succeeds.
    """

    target = Path(destination).expanduser()

    if target.exists() or target.is_symlink():
        raise ArtifactAlreadyExistsError(f"Artifact destination already exists: {target}")

    parent = target.parent
    staging: Path | None = None

    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}-",
                suffix=".tmp",
                dir=parent,
            )
        )

        metadata_path = staging / METADATA_FILE_NAME
        model_path = staging / MODEL_FILE_NAME
        manifest_path = staging / MANIFEST_FILE_NAME

        joblib.dump(model, model_path)
        _write_json(metadata_path, metadata.model_dump_json(indent=2))

        manifest = ArtifactManifest(
            artifact_version=metadata.artifact_version,
            files=(
                _file_record(metadata_path),
                _file_record(model_path),
            ),
        )
        _write_json(manifest_path, manifest.model_dump_json(indent=2))

        if target.exists() or target.is_symlink():
            raise ArtifactAlreadyExistsError(f"Artifact destination already exists: {target}")

        staging.rename(target)
        staging = None

        return manifest
    except ArtifactStorageError:
        raise
    except Exception as error:
        raise ArtifactStorageError(f"Could not save artifact to {target}.") from error
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _read_manifest(path: Path) -> ArtifactManifest:
    if not path.is_file() or path.is_symlink():
        raise ArtifactIntegrityError(f"Artifact manifest is missing or invalid: {path}")

    try:
        return ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ArtifactIntegrityError(f"Artifact manifest is invalid: {path}") from error


def _verify_file(root: Path, record: ArtifactFileRecord) -> Path:
    path = root / record.file_name

    if not path.is_file() or path.is_symlink():
        raise ArtifactIntegrityError(f"Artifact file is missing or invalid: {record.file_name}")

    actual_size = path.stat().st_size

    if actual_size != record.size_bytes:
        raise ArtifactIntegrityError(
            f"Artifact file size mismatch for {record.file_name}: "
            f"expected {record.size_bytes}, found {actual_size}."
        )

    actual_checksum = _sha256(path)

    if actual_checksum != record.sha256:
        raise ArtifactIntegrityError(f"Artifact checksum mismatch for {record.file_name}.")

    return path


def _read_metadata(path: Path) -> ArtifactMetadata:
    try:
        return ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ArtifactIntegrityError(f"Artifact metadata is invalid: {path}") from error


def load_artifact(source: str | Path) -> LoadedArtifact:
    """Verify and load a trusted NewsLens artifact.

    Joblib uses pickle internally. Checksums detect accidental modification but
    do not make an untrusted pickle safe. Only load artifacts created by a
    trusted NewsLens training workflow.
    """

    root = Path(source).expanduser()

    if not root.is_dir() or root.is_symlink():
        raise ArtifactNotFoundError(f"Artifact directory was not found: {root}")

    manifest = _read_manifest(root / MANIFEST_FILE_NAME)

    verified_files = {record.file_name: _verify_file(root, record) for record in manifest.files}
    metadata = _read_metadata(verified_files[METADATA_FILE_NAME])

    if metadata.artifact_version != manifest.artifact_version:
        raise ArtifactIntegrityError("Artifact version mismatch between metadata and manifest.")

    try:
        model = joblib.load(verified_files[MODEL_FILE_NAME])
    except Exception as error:
        raise ArtifactIntegrityError("The verified model payload could not be loaded.") from error

    return LoadedArtifact(
        path=root.resolve(),
        model=model,
        metadata=metadata,
        manifest=manifest,
    )
