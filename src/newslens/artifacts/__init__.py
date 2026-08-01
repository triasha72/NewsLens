"""Versioned model-artifact contracts for NewsLens."""

from .manifest import (
    REQUIRED_ARTIFACT_FILES,
    ArtifactFileRecord,
    ArtifactManifest,
)
from .metadata import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactMetadata,
    TfidfArtifactParameters,
)
from .storage import (
    MANIFEST_FILE_NAME,
    METADATA_FILE_NAME,
    MODEL_FILE_NAME,
    ArtifactAlreadyExistsError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactStorageError,
    LoadedArtifact,
    load_artifact,
    save_artifact,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MANIFEST_FILE_NAME",
    "METADATA_FILE_NAME",
    "MODEL_FILE_NAME",
    "REQUIRED_ARTIFACT_FILES",
    "ArtifactAlreadyExistsError",
    "ArtifactFileRecord",
    "ArtifactIntegrityError",
    "ArtifactManifest",
    "ArtifactMetadata",
    "ArtifactNotFoundError",
    "ArtifactStorageError",
    "LoadedArtifact",
    "TfidfArtifactParameters",
    "load_artifact",
    "save_artifact",
]
