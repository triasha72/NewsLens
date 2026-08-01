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

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "REQUIRED_ARTIFACT_FILES",
    "ArtifactFileRecord",
    "ArtifactManifest",
    "ArtifactMetadata",
    "TfidfArtifactParameters",
]
