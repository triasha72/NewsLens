"""File manifest contract for versioned NewsLens model artifacts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from .metadata import ARTIFACT_SCHEMA_VERSION, SEMANTIC_VERSION_PATTERN

SHA256_PATTERN = r"^[0-9a-f]{64}$"
REQUIRED_ARTIFACT_FILES = frozenset({"metadata.json", "model.joblib"})


class ArtifactFileRecord(BaseModel):
    """Integrity information for one file in an artifact directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_name: str
    size_bytes: PositiveInt
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, value: str) -> str:
        """Require a nonempty file name without directory traversal."""

        normalized = value.strip()

        if not normalized:
            raise ValueError("file_name cannot be empty.")

        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError("file_name must be a relative base name without directories.")

        return normalized


class ArtifactManifest(BaseModel):
    """Validated list of files belonging to one NewsLens artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = ARTIFACT_SCHEMA_VERSION
    artifact_version: str = Field(pattern=SEMANTIC_VERSION_PATTERN)
    files: tuple[ArtifactFileRecord, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        """Require unique entries and the minimum artifact file set."""

        file_names = [record.file_name for record in self.files]

        if len(file_names) != len(set(file_names)):
            raise ValueError("Artifact manifest file names must be unique.")

        missing = REQUIRED_ARTIFACT_FILES.difference(file_names)

        if missing:
            formatted = ", ".join(sorted(missing))
            raise ValueError(f"Artifact manifest is missing required files: {formatted}.")

        return self

    def file(self, file_name: str) -> ArtifactFileRecord:
        """Return the manifest entry for ``file_name``."""

        for record in self.files:
            if record.file_name == file_name:
                return record

        raise KeyError(f"Artifact manifest does not contain '{file_name}'.")
