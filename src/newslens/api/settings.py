"""Environment-backed runtime settings for the NewsLens API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ARTIFACT_PATH_ENVIRONMENT_VARIABLE = "NEWSLENS_ARTIFACT_PATH"


class ApiSettingsError(ValueError):
    """Raised when API runtime configuration is invalid."""


@dataclass(frozen=True)
class ApiSettings:
    """Configuration required to start the NewsLens API."""

    artifact_path: Path | None = None

    @classmethod
    def from_environment(cls) -> ApiSettings:
        """Build settings from process environment variables."""

        raw_artifact_path = os.getenv(ARTIFACT_PATH_ENVIRONMENT_VARIABLE)

        if raw_artifact_path is None:
            return cls()

        normalized_path = raw_artifact_path.strip()

        if not normalized_path:
            raise ApiSettingsError(f"{ARTIFACT_PATH_ENVIRONMENT_VARIABLE} cannot be empty.")

        return cls(artifact_path=Path(normalized_path).expanduser())
