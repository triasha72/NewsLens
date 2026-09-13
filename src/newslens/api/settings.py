"""Environment-backed runtime settings for the NewsLens API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ARTIFACT_PATH_ENVIRONMENT_VARIABLE = "NEWSLENS_ARTIFACT_PATH"
REALTIME_DATABASE_URL_ENVIRONMENT_VARIABLE = "NEWSLENS_REALTIME_DATABASE_URL"
REALTIME_CANDIDATE_LIMIT_ENVIRONMENT_VARIABLE = "NEWSLENS_REALTIME_CANDIDATE_LIMIT"
DEMO_MODE_ENVIRONMENT_VARIABLE = "NEWSLENS_DEMO_MODE"


class ApiSettingsError(ValueError):
    """Raised when API runtime configuration is invalid."""


@dataclass(frozen=True)
class ApiSettings:
    """Configuration required to start the NewsLens API."""

    artifact_path: Path | None = None
    realtime_database_url: str | None = None
    realtime_candidate_limit: int = 200
    demo_mode: bool = False

    @classmethod
    def from_environment(cls) -> ApiSettings:
        """Build settings from process environment variables."""

        raw_artifact_path = os.getenv(ARTIFACT_PATH_ENVIRONMENT_VARIABLE)

        release_root = os.getenv("NEWSLENS_RELEASE_ROOT")
        if release_root is not None:
            if not release_root.strip() or raw_artifact_path is not None:
                raise ApiSettingsError(
                    "Set one nonempty NEWSLENS_RELEASE_ROOT or NEWSLENS_ARTIFACT_PATH."
                )
            from newslens.operations.lifecycle import serving_path

            raw_artifact_path = str(serving_path(Path(release_root).expanduser()))

        raw_database_url = os.getenv(REALTIME_DATABASE_URL_ENVIRONMENT_VARIABLE)
        raw_candidate_limit = os.getenv(REALTIME_CANDIDATE_LIMIT_ENVIRONMENT_VARIABLE)
        raw_demo_mode = os.getenv(DEMO_MODE_ENVIRONMENT_VARIABLE)

        artifact_path: Path | None = None

        if raw_artifact_path is not None:
            normalized_path = raw_artifact_path.strip()

            if not normalized_path:
                raise ApiSettingsError(f"{ARTIFACT_PATH_ENVIRONMENT_VARIABLE} cannot be empty.")

            artifact_path = Path(normalized_path).expanduser()

        database_url: str | None = None

        if raw_database_url is not None:
            database_url = raw_database_url.strip()

            if not database_url:
                raise ApiSettingsError(
                    f"{REALTIME_DATABASE_URL_ENVIRONMENT_VARIABLE} cannot be empty."
                )

        candidate_limit = 200

        if raw_candidate_limit is not None:
            try:
                candidate_limit = int(raw_candidate_limit)
            except ValueError as error:
                raise ApiSettingsError(
                    f"{REALTIME_CANDIDATE_LIMIT_ENVIRONMENT_VARIABLE} must be an integer."
                ) from error

            if candidate_limit <= 0 or candidate_limit > 10_000:
                raise ApiSettingsError(
                    f"{REALTIME_CANDIDATE_LIMIT_ENVIRONMENT_VARIABLE} must be between 1 and 10000."
                )

        demo_mode = False
        if raw_demo_mode is not None:
            normalized_demo_mode = raw_demo_mode.strip().lower()
            if normalized_demo_mode not in {"true", "false"}:
                raise ApiSettingsError(
                    f"{DEMO_MODE_ENVIRONMENT_VARIABLE} must be true or false."
                )
            demo_mode = normalized_demo_mode == "true"

        if demo_mode and database_url is not None:
            raise ApiSettingsError(
                f"{DEMO_MODE_ENVIRONMENT_VARIABLE} cannot be combined with "
                f"{REALTIME_DATABASE_URL_ENVIRONMENT_VARIABLE}."
            )

        return cls(
            artifact_path=artifact_path,
            realtime_database_url=database_url,
            realtime_candidate_limit=candidate_limit,
            demo_mode=demo_mode,
        )
