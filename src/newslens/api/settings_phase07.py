"""Environment configuration for Phase-07 production serving."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SERVING_BUNDLE_ENV = "NEWSLENS_SERVING_BUNDLE_PATH"


@dataclass(frozen=True, slots=True)
class ProductionApiSettings:
    """Production API configuration."""

    serving_bundle_path: Path | None = None

    @classmethod
    def from_environment(
        cls,
    ) -> ProductionApiSettings:
        raw = os.getenv(
            SERVING_BUNDLE_ENV
        )

        if raw is None:
            return cls()

        normalized = raw.strip()

        if not normalized:
            raise ValueError(
                f"{SERVING_BUNDLE_ENV} cannot be empty."
            )

        return cls(
            serving_bundle_path=Path(
                normalized
            ).expanduser()
        )
