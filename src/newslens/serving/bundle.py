"""Utilities for Phase-07 serving bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ServingBundleError(RuntimeError):
    """Raised when a serving bundle fails validation."""


def sha256_file(
    path: Path,
) -> str:
    """Return SHA-256 for one file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Load one JSON object."""

    payload = json.loads(
        path.read_text()
    )

    if not isinstance(payload, dict):
        raise ServingBundleError(
            f"{path} does not contain a JSON object."
        )

    return payload


def verify_file_sha256(
    path: Path,
    expected_sha256: str,
) -> None:
    """Require one file to match its frozen fingerprint."""

    if not path.is_file():
        raise ServingBundleError(
            f"Required serving file does not exist: {path}"
        )

    actual = sha256_file(path)

    if actual != expected_sha256:
        raise ServingBundleError(
            f"SHA-256 mismatch for {path.name}: "
            f"expected {expected_sha256}, received {actual}."
        )
