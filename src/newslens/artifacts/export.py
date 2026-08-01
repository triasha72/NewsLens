"""Production training and export for the NewsLens fallback model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .. import __version__
from ..data import MindDataValidationError, parse_impressions
from ..models import (
    ContentBasedRecommender,
    ContentModelError,
    ContentPopularityFallbackRecommender,
    FallbackModelError,
    PopularityModelError,
    PopularityRecommender,
)
from .manifest import ArtifactManifest
from .metadata import ArtifactMetadata, TfidfArtifactParameters
from .storage import save_artifact


class ArtifactExportError(ValueError):
    """Raised when production model training or export cannot be completed."""


@dataclass(frozen=True)
class ArtifactExportResult:
    """Summary of one successfully exported production model."""

    path: Path
    metadata: ArtifactMetadata
    manifest: ArtifactManifest

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible export summary."""

        return {
            "artifact_path": str(self.path),
            "metadata": self.metadata.model_dump(mode="json"),
            "manifest": self.manifest.model_dump(mode="json"),
        }


def _prepare_catalog(news: pd.DataFrame) -> frozenset[str]:
    required_columns = {"news_id", "title"}
    missing_columns = required_columns.difference(news.columns)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise ArtifactExportError(f"Missing required article columns: {formatted}.")

    if news.empty:
        raise ArtifactExportError("At least one article is required.")

    news_ids = news["news_id"].fillna("").astype(str).str.strip()

    if news_ids.eq("").any():
        raise ArtifactExportError("Article identifiers cannot be empty.")

    duplicate_ids = news_ids[news_ids.duplicated(keep=False)].unique()

    if len(duplicate_ids) > 0:
        raise ArtifactExportError(
            f"Duplicate article identifiers found: {', '.join(duplicate_ids)}"
        )

    return frozenset(news_ids)


def _parse_history(value: object) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()

    return tuple(str(value).split())


def _training_article_ids(
    behaviors: pd.DataFrame,
    catalog: frozenset[str],
) -> frozenset[str]:
    required_columns = {
        "impression_id",
        "timestamp",
        "history",
        "impressions",
    }
    missing_columns = required_columns.difference(behaviors.columns)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise ArtifactExportError(f"Missing required behavior columns: {formatted}.")

    if behaviors.empty:
        raise ArtifactExportError("At least one behavior record is required.")

    referenced_ids: set[str] = set()

    for row in behaviors.itertuples(index=False):
        impression_id = str(row.impression_id)

        try:
            parsed_impressions = parse_impressions(str(row.impressions))
        except MindDataValidationError as error:
            raise ArtifactExportError(
                f"Invalid training impression '{impression_id}': {error}"
            ) from error

        candidate_ids = [news_id for news_id, _ in parsed_impressions]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ArtifactExportError(
                f"Training impression '{impression_id}' contains duplicate candidate IDs."
            )

        referenced_ids.update(_parse_history(row.history))
        referenced_ids.update(candidate_ids)

    unknown_ids = referenced_ids - catalog

    if unknown_ids:
        unknown_preview = ", ".join(sorted(unknown_ids)[:3])
        raise ArtifactExportError(
            f"Training interactions reference articles missing from the catalog: {unknown_preview}."
        )

    if not referenced_ids:
        raise ArtifactExportError("Training interactions did not reference any catalog articles.")

    return frozenset(referenced_ids)


def _training_cutoff(behaviors: pd.DataFrame) -> datetime:
    try:
        timestamps = pd.to_datetime(
            behaviors["timestamp"],
            format="mixed",
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ArtifactExportError("Training timestamps are invalid.") from error

    if timestamps.isna().any():
        raise ArtifactExportError("Training timestamps contain missing values.")

    return pd.Timestamp(timestamps.max()).to_pydatetime()


def export_fallback_artifact(
    news: pd.DataFrame,
    behaviors: pd.DataFrame,
    destination: str | Path,
    *,
    artifact_version: str = "0.3.0",
    ranking_cutoff: int = 10,
    max_features: int = 50_000,
    created_at_utc: datetime | None = None,
) -> ArtifactExportResult:
    """Fit on the full MIND-small training split and export one artifact."""

    catalog = _prepare_catalog(news)
    vocabulary_news_ids = _training_article_ids(behaviors, catalog)
    cutoff = _training_cutoff(behaviors)

    try:
        content_model = ContentBasedRecommender(max_features=max_features).fit(
            news,
            vocabulary_news_ids=vocabulary_news_ids,
        )
        popularity_model = PopularityRecommender().fit(behaviors)
        model = ContentPopularityFallbackRecommender(
            content_model,
            popularity_model,
        )
    except (
        ContentModelError,
        PopularityModelError,
        FallbackModelError,
    ) as error:
        raise ArtifactExportError(str(error)) from error

    metadata = ArtifactMetadata(
        artifact_version=artifact_version,
        package_version=__version__,
        model_name="tfidf_content_with_popularity_fallback",
        created_at_utc=created_at_utc or datetime.now(UTC),
        training_cutoff=cutoff,
        training_records=len(behaviors),
        indexed_article_count=content_model.indexed_article_count,
        vocabulary_article_count=(content_model.vocabulary_article_count),
        vocabulary_size=content_model.vocabulary_size,
        ranking_cutoff=ranking_cutoff,
        tfidf=TfidfArtifactParameters(max_features=max_features),
    )

    artifact_path = Path(destination).expanduser()
    manifest = save_artifact(
        artifact_path,
        model=model,
        metadata=metadata,
    )

    return ArtifactExportResult(
        path=artifact_path.resolve(),
        metadata=metadata,
        manifest=manifest,
    )
