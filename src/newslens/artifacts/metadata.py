"""Typed metadata contract for versioned NewsLens model artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

ARTIFACT_SCHEMA_VERSION = "1.0"
SEMANTIC_VERSION_PATTERN = (
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


class TfidfArtifactParameters(BaseModel):
    """TF-IDF configuration required to interpret a saved model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stop_words: Literal["english"] = "english"
    ngram_range: tuple[PositiveInt, PositiveInt] = (1, 2)
    min_document_frequency: PositiveInt = 1
    max_features: PositiveInt = 50_000
    sublinear_term_frequency: bool = True
    normalization: Literal["l2"] = "l2"

    @model_validator(mode="after")
    def validate_ngram_range(self) -> Self:
        """Require the lower n-gram bound to precede the upper bound."""

        lower, upper = self.ngram_range

        if lower > upper:
            raise ValueError("ngram_range lower bound cannot exceed its upper bound.")

        return self


class ArtifactMetadata(BaseModel):
    """Reproducibility and compatibility metadata for one model artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = ARTIFACT_SCHEMA_VERSION
    artifact_version: str = Field(pattern=SEMANTIC_VERSION_PATTERN)
    package_version: str = Field(pattern=SEMANTIC_VERSION_PATTERN)
    model_name: Literal["tfidf_content_with_popularity_fallback"]
    created_at_utc: datetime
    dataset_name: Literal["MIND-small"] = "MIND-small"
    source_split: Literal["train"] = "train"
    training_cutoff: datetime
    training_records: PositiveInt
    indexed_article_count: PositiveInt
    vocabulary_article_count: PositiveInt
    vocabulary_size: PositiveInt
    ranking_cutoff: PositiveInt = 10
    tfidf: TfidfArtifactParameters = Field(default_factory=TfidfArtifactParameters)
    content_profile_method: Literal["mean_history_tfidf"] = "mean_history_tfidf"
    fallback_policy: Literal["training_click_count_popularity"] = "training_click_count_popularity"

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_at_utc(cls, value: datetime) -> datetime:
        """Require an explicit UTC timestamp for artifact creation."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must include UTC timezone information.")

        if value.utcoffset() != timedelta(0):
            raise ValueError("created_at_utc must use a UTC offset of +00:00.")

        return value

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Validate relationships among recorded model dimensions."""

        if self.vocabulary_article_count > self.indexed_article_count:
            raise ValueError("vocabulary_article_count cannot exceed indexed_article_count.")

        if self.vocabulary_size > self.tfidf.max_features:
            raise ValueError("vocabulary_size cannot exceed tfidf.max_features.")

        return self
