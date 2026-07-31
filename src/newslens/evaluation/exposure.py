"""Model-independent ranking evaluation by training-item exposure."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .evaluator import (
    RankingEvaluationError,
    RankingEvaluationResult,
    RankingExample,
    evaluate_rankings,
)


class ExposureEvaluationError(ValueError):
    """Raised when training-exposure evaluation cannot be completed."""


@dataclass(frozen=True)
class TrainingExposureBand:
    """One inclusive interval of chronological-training candidate exposures."""

    name: str
    minimum: int
    maximum: int | None

    def __post_init__(self) -> None:
        name = str(self.name).strip()

        if not name:
            raise ExposureEvaluationError("Exposure-band name must not be empty.")

        if isinstance(self.minimum, bool) or not isinstance(self.minimum, int):
            raise ExposureEvaluationError("Exposure-band minimum must be a non-negative integer.")

        if self.minimum < 0:
            raise ExposureEvaluationError("Exposure-band minimum must be a non-negative integer.")

        if self.maximum is not None:
            if isinstance(self.maximum, bool) or not isinstance(self.maximum, int):
                raise ExposureEvaluationError(
                    "Exposure-band maximum must be a non-negative integer or None."
                )

            if self.maximum < self.minimum:
                raise ExposureEvaluationError(
                    "Exposure-band maximum must be greater than or equal to its minimum."
                )

        object.__setattr__(self, "name", name)

    def contains(self, exposures: int) -> bool:
        """Return whether a training-exposure count belongs to this band."""

        return exposures >= self.minimum and (self.maximum is None or exposures <= self.maximum)

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-compatible band definition."""

        return {
            "name": self.name,
            "minimum_training_exposures": self.minimum,
            "maximum_training_exposures": self.maximum,
        }


DEFAULT_TRAINING_EXPOSURE_BANDS: tuple[TrainingExposureBand, ...] = (
    TrainingExposureBand("unseen", 0, 0),
    TrainingExposureBand("low_exposure", 1, 9),
    TrainingExposureBand("medium_exposure", 10, 99),
    TrainingExposureBand("high_exposure", 100, None),
)


@dataclass(frozen=True)
class ExposureBandResult:
    """Relevance and recommendation results for one exposure band."""

    definition: TrainingExposureBand
    catalog_articles: int
    relevant_impressions: int
    relevant_impression_fraction: float
    relevant_item_occurrences: int
    unique_relevant_items: int
    recommended_occurrences_at_k: int
    unique_recommended_items_at_k: int
    catalog_coverage_at_k: float
    meets_minimum_support: bool
    metrics: RankingEvaluationResult | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible exposure-band result."""

        return {
            **self.definition.to_dict(),
            "catalog_articles": self.catalog_articles,
            "relevant_impressions": self.relevant_impressions,
            "relevant_impression_fraction": self.relevant_impression_fraction,
            "relevant_item_occurrences": self.relevant_item_occurrences,
            "unique_relevant_items": self.unique_relevant_items,
            "recommended_occurrences_at_k": self.recommended_occurrences_at_k,
            "unique_recommended_items_at_k": self.unique_recommended_items_at_k,
            "catalog_coverage_at_k": self.catalog_coverage_at_k,
            "meets_minimum_support": self.meets_minimum_support,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class ExposureEvaluationReport:
    """Overall metrics and overlapping clicked-item exposure cohorts."""

    overall_metrics: RankingEvaluationResult
    minimum_relevant_impressions: int
    clicked_impressions: int
    multi_band_clicked_impressions: int
    impression_band_pairs: int
    bands: tuple[ExposureBandResult, ...]

    @property
    def k(self) -> int:
        """Return the ranking cutoff used throughout the report."""

        return self.overall_metrics.k

    @property
    def total_impressions(self) -> int:
        """Return the number of input impressions."""

        return self.overall_metrics.total_impressions

    @property
    def band_membership_is_overlapping(self) -> bool:
        """Return whether any impression belongs to multiple exposure bands."""

        return self.multi_band_clicked_impressions > 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible evaluation report."""

        return {
            "k": self.k,
            "total_impressions": self.total_impressions,
            "clicked_impressions": self.clicked_impressions,
            "minimum_relevant_impressions": self.minimum_relevant_impressions,
            "multi_band_clicked_impressions": self.multi_band_clicked_impressions,
            "impression_band_pairs": self.impression_band_pairs,
            "band_membership_is_overlapping": self.band_membership_is_overlapping,
            "overall_metrics": self.overall_metrics.to_dict(),
            "bands": [band.to_dict() for band in self.bands],
        }


def _validate_band_definitions(
    definitions: tuple[TrainingExposureBand, ...],
) -> None:
    if not definitions:
        raise ExposureEvaluationError("At least one exposure band is required.")

    names = [definition.name for definition in definitions]

    if len(names) != len(set(names)):
        raise ExposureEvaluationError("Exposure-band names must be unique.")

    if definitions[0].minimum != 0:
        raise ExposureEvaluationError("Exposure bands must begin at zero.")

    for index, definition in enumerate(definitions):
        is_last = index == len(definitions) - 1

        if definition.maximum is None and not is_last:
            raise ExposureEvaluationError("Only the final exposure band may have no maximum.")

        if is_last:
            if definition.maximum is not None:
                raise ExposureEvaluationError("The final exposure band must have no maximum.")
            continue

        next_definition = definitions[index + 1]
        expected_minimum = definition.maximum + 1  # type: ignore[operator]

        if next_definition.minimum != expected_minimum:
            raise ExposureEvaluationError(
                "Exposure bands must be ordered, contiguous, and non-overlapping."
            )


def _normalize_catalog(catalog_items: Iterable[str]) -> frozenset[str]:
    catalog_values = list(catalog_items)

    if not catalog_values:
        raise ExposureEvaluationError("At least one catalog item is required.")

    if any(not isinstance(item, str) or not item.strip() for item in catalog_values):
        raise ExposureEvaluationError("Catalog items must be non-empty string identifiers.")

    normalized = [item.strip() for item in catalog_values]

    if len(normalized) != len(set(normalized)):
        raise ExposureEvaluationError("Catalog item identifiers must be unique.")

    return frozenset(normalized)


def _normalize_training_exposures(
    training_exposures: Mapping[str, int],
    catalog: frozenset[str],
) -> dict[str, int]:
    normalized: dict[str, int] = {}

    for raw_item_id, exposures in training_exposures.items():
        if not isinstance(raw_item_id, str) or not raw_item_id.strip():
            raise ExposureEvaluationError(
                "Training-exposure mappings must use non-empty string item IDs."
            )

        item_id = raw_item_id.strip()

        if item_id in normalized:
            raise ExposureEvaluationError(
                f"Training-exposure mapping contains duplicate item ID: {item_id}."
            )

        if isinstance(exposures, bool) or not isinstance(exposures, int) or exposures < 0:
            raise ExposureEvaluationError("Training-exposure counts must be non-negative integers.")

        if item_id not in catalog:
            raise ExposureEvaluationError(
                f"Training-exposure item is missing from the catalog: {item_id}."
            )

        normalized[item_id] = exposures

    return {item_id: normalized.get(item_id, 0) for item_id in catalog}


def evaluate_training_exposure_bands(
    examples: Iterable[RankingExample],
    catalog_items: Iterable[str],
    training_exposures: Mapping[str, int],
    *,
    k: int,
    bands: Iterable[TrainingExposureBand] = DEFAULT_TRAINING_EXPOSURE_BANDS,
    minimum_relevant_impressions: int = 1,
) -> ExposureEvaluationReport:
    """Evaluate rankings in cohorts defined by clicked-item training exposure.

    Items absent from ``training_exposures`` are assigned zero exposures. An
    impression belongs to every band represented by its relevant items. Band
    quality metrics preserve original global ranking positions.
    """

    if (
        isinstance(minimum_relevant_impressions, bool)
        or not isinstance(minimum_relevant_impressions, int)
        or minimum_relevant_impressions <= 0
    ):
        raise ExposureEvaluationError("minimum_relevant_impressions must be a positive integer.")

    example_list = list(examples)

    if not example_list:
        raise ExposureEvaluationError("At least one ranking example is required.")

    if any(not isinstance(example, RankingExample) for example in example_list):
        raise ExposureEvaluationError("examples must contain only RankingExample values.")

    definitions = tuple(bands)
    _validate_band_definitions(definitions)
    catalog = _normalize_catalog(catalog_items)
    exposure_by_item = _normalize_training_exposures(training_exposures, catalog)
    referenced_items = {
        item_id
        for example in example_list
        for item_id in (*example.ranked_items, *example.relevant_items)
    }
    missing_items = referenced_items - catalog

    if missing_items:
        preview = ", ".join(sorted(missing_items)[:3])
        raise ExposureEvaluationError(
            f"Ranking examples reference items missing from the catalog: {preview}."
        )

    try:
        overall_metrics = evaluate_rankings(example_list, catalog, k=k)
    except RankingEvaluationError as error:
        raise ExposureEvaluationError(str(error)) from error

    item_band: dict[str, TrainingExposureBand] = {}

    for item_id, exposure_count in exposure_by_item.items():
        matching_bands = [
            definition for definition in definitions if definition.contains(exposure_count)
        ]

        if len(matching_bands) != 1:
            raise ExposureEvaluationError(
                "Exposure-band definitions did not assign every catalog item exactly once."
            )

        item_band[item_id] = matching_bands[0]

    relevant_bands_by_impression = [
        {item_band[item_id].name for item_id in example.relevant_items} for example in example_list
    ]
    clicked_impressions = sum(bool(names) for names in relevant_bands_by_impression)
    multi_band_clicked_impressions = sum(len(names) > 1 for names in relevant_bands_by_impression)
    impression_band_pairs = sum(len(names) for names in relevant_bands_by_impression)
    results: list[ExposureBandResult] = []

    for definition in definitions:
        band_catalog = {
            item_id for item_id, assigned_band in item_band.items() if assigned_band == definition
        }
        band_examples: list[RankingExample] = []
        relevant_items: list[str] = []

        for example in example_list:
            band_relevance = frozenset(
                item_id for item_id in example.relevant_items if item_band[item_id] == definition
            )

            if not band_relevance:
                continue

            relevant_items.extend(band_relevance)
            band_examples.append(
                RankingExample(
                    impression_id=example.impression_id,
                    ranked_items=example.ranked_items,
                    relevant_items=band_relevance,
                )
            )

        metrics: RankingEvaluationResult | None = None

        if band_examples:
            try:
                metrics = evaluate_rankings(band_examples, catalog, k=k)
            except RankingEvaluationError as error:
                raise ExposureEvaluationError(str(error)) from error

        recommended_items = [
            item_id
            for example in example_list
            for item_id in example.ranked_items[:k]
            if item_band[item_id] == definition
        ]
        unique_recommended_items = set(recommended_items)
        coverage = len(unique_recommended_items) / len(band_catalog) if band_catalog else 0.0

        results.append(
            ExposureBandResult(
                definition=definition,
                catalog_articles=len(band_catalog),
                relevant_impressions=len(band_examples),
                relevant_impression_fraction=(len(band_examples) / clicked_impressions),
                relevant_item_occurrences=len(relevant_items),
                unique_relevant_items=len(set(relevant_items)),
                recommended_occurrences_at_k=len(recommended_items),
                unique_recommended_items_at_k=len(unique_recommended_items),
                catalog_coverage_at_k=coverage,
                meets_minimum_support=(len(band_examples) >= minimum_relevant_impressions),
                metrics=metrics,
            )
        )

    return ExposureEvaluationReport(
        overall_metrics=overall_metrics,
        minimum_relevant_impressions=minimum_relevant_impressions,
        clicked_impressions=clicked_impressions,
        multi_band_clicked_impressions=multi_band_clicked_impressions,
        impression_band_pairs=impression_band_pairs,
        bands=tuple(results),
    )
