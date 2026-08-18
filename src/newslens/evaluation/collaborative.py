"""Chronological offline evaluation for BPR collaborative filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from newslens.data.recsys_training import load_bpr_triples
from newslens.models.collaborative import CollaborativeRecommender

from .evaluator import RankingEvaluationResult, RankingExample, evaluate_rankings


class CollaborativeEvaluationError(ValueError):
    """Raised when collaborative-filtering evaluation cannot be completed."""


@dataclass(frozen=True, slots=True)
class ValidationImpression:
    """One validation impression reconstructed from the warehouse."""

    impression_id: str
    user_id: str
    candidate_news_ids: tuple[str, ...]
    relevant_items: frozenset[str]


@dataclass(frozen=True)
class CollaborativeEvaluationReport:
    """Metrics and cold-start accounting for one BPR evaluation."""

    model_name: str
    cutoff_timestamp: str
    k: int
    embedding_dim: int
    epochs: int
    seed: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_negatives_per_positive: int
    training_triples: int
    training_users: int
    training_items: int
    validation_impressions: int
    candidate_occurrences: int
    model_known_candidate_occurrences: int
    unknown_candidate_occurrences: int
    cold_start_user_impressions: int
    known_user_impressions: int
    impressions_with_no_model_candidates: int
    clicked_item_occurrences: int
    unknown_clicked_item_occurrences: int
    metrics: RankingEvaluationResult

    @property
    def cold_start_user_fraction(self) -> float:
        if self.validation_impressions == 0:
            return 0.0

        return self.cold_start_user_impressions / self.validation_impressions

    @property
    def unknown_candidate_fraction(self) -> float:
        if self.candidate_occurrences == 0:
            return 0.0

        return self.unknown_candidate_occurrences / self.candidate_occurrences

    @property
    def unknown_clicked_item_fraction(self) -> float:
        if self.clicked_item_occurrences == 0:
            return 0.0

        return (
            self.unknown_clicked_item_occurrences
            / self.clicked_item_occurrences
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report."""

        return {
            "model_name": self.model_name,
            "protocol": {
                "cutoff_timestamp": self.cutoff_timestamp,
                "training_boundary": "event_timestamp < cutoff",
                "validation_boundary": "event_timestamp >= cutoff",
                "candidate_set_evaluation": True,
                "cold_start_user_policy": "empty ranking; no fallback",
                "unknown_item_policy": (
                    "not scored by BPR; retained in validation relevance"
                ),
            },
            "hyperparameters": {
                "k": self.k,
                "embedding_dim": self.embedding_dim,
                "epochs": self.epochs,
                "seed": self.seed,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "max_negatives_per_positive": (
                    self.max_negatives_per_positive
                ),
            },
            "training": {
                "triples": self.training_triples,
                "users": self.training_users,
                "items": self.training_items,
            },
            "validation": {
                "impressions": self.validation_impressions,
                "candidate_occurrences": self.candidate_occurrences,
                "model_known_candidate_occurrences": (
                    self.model_known_candidate_occurrences
                ),
                "unknown_candidate_occurrences": (
                    self.unknown_candidate_occurrences
                ),
                "unknown_candidate_fraction": (
                    self.unknown_candidate_fraction
                ),
                "known_user_impressions": self.known_user_impressions,
                "cold_start_user_impressions": (
                    self.cold_start_user_impressions
                ),
                "cold_start_user_fraction": (
                    self.cold_start_user_fraction
                ),
                "impressions_with_no_model_candidates": (
                    self.impressions_with_no_model_candidates
                ),
                "clicked_item_occurrences": (
                    self.clicked_item_occurrences
                ),
                "unknown_clicked_item_occurrences": (
                    self.unknown_clicked_item_occurrences
                ),
                "unknown_clicked_item_fraction": (
                    self.unknown_clicked_item_fraction
                ),
            },
            "metrics": self.metrics.to_dict(),
        }


def _load_catalog(database: str | Path) -> tuple[str, ...]:
    with duckdb.connect(str(database), read_only=True) as connection:
        rows = connection.execute(
            """
            SELECT news_id
            FROM articles
            ORDER BY news_id
            """
        ).fetchall()

    catalog = tuple(str(row[0]) for row in rows)

    if not catalog:
        raise CollaborativeEvaluationError("Article catalog is empty.")

    return catalog


def _load_validation_impressions(
    database: str | Path,
    *,
    cutoff_timestamp: datetime | str,
) -> list[ValidationImpression]:
    query = """
        SELECT
            b.impression_id,
            b.user_id,
            c.news_id,
            c.clicked
        FROM behavior_events AS b
        JOIN candidate_interactions AS c
          ON b.impression_id = c.impression_id
        WHERE b.event_timestamp >= ?
        ORDER BY
            b.event_timestamp,
            b.impression_id,
            c.candidate_position
    """

    with duckdb.connect(str(database), read_only=True) as connection:
        rows = connection.execute(
            query,
            [str(cutoff_timestamp)],
        ).fetchall()

    if not rows:
        raise CollaborativeEvaluationError(
            "No validation interactions were found."
        )

    grouped: dict[
        tuple[str, str],
        list[tuple[str, bool]],
    ] = {}

    for impression_id, user_id, news_id, clicked in rows:
        key = (str(impression_id), str(user_id))
        grouped.setdefault(key, []).append(
            (str(news_id), bool(clicked))
        )

    impressions: list[ValidationImpression] = []

    for (impression_id, user_id), interactions in grouped.items():
        candidate_ids = tuple(item for item, _ in interactions)

        if len(candidate_ids) != len(set(candidate_ids)):
            raise CollaborativeEvaluationError(
                f"Duplicate candidates in impression '{impression_id}'."
            )

        relevant_items = frozenset(
            item
            for item, clicked in interactions
            if clicked
        )

        impressions.append(
            ValidationImpression(
                impression_id=impression_id,
                user_id=user_id,
                candidate_news_ids=candidate_ids,
                relevant_items=relevant_items,
            )
        )

    return impressions


def evaluate_collaborative_model(
    database: str | Path,
    *,
    cutoff_timestamp: datetime | str,
    k: int = 10,
    embedding_dim: int = 64,
    epochs: int = 10,
    batch_size: int = 2048,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-6,
    max_negatives_per_positive: int = 1,
    seed: int = 42,
) -> tuple[
    CollaborativeEvaluationReport,
    list[RankingExample],
]:
    """Train BPR before the cutoff and evaluate later impressions."""

    if k <= 0:
        raise CollaborativeEvaluationError("k must be positive.")

    triples = load_bpr_triples(
        database,
        cutoff_timestamp=cutoff_timestamp,
        max_negatives_per_positive=max_negatives_per_positive,
        seed=seed,
    )

    if not triples:
        raise CollaborativeEvaluationError(
            "No BPR training triples were produced."
        )

    model = CollaborativeRecommender(
        embedding_dim=embedding_dim,
        seed=seed,
    ).fit(
        triples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    validation = _load_validation_impressions(
        database,
        cutoff_timestamp=cutoff_timestamp,
    )
    catalog = _load_catalog(database)

    examples: list[RankingExample] = []

    candidate_occurrences = 0
    model_known_candidate_occurrences = 0
    cold_start_user_impressions = 0
    impressions_with_no_model_candidates = 0
    clicked_item_occurrences = 0
    unknown_clicked_item_occurrences = 0

    for impression in validation:
        candidate_ids = impression.candidate_news_ids
        candidate_occurrences += len(candidate_ids)

        known_candidates = tuple(
            item
            for item in candidate_ids
            if item in model.item_to_index
        )

        model_known_candidate_occurrences += len(known_candidates)

        if not known_candidates:
            impressions_with_no_model_candidates += 1

        if impression.user_id not in model.user_to_index:
            cold_start_user_impressions += 1

        clicked_item_occurrences += len(impression.relevant_items)
        unknown_clicked_item_occurrences += sum(
            item not in model.item_to_index
            for item in impression.relevant_items
        )

        recommendations = model.recommend_for_user(
            impression.user_id,
            candidate_news_ids=candidate_ids,
            top_k=k,
        )

        examples.append(
            RankingExample(
                impression_id=impression.impression_id,
                ranked_items=tuple(
                    recommendation.news_id
                    for recommendation in recommendations
                ),
                relevant_items=impression.relevant_items,
            )
        )

    metrics = evaluate_rankings(
        examples,
        catalog,
        k=k,
    )

    unknown_candidate_occurrences = (
        candidate_occurrences
        - model_known_candidate_occurrences
    )

    report = CollaborativeEvaluationReport(
        model_name="bpr_matrix_factorization",
        cutoff_timestamp=str(cutoff_timestamp),
        k=k,
        embedding_dim=embedding_dim,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_negatives_per_positive=max_negatives_per_positive,
        training_triples=len(triples),
        training_users=len(model.user_to_index),
        training_items=len(model.item_to_index),
        validation_impressions=len(validation),
        candidate_occurrences=candidate_occurrences,
        model_known_candidate_occurrences=(
            model_known_candidate_occurrences
        ),
        unknown_candidate_occurrences=unknown_candidate_occurrences,
        cold_start_user_impressions=cold_start_user_impressions,
        known_user_impressions=(
            len(validation) - cold_start_user_impressions
        ),
        impressions_with_no_model_candidates=(
            impressions_with_no_model_candidates
        ),
        clicked_item_occurrences=clicked_item_occurrences,
        unknown_clicked_item_occurrences=(
            unknown_clicked_item_occurrences
        ),
        metrics=metrics,
    )

    return report, examples
