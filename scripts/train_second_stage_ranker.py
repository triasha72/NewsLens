"""Train the frozen Phase-05 learned second-stage ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)

from newslens.data import (
    load_behaviors,
    load_news,
)
from newslens.data.ranker_training import (
    build_ranker_training_matrix,
)
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.models.popularity import (
    PopularityRecommender,
)
from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
)
from newslens.ranking import (
    SECOND_STAGE_FEATURE_NAMES,
    SecondStageFeatureBuilder,
    SecondStageRanker,
    SecondStageRankerConfig,
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)


def _sha256(
    path: Path,
) -> str:
    """Return SHA-256 fingerprint for one artifact."""

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--embedding-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--audit-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--phase03-validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--ranker-validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--max-negatives-per-positive",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-training-impressions",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    embedding_report = json.loads(
        args.embedding_report.read_text()
    )

    audit_report = json.loads(
        args.audit_report.read_text()
    )

    checkpoint_sha = _sha256(
        args.checkpoint
    )

    catalog_sha = _sha256(
        args.catalog
    )

    if (
        checkpoint_sha
        != embedding_report[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint SHA-256 does not match "
            "the frozen Phase-04 embedding report."
        )

    if (
        catalog_sha
        != embedding_report[
            "artifact_sha256"
        ]
    ):
        raise RuntimeError(
            "Catalog SHA-256 does not match "
            "the frozen Phase-04 embedding report."
        )

    if (
        audit_report[
            "official_dev_used"
        ]
        is not False
    ):
        raise RuntimeError(
            "Phase-05 audit unexpectedly used official dev."
        )

    if (
        audit_report[
            "phase05"
        ][
            "is_leakage_safe"
        ]
        is not True
    ):
        raise RuntimeError(
            "Phase-05 audit does not certify a leakage-safe split."
        )

    root = (
        args.data_dir
        / "MINDsmall_train"
    )

    print(
        "Loading MIND-small train..."
    )

    news = load_news(
        root
        / "news.tsv"
    )

    behaviors = load_behaviors(
        root
        / "behaviors.tsv"
    )

    phase03 = (
        chronological_train_validation_split(
            behaviors,
            validation_fraction=(
                args.phase03_validation_fraction
            ),
        )
    )

    phase05 = (
        chronological_train_validation_split(
            phase03.validation,
            validation_fraction=(
                args.ranker_validation_fraction
            ),
        )
    )

    if (
        phase03.cutoff.isoformat()
        != audit_report[
            "phase03"
        ][
            "cutoff"
        ]
    ):
        raise RuntimeError(
            "Phase-03 cutoff differs from the audited split."
        )

    if (
        phase05.cutoff.isoformat()
        != audit_report[
            "phase05"
        ][
            "cutoff"
        ]
    ):
        raise RuntimeError(
            "Phase-05 cutoff differs from the audited split."
        )

    if (
        len(
            phase05.train
        )
        != audit_report[
            "phase05"
        ][
            "ranker_training_impressions"
        ]
    ):
        raise RuntimeError(
            "Phase-05 training impression count differs "
            "from the audited protocol."
        )

    if (
        len(
            phase05.validation
        )
        != audit_report[
            "phase05"
        ][
            "ranker_validation_impressions"
        ]
    ):
        raise RuntimeError(
            "Phase-05 validation impression count differs "
            "from the audited protocol."
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    network = TwoTowerNetwork(
        TwoTowerConfig(
            **checkpoint[
                "network_config"
            ]
        )
    )

    network.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    catalog = (
        RetrievalCatalog.load_npz(
            args.catalog
        )
    )

    popularity = (
        PopularityRecommender()
        .fit(
            phase03.train
        )
    )

    builder = SecondStageFeatureBuilder(
        catalog=catalog,
        network=network,
        news=news,
        popularity_model=popularity,
        max_history_length=int(
            checkpoint[
                "protocol"
            ][
                "max_history_length"
            ]
        ),
    )

    print(
        "Building Phase-05 ranker training matrix..."
    )

    training = (
        build_ranker_training_matrix(
            phase05.train,
            feature_builder=builder,
            max_negatives_per_positive=(
                args.max_negatives_per_positive
            ),
            max_impressions=(
                args.max_training_impressions
            ),
        )
    )

    if not training.limited_training_run:
        audited_matrix = (
            audit_report[
                "training_matrix"
            ]
        )

        if (
            training.row_count
            != audited_matrix[
                "training_rows"
            ]
        ):
            raise RuntimeError(
                "Full training row count differs from "
                "the Phase-05A audited matrix."
            )

        if (
            training.positive_rows
            != audited_matrix[
                "positive_rows"
            ]
        ):
            raise RuntimeError(
                "Positive-row count differs from "
                "the Phase-05A audited matrix."
            )

        if (
            training.selected_negative_rows
            != audited_matrix[
                "selected_negative_rows"
            ]
        ):
            raise RuntimeError(
                "Selected-negative count differs from "
                "the Phase-05A audited matrix."
            )

    config = (
        SecondStageRankerConfig(
            learning_rate=(
                args.learning_rate
            ),
            max_iter=(
                args.max_iter
            ),
            max_leaf_nodes=(
                args.max_leaf_nodes
            ),
            min_samples_leaf=(
                args.min_samples_leaf
            ),
            l2_regularization=(
                args.l2_regularization
            ),
            seed=args.seed,
        )
    )

    print(
        "Training HistGradientBoosting second-stage ranker..."
    )

    ranker = SecondStageRanker(
        config
    )

    started = (
        time.perf_counter()
    )

    ranker.fit(
        training.features,
        training.labels,
    )

    runtime_seconds = (
        time.perf_counter()
        - started
    )

    training_scores = ranker.score(
        training.features
    )

    diagnostics = {
        "training_log_loss": float(
            log_loss(
                training.labels,
                training_scores,
            )
        ),
        "training_roc_auc": float(
            roc_auc_score(
                training.labels,
                training_scores,
            )
        ),
        "score_minimum": float(
            training_scores.min()
        ),
        "score_maximum": float(
            training_scores.max()
        ),
        "score_mean": float(
            training_scores.mean()
        ),
    }

    metadata = {
        "phase": (
            "phase05"
        ),
        "dataset": (
            "MINDsmall_train"
        ),
        "official_dev_used": False,
        "phase03_checkpoint_sha256": (
            checkpoint_sha
        ),
        "phase04_catalog_sha256": (
            catalog_sha
        ),
        "phase03_cutoff": (
            phase03.cutoff.isoformat()
        ),
        "phase05_cutoff": (
            phase05.cutoff.isoformat()
        ),
        "max_history_length": (
            builder.max_history_length
        ),
        "max_negatives_per_positive": (
            args.max_negatives_per_positive
        ),
        "limited_training_run": (
            training.limited_training_run
        ),
    }

    ranker.save(
        args.artifact,
        metadata=metadata,
    )

    artifact_sha = _sha256(
        args.artifact
    )

    payload = {
        "experiment": (
            "phase05_second_stage_ranker_training"
        ),
        "protocol": {
            "dataset": (
                "MINDsmall_train"
            ),
            "official_dev_used": False,
            "phase03_training_impressions": len(
                phase03.train
            ),
            "phase03_development_pool_impressions": len(
                phase03.validation
            ),
            "phase03_cutoff": (
                phase03.cutoff.isoformat()
            ),
            "ranker_training_impressions": len(
                phase05.train
            ),
            "ranker_validation_impressions": len(
                phase05.validation
            ),
            "ranker_cutoff": (
                phase05.cutoff.isoformat()
            ),
            "ranker_validation_fraction": (
                args.ranker_validation_fraction
            ),
            "limited_training_run": (
                training.limited_training_run
            ),
        },
        "feature_names": list(
            SECOND_STAGE_FEATURE_NAMES
        ),
        "model_config": (
            config.to_dict()
        ),
        "training_matrix": (
            training.to_dict()
        ),
        "diagnostics": (
            diagnostics
        ),
        "runtime_seconds": (
            runtime_seconds
        ),
        "artifact": {
            "path": str(
                args.artifact
            ),
            "sha256": (
                artifact_sha
            ),
        },
        "frozen_inputs": {
            "phase03_checkpoint_sha256": (
                checkpoint_sha
            ),
            "phase04_catalog_sha256": (
                catalog_sha
            ),
            "phase05_audit_report": str(
                args.audit_report
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 78)
    print("PHASE 05 — SECOND-STAGE RANKER TRAINING")
    print("=" * 78)

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
