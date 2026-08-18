"""Audit the leakage-safe Phase-05 ranker training matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

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
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

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
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    embedding_report = json.loads(
        args.embedding_report.read_text()
    )

    if (
        _sha256(args.checkpoint)
        != embedding_report[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint SHA mismatch."
        )

    if (
        _sha256(args.catalog)
        != embedding_report[
            "artifact_sha256"
        ]
    ):
        raise RuntimeError(
            "Catalog SHA mismatch."
        )

    root = (
        args.data_dir
        / "MINDsmall_train"
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

    result = (
        build_ranker_training_matrix(
            phase05.train,
            feature_builder=builder,
            max_negatives_per_positive=(
                args.max_negatives_per_positive
            ),
        )
    )

    feature_summary = {}

    for index, name in enumerate(
        SECOND_STAGE_FEATURE_NAMES
    ):
        values = result.features[
            :,
            index
        ]

        feature_summary[
            name
        ] = {
            "minimum": float(
                values.min()
            ),
            "maximum": float(
                values.max()
            ),
            "mean": float(
                values.mean()
            ),
            "std": float(
                values.std()
            ),
        }

    payload = {
        "experiment": (
            "phase05_ranker_data_audit"
        ),
        "official_dev_used": False,
        "phase03": {
            "training_impressions": len(
                phase03.train
            ),
            "development_pool_impressions": len(
                phase03.validation
            ),
            "cutoff": (
                phase03.cutoff.isoformat()
            ),
        },
        "phase05": {
            "ranker_training_impressions": len(
                phase05.train
            ),
            "ranker_validation_impressions": len(
                phase05.validation
            ),
            "cutoff": (
                phase05.cutoff.isoformat()
            ),
            "is_leakage_safe": (
                phase05.is_leakage_safe
            ),
        },
        "training_matrix": (
            result.to_dict()
        ),
        "feature_names": list(
            SECOND_STAGE_FEATURE_NAMES
        ),
        "feature_summary": (
            feature_summary
        ),
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

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
