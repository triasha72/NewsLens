"""Export the frozen Phase-03 article retrieval embedding catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from newslens.data import (
    load_behaviors,
    load_news,
)
from newslens.evaluation.content import (
    _prepare_catalog,
    _training_vocabulary_news_ids,
)
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.features import (
    ArticleTextFeatureEncoder,
)
from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)


def _sha256(
    path: Path,
) -> str:
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


def _encode_articles(
    network: TwoTowerNetwork,
    values: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    network.to(
        "cpu"
    )

    network.eval()

    batches: list[
        np.ndarray
    ] = []

    with torch.no_grad():
        for start in range(
            0,
            values.shape[0],
            batch_size,
        ):
            tensor = (
                torch.from_numpy(
                    values[
                        start:
                        start + batch_size
                    ]
                )
            )

            encoded = (
                network.encode_articles(
                    tensor
                )
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            batches.append(
                encoded
            )

    return np.concatenate(
        batches,
        axis=0,
    )


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
        "--training-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cutoff",
        default="2019-11-13T20:36:26",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    args = parser.parse_args()

    training_report = json.loads(
        args.training_report.read_text()
    )

    if training_report[
        "protocol"
    ]["limited_training_run"]:
        raise RuntimeError(
            "Refusing to export from a limited training run."
        )

    expected_sha = (
        training_report[
            "checkpoint"
        ]["sha256"]
    )

    actual_sha = _sha256(
        args.checkpoint
    )

    if actual_sha != expected_sha:
        raise RuntimeError(
            "Frozen checkpoint SHA mismatch."
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint[
            "protocol"
        ]["cutoff_timestamp"]
        != args.cutoff
    ):
        raise RuntimeError(
            "Checkpoint cutoff mismatch."
        )

    data_path = (
        args.data_dir
        / "MINDsmall_train"
    )

    print(
        "Loading MIND-small train..."
    )

    news = load_news(
        data_path
        / "news.tsv"
    )

    behaviors = (
        load_behaviors(
            data_path
            / "behaviors.tsv"
        )
    )

    split = (
        chronological_train_validation_split(
            behaviors,
            validation_fraction=(
                args.validation_fraction
            ),
        )
    )

    if (
        split.cutoff.isoformat()
        != args.cutoff
    ):
        raise RuntimeError(
            "Chronological cutoff mismatch."
        )

    content_catalog = (
        _prepare_catalog(
            news
        )
    )

    fitting_news_ids = (
        _training_vocabulary_news_ids(
            split.train,
            content_catalog,
        )
    )

    feature_config = (
        checkpoint[
            "feature_encoder"
        ]
    )

    print(
        "Reconstructing frozen article features..."
    )

    encoder = (
        ArticleTextFeatureEncoder(
            max_features=int(
                feature_config[
                    "max_features"
                ]
            ),
            svd_components=int(
                feature_config[
                    "svd_components"
                ]
            ),
            seed=int(
                feature_config[
                    "seed"
                ]
            ),
        )
    )

    features = (
        encoder.fit_transform(
            news,
            fitting_news_ids=(
                fitting_news_ids
            ),
        )
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

    print(
        "Encoding frozen article embeddings..."
    )

    vectors = _encode_articles(
        network,
        features.values,
        batch_size=(
            args.batch_size
        ),
    )

    catalog = RetrievalCatalog(
        news_ids=(
            features.news_ids
        ),
        vectors=vectors,
    )

    catalog.save_npz(
        args.output
    )

    artifact_sha = _sha256(
        args.output
    )

    norms = np.linalg.norm(
        catalog.vectors,
        axis=1,
    )

    payload = {
        "experiment": (
            "phase04_frozen_embedding_export"
        ),
        "dataset": (
            "MINDsmall_train"
        ),
        "cutoff_timestamp": (
            args.cutoff
        ),
        "checkpoint_sha256": (
            actual_sha
        ),
        "artifact_sha256": (
            artifact_sha
        ),
        "article_count": (
            catalog.article_count
        ),
        "embedding_dim": (
            catalog.embedding_dim
        ),
        "fit_article_count": (
            encoder.fit_article_count
        ),
        "training_impressions": (
            len(split.train)
        ),
        "validation_impressions": (
            len(split.validation)
        ),
        "vector_norm": {
            "minimum": float(
                norms.min()
            ),
            "maximum": float(
                norms.max()
            ),
            "mean": float(
                norms.mean()
            ),
        },
        "artifact_path": str(
            args.output
        ),
    }

    args.metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.metadata.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "PHASE 04A EMBEDDING EXPORT"
    )

    print(
        "Articles:",
        catalog.article_count,
    )

    print(
        "Dimension:",
        catalog.embedding_dim,
    )

    print(
        "Norm range:",
        float(
            norms.min()
        ),
        "to",
        float(
            norms.max()
        ),
    )

    print(
        "Artifact SHA-256:",
        artifact_sha,
    )


if __name__ == "__main__":
    main()
