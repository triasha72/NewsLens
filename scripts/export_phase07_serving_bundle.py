"""Export the frozen Phase-07 production serving bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from newslens.data import load_behaviors
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.models.popularity import (
    PopularityRecommender,
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)
from newslens.retrieval.faiss_flat import (
    FaissFlatIPRetriever,
)
from newslens.serving.bundle import (
    sha256_file,
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
        "--catalog",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--phase05-audit-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--phase06-final-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--phase07-optimization-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    phase05 = json.loads(
        args.phase05_audit_report.read_text()
    )

    phase06 = json.loads(
        args.phase06_final_report.read_text()
    )

    phase07 = json.loads(
        args.phase07_optimization_report.read_text()
    )

    if (
        phase06["selected_lambda"]
        != 0.80
    ):
        raise RuntimeError(
            "Frozen Phase-06 lambda is not 0.80."
        )

    if (
        phase06["retrieval_backend"]
        != "faiss_flat"
    ):
        raise RuntimeError(
            "Frozen retrieval backend is not faiss_flat."
        )

    if (
        phase07[
            "promotion_gate"
        ][
            "passed"
        ]
        is not True
    ):
        raise RuntimeError(
            "Phase-07 MMR optimization was not promoted."
        )

    checkpoint_sha = sha256_file(
        args.checkpoint
    )

    catalog_sha = sha256_file(
        args.catalog
    )

    if (
        checkpoint_sha
        != phase07[
            "frozen_inputs"
        ][
            "phase03_checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint differs from frozen Phase-07 benchmark."
        )

    if (
        catalog_sha
        != phase07[
            "frozen_inputs"
        ][
            "phase04_catalog_sha256"
        ]
    ):
        raise RuntimeError(
            "Catalog differs from frozen Phase-07 benchmark."
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    behaviors = load_behaviors(
        args.data_dir
        / "MINDsmall_train"
        / "behaviors.tsv"
    )

    phase03 = (
        chronological_train_validation_split(
            behaviors,
            validation_fraction=0.20,
        )
    )

    if (
        phase03.cutoff.isoformat()
        != phase05[
            "phase03"
        ][
            "cutoff"
        ]
    ):
        raise RuntimeError(
            "Phase-03 training boundary changed."
        )

    popularity = (
        PopularityRecommender()
        .fit(
            phase03.train
        )
    )

    catalog = (
        RetrievalCatalog.load_npz(
            args.catalog
        )
    )

    bundle_dir = (
        args.bundle_dir
        .expanduser()
        .resolve()
    )

    bundle_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_name = (
        "two_tower.pt"
    )

    catalog_name = (
        "article_embeddings.npz"
    )

    index_name = (
        "faiss_flat.index"
    )

    popularity_name = (
        "popularity.json"
    )

    checkpoint_target = (
        bundle_dir
        / checkpoint_name
    )

    catalog_target = (
        bundle_dir
        / catalog_name
    )

    index_target = (
        bundle_dir
        / index_name
    )

    popularity_target = (
        bundle_dir
        / popularity_name
    )

    shutil.copy2(
        args.checkpoint,
        checkpoint_target,
    )

    shutil.copy2(
        args.catalog,
        catalog_target,
    )

    retriever = (
        FaissFlatIPRetriever(
            catalog
        )
    )

    retriever.save(
        index_target
    )

    popularity_payload = {
        "schema_version": "1.0.0",
        "source": (
            "phase03_train_candidate_click_counts"
        ),
        "clicks": {
            news_id: int(
                popularity.statistics(
                    news_id
                ).clicks
            )
            for news_id
            in sorted(
                catalog.news_ids
            )
        },
    }

    popularity_target.write_text(
        json.dumps(
            popularity_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": (
            "newslens_phase07_serving_bundle"
        ),
        "config": {
            "artifact_version": (
                "phase07-v0.1"
            ),
            "selected_policy": (
                "mmr_lambda_0.80"
            ),
            "retrieval_backend": (
                "faiss_flat"
            ),
            "lambda_weight": 0.80,
            "retrieval_k": 100,
            "final_k": 10,
            "temperature": float(
                checkpoint[
                    "network_config"
                ][
                    "temperature"
                ]
            ),
            "max_history_length": int(
                checkpoint[
                    "protocol"
                ][
                    "max_history_length"
                ]
            ),
            "faiss_threads": 1,
        },
        "files": {
            "checkpoint": {
                "name": checkpoint_name,
                "sha256": (
                    sha256_file(
                        checkpoint_target
                    )
                ),
            },
            "catalog": {
                "name": catalog_name,
                "sha256": (
                    sha256_file(
                        catalog_target
                    )
                ),
            },
            "faiss_index": {
                "name": index_name,
                "sha256": (
                    sha256_file(
                        index_target
                    )
                ),
            },
            "popularity": {
                "name": popularity_name,
                "sha256": (
                    sha256_file(
                        popularity_target
                    )
                ),
            },
        },
        "evidence": {
            "phase06_final_report_sha256": (
                sha256_file(
                    args.phase06_final_report
                )
            ),
            "phase07_optimization_report_sha256": (
                sha256_file(
                    args.phase07_optimization_report
                )
            ),
        },
        "integrity": {
            "checkpoint_matches_phase07": True,
            "catalog_matches_phase07": True,
            "phase07_promotion_gate_passed": True,
            "passed": True,
        },
    }

    (
        bundle_dir
        / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    args.report_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report_output.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
