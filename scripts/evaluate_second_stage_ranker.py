"""Evaluate the frozen Phase-05 learned second-stage ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from newslens.data import (
    load_behaviors,
    load_news,
    parse_impressions,
)
from newslens.evaluation.comparison import (
    paired_bootstrap_ranking_comparison,
)
from newslens.evaluation.evaluator import (
    RankingExample,
    evaluate_rankings,
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
    SecondStageFeatureBuilder,
    SecondStageRanker,
)
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)


def _sha256(
    path: Path,
) -> str:
    """Return SHA-256 for one artifact."""

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


def _parse_history(
    value: object,
) -> tuple[str, ...]:
    """Parse one MIND history field."""

    if (
        value is None
        or pd.isna(value)
    ):
        return ()

    text = str(
        value
    ).strip()

    if not text:
        return ()

    return tuple(
        text.split()
    )


def _history_bin(
    length: int,
) -> str:
    """Return fixed history-size subgroup."""

    if length == 0:
        return "fallback"

    if length <= 5:
        return "1_5"

    if length <= 10:
        return "6_10"

    return "11_20"


def _candidate_bin(
    count: int,
) -> str:
    """Return fixed logged-candidate-count subgroup."""

    if count <= 10:
        return "le_10"

    if count <= 25:
        return "11_25"

    if count <= 50:
        return "26_50"

    return "gt_50"


def _subgroup_metrics(
    baseline_groups: dict[
        str,
        list[RankingExample],
    ],
    ranker_groups: dict[
        str,
        list[RankingExample],
    ],
    catalog_items: tuple[str, ...],
    *,
    k: int,
) -> dict[str, object]:
    """Return point metrics for fixed diagnostic subgroups."""

    payload: dict[
        str,
        object,
    ] = {}

    for group_name in sorted(
        baseline_groups
    ):
        baseline = (
            baseline_groups[
                group_name
            ]
        )

        candidate = (
            ranker_groups[
                group_name
            ]
        )

        if not any(
            example.relevant_items
            for example in baseline
        ):
            continue

        baseline_metrics = (
            evaluate_rankings(
                baseline,
                catalog_items,
                k=k,
            )
        )

        candidate_metrics = (
            evaluate_rankings(
                candidate,
                catalog_items,
                k=k,
            )
        )

        payload[
            group_name
        ] = {
            "impressions": len(
                baseline
            ),
            "baseline": (
                baseline_metrics.to_dict()
            ),
            "ranker": (
                candidate_metrics.to_dict()
            ),
            "point_delta": {
                "ndcg_at_k": (
                    candidate_metrics.ndcg_at_k
                    - baseline_metrics.ndcg_at_k
                ),
                "mrr_at_k": (
                    candidate_metrics.mrr_at_k
                    - baseline_metrics.mrr_at_k
                ),
                "recall_at_k": (
                    candidate_metrics.recall_at_k
                    - baseline_metrics.recall_at_k
                ),
                "hit_rate_at_k": (
                    candidate_metrics.hit_rate_at_k
                    - baseline_metrics.hit_rate_at_k
                ),
                "catalog_coverage_at_k": (
                    candidate_metrics.catalog_coverage_at_k
                    - baseline_metrics.catalog_coverage_at_k
                ),
            },
        }

    return payload


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
        "--ranker-artifact",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--training-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--audit-report",
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
        "--k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
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

    training_report = json.loads(
        args.training_report.read_text()
    )

    audit_report = json.loads(
        args.audit_report.read_text()
    )

    if (
        training_report[
            "protocol"
        ][
            "limited_training_run"
        ]
        is not False
    ):
        raise RuntimeError(
            "Refusing to evaluate a limited ranker."
        )

    if (
        training_report[
            "protocol"
        ][
            "official_dev_used"
        ]
        is not False
    ):
        raise RuntimeError(
            "Unexpected official-dev usage."
        )

    checkpoint_sha = _sha256(
        args.checkpoint
    )

    catalog_sha = _sha256(
        args.catalog
    )

    ranker_sha = _sha256(
        args.ranker_artifact
    )

    if (
        checkpoint_sha
        != embedding_report[
            "checkpoint_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen Phase-03 checkpoint SHA mismatch."
        )

    if (
        catalog_sha
        != embedding_report[
            "artifact_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen Phase-04 catalog SHA mismatch."
        )

    if (
        ranker_sha
        != training_report[
            "artifact"
        ][
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen Phase-05 ranker SHA mismatch."
        )

    root = (
        args.data_dir
        / "MINDsmall_train"
    )

    print(
        "Loading chronological MIND-small data..."
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
        ]["cutoff"]
    ):
        raise RuntimeError(
            "Phase-03 split differs from audit."
        )

    if (
        phase05.cutoff.isoformat()
        != audit_report[
            "phase05"
        ]["cutoff"]
    ):
        raise RuntimeError(
            "Phase-05 split differs from audit."
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
            "Phase-05 validation count differs from audit."
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

    network.eval()

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

    ranker, ranker_metadata = (
        SecondStageRanker.load(
            args.ranker_artifact
        )
    )

    if (
        ranker_metadata[
            "phase05_cutoff"
        ]
        != phase05.cutoff.isoformat()
    ):
        raise RuntimeError(
            "Ranker metadata cutoff mismatch."
        )

    if (
        ranker_metadata[
            "limited_training_run"
        ]
        is not False
    ):
        raise RuntimeError(
            "Loaded ranker is a limited run."
        )

    if (
        ranker.config.to_dict()
        != training_report[
            "model_config"
        ]
    ):
        raise RuntimeError(
            "Loaded ranker configuration differs "
            "from the training report."
        )

    baseline_examples: list[
        RankingExample
    ] = []

    ranker_examples: list[
        RankingExample
    ] = []

    baseline_history_groups: dict[
        str,
        list[RankingExample],
    ] = defaultdict(
        list
    )

    ranker_history_groups: dict[
        str,
        list[RankingExample],
    ] = defaultdict(
        list
    )

    baseline_candidate_groups: dict[
        str,
        list[RankingExample],
    ] = defaultdict(
        list
    )

    ranker_candidate_groups: dict[
        str,
        list[RankingExample],
    ] = defaultdict(
        list
    )

    catalog_ids = set(
        catalog.news_ids
    )

    ranker_scored_impressions = 0
    popularity_fallback_impressions = 0
    truncated_history_impressions = 0
    changed_rankings = 0
    candidate_occurrences = 0

    print(
        "Scoring Phase-05 chronological validation impressions..."
    )

    for row in phase05.validation.itertuples(
        index=False
    ):
        impression_id = str(
            row.impression_id
        )

        history_ids = _parse_history(
            row.history
        )

        parsed = parse_impressions(
            str(
                row.impressions
            )
        )

        candidate_ids = tuple(
            news_id
            for news_id, _
            in parsed
        )

        if (
            len(candidate_ids)
            != len(
                set(candidate_ids)
            )
        ):
            raise RuntimeError(
                "Duplicate candidate IDs in validation impression."
            )

        missing_candidates = [
            news_id
            for news_id in candidate_ids
            if news_id
            not in catalog_ids
        ]

        if missing_candidates:
            raise RuntimeError(
                "Validation candidate missing from frozen catalog."
            )

        relevant_items = frozenset(
            news_id
            for news_id, label
            in parsed
            if label == 1
        )

        candidate_occurrences += len(
            candidate_ids
        )

        context = (
            builder.build_context(
                history_ids
            )
        )

        if context is None:
            popularity_fallback_impressions += 1

            baseline_items = tuple(
                popularity.rank_candidates(
                    candidate_ids,
                    top_k=args.k,
                    exclude_news_ids=(
                        history_ids
                    ),
                )
            )

            ranker_items = (
                baseline_items
            )

            usable_history_length = 0

        else:
            if (
                context.source_usable_history_count
                > builder.max_history_length
            ):
                truncated_history_impressions += 1

            features = (
                builder.features_for_candidates(
                    context,
                    candidate_ids,
                )
            )

            # Feature zero is the raw user/article inner product.
            # Phase-03 score_candidates divides by a positive
            # temperature constant, so ordering is identical.
            two_tower_scores = (
                features[
                    :,
                    0,
                ]
            )

            baseline_items = tuple(
                news_id
                for news_id, _
                in sorted(
                    zip(
                        candidate_ids,
                        two_tower_scores.tolist(),
                        strict=True,
                    ),
                    key=lambda item: (
                        -float(
                            item[1]
                        ),
                        item[0],
                    ),
                )[
                    :args.k
                ]
            )

            ranker_items = tuple(
                ranker.rank(
                    candidate_ids,
                    features,
                    top_k=args.k,
                )
            )

            ranker_scored_impressions += 1

            usable_history_length = len(
                context.usable_history_news_ids
            )

        baseline_example = (
            RankingExample(
                impression_id=(
                    impression_id
                ),
                ranked_items=(
                    baseline_items
                ),
                relevant_items=(
                    relevant_items
                ),
            )
        )

        ranker_example = (
            RankingExample(
                impression_id=(
                    impression_id
                ),
                ranked_items=(
                    ranker_items
                ),
                relevant_items=(
                    relevant_items
                ),
            )
        )

        baseline_examples.append(
            baseline_example
        )

        ranker_examples.append(
            ranker_example
        )

        if (
            baseline_items
            != ranker_items
        ):
            changed_rankings += 1

        history_group = _history_bin(
            usable_history_length
        )

        candidate_group = (
            _candidate_bin(
                len(
                    candidate_ids
                )
            )
        )

        baseline_history_groups[
            history_group
        ].append(
            baseline_example
        )

        ranker_history_groups[
            history_group
        ].append(
            ranker_example
        )

        baseline_candidate_groups[
            candidate_group
        ].append(
            baseline_example
        )

        ranker_candidate_groups[
            candidate_group
        ].append(
            ranker_example
        )

    baseline_metrics = (
        evaluate_rankings(
            baseline_examples,
            catalog.news_ids,
            k=args.k,
        )
    )

    ranker_metrics = (
        evaluate_rankings(
            ranker_examples,
            catalog.news_ids,
            k=args.k,
        )
    )

    comparison = (
        paired_bootstrap_ranking_comparison(
            baseline_examples,
            ranker_examples,
            baseline_model_name=(
                "phase03_two_tower_popularity"
            ),
            candidate_model_name=(
                "phase05_second_stage_ranker"
            ),
            k=args.k,
            bootstrap_samples=(
                args.bootstrap_samples
            ),
            confidence_level=(
                args.confidence_level
            ),
            random_seed=(
                args.seed
            ),
        )
    )

    ndcg_significantly_positive = (
        comparison.ndcg_at_k.lower_bound
        > 0.0
    )

    mrr_not_significantly_worse = (
        comparison.mrr_at_k.upper_bound
        >= 0.0
    )

    recall_not_significantly_worse = (
        comparison.recall_at_k.upper_bound
        >= 0.0
    )

    hit_not_significantly_worse = (
        comparison.hit_rate_at_k.upper_bound
        >= 0.0
    )

    guardrails_passed = (
        mrr_not_significantly_worse
        and recall_not_significantly_worse
        and hit_not_significantly_worse
    )

    selected = (
        ndcg_significantly_positive
        and guardrails_passed
    )

    if selected:
        selected_model = (
            "phase05_second_stage_ranker"
        )

        reason = (
            "Ranker satisfies the preregistered "
            "Phase-05 selection rule."
        )

    else:
        selected_model = (
            "phase03_two_tower_popularity"
        )

        reason = (
            "Ranker does not satisfy the preregistered "
            "Phase-05 selection rule."
        )

    coverage_delta = (
        ranker_metrics.catalog_coverage_at_k
        - baseline_metrics.catalog_coverage_at_k
    )

    payload = {
        "experiment": (
            "phase05_second_stage_ranker_evaluation"
        ),
        "protocol": {
            "dataset": (
                "MINDsmall_train"
            ),
            "official_dev_used": False,
            "phase03_cutoff": (
                phase03.cutoff.isoformat()
            ),
            "ranker_cutoff": (
                phase05.cutoff.isoformat()
            ),
            "ranker_validation_impressions": len(
                phase05.validation
            ),
            "k": args.k,
            "bootstrap_samples": (
                args.bootstrap_samples
            ),
            "confidence_level": (
                args.confidence_level
            ),
            "seed": args.seed,
            "baseline_score": (
                "raw_inner_product; ranking-equivalent "
                "to Phase-03 temperature-scaled score"
            ),
        },
        "frozen_inputs": {
            "phase03_checkpoint_sha256": (
                checkpoint_sha
            ),
            "phase04_catalog_sha256": (
                catalog_sha
            ),
            "phase05_ranker_sha256": (
                ranker_sha
            ),
        },
        "accounting": {
            "candidate_occurrences": (
                candidate_occurrences
            ),
            "ranker_scored_impressions": (
                ranker_scored_impressions
            ),
            "popularity_fallback_impressions": (
                popularity_fallback_impressions
            ),
            "truncated_history_impressions": (
                truncated_history_impressions
            ),
            "changed_rankings": (
                changed_rankings
            ),
            "changed_ranking_fraction": (
                changed_rankings
                / len(
                    phase05.validation
                )
            ),
        },
        "metrics": {
            "phase03_two_tower_popularity": (
                baseline_metrics.to_dict()
            ),
            "phase05_second_stage_ranker": (
                ranker_metrics.to_dict()
            ),
            "catalog_coverage_point_delta": (
                coverage_delta
            ),
        },
        "comparison": (
            comparison.to_dict()
        ),
        "subgroups": {
            "history_length": (
                _subgroup_metrics(
                    baseline_history_groups,
                    ranker_history_groups,
                    catalog.news_ids,
                    k=args.k,
                )
            ),
            "candidate_count": (
                _subgroup_metrics(
                    baseline_candidate_groups,
                    ranker_candidate_groups,
                    catalog.news_ids,
                    k=args.k,
                )
            ),
        },
        "selection": {
            "selected_model": (
                selected_model
            ),
            "ndcg_significantly_positive": (
                ndcg_significantly_positive
            ),
            "mrr_not_significantly_worse": (
                mrr_not_significantly_worse
            ),
            "recall_not_significantly_worse": (
                recall_not_significantly_worse
            ),
            "hit_rate_not_significantly_worse": (
                hit_not_significantly_worse
            ),
            "guardrails_passed": (
                guardrails_passed
            ),
            "reason": reason,
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
    print("PHASE 05D — SECOND-STAGE RANKER EVALUATION")
    print("=" * 78)

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
