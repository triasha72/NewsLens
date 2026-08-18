"""Freeze the final Phase-05 learned-ranking decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audit-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--training-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--evaluation-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--phase04-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    audit = json.loads(
        args.audit_report.read_text()
    )

    training = json.loads(
        args.training_report.read_text()
    )

    evaluation = json.loads(
        args.evaluation_report.read_text()
    )

    phase04 = json.loads(
        args.phase04_report.read_text()
    )

    if (
        training[
            "protocol"
        ][
            "limited_training_run"
        ]
        is not False
    ):
        raise RuntimeError(
            "Cannot freeze a limited Phase-05 training run."
        )

    if (
        audit[
            "phase05"
        ][
            "is_leakage_safe"
        ]
        is not True
    ):
        raise RuntimeError(
            "Phase-05 split is not certified leakage-safe."
        )

    if (
        evaluation[
            "protocol"
        ][
            "official_dev_used"
        ]
        is not False
    ):
        raise RuntimeError(
            "Unexpected official-dev usage."
        )

    selected_model = (
        evaluation[
            "selection"
        ][
            "selected_model"
        ]
    )

    expected_baseline = (
        "phase03_two_tower_popularity"
    )

    if selected_model != expected_baseline:
        raise RuntimeError(
            "This freeze expects the preregistered "
            "Phase-05 evaluation result to retain "
            "the Phase-03 baseline."
        )

    phase04_selected_index = (
        phase04.get(
            "selected_index"
        )
        or phase04.get(
            "selection",
            {},
        ).get(
            "selected_index"
        )
    )

    comparison = (
        evaluation[
            "comparison"
        ][
            "metrics"
        ]
    )

    baseline = (
        evaluation[
            "metrics"
        ][
            "phase03_two_tower_popularity"
        ]
    )

    candidate = (
        evaluation[
            "metrics"
        ][
            "phase05_second_stage_ranker"
        ]
    )

    payload = {
        "phase": "phase05",
        "status": "frozen",
        "selected_model": (
            selected_model
        ),
        "selected_retrieval_backend": (
            phase04_selected_index
        ),
        "learned_ranker_promoted": False,
        "information_boundary": {
            "dataset": (
                "MINDsmall_train"
            ),
            "official_dev_used": False,
            "phase03_cutoff": (
                audit[
                    "phase03"
                ][
                    "cutoff"
                ]
            ),
            "phase05_cutoff": (
                audit[
                    "phase05"
                ][
                    "cutoff"
                ]
            ),
            "ranker_training_impressions": (
                audit[
                    "phase05"
                ][
                    "ranker_training_impressions"
                ]
            ),
            "ranker_validation_impressions": (
                audit[
                    "phase05"
                ][
                    "ranker_validation_impressions"
                ]
            ),
        },
        "training": {
            "model_config": (
                training[
                    "model_config"
                ]
            ),
            "training_matrix": (
                training[
                    "training_matrix"
                ]
            ),
            "ranker_artifact_sha256": (
                training[
                    "artifact"
                ][
                    "sha256"
                ]
            ),
        },
        "evaluation": {
            "baseline": (
                baseline
            ),
            "learned_ranker": (
                candidate
            ),
            "paired_comparison": (
                evaluation[
                    "comparison"
                ]
            ),
            "catalog_coverage_point_delta": (
                evaluation[
                    "metrics"
                ][
                    "catalog_coverage_point_delta"
                ]
            ),
            "changed_ranking_fraction": (
                evaluation[
                    "accounting"
                ][
                    "changed_ranking_fraction"
                ]
            ),
        },
        "selection": (
            evaluation[
                "selection"
            ]
        ),
        "serving_decision": {
            "phase05_ranker_integrated": False,
            "reason": (
                "The learned second-stage ranker failed "
                "the preregistered offline quality gate. "
                "The Phase-03 two-tower ordering and "
                "Phase-04 retrieval backend remain selected."
            ),
        },
    }

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.json_output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    markdown = f"""## Phase 05F: Final learned-ranking decision

Phase 05 evaluated a learned second-stage ranker over the frozen
Phase-03 two-tower representation.

### Information boundary

- ranker training impressions:
  {audit["phase05"]["ranker_training_impressions"]:,}
- ranker validation impressions:
  {audit["phase05"]["ranker_validation_impressions"]:,}
- Phase-05 cutoff:
  `{audit["phase05"]["cutoff"]}`
- MINDsmall_dev used:
  False

### Training

The candidate ranker used the frozen 11-feature contract and
HistGradientBoosting configuration.

Training rows:

- {training["training_matrix"]["training_rows"]:,}

Clicked rows:

- {training["training_matrix"]["positive_rows"]:,}

Selected hard nonclick rows:

- {training["training_matrix"]["selected_negative_rows"]:,}

### Ranking quality

| Model | NDCG@10 | MRR@10 | Recall@10 | Hit@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Phase-03 two-tower + popularity | {baseline["ndcg_at_k"]:.6f} | {baseline["mrr_at_k"]:.6f} | {baseline["recall_at_k"]:.6f} | {baseline["hit_rate_at_k"]:.6f} | {baseline["catalog_coverage_at_k"]:.6f} |
| Phase-05 learned ranker | {candidate["ndcg_at_k"]:.6f} | {candidate["mrr_at_k"]:.6f} | {candidate["recall_at_k"]:.6f} | {candidate["hit_rate_at_k"]:.6f} | {candidate["catalog_coverage_at_k"]:.6f} |

### Paired candidate-minus-baseline differences

- NDCG@10:
  {comparison["ndcg_at_k"]["point_difference"]:+.6f}
  [{comparison["ndcg_at_k"]["lower_bound"]:+.6f},
   {comparison["ndcg_at_k"]["upper_bound"]:+.6f}]
- MRR@10:
  {comparison["mrr_at_k"]["point_difference"]:+.6f}
  [{comparison["mrr_at_k"]["lower_bound"]:+.6f},
   {comparison["mrr_at_k"]["upper_bound"]:+.6f}]
- Recall@10:
  {comparison["recall_at_k"]["point_difference"]:+.6f}
  [{comparison["recall_at_k"]["lower_bound"]:+.6f},
   {comparison["recall_at_k"]["upper_bound"]:+.6f}]
- Hit Rate@10:
  {comparison["hit_rate_at_k"]["point_difference"]:+.6f}
  [{comparison["hit_rate_at_k"]["lower_bound"]:+.6f},
   {comparison["hit_rate_at_k"]["upper_bound"]:+.6f}]

### Selection

**{selected_model}**

The learned second-stage ranker is rejected.

All four paired ranking metrics deteriorated, and the preregistered
selection guardrails failed.

The candidate changed
{evaluation["accounting"]["changed_ranking_fraction"]:.2%}
of validation rankings while also reducing catalog coverage.

### Serving decision

The rejected Phase-05 ranker is not promoted into the serving path.

The selected system remains:

- frozen Phase-03 hard-negative two-tower;
- popularity fallback for unusable histories;
- Phase-04 `{phase04_selected_index}` retrieval backend.

No Phase-05 hyperparameters were changed after observing validation.

No MINDsmall_dev result was used.

Phase 05 is frozen as a negative learned-reranking result.
"""

    args.markdown_output.write_text(
        markdown
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
