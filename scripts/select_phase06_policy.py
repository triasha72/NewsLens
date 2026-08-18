"""Apply the preregistered Phase-06 MMR selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CANDIDATE_KEYS: tuple[str, ...] = (
    "0.95",
    "0.90",
    "0.85",
    "0.80",
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sweep-report",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--baseline-report",
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

    sweep = json.loads(
        args.sweep_report.read_text()
    )

    baseline_report = json.loads(
        args.baseline_report.read_text()
    )

    if sweep["baseline_parity"]["passed"] is not True:
        raise RuntimeError(
            "Cannot select from a sweep with failed baseline parity."
        )

    if sweep["protocol"]["selection_performed"] is not False:
        raise RuntimeError(
            "Sweep unexpectedly reports prior selection."
        )

    if sweep["protocol"]["lambdas"] != [
        1.0,
        0.95,
        0.90,
        0.85,
        0.80,
    ]:
        raise RuntimeError(
            "Sweep lambda grid differs from frozen Phase-06 protocol."
        )

    baseline = sweep["policies"]["1.00"]

    baseline_ndcg = float(
        baseline["relevance"]["ndcg_at_k"]
    )

    baseline_recall = float(
        baseline["relevance"]["recall_at_k"]
    )

    baseline_ild = float(
        baseline["diversity"][
            "mean_intra_list_diversity"
        ]
    )

    baseline_gini = float(
        baseline["exposure"]["exposure_gini"]
    )

    if baseline_report["integrity"]["passed"] is not True:
        raise RuntimeError(
            "Frozen Phase-06B baseline audit did not pass."
        )

    ndcg_threshold = (
        0.98 * baseline_ndcg
    )

    recall_threshold = (
        0.98 * baseline_recall
    )

    ndcg_ci_lower_threshold = (
        -0.02 * baseline_ndcg
    )

    evaluations: dict[
        str,
        dict[str, object],
    ] = {}

    eligible_keys: list[str] = []

    for key in CANDIDATE_KEYS:
        policy = sweep["policies"][key]

        comparison = (
            sweep[
                "paired_comparisons_vs_lambda_1"
            ][key]
        )

        ndcg = float(
            policy["relevance"]["ndcg_at_k"]
        )

        recall = float(
            policy["relevance"]["recall_at_k"]
        )

        ild = float(
            policy["diversity"][
                "mean_intra_list_diversity"
            ]
        )

        gini = float(
            policy["exposure"]["exposure_gini"]
        )

        ndcg_lower = float(
            comparison["metrics"]["ndcg_at_k"][
                "lower_bound"
            ]
        )

        checks = {
            "ndcg_retention_passed": (
                ndcg >= ndcg_threshold
            ),
            "recall_retention_passed": (
                recall >= recall_threshold
            ),
            "ndcg_ci_noninferiority_passed": (
                ndcg_lower
                >= ndcg_ci_lower_threshold
            ),
            "ild_improved": (
                ild > baseline_ild
            ),
            "gini_improved": (
                gini < baseline_gini
            ),
        }

        eligible = all(
            checks.values()
        )

        if eligible:
            eligible_keys.append(
                key
            )

        evaluations[key] = {
            "lambda": float(key),
            "eligible": eligible,
            "checks": checks,
            "metrics": {
                "ndcg_at_k": ndcg,
                "recall_at_k": recall,
                "mean_intra_list_diversity": ild,
                "exposure_gini": gini,
                "ndcg_ci_lower_bound": (
                    ndcg_lower
                ),
                "changed_ranking_fraction": (
                    policy[
                        "changed_ranking_fraction_vs_lambda_1"
                    ]
                ),
            },
            "deltas_vs_baseline": {
                "ndcg_at_k": (
                    ndcg
                    - baseline_ndcg
                ),
                "recall_at_k": (
                    recall
                    - baseline_recall
                ),
                "mean_intra_list_diversity": (
                    ild
                    - baseline_ild
                ),
                "exposure_gini": (
                    gini
                    - baseline_gini
                ),
            },
        }

    if eligible_keys:
        selected_key = max(
            eligible_keys,
            key=lambda key: (
                float(
                    sweep["policies"][key][
                        "diversity"
                    ][
                        "mean_intra_list_diversity"
                    ]
                ),
                float(
                    sweep["policies"][key][
                        "relevance"
                    ][
                        "ndcg_at_k"
                    ]
                ),
                -float(
                    sweep["policies"][key][
                        "exposure"
                    ][
                        "exposure_gini"
                    ]
                ),
                float(key),
            ),
        )

        selected_lambda = float(
            selected_key
        )

        selected_policy = (
            f"mmr_lambda_{selected_key}"
        )

    else:
        selected_key = "1.00"
        selected_lambda = 1.0
        selected_policy = (
            "phase03_two_tower_ordering"
        )

    selected = sweep["policies"][
        selected_key
    ]

    payload = {
        "experiment": (
            "phase06_preregistered_policy_selection"
        ),
        "protocol": {
            "selection_rule": (
                "98_percent_relevance_retention_"
                "plus_diversity_and_gini_improvement"
            ),
            "tie_break_order": [
                "higher_mean_intra_list_diversity",
                "higher_ndcg_at_k",
                "lower_exposure_gini",
                "larger_lambda",
            ],
            "official_dev_used": False,
        },
        "thresholds": {
            "baseline_ndcg_at_k": (
                baseline_ndcg
            ),
            "minimum_ndcg_at_k": (
                ndcg_threshold
            ),
            "baseline_recall_at_k": (
                baseline_recall
            ),
            "minimum_recall_at_k": (
                recall_threshold
            ),
            "minimum_ndcg_ci_lower_bound": (
                ndcg_ci_lower_threshold
            ),
            "baseline_mean_intra_list_diversity": (
                baseline_ild
            ),
            "baseline_exposure_gini": (
                baseline_gini
            ),
        },
        "candidates": evaluations,
        "eligible_lambdas": [
            float(key)
            for key in eligible_keys
        ],
        "selection": {
            "selected_policy": (
                selected_policy
            ),
            "selected_lambda": (
                selected_lambda
            ),
            "baseline_retained": (
                selected_lambda == 1.0
            ),
            "metrics": {
                "ndcg_at_k": (
                    selected[
                        "relevance"
                    ][
                        "ndcg_at_k"
                    ]
                ),
                "recall_at_k": (
                    selected[
                        "relevance"
                    ][
                        "recall_at_k"
                    ]
                ),
                "mean_intra_list_diversity": (
                    selected[
                        "diversity"
                    ][
                        "mean_intra_list_diversity"
                    ]
                ),
                "exposure_gini": (
                    selected[
                        "exposure"
                    ][
                        "exposure_gini"
                    ]
                ),
                "catalog_coverage_at_k": (
                    selected[
                        "relevance"
                    ][
                        "catalog_coverage_at_k"
                    ]
                ),
                "changed_ranking_fraction": (
                    selected[
                        "changed_ranking_fraction_vs_lambda_1"
                    ]
                ),
            },
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

    markdown = f"""## Phase 06D: Constrained MMR selection

The preregistered Phase-06 selection rule was applied without changing
the lambda grid or relevance budget after observing the sweep.

Eligible non-baseline lambdas:

{", ".join(eligible_keys) if eligible_keys else "none"}

Selected policy:

**{selected_policy}**

Selected lambda:

**{selected_lambda:.2f}**

Selected metrics:

- NDCG@10: {selected["relevance"]["ndcg_at_k"]:.6f}
- Recall@10: {selected["relevance"]["recall_at_k"]:.6f}
- mean intra-list diversity:
  {selected["diversity"]["mean_intra_list_diversity"]:.6f}
- exposure Gini:
  {selected["exposure"]["exposure_gini"]:.6f}
- catalog coverage@10:
  {selected["relevance"]["catalog_coverage_at_k"]:.6f}
- changed rankings:
  {selected["changed_ranking_fraction_vs_lambda_1"]:.2%}

The selection is determined by the frozen Phase-06 rule rather than
post-hoc preference among the observed policies.
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
