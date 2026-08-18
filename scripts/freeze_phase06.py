"""Freeze the final Phase-06 diversity and exposure result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    """Load one JSON report."""
    return json.loads(
        path.read_text()
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sweep-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--serving-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--phase05-final-report",
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

    baseline = _load(
        args.baseline_report
    )
    sweep = _load(
        args.sweep_report
    )
    selection = _load(
        args.selection_report
    )
    serving = _load(
        args.serving_report
    )
    phase05 = _load(
        args.phase05_final_report
    )

    if (
        baseline["integrity"]["passed"]
        is not True
    ):
        raise RuntimeError(
            "Phase-06B baseline audit did not pass."
        )

    if (
        sweep["baseline_parity"]["passed"]
        is not True
    ):
        raise RuntimeError(
            "Phase-06C baseline parity did not pass."
        )

    if (
        selection["selection"]["selected_lambda"]
        != 0.80
    ):
        raise RuntimeError(
            "Unexpected Phase-06 selected lambda."
        )

    if (
        selection["selection"]["selected_policy"]
        != "mmr_lambda_0.80"
    ):
        raise RuntimeError(
            "Unexpected Phase-06 selected policy."
        )

    if (
        serving["integrity"]["passed"]
        is not True
    ):
        raise RuntimeError(
            "Phase-06E serving benchmark did not pass."
        )

    if (
        serving["protocol"]["selected_lambda"]
        != 0.80
    ):
        raise RuntimeError(
            "Phase-06E used a different MMR lambda."
        )

    if (
        serving["protocol"][
            "quality_claim_for_global_candidates"
        ]
        != "none"
    ):
        raise RuntimeError(
            "Global FAISS benchmark made "
            "an unsupported relevance claim."
        )

    if (
        phase05["selected_retrieval_backend"]
        != "faiss_flat"
    ):
        raise RuntimeError(
            "Unexpected frozen retrieval backend."
        )

    logged_baseline = (
        sweep["policies"]["1.00"]
    )

    logged_selected = (
        sweep["policies"]["0.80"]
    )

    logged_comparison = (
        sweep[
            "paired_comparisons_vs_lambda_1"
        ][
            "0.80"
        ]
    )

    global_baseline = (
        serving[
            "baseline_relevance_top10"
        ]
    )

    global_selected = (
        serving[
            "selected_mmr_top10"
        ]
    )

    global_deltas = (
        serving[
            "deltas_selected_minus_baseline"
        ]
    )

    global_ild_improved = (
        float(
            global_deltas[
                "mean_intra_list_diversity"
            ]
        )
        > 0.0
    )

    global_gini_improved = (
        float(
            global_deltas[
                "exposure_gini"
            ]
        )
        < 0.0
    )

    global_coverage_improved = (
        float(
            global_deltas[
                "catalog_coverage"
            ]
        )
        > 0.0
    )

    payload = {
        "phase": "phase06",
        "status": "frozen",
        "offline_selected_policy": (
            "mmr_lambda_0.80"
        ),
        "selected_lambda": 0.80,
        "retrieval_backend": (
            "faiss_flat"
        ),
        "information_boundary": {
            "dataset": (
                "MINDsmall_train"
            ),
            "official_dev_used": False,
            "logged_candidate_benchmark_status": (
                "previously_observed_development_benchmark"
            ),
            "global_retrieval_quality_claim": (
                "none"
            ),
        },
        "logged_candidate_result": {
            "baseline": (
                logged_baseline
            ),
            "selected": (
                logged_selected
            ),
            "paired_comparison": (
                logged_comparison
            ),
            "selection": (
                selection["selection"]
            ),
        },
        "global_faiss_result": {
            "query_count": (
                serving[
                    "protocol"
                ][
                    "query_count"
                ]
            ),
            "retrieval_k": (
                serving[
                    "protocol"
                ][
                    "retrieval_k"
                ]
            ),
            "final_k": (
                serving[
                    "protocol"
                ][
                    "final_k"
                ]
            ),
            "baseline": (
                global_baseline
            ),
            "selected_mmr": (
                global_selected
            ),
            "deltas": (
                global_deltas
            ),
            "changed_top10_fraction": (
                serving[
                    "accounting"
                ][
                    "changed_top10_fraction"
                ]
            ),
            "mean_top10_set_overlap_fraction": (
                serving[
                    "accounting"
                ][
                    "mean_top10_set_overlap_fraction"
                ]
            ),
        },
        "systems_result": {
            "latency_scope": (
                serving[
                    "protocol"
                ][
                    "latency_scope"
                ]
            ),
            "retrieval_top100": (
                serving[
                    "latency_ms"
                ][
                    "retrieval_top100"
                ]
            ),
            "mmr_rerank_top100_to_top10": (
                serving[
                    "latency_ms"
                ][
                    "mmr_rerank_top100_to_top10"
                ]
            ),
            "baseline_post_embedding": (
                serving[
                    "latency_ms"
                ][
                    "baseline_post_embedding_end_to_end"
                ]
            ),
            "mmr_post_embedding": (
                serving[
                    "latency_ms"
                ][
                    "mmr_post_embedding_end_to_end"
                ]
            ),
            "p95_mmr_over_baseline_ratio": (
                serving[
                    "latency_ms"
                ][
                    "p95_mmr_over_baseline_ratio"
                ]
            ),
        },
        "conclusions": {
            "logged_candidate_relevance_budget_passed": (
                True
            ),
            "global_semantic_diversity_improved": (
                global_ild_improved
            ),
            "global_exposure_gini_improved": (
                global_gini_improved
            ),
            "global_catalog_coverage_improved": (
                global_coverage_improved
            ),
            "production_serving_readiness_established": (
                False
            ),
            "production_serving_reason": (
                "The frozen Phase-06 policy improved "
                "semantic diversity, but the correctness-first "
                "Python MMR implementation introduced substantial "
                "post-embedding latency. Phase 07 will optimize "
                "the same policy under exact ranking parity."
            ),
        },
        "limitations": [
            (
                "Global FAISS candidates do not have valid "
                "logged MIND relevance labels."
            ),
            (
                "Phase-06E therefore provides systems, "
                "diversity, and exposure evidence only."
            ),
            (
                "Global MMR improved semantic diversity but "
                "did not reduce aggregate exposure Gini."
            ),
            (
                "Global MMR did not improve catalog coverage."
            ),
            (
                "Current MMR implementation prioritizes "
                "deterministic correctness over serving latency."
            ),
            (
                "Latency component percentiles were measured "
                "in separate benchmark passes and are not "
                "additive percentiles."
            ),
        ],
    }

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.markdown_output.parent.mkdir(
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

    logged_ndcg_baseline = float(
        logged_baseline[
            "relevance"
        ][
            "ndcg_at_k"
        ]
    )

    logged_ndcg_selected = float(
        logged_selected[
            "relevance"
        ][
            "ndcg_at_k"
        ]
    )

    logged_recall_baseline = float(
        logged_baseline[
            "relevance"
        ][
            "recall_at_k"
        ]
    )

    logged_recall_selected = float(
        logged_selected[
            "relevance"
        ][
            "recall_at_k"
        ]
    )

    global_ild_baseline = float(
        global_baseline[
            "diversity"
        ][
            "mean_intra_list_diversity"
        ]
    )

    global_ild_selected = float(
        global_selected[
            "diversity"
        ][
            "mean_intra_list_diversity"
        ]
    )

    global_gini_baseline = float(
        global_baseline[
            "exposure"
        ][
            "exposure_gini"
        ]
    )

    global_gini_selected = float(
        global_selected[
            "exposure"
        ][
            "exposure_gini"
        ]
    )

    global_coverage_baseline = float(
        global_baseline[
            "exposure"
        ][
            "catalog_coverage"
        ]
    )

    global_coverage_selected = float(
        global_selected[
            "exposure"
        ][
            "catalog_coverage"
        ]
    )

    baseline_p95 = float(
        serving[
            "latency_ms"
        ][
            "baseline_post_embedding_end_to_end"
        ][
            "p95"
        ]
    )

    mmr_p95 = float(
        serving[
            "latency_ms"
        ][
            "mmr_post_embedding_end_to_end"
        ][
            "p95"
        ]
    )

    markdown = f"""## Phase 06F: Final diversity and exposure result

Phase 06 evaluated deterministic MMR reranking over the frozen
Phase-03 and Phase-04 recommendation stack.

### Selected offline policy

**MMR lambda = 0.80**

The policy was selected mechanically using the preregistered
relevance-retention, diversity, and exposure constraints.

### Logged-candidate benchmark

| Metric | Baseline | MMR 0.80 |
|---|---:|---:|
| NDCG@10 | {logged_ndcg_baseline:.6f} | {logged_ndcg_selected:.6f} |
| Recall@10 | {logged_recall_baseline:.6f} | {logged_recall_selected:.6f} |
| Mean ILD | {logged_baseline["diversity"]["mean_intra_list_diversity"]:.6f} | {logged_selected["diversity"]["mean_intra_list_diversity"]:.6f} |
| Exposure Gini | {logged_baseline["exposure"]["exposure_gini"]:.6f} | {logged_selected["exposure"]["exposure_gini"]:.6f} |

The selected policy remained inside the frozen relevance
non-inferiority budget.

### Global FAISS top-100 benchmark

The global benchmark makes no relevance-quality claim because
arbitrary FAISS candidates do not have valid logged click labels.

| Metric | Relevance top-10 | MMR 0.80 |
|---|---:|---:|
| Mean ILD | {global_ild_baseline:.6f} | {global_ild_selected:.6f} |
| Mean unique categories | {global_baseline["diversity"]["mean_unique_categories"]:.6f} | {global_selected["diversity"]["mean_unique_categories"]:.6f} |
| Mean unique subcategories | {global_baseline["diversity"]["mean_unique_subcategories"]:.6f} | {global_selected["diversity"]["mean_unique_subcategories"]:.6f} |
| Exposure Gini | {global_gini_baseline:.6f} | {global_gini_selected:.6f} |
| Catalog coverage | {global_coverage_baseline:.6f} | {global_coverage_selected:.6f} |

MMR changed
{serving["accounting"]["changed_top10_fraction"]:.2%}
of top-10 rankings while retaining mean top-10 set overlap of
{serving["accounting"]["mean_top10_set_overlap_fraction"]:.2%}.

Global semantic diversity improved.

Global aggregate exposure concentration did not improve.

### Systems result

Post-user-embedding p95:

- relevance-only path: {baseline_p95:.3f} ms
- current MMR path: {mmr_p95:.3f} ms

The current deterministic Python MMR implementation is not yet
considered production-serving ready.

Phase 06 did not preregister a latency promotion threshold, so this
is recorded as a systems limitation rather than a post-hoc policy
rejection.

### Final interpretation

Phase 06 freezes MMR lambda=0.80 as the selected offline diversity
policy.

The logged-candidate experiment preserved relevance inside the
predeclared non-inferiority budget.

The global FAISS benchmark showed a larger semantic-diversity effect,
but did not improve aggregate exposure concentration or catalog
coverage.

Phase 07 will optimize the same frozen MMR policy under exact ranking
parity and establish serving performance, observability,
containerization, Kubernetes deployment, and online experiment design.
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
