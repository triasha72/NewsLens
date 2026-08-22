from __future__ import annotations

import pytest

from newslens.experimentation import (
    ExperimentDesignError,
    Guardrail,
    MetricDirection,
    build_experiment_plan,
    check_sample_ratio_mismatch,
    compare_offline_online_effects,
    team_draft_interleave,
)


def _plan():
    return build_experiment_plan(
        experiment_id="mmr-080-vs-relevance",
        hypothesis="MMR improves retained reading without harming reliability.",
        eligibility="Signed-in users with at least one known history article.",
        assignment_unit="user_id",
        control="frozen two-tower ranking",
        treatment="frozen MMR lambda=0.80 ranking",
        primary_metric="satisfied_read_rate",
        baseline_rate=0.20,
        minimum_detectable_relative_effect=0.05,
        guardrails=(
            Guardrail("p95_latency_ms", MetricDirection.DECREASE, 0.10),
            Guardrail("hide_rate", MetricDirection.DECREASE, 0.02),
        ),
    )


def test_plan_freezes_power_and_decision_contract() -> None:
    plan = _plan()

    assert plan.required_users_per_arm == 25583
    assert plan.required_total_users == 51166
    assert plan.minimum_detectable_absolute_effect == pytest.approx(0.01)
    assert plan.sequential_peeking_allowed is False
    assert plan.to_dict()["guardrails"][0]["direction"] == "decrease"


def test_plan_rejects_missing_guardrails() -> None:
    with pytest.raises(ExperimentDesignError, match="guardrail"):
        build_experiment_plan(
            experiment_id="x",
            hypothesis="h",
            eligibility="e",
            assignment_unit="user",
            control="a",
            treatment="b",
            primary_metric="click_rate",
            baseline_rate=0.2,
            minimum_detectable_relative_effect=0.05,
            guardrails=(),
        )


def test_sample_ratio_mismatch_distinguishes_noise_from_large_skew() -> None:
    balanced = check_sample_ratio_mismatch(5005, 4995)
    skewed = check_sample_ratio_mismatch(6000, 4000)

    assert balanced.mismatch_detected is False
    assert skewed.mismatch_detected is True
    assert skewed.p_value < 0.001


def test_offline_online_divergence_is_explicit() -> None:
    comparison = compare_offline_online_effects(0.02, -0.01)

    assert comparison.signs_agree is False
    assert "do not use" in comparison.interpretation


def test_team_draft_interleaving_preserves_attribution() -> None:
    result = team_draft_interleave(
        ["a", "b", "c", "d"],
        ["b", "e", "a", "f"],
        k=5,
    )

    assert result.ranking == ("a", "b", "c", "e", "d")
    assert len(set(result.ranking)) == len(result.ranking)
    assert result.credit({"a", "e", "f"}) == {"control": 1, "treatment": 1}
