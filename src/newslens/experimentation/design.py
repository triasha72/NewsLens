"""Deterministic, pre-registered A/B-test planning for ranking experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil, isfinite, sqrt
from statistics import NormalDist


class ExperimentDesignError(ValueError):
    """Raised when an experiment plan is incomplete or internally inconsistent."""


class MetricDirection(StrEnum):
    """Direction in which a metric is considered beneficial."""

    INCREASE = "increase"
    DECREASE = "decrease"


@dataclass(frozen=True)
class Guardrail:
    """A metric and the largest tolerated adverse relative change."""

    name: str
    direction: MetricDirection
    maximum_adverse_relative_change: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ExperimentDesignError("Guardrail name must not be empty.")
        value = self.maximum_adverse_relative_change
        if not isfinite(value) or not 0.0 <= value < 1.0:
            raise ExperimentDesignError("maximum_adverse_relative_change must be in [0, 1).")


@dataclass(frozen=True)
class ExperimentPlan:
    """Frozen decision contract created before an online experiment begins."""

    experiment_id: str
    hypothesis: str
    eligibility: str
    assignment_unit: str
    control: str
    treatment: str
    primary_metric: str
    primary_metric_direction: MetricDirection
    baseline_rate: float
    minimum_detectable_absolute_effect: float
    minimum_detectable_relative_effect: float
    significance_level: float
    statistical_power: float
    allocation_control: float
    allocation_treatment: float
    required_users_per_arm: int
    required_total_users: int
    guardrails: tuple[Guardrail, ...]
    novelty_washout_days: int
    maximum_duration_days: int
    decision_rule: str
    sample_ratio_mismatch_alpha: float
    sequential_peeking_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["primary_metric_direction"] = self.primary_metric_direction.value
        result["guardrails"] = [
            {**asdict(g), "direction": g.direction.value} for g in self.guardrails
        ]
        return result


def _required_users_per_arm(
    *, baseline_rate: float, absolute_effect: float, alpha: float, power: float
) -> int:
    treatment_rate = baseline_rate + absolute_effect
    pooled_rate = (baseline_rate + treatment_rate) / 2.0
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    numerator = (
        z_alpha * sqrt(2.0 * pooled_rate * (1.0 - pooled_rate))
        + z_power
        * sqrt(baseline_rate * (1.0 - baseline_rate) + treatment_rate * (1.0 - treatment_rate))
    ) ** 2
    return ceil(numerator / absolute_effect**2)


def build_experiment_plan(
    *,
    experiment_id: str,
    hypothesis: str,
    eligibility: str,
    assignment_unit: str,
    control: str,
    treatment: str,
    primary_metric: str,
    baseline_rate: float,
    minimum_detectable_relative_effect: float,
    guardrails: tuple[Guardrail, ...],
    significance_level: float = 0.05,
    statistical_power: float = 0.80,
    novelty_washout_days: int = 7,
    maximum_duration_days: int = 28,
    sample_ratio_mismatch_alpha: float = 0.001,
) -> ExperimentPlan:
    """Build a two-sided, equal-allocation plan without looking at outcomes."""

    required_text = {
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "eligibility": eligibility,
        "assignment_unit": assignment_unit,
        "control": control,
        "treatment": treatment,
        "primary_metric": primary_metric,
    }
    empty = [name for name, value in required_text.items() if not value.strip()]
    if empty:
        raise ExperimentDesignError(f"Required text fields are empty: {', '.join(empty)}.")
    if control.strip() == treatment.strip():
        raise ExperimentDesignError("Control and treatment must be distinct.")
    for name, value in (
        ("baseline_rate", baseline_rate),
        ("minimum_detectable_relative_effect", minimum_detectable_relative_effect),
        ("significance_level", significance_level),
        ("statistical_power", statistical_power),
        ("sample_ratio_mismatch_alpha", sample_ratio_mismatch_alpha),
    ):
        if not isfinite(value) or not 0.0 < value < 1.0:
            raise ExperimentDesignError(f"{name} must be finite and between 0 and 1.")
    if statistical_power <= 0.5:
        raise ExperimentDesignError("statistical_power must be greater than 0.5.")
    absolute_effect = baseline_rate * minimum_detectable_relative_effect
    if baseline_rate + absolute_effect >= 1.0:
        raise ExperimentDesignError("The requested effect would make the treatment rate >= 1.")
    if not guardrails:
        raise ExperimentDesignError("At least one guardrail metric is required.")
    if novelty_washout_days < 0 or maximum_duration_days <= novelty_washout_days:
        raise ExperimentDesignError("maximum_duration_days must exceed novelty_washout_days.")
    users_per_arm = _required_users_per_arm(
        baseline_rate=baseline_rate,
        absolute_effect=absolute_effect,
        alpha=significance_level,
        power=statistical_power,
    )
    return ExperimentPlan(
        experiment_id.strip(),
        hypothesis.strip(),
        eligibility.strip(),
        assignment_unit.strip(),
        control.strip(),
        treatment.strip(),
        primary_metric.strip(),
        MetricDirection.INCREASE,
        baseline_rate,
        absolute_effect,
        minimum_detectable_relative_effect,
        significance_level,
        statistical_power,
        0.5,
        0.5,
        users_per_arm,
        users_per_arm * 2,
        guardrails,
        novelty_washout_days,
        maximum_duration_days,
        "Analyze once after the required sample and washout are complete. Ship only when the two-sided primary test passes and no guardrail exceeds its preregistered adverse-change limit.",
        sample_ratio_mismatch_alpha,
    )
