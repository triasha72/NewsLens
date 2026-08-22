"""Diagnostics that prevent invalid online ranking conclusions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, isfinite, sqrt


class ExperimentDiagnosticError(ValueError):
    """Raised when online experiment diagnostics receive invalid data."""


@dataclass(frozen=True)
class SampleRatioMismatch:
    observed_control: int
    observed_treatment: int
    expected_treatment_fraction: float
    z_score: float
    p_value: float
    alpha: float
    mismatch_detected: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OfflineOnlineComparison:
    offline_effect: float
    online_effect: float
    signs_agree: bool
    interpretation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def check_sample_ratio_mismatch(
    observed_control: int,
    observed_treatment: int,
    *,
    expected_treatment_fraction: float = 0.5,
    alpha: float = 0.001,
) -> SampleRatioMismatch:
    """Run a two-sided normal test for assignment imbalance."""

    if observed_control < 0 or observed_treatment < 0:
        raise ExperimentDiagnosticError("Observed assignment counts must be non-negative.")
    total = observed_control + observed_treatment
    if total == 0:
        raise ExperimentDiagnosticError("At least one assignment is required.")
    if not 0.0 < expected_treatment_fraction < 1.0 or not 0.0 < alpha < 1.0:
        raise ExperimentDiagnosticError("Expected fraction and alpha must be in (0, 1).")
    expected = total * expected_treatment_fraction
    standard_error = sqrt(total * expected_treatment_fraction * (1.0 - expected_treatment_fraction))
    z_score = (observed_treatment - expected) / standard_error
    p_value = 1.0 - erf(abs(z_score) / sqrt(2.0))
    return SampleRatioMismatch(
        observed_control,
        observed_treatment,
        expected_treatment_fraction,
        z_score,
        p_value,
        alpha,
        p_value < alpha,
    )


def compare_offline_online_effects(
    offline_effect: float, online_effect: float
) -> OfflineOnlineComparison:
    """Make offline-online disagreement explicit instead of hiding it in prose."""

    if not isfinite(offline_effect) or not isfinite(online_effect):
        raise ExperimentDiagnosticError("Effects must be finite.")
    signs_agree = offline_effect == 0.0 == online_effect or offline_effect * online_effect > 0.0
    interpretation = (
        "Offline and online effects point in the same direction."
        if signs_agree
        else "Offline and online effects disagree; do not use the offline result as a product-impact proxy."
    )
    return OfflineOnlineComparison(offline_effect, online_effect, signs_agree, interpretation)
