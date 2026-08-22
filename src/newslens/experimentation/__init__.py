"""Pre-registered online-experiment design utilities."""

from .design import (
    ExperimentDesignError,
    ExperimentPlan,
    Guardrail,
    MetricDirection,
    build_experiment_plan,
)
from .diagnostics import (
    ExperimentDiagnosticError,
    OfflineOnlineComparison,
    SampleRatioMismatch,
    check_sample_ratio_mismatch,
    compare_offline_online_effects,
)
from .interleaving import TeamDraftResult, team_draft_interleave

__all__ = [
    "ExperimentDesignError",
    "ExperimentDiagnosticError",
    "ExperimentPlan",
    "Guardrail",
    "MetricDirection",
    "OfflineOnlineComparison",
    "SampleRatioMismatch",
    "TeamDraftResult",
    "build_experiment_plan",
    "check_sample_ratio_mismatch",
    "compare_offline_online_effects",
    "team_draft_interleave",
]
