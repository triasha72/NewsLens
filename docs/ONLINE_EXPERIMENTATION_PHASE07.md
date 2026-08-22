# Phase 07 — Online ranking experiment contract

## Question

What evidence would be required before the frozen MMR policy could be said to
improve user or product outcomes?

Offline NDCG and diversity results cannot answer that question. Phase 07 adds a
pre-registration contract without claiming that a live experiment has occurred.

## Implemented contract

`newslens.experimentation` now provides:

- two-sided sample-size planning from a baseline rate, minimum detectable
  effect, significance level, and power;
- explicit eligibility, assignment unit, control, treatment, primary metric,
  novelty washout, maximum duration, and decision rules;
- required guardrail metrics and tolerated adverse movement;
- a sample-ratio-mismatch diagnostic;
- deterministic team-draft interleaving with click attribution; and
- an explicit offline-versus-online direction check.

The default design is deliberately fixed-horizon. Repeated significance checks
are not allowed by the generated plan.

## Proposed first experiment

The first eligible comparison is the frozen two-tower relevance order against
the frozen Phase-06 MMR `lambda=0.80` policy. A real deployment must choose its
baseline rate and minimum detectable effect before assignment begins.

Recommended metric families:

| Role | Metric |
|---|---|
| Primary | Satisfied-read rate with a preregistered dwell threshold |
| Secondary | Save rate, return rate, category discovery |
| Guardrail | Hide/report rate, empty-result rate, p95 latency, crash/error rate |
| Integrity | Sample-ratio mismatch and missing exposure/outcome joins |

Novelty-period outcomes should be reported separately. They must not be
silently pooled into the steady-state decision window.

## Non-claims

This module demonstrates experiment-design and diagnostic behavior using
hand-checkable tests. NewsLens has not assigned real users, collected product
events, or established causal impact.
