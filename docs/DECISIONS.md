# Technical decisions

## Decision template

### Decision

State the technical choice.

### Context

Explain the problem, constraints, and alternatives.

### Rationale

Explain why this option was selected.

### Consequences

Describe benefits, limitations, and future work.

---

## D001 - Begin with transparent baselines

### Decision

Implement popularity and TF-IDF baselines before neural retrieval or ranking.

### Context

The project needs a trustworthy reference point and a way to identify whether
later complexity produces a measurable improvement.

### Rationale

Simple baselines are fast, interpretable, and easier to audit for leakage.

### Consequences

Early results may be modest. Neural or embedding models will be added only
after the evaluation pipeline is validated.
