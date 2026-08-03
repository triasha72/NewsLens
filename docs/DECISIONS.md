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

---

## D002 - Use DuckDB for the first relational data layer

### Decision

Materialize validated MIND records in a normalized DuckDB database and expose
SQL summaries and cutoff-aware training features through the NewsLens CLI.

### Context

Repeated TSV parsing makes exploratory analysis less explicit and encourages
one-off dataframe transformations. NewsLens needs durable relational structure,
queryable provenance, and a way to express time-bounded feature extraction in
SQL. It does not currently need concurrent writers or a continuously available
database server.

### Rationale

DuckDB provides standard analytical SQL, transactions, constraints, and a
portable database file without adding an external service to a single-machine
research workflow. Keeping the original validated loaders as the ingestion
boundary avoids creating a second interpretation of the MIND format.

### Consequences

Warehouse builds are local batch operations. Source digests and atomic
replacement improve reproducibility and failure safety, but DuckDB is not being
used as an online feature store. A PostgreSQL service should be considered only
if future work establishes a need for shared access, concurrent writes, or
operational event ingestion.
