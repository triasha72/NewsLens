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

---

## D003 - Separate real-time transport from offline recommendation research

### Decision

Use a small Go service and Kafka consumer group for streamed article ingestion,
PostgreSQL for the searchable write model, and the existing Python API for query
understanding and ranking.

### Context

The MIND pipeline is a batch research workflow whose DuckDB database and model
artifacts are deliberately local and immutable. New articles introduce concurrent
writes, bursts, redelivery, failure recovery, and freshness expectations. Putting
those concerns into the recommendation process would couple model readiness to
event transport and create an unbounded in-memory failure boundary.

### Rationale

Go provides a small statically linked server for the I/O path. Kafka retains
bursts, preserves per-article key order, and distributes three partitions across
a consumer group. PostgreSQL supplies concurrent transactions and makes the event
ledger plus article update one atomic idempotency boundary. Python remains the
right place for transparent search features and same-repository evaluation.

### Consequences

The system now has independent recommendation and real-time readiness contracts.
Its at-least-once Kafka delivery can redeliver records during the one-second
offset-commit window, so stable event IDs are part of the API contract. A late
event cannot overwrite a newer article version. Invalid or exhausted records
require an operated DLQ review and replay process.

The local Compose system has one broker and one PostgreSQL instance. The split
creates clear production scaling and recovery boundaries, but it does not claim
broker replication, database high availability, or multi-zone operation.
