# NewsLens analytical data warehouse

NewsLens includes a local DuckDB data layer for reproducible SQL analysis of the
licensed MIND-small dataset. The warehouse is deliberately separate from the
model artifact: it supports data validation, feature generation, and exploratory
analysis, while the artifact contains only the fitted objects needed for serving.

DuckDB was selected because the project is currently a single-machine research
system. It provides a real relational database and standard SQL without requiring
users to provision a network service. PostgreSQL would be a reasonable next step
if NewsLens later needed concurrent writers, shared access, or an online event
store; none of those capabilities is claimed here.

## Schema

The warehouse normalizes a validated MIND split into four core tables:

| Table | Grain | Purpose |
|---|---|---|
| `articles` | one row per article | category, subcategory, text, URL, and entity JSON |
| `behavior_events` | one row per impression | user, timestamp, history size, and candidate count |
| `user_history` | one row per history position | ordered article history for each impression |
| `candidate_interactions` | one row per candidate position | candidate article and click label |

`warehouse_metadata` records the schema version, split, build time, and SHA-256
digests of the two source TSV files. The persisted `article_engagement` SQL view
reports candidate exposures, clicks, and click-through rate by article.

Foreign keys and primary keys preserve the relationships among articles,
impressions, histories, and candidates. Builds fail if a behavior row references
an article that is absent from `news.tsv`.

## Build a warehouse

After placing the licensed files under `data/` as described in the main README,
run:

```bash
python -m newslens build-warehouse \
  --data-dir data \
  --split train \
  --output warehouses/mindsmall_train.duckdb
```

The command first uses the existing validated MIND loaders. It then builds the
database at a temporary sibling path and atomically moves it into place only
after schema creation, reference validation, and all inserts succeed. Existing
databases are protected unless `--overwrite` is supplied.

Warehouse files are derived from licensed data and are excluded from Git and the
Docker image.

## Inspect the database through NewsLens

```bash
python -m newslens warehouse-summary \
  --database warehouses/mindsmall_train.duckdb
```

The JSON summary is computed with SQL and includes the number of articles,
impressions, users, history events, candidate interactions, clicks, and the
observed time range.

### Verified MIND-small training build

The complete MIND-small training split was materialized locally and summarized
through the command above:

| Warehouse measure | Verified result |
|---|---:|
| Schema version | `1.0.0` |
| Articles | 51,282 |
| Behavior events | 156,965 |
| Users | 50,000 |
| Ordered history events | 5,107,639 |
| Candidate interactions | 5,843,444 |
| Clicks | 236,344 |

The earliest stored behavior timestamp is `2019-11-09T00:00:19`, and the latest
is `2019-11-14T23:59:13`. The generated database is not distributed because it
is derived from licensed MIND data; the commands, schema, tests, and verified
aggregate results are reproducible by users with authorized dataset access.

## Query it directly with SQL

The DuckDB Python client can execute ad hoc queries without a separate server:

```bash
python - <<'PY'
import duckdb

database = "warehouses/mindsmall_train.duckdb"

with duckdb.connect(database, read_only=True) as connection:
    rows = connection.execute(
        """
        SELECT
            category,
            SUM(candidate_exposures) AS exposures,
            SUM(clicks) AS clicks,
            SUM(clicks)::DOUBLE / NULLIF(SUM(candidate_exposures), 0) AS ctr
        FROM article_engagement
        GROUP BY category
        HAVING SUM(candidate_exposures) > 0
        ORDER BY clicks DESC
        LIMIT 10
        """
    ).fetchall()

for row in rows:
    print(row)
PY
```

## Export leakage-safe training features

Feature extraction applies an exclusive timestamp cutoff, so interactions at or
after the validation boundary do not contribute to training features:

```bash
python -m newslens export-training-features \
  --database warehouses/mindsmall_train.duckdb \
  --cutoff-timestamp 2019-11-13T20:36:26 \
  --output reports/article_training_features.csv
```

The exported article-level table contains pre-cutoff exposures, clicks, and
click-through rate. Articles with no pre-cutoff interactions remain present with
zero-valued features, which makes the cold-start population explicit.

## Reproducibility and scope

- Raw TSV files are validated before SQL materialization.
- Source SHA-256 digests preserve build provenance.
- Atomic replacement prevents a failed rebuild from destroying a valid database.
- The schema is versioned and distributed with the Python package.
- Synthetic tests exercise schema constraints, SQL summaries, cutoff semantics,
  failure recovery, and all three warehouse CLI commands.

This is an analytical batch layer. It is not a hosted database, streaming
pipeline, online feature store, or multi-user production data platform.
