# NewsLens research and engineering roadmap

NewsLens began as an investigation into a deceptively simple problem: evaluating a news recommender without letting future behavior influence past recommendations. Each later component was added because an earlier result raised another question.

The current release is `v0.3.0`. The main branch now also includes a verified
normalized DuckDB analytical layer alongside the leakage-aware evaluation,
TF-IDF history model with training-only popularity fallback, uncertainty
analysis, versioned artifacts, FastAPI inference, observability, Docker
deployment, CI, and multi-platform container publishing.

This roadmap records questions worth testing next. It is intentionally not a list of technologies to add for their own sake.

## Current evidence

### Chronological evaluation

A strict chronological 80/20 split produces 125,572 training records and 31,393 validation records with no temporal overlap. The official MIND-small development split remains unused as a final holdout.

### Model behavior

The selected content-plus-fallback system reaches:

| Metric @10 | Result |
|---|---:|
| NDCG | 0.3664 |
| MRR | 0.3179 |
| Recall | 0.5955 |
| Hit Rate | 0.6762 |
| Catalog coverage | 0.0719 |

The content route handles 97.05% of validation impressions. Popularity recovers the remaining 2.95%, including empty histories and zero-signal profiles, so the final system returns no empty rankings under the current protocol.

### Uncertainty and diagnostics

Paired bootstrap intervals favor the fallback system over content-only ranking for all four reported relevance metrics. History-length, category, training-exposure, and high-score failure analyses show that aggregate metrics hide meaningful differences in behavior.

### Reproducible serving

The selected model can be exported as a versioned, checksummed artifact, loaded by a typed API, mounted read-only into a non-root container, validated in CI, and published for both `linux/amd64` and `linux/arm64`.

### Reproducible analytical data

Validated MIND records can be materialized into normalized DuckDB tables for
articles, impressions, ordered histories, and candidate interactions. The build
records source SHA-256 digests, replaces databases atomically, exposes a persisted
SQL engagement view, and produces pre-cutoff article features without using
validation-time events.

The verified MIND-small training build contains 51,282 articles, 156,965
behavior events from 50,000 users, 5,107,639 ordered history events, and
5,843,444 candidate interactions, including 236,344 clicks. The database spans
`2019-11-09T00:00:19` through `2019-11-14T23:59:13` and uses schema version
`1.0.0`. The generated database remains excluded from Git as a derivative of
licensed data.

Detailed evidence is recorded in:

- [`docs/EVALUATION.md`](docs/EVALUATION.md)
- [`docs/DECISIONS.md`](docs/DECISIONS.md)
- [`docs/API.md`](docs/API.md)
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
- [`docs/DATA_WAREHOUSE.md`](docs/DATA_WAREHOUSE.md)
- [`docs/RESEARCH_QUESTIONS.md`](docs/RESEARCH_QUESTIONS.md)
- [`reports/fallback_metrics.json`](reports/fallback_metrics.json)

## Questions already investigated

### Does a chronological split change what can be claimed?

Yes. It makes the information boundary explicit and prevents later interactions from entering model fitting. This does not eliminate every source of bias, but it removes a common and avoidable form of leakage.

### Is content similarity enough for every request?

No. TF-IDF produces useful rankings for most impressions, but it abstains for empty histories and zero-signal profiles. A deterministic popularity fallback recovers those cases without mixing incomparable score scales.

### Is the fallback improvement only a point-estimate artifact?

The paired impression-level bootstrap intervals exclude zero under the current fixed protocol. This is evidence of an offline improvement on these impressions, not evidence of causal or online impact.

### Do aggregate metrics describe all users and articles equally well?

No. Cold-start histories are weaker than longer histories, and performance varies substantially by article category and training exposure. Those differences motivate the next experiments.

### Can the fitted behavior be reproduced outside the evaluation script?

Yes. The artifact contract records configuration and metadata, verifies file integrity, and reproduces model-backed API rankings after load.

### Can the raw interaction structure be queried without repeating TSV parsing?

Yes. The DuckDB warehouse provides a versioned relational schema, SQL-derived
summaries, and cutoff-aware feature export. It remains a local analytical layer;
a shared PostgreSQL service or online feature store would require a distinct use
case and operational design.

## Next investigations

### 1. Untouched holdout evaluation

**Question:** Do the selected model and routing rules retain their advantage on the reserved MIND-small development split?

**Plan:**

- freeze the current model, preprocessing, routing, and metric definitions;
- document a one-time holdout protocol before reading results;
- evaluate popularity, content-only, and fallback systems once;
- report paired differences and uncertainty without retuning on the holdout; and
- keep the original validation results alongside the holdout results.

**Completion evidence:**

- a dated protocol;
- a deterministic report;
- explicit confirmation that no post-holdout tuning occurred; and
- documented discrepancies between validation and holdout behavior.

### 2. History recency and weighting

**Question:** Does treating every history article equally discard useful temporal information?

**Plan:**

- add recency-decayed and position-weighted history profiles;
- keep the same split, candidates, and ranking metrics;
- compare against the equal-weight profile with paired intervals; and
- inspect effects by history length.

A weighting scheme should be retained only if it improves more than the aggregate score while avoiding a material regression in cold-start and short-history cohorts.

### 3. Stronger sparse-text baselines

**Question:** Are the current results specific to TF-IDF, or do other transparent lexical methods produce a better quality–complexity tradeoff?

**Candidates:**

- BM25;
- sublinear TF scaling;
- title-only versus title-plus-abstract representations;
- category-aware lexical features; and
- query or history term weighting.

The goal is to establish a stronger interpretable baseline before introducing dense representations.

### 4. Semantic representations

**Question:** Which failures come from lexical mismatch rather than insufficient user signal?

**Plan:**

- identify failure examples with low lexical overlap but plausible semantic relation;
- evaluate a compact sentence-embedding baseline;
- keep candidate sets and evaluation semantics unchanged;
- measure memory, fitting time, artifact size, and inference latency; and
- compare quality gains with operational cost.

A semantic model should not replace the lexical baseline unless the added complexity is supported by both ranking and diagnostic evidence.

### 5. Routing and score calibration

**Question:** Can routing decisions use evidence beyond the binary presence of a positive TF-IDF signal?

**Possible tests:**

- margin between the top two content scores;
- profile norm and known-history fraction;
- calibrated route-specific confidence estimates;
- learned abstention on training-only features; and
- per-route error and coverage curves.

Content and popularity scores must remain separate unless a defensible calibration maps them onto a comparable interpretation.

### 6. Performance and reliability

**Question:** Where are the serving limits of the current artifact and API?

**Plan:**

- benchmark startup time, memory, throughput, and p50/p95/p99 latency;
- test multiple history and candidate-set sizes;
- exercise corrupt, missing, and incompatible artifacts;
- add graceful shutdown and concurrency tests;
- publish the hardware and workload used for every benchmark; and
- define thresholds before optimizing.

The same investigation should benchmark warehouse build time, database size, and
representative analytical queries before considering a shared PostgreSQL backend.

### 7. Online-experiment design

**Question:** What evidence would be needed before claiming user or product impact?

Before any online test, define:

- eligibility and assignment;
- exposure and outcome events;
- primary, secondary, and guardrail metrics;
- sample-ratio-mismatch checks;
- power and minimum-detectable-effect assumptions;
- stopping and decision rules; and
- privacy and retention constraints.

No online experiment has been conducted, and the roadmap does not treat offline improvements as a substitute for one.

## Working method

Every experiment should begin with a question and a written protocol. A change is complete when it includes:

1. the hypothesis or engineering risk;
2. the information available at decision time;
3. a deterministic implementation;
4. synthetic, hand-checkable tests;
5. a same-protocol comparison;
6. machine-readable results;
7. failure or subgroup inspection;
8. limitations and non-claims; and
9. a focused pull request.

Negative results should be retained when they change the understanding of the problem.

## Constraints

- Raw licensed MIND data is never committed or copied into published images.
- DuckDB database files and exported warehouse features derived from licensed
  records are never committed or copied into published images.
- Generated model artifacts remain outside Git.
- The official development split stays untouched until a written holdout protocol is frozen.
- New models do not get a different split, catalog, candidate policy, or metric implementation merely because it improves their score.
- Checksums detect corruption; they do not make untrusted pickle-compatible artifacts safe.
- A passing test suite and a healthy container do not establish production impact.
