# NewsLens roadmap

NewsLens is developed through small, testable feature branches and pull
requests. A milestone is complete only when its implementation, tests,
documentation, reproducible outputs, and continuous-integration checks pass.

The offline recommendation-system evaluation milestone is complete. The
current focus is production-style model packaging and serving.

## Current status

- [x] Phase 1 — Project foundation
- [x] Phase 2 — Validated MIND-small ingestion and dataset auditing
- [x] Phase 3 — Evaluation-safe search and recommendation baselines
- [x] Phase 4 — Offline ranking evaluation and model selection
- [x] Phase 5 — Rule-based content and popularity fallback candidate
- [ ] Phase 6 — Model packaging and serving *(in progress)*
- [ ] Phase 7 — MLOps, containerization, and deployment
- [ ] Phase 8 — Controlled online-experiment design

Current serving status:

- [x] Package and API version aligned to `0.2.0`
- [x] FastAPI application factory
- [x] Typed `/health` endpoint
- [x] Typed `/model-info` endpoint
- [x] Automatic OpenAPI and Swagger documentation
- [x] API tests
- [x] API usage documentation
- [ ] Versioned model-artifact contract
- [ ] Model serialization and loading
- [ ] Artifact compatibility and integrity validation
- [ ] Model-aware readiness endpoint
- [ ] `/search` inference endpoint
- [ ] `/recommend` inference endpoint
- [ ] Containerized deployment

The current API is a tested service foundation. It does not yet load a trained
model artifact or provide search and recommendation inference. Model readiness
must remain false until artifact loading and validation are implemented.

## Engineering principles

1. Fit models only with information allowed by the evaluation protocol.
2. Keep the official MIND-small development split reserved until a deliberate
   final-holdout decision is made.
3. Use deterministic ordering and explicit random seeds.
4. Validate behavior with synthetic, hand-checkable tests.
5. Preserve machine-readable reports for important experiments.
6. Document limitations and negative findings alongside positive findings.
7. Never commit raw licensed MIND data, trained redistributable artifacts
   without permission, credentials, or secrets.
8. Separate liveness, readiness, model loading, and inference concerns.
9. Publish work through reviewable feature branches and pull requests.

---

## Phase 1 — Project foundation

Status: **completed**

- [x] Create the Python `src/` package layout
- [x] Add isolated environment instructions
- [x] Add smoke tests
- [x] Configure Ruff formatting and linting
- [x] Add GitHub Actions continuous integration
- [x] Create the public GitHub repository
- [x] Establish feature-branch and pull-request development
- [x] Add a license, Makefile, and developer documentation
- [x] Exclude environments, caches, generated outputs, and raw data from Git

Evidence:

```text
pyproject.toml
Makefile
.github/workflows/ci.yml
src/newslens/
tests/test_smoke.py
README.md
START_HERE.md
```

---

## Phase 2 — Validated MIND-small ingestion and dataset auditing

Status: **completed**

### Loading and validation

- [x] Parse `news.tsv`
- [x] Parse `behaviors.tsv`
- [x] Validate expected TSV schemas
- [x] Reject malformed rows and invalid click labels
- [x] Reject duplicate article and impression identifiers
- [x] Parse behavior timestamps, histories, and candidate impressions
- [x] Preserve empty histories as valid cold-start records
- [x] Validate article references
- [x] Add synthetic loader fixtures and tests

### Dataset audit

- [x] Report users, articles, impressions, clicks, and non-clicks
- [x] Report missing titles and abstracts
- [x] Check candidate and history references against article metadata
- [x] Report empty histories, time coverage, and categories
- [x] Add a reproducible audit command
- [x] Audit MIND-small train and development data
- [x] Publish deterministic JSON audit reports
- [x] Document assumptions, limitations, and non-claims
- [x] Keep licensed raw data outside Git

Evidence:

```text
src/newslens/data/mind.py
src/newslens/data/audit.py
tests/test_mind_loader.py
tests/test_audit.py
tests/test_cli.py
reports/mindsmall_train_audit.json
reports/mindsmall_dev_audit.json
```

Optional hardening:

- [ ] Add streaming ingestion for datasets larger than memory
- [ ] Add checksums for locally downloaded archives
- [ ] Produce structured row-level validation-error reports
- [ ] Add immutable typed domain records

---

## Phase 3 — Evaluation-safe search and recommendation baselines

Status: **completed**

### Chronological splitting

- [x] Sort behavior records deterministically
- [x] Keep identical timestamps within one partition
- [x] Require strict temporal separation
- [x] Test temporal-leakage invariants
- [x] Confirm the input data are not modified
- [x] Run the split on real MIND-small training records

| Property | Value |
|---|---:|
| Total behavior records | 156,965 |
| Training records | 125,572 |
| Validation records | 31,393 |
| Actual validation fraction | 20% |
| Cutoff | 2019-11-13 20:36:26 |
| Final training timestamp | 2019-11-13 20:36:19 |
| First validation timestamp | 2019-11-13 20:36:26 |
| Temporal overlap | None |

The official MIND-small development split remains reserved and has not been
used for model selection.

### Baselines

- [x] Training-only click-count popularity recommender
- [x] TF-IDF article search
- [x] TF-IDF user-history content recommender
- [x] Training-only popularity fallback
- [x] Candidate-set restrictions
- [x] History-article exclusions
- [x] Deterministic article-ID tie-breaking
- [x] Empty-history handling
- [x] Unknown-history handling
- [x] Zero-profile and zero-signal handling
- [x] Synthetic and integration tests
- [x] Version-controlled parameters
- [x] Baseline assumptions and limitations documentation

Evidence:

```text
configs/baselines.json
docs/BASELINES.md
src/newslens/evaluation/split.py
src/newslens/models/popularity.py
src/newslens/models/tfidf.py
src/newslens/models/content.py
src/newslens/models/fallback.py
tests/test_split.py
tests/test_popularity.py
tests/test_tfidf.py
tests/test_content.py
tests/test_fallback.py
```

---

## Phase 4 — Offline ranking evaluation and model selection

Status: **completed**

Popularity, TF-IDF history content, and content with popularity fallback were
evaluated under one chronological protocol. The milestone also includes
subgroup analysis, uncertainty estimation, paired comparison, and
high-confidence failure inspection.

### Evaluation framework

- [x] Binary NDCG@K
- [x] Mean Reciprocal Rank@K
- [x] Recall@K
- [x] Hit Rate@K
- [x] Catalog coverage
- [x] Multiple-click and no-click semantics
- [x] Rankings shorter than `K`
- [x] Duplicate recommendation validation
- [x] Hand-calculated metric tests
- [x] Model-independent candidate-ranking evaluator
- [x] Empty-ranking and skipped-impression accounting
- [x] Deterministic evaluation behavior

### Primary model results

All systems were evaluated on the same 31,393 chronological validation
impressions at `K = 10`.

| Metric | Popularity | TF-IDF content | Content + popularity fallback |
|---|---:|---:|---:|
| NDCG@10 | 0.2853 | 0.3594 | **0.3664** |
| MRR@10 | 0.2308 | 0.3133 | **0.3179** |
| Recall@10 | 0.5047 | 0.5819 | **0.5955** |
| Hit Rate@10 | 0.5705 | 0.6610 | **0.6762** |
| Catalog Coverage@10 | 0.0402 | **0.0719** | **0.0719** |
| Unique recommended articles | 2,061 | **3,687** | **3,687** |
| Empty rankings | 0 | 927 | **0** |

Selected serving candidate:

```text
tfidf_content_with_popularity_fallback
```

Selection rationale:

- strongest recorded NDCG, MRR, Recall, and Hit Rate;
- recovers every content-model abstention;
- preserves content catalog coverage;
- uses an explicit, deterministic, tested cold-start policy; and
- is practical for the serving milestone.

This is an offline model-selection result, not evidence of online impact.

### Fallback routing

| Route | Impressions | Share |
|---|---:|---:|
| Content | 30,466 | 97.05% |
| Popularity fallback | 927 | 2.95% |
| Empty-history fallback | 801 | 2.55% |
| Zero-signal fallback | 126 | 0.40% |
| Unknown-history fallback | 0 | 0.00% |
| Zero-profile fallback | 0 | 0.00% |

- [x] Recover all 927 content abstentions
- [x] Produce no empty fallback rankings
- [x] Preserve candidate restrictions
- [x] Preserve temporal-leakage protections

### History-length analysis

- [x] Define mutually exclusive and exhaustive history groups
- [x] Assign every validation impression exactly once
- [x] Keep membership independent of validation clicks
- [x] Preserve overall metrics after segmentation

| Segment | History length | Impressions | Share | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold start | 0 | 801 | 2.55% | 0.2954 | 0.2334 | 0.5339 | 0.6017 |
| Short | 1–4 | 3,379 | 10.76% | 0.3471 | 0.2809 | 0.5911 | 0.6324 |
| Medium | 5–9 | 4,994 | 15.91% | **0.3767** | 0.3184 | **0.6148** | 0.6676 |
| Long | 10+ | 22,219 | 70.78% | 0.3695 | **0.3265** | 0.5940 | **0.6875** |

These are descriptive subgroup findings, not causal effects of history length.

### Article-category analysis

- [x] Define cohorts from clicked-article metadata
- [x] Preserve global ranking positions within cohorts
- [x] Record overlapping category membership
- [x] Apply minimum-support rules
- [x] Preserve overall and history-segment results

Key findings:

- 43,413 impression-category memberships occurred across 31,393 impressions.
- 8,074 impressions contained clicks from multiple categories.
- Weather recorded the strongest supported relevance metrics.
- Lifestyle recorded the broadest supported within-category coverage.
- Travel recorded the weakest supported relevance metrics.
- `kids`, `middleeast`, and `northamerica` had no clicked validation support.

Category memberships overlap, so their shares must not be summed. These are
descriptive, outcome-conditioned diagnostics.

### Training-exposure analysis

- [x] Define unseen as zero chronological-training candidate exposures
- [x] Define low, medium, and high exposure bands
- [x] Preserve global ranking positions
- [x] Record overlapping band membership
- [x] Preserve overall, history, and category results

| Band | Exposures | Catalog articles | Relevant impressions | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unseen | 0 | 34,450 | 15,003 | **0.3924** | **0.3291** | **0.6298** | **0.6655** | 0.0577 |
| Low | 1–9 | 9,648 | 13,038 | 0.2807 | 0.2159 | 0.5162 | 0.5512 | 0.0470 |
| Medium | 10–99 | 4,405 | 3,012 | 0.2916 | 0.2246 | 0.5136 | 0.5196 | 0.0854 |
| High | 100+ | 2,779 | 8,580 | 0.2740 | 0.2265 | 0.4697 | 0.5169 | **0.3127** |

The unseen result is compatible with lexical generalization to articles whose
metadata are available but which had no chronological-training candidate
exposures. It is not evidence that low exposure causes stronger performance.

### Bootstrap uncertainty

- [x] Resample evaluated impressions
- [x] Run 1,000 bootstrap samples
- [x] Use 95% percentile intervals
- [x] Use random seed 42
- [x] Record standard errors
- [x] Reproduce evaluator point estimates
- [x] Confirm deterministic repeated reports

| Metric | Estimate | Lower 95% | Upper 95% | Standard error |
|---|---:|---:|---:|---:|
| NDCG@10 | 0.3664 | 0.3627 | 0.3703 | 0.0019 |
| MRR@10 | 0.3179 | 0.3140 | 0.3218 | 0.0020 |
| Recall@10 | 0.5955 | 0.5907 | 0.6003 | 0.0026 |
| Hit Rate@10 | 0.6762 | 0.6712 | 0.6815 | 0.0027 |

These intervals quantify impression-sampling variability conditional on the
fixed split, model, candidates, and metric definitions. They do not quantify
temporal drift, model-selection uncertainty, or online impact.

### Paired comparison

- [x] Compare aligned content and fallback rankings
- [x] Resample aligned impression pairs
- [x] Report fallback-minus-content differences
- [x] Confirm all reported 95% intervals exclude zero

| Metric | Content | Fallback | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|
| NDCG@10 | 0.3594 | 0.3664 | +0.0069 | [0.0058, 0.0081] |
| MRR@10 | 0.3133 | 0.3179 | +0.0046 | [0.0034, 0.0058] |
| Recall@10 | 0.5819 | 0.5955 | +0.0135 | [0.0119, 0.0152] |
| Hit Rate@10 | 0.6610 | 0.6762 | +0.0152 | [0.0135, 0.0170] |

The comparison provides evidence of an offline improvement under this exact
protocol. It does not establish production or causal impact.

### High-confidence failure analysis

- [x] Define source-specific score thresholds
- [x] Use the 90th percentile of top scores within each source
- [x] Identify high-score top-K misses
- [x] Keep content and popularity score interpretations separate
- [x] Retain deterministic examples for inspection
- [x] Document score-confidence limitations

| Property | Result |
|---|---:|
| Score-eligible impressions | 31,162 |
| Top-K misses | 10,164 (32.38%) |
| High-score misses | 1,222 (3.92% of eligible impressions) |
| Retained examples | 50 |
| Content threshold | 0.1918 |
| Popularity threshold | 1,007 training clicks |

High model scores are not calibrated probabilities. These examples are
debugging evidence, not proof of real-world confidence.

### Phase 4 evidence

```text
src/newslens/evaluation/metrics.py
src/newslens/evaluation/evaluator.py
src/newslens/evaluation/popularity.py
src/newslens/evaluation/content.py
src/newslens/evaluation/fallback.py
src/newslens/evaluation/segments.py
src/newslens/evaluation/categories.py
src/newslens/evaluation/exposure.py
src/newslens/evaluation/uncertainty.py
src/newslens/evaluation/comparison.py
src/newslens/evaluation/failures.py
reports/popularity_metrics.json
reports/content_metrics.json
reports/fallback_metrics.json
reports/baseline_comparison.json
docs/EVALUATION.md
```

### Closure checklist

- [x] Evaluate models on identical validation impressions
- [x] Publish deterministic machine-readable reports
- [x] Analyze history, category, and exposure subgroups
- [x] Estimate overall uncertainty
- [x] Run a paired model comparison
- [x] Inspect high-confidence failures
- [x] Document metric and comparison limitations
- [x] Select the serving candidate
- [x] Pass the complete offline-evaluation test suite
- [x] Tag the offline evaluation milestone as `v0.2.0`
- [x] Keep the official development split reserved

Deliberately deferred:

- [ ] Run one final evaluation on the official MIND-small development split

Before using the reserved split, freeze the model, configuration, metrics,
reporting plan, and no-further-tuning policy.

---

## Phase 5 — Rule-based hybrid fallback candidate

Status: **completed for the selected rule-based system**

The selected system combines TF-IDF history-content ranking with
training-only popularity whenever content cannot produce a usable ranking.
This is deterministic routing, not learned score fusion.

- [x] Define routing rules
- [x] Track recommendation source
- [x] Compare against both component models
- [x] Analyze cold-start and warm-start behavior
- [x] Quantify uncertainty
- [x] Inspect failure cases
- [x] Select the rule-based system for serving

Optional model research:

- [ ] Weighted content, popularity, and recency fusion
- [ ] Learning-to-rank baseline
- [ ] Dense text embeddings
- [ ] Neural user representation
- [ ] Diversity-aware re-ranking
- [ ] Calibrated confidence estimates

These experiments are optional and do not block production-ML engineering
work with the selected candidate.

---

## Phase 6 — Model packaging and serving

Status: **in progress**

### 6A. FastAPI foundation

- [x] Align package and API version metadata to `0.2.0`
- [x] Add FastAPI, Pydantic, Uvicorn, and API-test dependencies
- [x] Implement an application factory
- [x] Define typed response schemas
- [x] Implement `GET /health`
- [x] Implement `GET /model-info`
- [x] Generate `/docs` and `/openapi.json`
- [x] Add API tests
- [x] Verify the routes manually with Uvicorn
- [x] Document current usage in `docs/API.md`
- [x] Document current non-claims

Evidence:

```text
src/newslens/api/__init__.py
src/newslens/api/app.py
src/newslens/api/schemas.py
tests/test_api.py
docs/API.md
```

Current routes:

| Route | Purpose | Expected status |
|---|---|---:|
| `GET /health` | Process liveness | 200 |
| `GET /model-info` | Package and configured-model metadata | 200 |
| `GET /docs` | Swagger documentation | 200 |
| `GET /openapi.json` | OpenAPI schema | 200 |

The complete suite contains 348 tests after the API foundation was added.

### 6B. Versioned model artifacts — next milestone

- [ ] Define an artifact schema version
- [ ] Define typed metadata and a manifest
- [ ] Record package and dependency versions
- [ ] Record model name, configuration, and training cutoff
- [ ] Record vocabulary and indexed-catalog sizes
- [ ] Serialize the TF-IDF content model
- [ ] Serialize the popularity fallback
- [ ] Store inference-required article metadata
- [ ] Record file sizes and SHA-256 checksums
- [ ] Validate checksums before loading
- [ ] Reject missing or corrupt files
- [ ] Reject unsupported schema versions
- [ ] Reject incompatible package versions clearly
- [ ] Test save/load round trips with synthetic fixtures
- [ ] Confirm rankings survive round trips exactly
- [ ] Add a reproducible artifact-build command
- [ ] Document artifact security assumptions
- [ ] Keep licensed data and non-redistributable artifacts outside Git

Planned files:

```text
src/newslens/artifacts/__init__.py
src/newslens/artifacts/metadata.py
src/newslens/artifacts/manifest.py
src/newslens/artifacts/io.py
scripts/build_artifact.py
tests/test_artifacts.py
docs/ARTIFACTS.md
```

### 6C. Model-aware startup and readiness

- [ ] Configure an artifact directory through an environment variable
- [ ] Load and validate the artifact during FastAPI lifespan startup
- [ ] Store the loaded service in application state
- [ ] Keep `/health` independent of model state
- [ ] Add `GET /ready`
- [ ] Return readiness failure when no valid model is loaded
- [ ] Return artifact metadata from `/model-info`
- [ ] Test valid, missing, corrupt, and incompatible artifacts
- [ ] Document startup failure policy

### 6D. Search inference

- [ ] Define typed search requests and ranked results
- [ ] Implement `POST /search`
- [ ] Validate empty and oversized queries
- [ ] Validate `top_k`
- [ ] Support result exclusions
- [ ] Return deterministic results
- [ ] Return model and artifact versions
- [ ] Return 503 when the model is unavailable
- [ ] Add unit and API integration tests

### 6E. Recommendation inference

- [ ] Define typed recommendation requests and results
- [ ] Accept history article identifiers
- [ ] Support optional candidate restrictions
- [ ] Implement `POST /recommend`
- [ ] Preserve cold-start fallback behavior
- [ ] Return content or popularity source
- [ ] Validate unknown and duplicate identifiers
- [ ] Validate `top_k`
- [ ] Return model and artifact versions
- [ ] Return 503 when the model is unavailable
- [ ] Add unit and API integration tests

### 6F. Serving documentation

- [x] Document foundation endpoints
- [x] Document current limitations
- [ ] Document artifact configuration
- [ ] Document liveness versus readiness
- [ ] Document search examples
- [ ] Document recommendation examples
- [ ] Document cold-start routing in responses
- [ ] Document error responses

---

## Phase 7 — MLOps, containerization, and deployment

Status: **planned**

### Reproducibility and artifacts

- [ ] Add a complete training command
- [ ] Record data, code, parameter, and artifact versions
- [ ] Add artifact-validation checks to CI
- [ ] Add lightweight experiment tracking
- [ ] Add a model card

### Containerization

- [ ] Add a production Dockerfile
- [ ] Use a non-root runtime user
- [ ] Add a container health check
- [ ] Exclude raw data and secrets
- [ ] Add a CI container-build check
- [ ] Test local startup from the image

### Observability and performance

- [ ] Add structured logs and request identifiers
- [ ] Track readiness and inference failures
- [ ] Measure artifact-loading time
- [ ] Run controlled load tests
- [ ] Report p50, p95, and p99 latency
- [ ] Report throughput and error rate
- [ ] Document test hardware and limitations

### Deployment

- [ ] Select a deployment target
- [ ] Store artifacts outside the source repository
- [ ] Configure secrets and environment variables
- [ ] Deploy a synthetic or redistribution-safe demonstration model
- [ ] Validate public health, readiness, and inference endpoints
- [ ] Document cost, security, and availability limitations

---

## Phase 8 — Controlled online-experiment design

Status: **planned**

- [ ] Define deterministic user assignment
- [ ] Define exposure and outcome events
- [ ] Define experiment eligibility
- [ ] Define primary and secondary metrics
- [ ] Define product, latency, and reliability guardrails
- [ ] Add sample-ratio-mismatch checks
- [ ] Implement power and sample-size analysis
- [ ] Calculate uncertainty intervals
- [ ] Add experiment-analysis tests
- [ ] Document stopping and decision rules

No online experiment has been conducted. This phase demonstrates sound design
without inventing production impact.

---

## Recommended pull-request sequence

1. `feat/api-foundation`
   - application factory, health, model information, schemas, tests, and docs
2. `feat/model-artifacts`
   - metadata, serialization, manifest, checksums, loader, and tests
3. `feat/api-model-loading`
   - lifespan loading, application state, and readiness
4. `feat/api-search`
   - search endpoint and tests
5. `feat/api-recommendation`
   - recommendation endpoint, fallback source, and tests
6. `feat/containerization`
   - Dockerfile, health check, and CI build
7. `feat/observability`
   - structured logs, request IDs, and latency reporting
8. `docs/model-card`
   - model card, reproduction guide, and serving limitations

Each pull request should contain focused tests, updated documentation, and an
explicit statement of what remains unimplemented.

---

## Portfolio completion criteria

### Data and evaluation

- [x] Validated real-world data ingestion
- [x] Reproducible dataset auditing
- [x] Leakage-safe chronological preparation
- [x] Three documented search or recommendation baselines
- [x] Explicit cold-start behavior
- [x] Hand-validated ranking metrics
- [x] Model-independent offline evaluator
- [x] Machine-readable experiment reports
- [x] Same-split model comparison
- [x] History, category, and exposure analysis
- [x] Overall uncertainty intervals
- [x] Paired comparison intervals
- [x] High-confidence failure analysis
- [x] Selected serving candidate

### Software engineering

- [x] Installable Python package
- [x] Object-oriented model implementations
- [x] Command-line workflows
- [x] Automated tests and continuous integration
- [x] Formatting and linting
- [x] Feature-branch and pull-request history
- [x] Tested FastAPI foundation
- [x] Typed API schemas
- [x] Liveness and model-information endpoints
- [x] Automatic OpenAPI documentation
- [ ] Versioned and validated model artifacts
- [ ] Model-aware readiness
- [ ] Tested search inference API
- [ ] Tested recommendation inference API
- [ ] Containerization
- [ ] Load and latency testing
- [ ] Structured observability
- [ ] Public redistribution-safe demonstration deployment

### Research and product discipline

- [x] Explicit evaluation protocol
- [x] Reserved final-holdout policy
- [x] Deterministic random seeds
- [x] Subgroup and failure analysis
- [x] Statistical uncertainty
- [x] Honest non-claims about causality and online impact
- [ ] Model card
- [ ] Complete artifact reproduction guide
- [ ] Controlled-experiment design document

NewsLens is already a credible offline machine-learning portfolio project. The
remaining work will demonstrate full-lifecycle ML engineering through artifact
management, reliable inference, containerization, observability, and
deployment.

---

## Immediate next action

Complete and merge `feat/api-foundation`. After it is merged:

```bash
git switch main
git pull --ff-only origin main
git fetch --prune origin
git status -sb
git switch -c feat/model-artifacts
```

On `feat/model-artifacts`, define and test the metadata and manifest contracts
before implementing the serializer or loader.
