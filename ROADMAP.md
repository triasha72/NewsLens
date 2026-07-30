# NewsLens roadmap

NewsLens is being developed through small, testable feature branches and pull
requests. A phase is complete only when its implementation, tests,
documentation, and continuous-integration checks pass.

## Current status

- Phase 1: Foundation — completed
- Phase 2: MIND ingestion and audit — completed
- Phase 3: Evaluation-safe baselines — completed
- Phase 4: Ranking evaluation — next
- Phase 5: Hybrid ranking — planned
- Phase 6: Serving — planned
- Phase 7: MLOps and deployment — planned
- Phase 8: Controlled experiments — planned

## Phase 1: Foundation

- [x] Create the Python package layout
- [x] Add isolated environment instructions
- [x] Add smoke tests
- [x] Configure Ruff
- [x] Add a GitHub Actions workflow
- [x] Create the initial GitHub repository
- [x] Establish a feature-branch and pull-request workflow

Completed commit:

```text
chore: initialize NewsLens project
```

## Phase 2: MIND ingestion and audit

- [x] Parse `news.tsv`
- [x] Parse `behaviors.tsv`
- [x] Validate the expected TSV schemas
- [x] Reject malformed rows
- [x] Reject invalid click labels
- [x] Reject duplicate article identifiers
- [x] Reject duplicate impression identifiers
- [x] Parse behavior timestamps
- [x] Parse user histories
- [x] Parse candidate impressions
- [x] Preserve empty histories as valid cold-start records
- [x] Report users, articles, impressions, clicks, and missing fields
- [x] Check whether referenced articles have corresponding metadata
- [x] Add unit tests with synthetic TSV fixtures
- [x] Add a reproducible dataset-audit command
- [x] Run the audit on MIND-small train and development data
- [x] Publish reproducible JSON audit reports
- [x] Exclude raw licensed data from Git
- [x] Document dataset limitations and non-claims

Completed commits include:

```text
feat: add validated MIND news loader
feat: add validated MIND behavior loader
feat: add MIND dataset audit metrics
feat: add reproducible MIND data audit command
style: format MIND data loader
docs: record MIND data audit and limitations
```

Phase 2 evidence:

```text
src/newslens/data/mind.py
src/newslens/data/audit.py
tests/test_mind_loader.py
tests/test_audit.py
tests/test_cli.py
reports/mindsmall_train_audit.json
reports/mindsmall_dev_audit.json
```

### Optional Phase 2 hardening

- [ ] Define typed `Article` and `Impression` domain records
- [ ] Add streaming support for datasets larger than memory
- [ ] Add checksums for locally downloaded dataset archives
- [ ] Add structured validation-error reports

These are useful improvements but are not blockers for the current tabular
baseline pipeline.

## Phase 3: Evaluation-safe baselines

### Chronological splitting

- [x] Define a chronological train/validation split
- [x] Sort behavior records deterministically
- [x] Keep identical timestamps in the same partition
- [x] Require strict separation between training and validation timestamps
- [x] Test the split for temporal leakage
- [x] Confirm that the input data are not modified
- [x] Run the split on real MIND-small training records

Observed split:

| Property | Value |
|---|---:|
| Total records | 156,965 |
| Training records | 125,572 |
| Validation records | 31,393 |
| Actual validation fraction | 20% |
| Cutoff | 2019-11-13 20:36:26 |
| Last training timestamp | 2019-11-13 20:36:19 |
| First validation timestamp | 2019-11-13 20:36:26 |
| Temporal overlap | None |

The official MIND-small development split remains reserved as a later final
holdout.

### Popularity baseline

- [x] Count candidate exposures from training records
- [x] Count clicks from training records
- [x] Rank articles by training-only click count
- [x] Add deterministic article-ID tie-breaking
- [x] Assign unseen articles a score of zero
- [x] Support candidate-set ranking
- [x] Support exclusion of history articles
- [x] Test that validation records do not affect fitted scores
- [x] Run a preliminary validation diagnostic

Preliminary diagnostic:

| Diagnostic | Result |
|---|---:|
| Validation impressions evaluated | 31,393 |
| Top-1 clicked-article rate | 11.48% |
| Top-10 clicked-article rate | 57.05% |
| Average best clicked-article rank | 15.72 |
| Unseen candidate occurrence rate | 30.36% |
| Temporal leakage detected | No |

These values are preliminary. Formal ranking metrics belong to Phase 4.

### TF-IDF article search

- [x] Combine title, abstract, category, and subcategory text
- [x] Build unigram and bigram TF-IDF representations
- [x] Rank search results using cosine similarity
- [x] Return article identifiers, titles, categories, and scores
- [x] Support result exclusions
- [x] Handle unknown query vocabulary
- [x] Add deterministic ranking behavior
- [x] Add synthetic search tests
- [x] Run example searches on the real article catalog

### Content-based history recommender

- [x] Build TF-IDF article representations
- [x] Create user profiles from mean history vectors
- [x] Rank candidate articles using cosine similarity
- [x] Exclude previously viewed history articles
- [x] Support training-only vocabulary fitting
- [x] Transform available candidate metadata without validation labels
- [x] Detect unknown histories
- [x] Detect histories with no usable vocabulary
- [x] Add synthetic recommendation tests
- [x] Run a real MIND-small recommendation example

### Cold-start fallback

- [x] Detect empty user histories
- [x] Detect unknown user histories
- [x] Detect zero content-similarity signal
- [x] Fall back to training-only popularity
- [x] Preserve candidate restrictions
- [x] Exclude history articles during fallback
- [x] Record which strategy produced each recommendation
- [x] Test content and popularity routing
- [x] Keep fallback separate from future weighted hybrid ranking

### Reproducibility and documentation

- [x] Store version-controlled baseline parameters
- [x] Document dataset and holdout policies
- [x] Document temporal leakage controls
- [x] Document baseline assumptions
- [x] Document preliminary diagnostics
- [x] Document known limitations and non-claims
- [x] Add unit and integration tests
- [x] Pass all 66 automated tests
- [x] Pass Ruff and GitHub Actions checks

Completed commits include:

```text
feat: add leakage-safe chronological split
feat: add training-only popularity baseline
feat: add TF-IDF article search baseline
feat: add content-based history recommender
feat: add popularity cold-start fallback
docs: document evaluation-safe baselines
```

Phase 3 evidence:

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

## Phase 4: Ranking evaluation

### Metric implementation

- [ ] Implement NDCG@K
- [ ] Implement mean reciprocal rank
- [ ] Implement Recall@K
- [ ] Implement Hit Rate@K
- [ ] Implement catalog coverage
- [ ] Define behavior for impressions with multiple clicked articles
- [ ] Define behavior for impressions without a clicked article
- [ ] Validate every metric with hand-calculated examples
- [ ] Add boundary and invalid-input tests

### Reproducible evaluation

- [ ] Create a common candidate-ranking interface
- [ ] Create a reproducible evaluation command
- [ ] Evaluate popularity on chronological validation data
- [ ] Evaluate content recommendations on chronological validation data
- [ ] Evaluate the cold-start fallback
- [ ] Record model coverage and fallback frequency
- [ ] Measure unseen-article performance
- [ ] Store machine-readable evaluation results
- [ ] Generate a baseline-comparison table
- [ ] Reserve MIND-small development data as the final holdout

### Analysis

- [ ] Compare performance across user-history lengths
- [ ] Compare warm-start and cold-start users
- [ ] Analyze performance by article category
- [ ] Analyze unseen and low-exposure articles
- [ ] Inspect high-confidence failures
- [ ] Document positive and negative findings
- [ ] Add uncertainty estimates or confidence intervals
- [ ] Document metric limitations

Expected outputs:

```text
src/newslens/evaluation/metrics.py
src/newslens/evaluation/evaluator.py
tests/test_metrics.py
tests/test_evaluator.py
reports/baseline_metrics.json
reports/baseline_comparison.md
```

Suggested commits:

```text
feat: add tested ranking metrics
feat: add reproducible baseline evaluator
docs: publish baseline comparison and error analysis
```

## Phase 5: Hybrid ranking

- [ ] Combine content, popularity, and recency features
- [ ] Normalize component scores
- [ ] Tune weights using only training and validation data
- [ ] Compare against every simple baseline
- [ ] Test robustness across user-history lengths
- [ ] Analyze cold-start performance
- [ ] Analyze article-category performance
- [ ] Publish positive and negative results
- [ ] Document known failure modes
- [ ] Avoid using the final holdout during tuning

Potential future extensions:

- [ ] Learn-to-rank model
- [ ] Dense text embeddings
- [ ] Neural user representation
- [ ] Diversity-aware re-ranking
- [ ] Calibration or uncertainty estimates

## Phase 6: Serving

- [ ] Save and validate trained model artifacts
- [ ] Implement a FastAPI health endpoint
- [ ] Implement an article-search endpoint
- [ ] Implement a recommendation endpoint
- [ ] Implement a feedback-event endpoint
- [ ] Define a SQL event schema
- [ ] Add request validation
- [ ] Add structured error handling
- [ ] Add API integration tests
- [ ] Document the API contract
- [ ] Document cold-start behavior
- [ ] Add model-version information to responses

Expected outputs:

```text
src/newslens/api/
src/newslens/service/
tests/test_api.py
docs/API.md
```

## Phase 7: MLOps and deployment

- [ ] Add MLflow experiment tracking
- [ ] Add data and model-version metadata
- [ ] Add reproducible model-training commands
- [ ] Create a Docker image
- [ ] Add a CI Docker-build check
- [ ] Add artifact-validation checks
- [ ] Add structured application logging
- [ ] Add model and data monitoring
- [ ] Add latency and reliability measurements
- [ ] Run a load test
- [ ] Report p50 and p95 latency
- [ ] Create a public deployment
- [ ] Add a model card
- [ ] Add complete reproduction instructions
- [ ] Document deployment limitations

## Phase 8: Controlled experiments

- [ ] Implement deterministic user assignment
- [ ] Define exposure events
- [ ] Define outcome events
- [ ] Add a sample-ratio-mismatch check
- [ ] Implement power analysis
- [ ] Calculate confidence intervals
- [ ] Define product guardrails
- [ ] Define latency guardrails
- [ ] Define reliability guardrails
- [ ] Add experiment-analysis tests
- [ ] Document experiment assumptions
- [ ] Document experiment limitations

## Portfolio completion criteria

NewsLens will be considered portfolio-ready when it includes:

- [x] validated real-world data ingestion;
- [x] leakage-safe data preparation;
- [x] reproducible dataset auditing;
- [x] at least three documented recommendation or search baselines;
- [x] an explicit cold-start policy;
- [x] automated tests and continuous integration;
- [x] documented assumptions and model limitations;
- [ ] correct ranking metrics with unit tests;
- [ ] a reproducible held-out baseline comparison;
- [ ] error and user-segment analysis;
- [ ] uncertainty estimates or confidence intervals;
- [ ] a tested inference API;
- [ ] containerization;
- [ ] experiment tracking;
- [ ] public deployment; and
- [ ] a model card with complete reproduction instructions.