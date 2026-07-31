# NewsLens roadmap

NewsLens is being developed through small, testable feature branches and pull
requests. A phase is complete only when its implementation, tests,
documentation, reproducible outputs, and continuous-integration checks pass.

## Current status

- Phase 1: Foundation — completed
- Phase 2: MIND ingestion and audit — completed
- Phase 3: Evaluation-safe baselines — completed
- Phase 4: Ranking evaluation — in progress
  - ranking metrics — completed
  - model-independent evaluator — completed
  - popularity evaluation — completed
  - TF-IDF content evaluation — completed
  - cold-start fallback evaluation — next
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

These improvements are useful but are not blockers for the current tabular
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

These preliminary results have now been confirmed and extended using the formal
Phase 4 ranking evaluator.

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
- [x] Pass all 66 Phase 3 automated tests
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

Phase 4 is currently in progress. Formal metric implementation, the common
evaluator, popularity evaluation, and TF-IDF history-content evaluation are
complete. Evaluation of the rule-based cold-start fallback and segment-level
error analysis remain pending.

### Metric implementation

- [x] Implement binary NDCG@K
- [x] Implement Mean Reciprocal Rank@K
- [x] Implement Recall@K
- [x] Implement Hit Rate@K
- [x] Implement catalog coverage
- [x] Define behavior for impressions with multiple clicked articles
- [x] Define behavior for impressions without a clicked article
- [x] Define behavior for rankings shorter than `K`
- [x] Reject duplicate recommended identifiers
- [x] Validate cutoff and catalog inputs
- [x] Validate every metric with hand-calculated examples
- [x] Add boundary and invalid-input tests
- [x] Export the metrics through the evaluation package

### Model-independent evaluation

- [x] Create a common candidate-ranking interface
- [x] Separate model ranking from metric computation
- [x] Evaluate multiple-click impressions consistently
- [x] Track evaluated and skipped impressions
- [x] Track empty rankings
- [x] Track unique recommended articles
- [x] Calculate catalog coverage
- [x] Add synthetic evaluator tests
- [x] Add invalid-ranker and invalid-ranking tests
- [x] Confirm deterministic evaluation behavior

### Reproducible popularity evaluation

- [x] Create a popularity-specific evaluation pipeline
- [x] Load the real MIND-small training data
- [x] Apply the strict chronological split
- [x] Fit popularity only on chronological training interactions
- [x] Rank later validation candidate sets
- [x] Evaluate popularity using all formal ranking metrics
- [x] Measure unseen candidate occurrences
- [x] Record training and validation sizes
- [x] Record the chronological cutoff
- [x] Add a reproducible CLI command
- [x] Write machine-readable evaluation results
- [x] Confirm identical results across repeated runs
- [x] Evaluate all 31,393 validation impressions
- [x] Produce no empty rankings
- [x] Add unit, integration, and CLI tests
- [x] Pass all 120 automated tests locally
- [x] Pass Ruff locally
- [ ] Pass GitHub Actions for the current Phase 4 pull request

Popularity results at `K = 10`:

| Metric | Result |
|---|---:|
| NDCG@10 | 0.2853 |
| MRR@10 | 0.2308 |
| Recall@10 | 0.5047 |
| Hit Rate@10 | 0.5705 |
| Catalog Coverage@10 | 0.0402 |
| Evaluated impressions | 31,393 |
| Empty rankings | 0 |
| Unique recommended articles | 2,061 |
| Unseen candidate occurrence rate | 30.36% |
| Temporal leakage detected | No |

Phase 4 popularity evidence:

```text
src/newslens/evaluation/metrics.py
src/newslens/evaluation/evaluator.py
src/newslens/evaluation/popularity.py
tests/test_metrics.py
tests/test_evaluator.py
tests/test_popularity_evaluation.py
tests/test_cli.py
tests/test_smoke.py
reports/popularity_metrics.json
docs/EVALUATION.md
```

### Reproducible TF-IDF content evaluation

- [x] Create a content-specific evaluation pipeline
- [x] Fit the TF-IDF vocabulary from training-referenced articles only
- [x] Transform candidate metadata without using validation interaction labels
- [x] Use the same chronological split and candidate sets as popularity
- [x] Evaluate all 31,393 validation impressions
- [x] Count empty rankings as zero-scoring outcomes
- [x] Record content-ranked, cold-start, and zero-signal impressions
- [x] Record vocabulary and indexed-catalog sizes
- [x] Add a reproducible CLI command
- [x] Write machine-readable evaluation results
- [x] Confirm identical results across repeated runs
- [x] Add unit, integration, and CLI tests
- [x] Pass all 141 automated tests locally
- [x] Pass Ruff locally
- [ ] Pass GitHub Actions for the current Phase 4 pull request

Popularity versus TF-IDF content results at `K = 10`:

| Metric | Popularity | TF-IDF content | Relative change |
|---|---:|---:|---:|
| NDCG@10 | 0.2853 | 0.3594 | +26.0% |
| MRR@10 | 0.2308 | 0.3133 | +35.7% |
| Recall@10 | 0.5047 | 0.5819 | +15.3% |
| Hit Rate@10 | 0.5705 | 0.6610 | +15.9% |
| Catalog Coverage@10 | 0.0402 | 0.0719 | +78.9% |
| Unique recommended articles | 2,061 | 3,687 | +78.9% |
| Empty rankings | 0 | 927 | — |

Content-evaluation accounting:

| Property | Result |
|---|---:|
| Content-ranked impressions | 30,466 (97.05%) |
| Empty-history impressions | 801 (2.55%) |
| Zero-signal impressions | 126 (0.40%) |
| Total abstentions | 927 (2.95%) |
| Vocabulary articles | 47,367 |
| Indexed articles | 51,282 |
| Maximum vocabulary size | 50,000 |
| Temporal leakage detected | No |

All content abstentions remain in the metric denominator as empty rankings.

Phase 4 content evidence:

```text
src/newslens/evaluation/content.py
src/newslens/cli.py
tests/test_content_evaluation.py
tests/test_cli.py
reports/content_metrics.json
docs/EVALUATION.md
```

### Remaining model evaluation

- [x] Evaluate the content-based history recommender
- [ ] Evaluate the cold-start fallback
- [x] Record content-model coverage
- [ ] Record fallback frequency
- [ ] Measure unseen-article ranking performance
- [ ] Evaluate warm-start and cold-start impressions separately
- [x] Generate a two-baseline comparison table
- [x] Compare popularity and content using identical validation impressions
- [x] Reserve the official MIND-small development data for final evaluation
- [ ] Run final evaluation on the reserved development split only after model
      selection is complete

### Remaining analysis

- [ ] Compare performance across user-history lengths
- [ ] Compare warm-start and cold-start users
- [ ] Analyze performance by article category
- [ ] Analyze unseen and low-exposure articles
- [ ] Inspect high-confidence failures
- [x] Document initial popularity findings
- [x] Document popularity limitations and non-claims
- [ ] Document positive and negative findings across all baselines
- [ ] Add uncertainty estimates or confidence intervals
- [ ] Document complete metric and comparison limitations

Expected remaining outputs:

```text
reports/baseline_comparison.json
reports/baseline_comparison.md
docs/ERROR_ANALYSIS.md
```

Suggested future commits:

```text
feat: evaluate cold-start fallback baseline
analysis: compare warm-start and cold-start segments
docs: publish baseline error analysis
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
- [x] correct ranking metrics with hand-validated tests;
- [x] a model-independent offline evaluator;
- [x] a reproducible formal popularity evaluation;
- [x] a machine-readable popularity evaluation report;
- [x] a reproducible formal TF-IDF content evaluation;
- [x] a machine-readable content evaluation report;
- [x] a reproducible held-out two-baseline comparison;
- [ ] error and user-segment analysis;
- [ ] uncertainty estimates or confidence intervals;
- [ ] a tested inference API;
- [ ] containerization;
- [ ] experiment tracking;
- [ ] public deployment; and
- [ ] a model card with complete reproduction instructions.
