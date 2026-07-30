# NewsLens roadmap

Each phase should be completed in its own feature branch and pull request.
A phase is considered complete only after its implementation, tests, and
documentation are committed and the continuous-integration checks pass.

## Phase 1 - Foundation

- [x] Package layout
- [x] Isolated Python environment instructions
- [x] Smoke tests
- [x] Ruff configuration
- [x] GitHub Actions workflow
- [x] Initial GitHub commit made by the project owner

Completed commit:

```text
chore: initialize NewsLens project
```

## Phase 2 - MIND ingestion and audit

- [ ] Define typed `Article` and `Impression` domain records
- [x] Parse `news.tsv`
- [x] Parse `behaviors.tsv`
- [x] Validate the expected TSV schemas
- [x] Reject malformed rows and invalid labels
- [x] Reject duplicate article and impression identifiers
- [x] Parse timestamps, histories, and candidate impressions
- [x] Report users, articles, impressions, click rate, and missing fields
- [x] Add unit tests with small synthetic TSV fixtures
- [x] Test the command-line audit workflow
- [x] Document why downloaded data are excluded from Git
- [x] Add a reproducible dataset-audit command
- [x] Run the audit on the MIND-small train and development splits
- [x] Publish reproducible JSON audit reports
- [x] Confirm that referenced articles have corresponding metadata
- [x] Document dataset limitations and non-claims

Completed commits include:

```text
feat: add validated MIND data loaders
feat: add MIND dataset audit metrics
feat: add reproducible MIND data audit command
style: format MIND data loader
docs: record MIND data audit and limitations
```

Phase 2 evidence:

```text
reports/mindsmall_train_audit.json
reports/mindsmall_dev_audit.json
```

Remaining improvement:

```text
Define typed Article and Impression domain records if they provide a clear
benefit over the currently validated tabular representation.
```

## Phase 3 - Evaluation-safe baselines

- [ ] Define a chronological train/validation split
- [ ] Test the split for temporal leakage
- [ ] Implement a popularity recommender
- [ ] Implement TF-IDF article search
- [ ] Implement a content-based user-profile recommender
- [ ] Add a cold-start fallback
- [ ] Store reproducible baseline configurations
- [ ] Add unit and integration tests
- [ ] Document baseline assumptions and limitations

Expected outputs:

```text
src/newslens/evaluation/split.py
src/newslens/models/popularity.py
src/newslens/models/tfidf.py
tests/test_split.py
tests/test_baselines.py
```

## Phase 4 - Ranking evaluation

- [ ] Implement NDCG@K
- [ ] Implement mean reciprocal rank
- [ ] Implement Recall@K
- [ ] Implement catalog coverage
- [ ] Validate metrics with hand-calculated test cases
- [ ] Create a reproducible evaluation command
- [ ] Produce a baseline-comparison table
- [ ] Perform error and user-segment analysis
- [ ] Document uncertainty and evaluation limitations

Expected outputs:

```text
src/newslens/evaluation/metrics.py
reports/baseline_metrics.json
reports/baseline_comparison.md
```

## Phase 5 - Hybrid ranking

- [ ] Combine content, popularity, and recency features
- [ ] Tune weights using only training and validation data
- [ ] Compare the hybrid method against every simple baseline
- [ ] Analyze cold-start performance
- [ ] Analyze performance across user-history segments
- [ ] Publish positive and negative findings
- [ ] Document known failure modes

## Phase 6 - Serving

- [ ] Save and validate a trained model artifact
- [ ] Implement a FastAPI health endpoint
- [ ] Implement an article-search endpoint
- [ ] Implement a recommendation endpoint
- [ ] Implement a feedback-event endpoint
- [ ] Define a SQL event schema
- [ ] Add API integration tests
- [ ] Add request validation and error handling
- [ ] Document the API contract

## Phase 7 - MLOps and deployment

- [ ] Add MLflow experiment tracking
- [ ] Create a Docker image
- [ ] Add CI checks for linting and tests
- [ ] Add a CI image-build check
- [ ] Add model-version metadata
- [ ] Add model and data validation checks
- [ ] Add monitoring and structured logging
- [ ] Run a load test and report p50 and p95 latency
- [ ] Create a public deployment
- [ ] Add a model card
- [ ] Add complete reproducibility instructions

## Phase 8 - ExperimentLab

- [ ] Implement deterministic user assignment
- [ ] Define exposure and outcome events
- [ ] Add a sample-ratio-mismatch check
- [ ] Implement power analysis
- [ ] Calculate confidence intervals
- [ ] Define product guardrails
- [ ] Define latency and reliability guardrails
- [ ] Document experiment assumptions and limitations

## Portfolio completion criteria

NewsLens will be considered portfolio-ready when it includes:

- [ ] validated, leakage-safe data preparation;
- [ ] at least three documented recommendation or search baselines;
- [ ] correct ranking metrics with unit tests;
- [ ] a reproducible comparison on held-out data;
- [ ] error and segment analysis;
- [ ] a tested inference API;
- [ ] continuous integration and containerization;
- [ ] documented model limitations; and
- [ ] reproducible setup and evaluation instructions.