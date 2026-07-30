# NewsLens roadmap

Each phase should be completed in its own feature branch and pull request.

## Phase 1 - Foundation

- [x] Package layout
- [x] Isolated Python environment instructions
- [x] Smoke tests
- [x] Ruff configuration
- [x] GitHub Actions workflow
- [ ] Initial GitHub commit made by the project owner

Suggested commit:

```text
chore: initialize NewsLens package and test environment
```

## Phase 2 - MIND ingestion and audit

- [ ] Define typed `Article` and `Impression` records
- [ ] Parse `news.tsv`
- [ ] Parse `behaviors.tsv`
- [ ] Reject malformed rows and invalid labels
- [ ] Report users, articles, impressions, click rate, and missing fields
- [ ] Add unit tests with tiny synthetic TSV fixtures
- [ ] Document why the downloaded dataset is excluded from Git

Suggested commits:

```text
feat: add validated MIND data loaders
test: cover malformed MIND rows and labels
docs: record MIND data audit and limitations
```

## Phase 3 - Evaluation-safe baselines

- [ ] Chronological train/validation split
- [ ] Popularity recommender
- [ ] TF-IDF article search
- [ ] Content-based user-profile recommender
- [ ] Cold-start fallback
- [ ] Leakage tests

## Phase 4 - Ranking evaluation

- [ ] NDCG@K
- [ ] MRR
- [ ] Recall@K
- [ ] Catalog coverage
- [ ] Baseline comparison table
- [ ] Error and segment analysis

## Phase 5 - Hybrid ranking

- [ ] Combine content, popularity, and recency features
- [ ] Tune weights only on training/validation data
- [ ] Compare against every simple baseline
- [ ] Publish positive and negative results

## Phase 6 - Serving

- [ ] Save and validate a trained model artifact
- [ ] FastAPI health, search, recommendation, and feedback endpoints
- [ ] SQL event schema
- [ ] API integration tests
- [ ] Request validation and error handling

## Phase 7 - MLOps and deployment

- [ ] MLflow experiment tracking
- [ ] Docker image
- [ ] CI checks for lint, tests, and image build
- [ ] Load test with p50 and p95 latency
- [ ] Public deployment with documented limitations
- [ ] Model card and reproducibility instructions

## Phase 8 - ExperimentLab

- [ ] Deterministic user assignment
- [ ] Exposure and outcome events
- [ ] Sample-ratio-mismatch check
- [ ] Power analysis and confidence intervals
- [ ] Product and latency guardrails
