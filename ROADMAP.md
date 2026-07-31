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
  - cold-start fallback evaluation — completed
  - history-length segment evaluation — completed
  - article-category evaluation — completed
  - unseen and training-exposure analysis — completed
  - uncertainty estimation — next
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
complete. Rule-based cold-start fallback evaluation is also complete.
History-length segment evaluation and overlapping article-category evaluation
are complete. Unseen and training-exposure analysis is also complete.
Uncertainty estimation and high-confidence failure inspection remain pending.

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
- [x] Merge the popularity-evaluation pull request into main

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
- [x] Merge the content-evaluation pull request into main

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

### Reproducible cold-start fallback evaluation

- [x] Create a fallback-specific evaluation pipeline
- [x] Fit content and popularity only from chronological training information
- [x] Preserve content rankings when positive similarity is available
- [x] Route content abstentions to training-only popularity
- [x] Record content and popularity routing frequencies
- [x] Record empty-history, unknown-history, zero-profile, and zero-signal routes
- [x] Evaluate all 31,393 validation impressions
- [x] Recover all 927 content abstentions
- [x] Produce no empty rankings
- [x] Add a reproducible CLI command
- [x] Write machine-readable evaluation results
- [x] Confirm identical results across repeated runs
- [x] Add unit, integration, and CLI tests
- [x] Pass all 159 automated tests in the fallback-evaluation scope locally
- [x] Pass Ruff locally
- [x] Pass GitHub Actions for the fallback-evaluation pull request

Three-model results at `K = 10`:

| Metric | Popularity | TF-IDF content | Content + fallback |
|---|---:|---:|---:|
| NDCG@10 | 0.2853 | 0.3594 | **0.3664** |
| MRR@10 | 0.2308 | 0.3133 | **0.3179** |
| Recall@10 | 0.5047 | 0.5819 | **0.5955** |
| Hit Rate@10 | 0.5705 | 0.6610 | **0.6762** |
| Catalog Coverage@10 | 0.0402 | **0.0719** | **0.0719** |
| Unique recommended articles | 2,061 | **3,687** | **3,687** |
| Empty rankings | 0 | 927 | **0** |

Fallback routing accounting:

| Property | Result |
|---|---:|
| Content-routed impressions | 30,466 (97.05%) |
| Popularity-routed impressions | 927 (2.95%) |
| Empty-history fallback routes | 801 |
| Unknown-history fallback routes | 0 |
| Zero-profile fallback routes | 0 |
| Zero-signal fallback routes | 126 |
| Recovered fallback impressions | 927 (100%) |
| Empty rankings after fallback | 0 |
| Temporal leakage detected | No |

Relative to content alone, fallback improves NDCG@10 by 1.92%, MRR@10 by
1.47%, Recall@10 by 2.33%, and Hit Rate@10 by 2.30%. Catalog coverage is
unchanged because fallback recommendations do not expand the union of articles
already recommended by the content system.

Phase 4 fallback evidence:

```text
src/newslens/evaluation/fallback.py
src/newslens/cli.py
tests/test_fallback_evaluation.py
tests/test_cli.py
reports/fallback_metrics.json
docs/EVALUATION.md
```

### Reproducible history-length segment evaluation

- [x] Define mutually exclusive, exhaustive history-length intervals
- [x] Assign every validation impression exactly once
- [x] Keep segment membership independent of validation click labels
- [x] Evaluate the same fallback rankings within every segment
- [x] Record per-segment NDCG, MRR, Recall, Hit Rate, and coverage
- [x] Preserve the overall fallback metrics after segmentation
- [x] Integrate segment results into the deterministic fallback JSON report
- [x] Add unit and integration tests
- [x] Pass all 171 automated tests locally
- [x] Pass Ruff locally

History-length results at `K = 10`:

| Segment | History articles | Impressions | Share | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold start | 0 | 801 | 2.55% | 0.2954 | 0.2334 | 0.5339 | 0.6017 |
| Short history | 1–4 | 3,379 | 10.76% | 0.3471 | 0.2809 | 0.5911 | 0.6324 |
| Medium history | 5–9 | 4,994 | 15.91% | **0.3767** | 0.3184 | **0.6148** | 0.6676 |
| Long history | 10+ | 22,219 | 70.78% | 0.3695 | **0.3265** | 0.5940 | **0.6875** |

Cold-start impressions are weakest across all four relevance metrics. Usable
history is associated with stronger ranking performance, but the pattern is
metric-specific rather than uniformly monotonic: medium histories lead NDCG
and Recall, while long histories lead MRR and Hit Rate. The 801 cold-start
impressions equal the empty-history fallback routes; the separate 126
zero-signal routes have nonempty histories and are assigned among the other
history groups.

The distribution is highly imbalanced because 70.78% of validation
impressions have long histories. These are descriptive subgroup findings, not
causal claims, and uncertainty estimates remain pending.

Phase 4 history-segment evidence:

```text
src/newslens/evaluation/segments.py
src/newslens/evaluation/fallback.py
tests/test_segments.py
tests/test_fallback_evaluation.py
reports/fallback_metrics.json
docs/EVALUATION.md
```

### Reproducible article-category evaluation

- [x] Define category cohorts from clicked article metadata
- [x] Preserve original global ranking positions within every cohort
- [x] Support impressions containing clicks from multiple categories
- [x] Record overlapping impression-category memberships explicitly
- [x] Compute per-category NDCG, MRR, Recall, and Hit Rate
- [x] Compute exposure using each category's own catalog denominator
- [x] Apply a minimum-support threshold without dropping categories
- [x] Preserve overall fallback and history-segment metrics
- [x] Integrate category results into the deterministic fallback JSON report
- [x] Add unit and integration tests
- [x] Pass all 195 automated tests locally
- [x] Pass Ruff locally

Category-evaluation accounting:

| Property | Result |
|---|---:|
| Validation impressions | 31,393 |
| Clicked impressions | 31,393 |
| Impression-category pairs | 43,413 |
| Multi-category clicked impressions | 8,074 (25.72%) |
| Minimum support | 100 relevant impressions |
| Supported categories | 14 |
| Zero-support categories | 3 |
| Overall fallback metrics changed | No |
| History-segment metrics changed | No |

Supported category results at `K = 10`:

| Category | Relevant impressions | Share | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| News | 10,338 | 32.93% | 0.3361 | 0.2680 | 0.5804 | 0.6106 | 0.0641 |
| Lifestyle | 6,594 | 21.00% | 0.3894 | 0.3297 | 0.6016 | 0.6175 | **0.1291** |
| Sports | 3,564 | 11.35% | 0.3151 | 0.2546 | 0.5287 | 0.5485 | 0.0540 |
| Finance | 3,464 | 11.03% | 0.2819 | 0.2169 | 0.5025 | 0.5141 | 0.1033 |
| Music | 2,967 | 9.45% | 0.2234 | 0.1550 | 0.4520 | 0.4604 | 0.1014 |
| Food and drink | 2,500 | 7.96% | 0.2577 | 0.2027 | 0.4530 | 0.4704 | 0.0894 |
| TV | 2,312 | 7.36% | 0.2779 | 0.2069 | 0.5168 | 0.5238 | 0.0810 |
| Health | 2,268 | 7.22% | 0.2124 | 0.1581 | 0.3978 | 0.4114 | 0.1119 |
| Travel | 1,920 | 6.12% | 0.1772 | 0.1267 | 0.3493 | 0.3563 | 0.0728 |
| Entertainment | 1,663 | 5.30% | 0.3230 | 0.2574 | 0.5441 | 0.5538 | 0.1227 |
| Weather | 1,640 | 5.22% | **0.4764** | **0.3932** | **0.7359** | **0.7366** | 0.0420 |
| Video | 1,558 | 4.96% | 0.1917 | 0.1346 | 0.3883 | 0.3928 | 0.0662 |
| Autos | 1,469 | 4.68% | 0.2296 | 0.1764 | 0.4210 | 0.4391 | 0.0811 |
| Movies | 1,156 | 3.68% | 0.2448 | 0.1705 | 0.4942 | 0.4957 | 0.1040 |

Weather leads every supported relevance metric but has only 0.0420 category
coverage. Lifestyle has the broadest supported category exposure at 0.1291.
Travel has the weakest supported relevance results. `kids`, `middleeast`, and
`northamerica` have no clicked validation support and therefore have no
relevance metrics.

The 43,413 memberships exceed the 31,393 impressions because an impression can
belong to multiple clicked-category cohorts. The shares are not a partition
and must not be summed. Results are descriptive, outcome-conditioned subgroup
diagnostics rather than causal or candidate-availability-adjusted comparisons.

Phase 4 category-evaluation evidence:

```text
src/newslens/evaluation/categories.py
src/newslens/evaluation/fallback.py
tests/test_categories.py
tests/test_fallback_evaluation.py
reports/fallback_metrics.json
docs/EVALUATION.md
```

### Reproducible training-exposure evaluation

- [x] Define unseen as zero chronological-training candidate exposures
- [x] Define contiguous low-, medium-, and high-exposure intervals
- [x] Fit exposure counts using only the earlier training partition
- [x] Preserve original global ranking positions within every cohort
- [x] Support impressions with clicked items from multiple exposure bands
- [x] Record overlapping impression-band memberships explicitly
- [x] Compute per-band NDCG, MRR, Recall, and Hit Rate
- [x] Compute recommendation coverage using each band's catalog denominator
- [x] Apply a minimum-support threshold without dropping bands
- [x] Preserve overall, history-segment, and category metrics
- [x] Integrate exposure results into the deterministic fallback report
- [x] Add unit and integration tests
- [x] Pass all 222 automated tests locally
- [x] Pass Ruff locally

Exposure-evaluation accounting:

| Property | Result |
|---|---:|
| Validation impressions | 31,393 |
| Clicked impressions | 31,393 |
| Impression-band pairs | 39,633 |
| Multi-band clicked impressions | 6,627 (21.11%) |
| Average band memberships per impression | 1.2625 |
| Minimum support | 100 relevant impressions |
| Supported bands | 4 |
| Overall fallback metrics changed | No |
| History-segment metrics changed | No |
| Article-category metrics changed | No |

Training-exposure results at `K = 10`:

| Band | Exposures | Catalog articles | Relevant impressions | Share | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unseen | 0 | 34,450 | 15,003 | 47.79% | **0.3924** | **0.3291** | **0.6298** | **0.6655** | 0.0577 |
| Low exposure | 1–9 | 9,648 | 13,038 | 41.53% | 0.2807 | 0.2159 | 0.5162 | 0.5512 | 0.0470 |
| Medium exposure | 10–99 | 4,405 | 3,012 | 9.59% | 0.2916 | 0.2246 | 0.5136 | 0.5196 | 0.0854 |
| High exposure | 100+ | 2,779 | 8,580 | 27.33% | 0.2740 | 0.2265 | 0.4697 | 0.5169 | **0.3127** |

Unseen clicked articles have the strongest recorded relevance metrics, while
high-exposure articles have the broadest within-band catalog coverage. The
result is compatible with TF-IDF generalizing to catalog text that had zero
training candidate appearances. It is not evidence that zero exposure causes
better ranking: recency, category mix, candidate difficulty, and lexical
alignment remain possible explanations.

The cohorts overlap because multi-click impressions can contain relevant items
from several exposure bands. “Unseen” refers specifically to zero candidate
exposures in chronological training, not missing article text or an
out-of-vocabulary document.

Phase 4 exposure-evaluation evidence:

```text
src/newslens/evaluation/exposure.py
src/newslens/evaluation/fallback.py
tests/test_exposure.py
tests/test_fallback_evaluation.py
reports/fallback_metrics.json
docs/EVALUATION.md
```

### Remaining model evaluation

- [x] Evaluate the content-based history recommender
- [x] Evaluate the cold-start fallback
- [x] Record content-model coverage
- [x] Record fallback frequency
- [x] Measure unseen-article ranking performance
- [x] Evaluate warm-start and cold-start impressions separately
- [x] Generate a three-model comparison table
- [x] Compare popularity, content, and fallback using identical validation
      impressions
- [x] Reserve the official MIND-small development data for final evaluation
- [ ] Run final evaluation on the reserved development split only after model
      selection is complete

### Remaining analysis

- [x] Compare performance across user-history lengths
- [x] Compare warm-start and cold-start users
- [x] Analyze performance by article category
- [x] Analyze unseen and low-exposure articles
- [ ] Inspect high-confidence failures
- [x] Document initial popularity findings
- [x] Document popularity limitations and non-claims
- [x] Document positive and negative findings across all evaluated baselines
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
analysis: inspect high-confidence ranking failures
analysis: add uncertainty estimates to baseline comparisons
```

Completed fallback-evaluation commits include:

```text
feat: add tested cold-start fallback evaluation
feat: add fallback evaluation command
results: record MIND fallback evaluation metrics
docs: publish fallback evaluation
```

Completed history-segment commits include:

```text
feat: add tested history-segment evaluator
feat: integrate history segments with fallback evaluation
results: record MIND history-segment metrics
docs: publish history-segment evaluation
```

Completed article-category commits include:

```text
feat: add tested article-category evaluator
feat: integrate category analysis with fallback evaluation
results: record MIND article-category metrics
docs: publish article-category evaluation
```

Completed training-exposure commits include:

```text
feat: add tested training-exposure evaluator
feat: integrate exposure analysis with fallback evaluation
results: record MIND training-exposure metrics
docs: publish training-exposure evaluation
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
- [x] a reproducible formal cold-start fallback evaluation;
- [x] a machine-readable fallback evaluation report;
- [x] a reproducible same-split three-model comparison;
- [x] history-length user-segment analysis;
- [x] category-level analysis;
- [x] unseen and low-exposure item analysis;
- [ ] high-confidence error analysis;
- [ ] uncertainty estimates or confidence intervals;
- [ ] a tested inference API;
- [ ] containerization;
- [ ] experiment tracking;
- [ ] public deployment; and
- [ ] a model card with complete reproduction instructions.
