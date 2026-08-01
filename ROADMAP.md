NewsLens roadmap

NewsLens is developed through small, testable feature branches and pullrequests. A milestone is complete only when its implementation, tests,documentation, reproducible outputs, and continuous-integration checks pass.

The offline recommendation-system evaluation milestone is complete. Thecurrent focus is production-style model packaging and serving.

Current status

Phase 1 — Project foundation

Phase 2 — Validated MIND-small ingestion and dataset auditing

Phase 3 — Evaluation-safe search and recommendation baselines

Phase 4 — Offline ranking evaluation and model selection

Phase 5 — Rule-based content and popularity fallback candidate

Phase 6 — Model packaging and serving (in progress)

Phase 7 — MLOps, containerization, and deployment

Phase 8 — Controlled online-experiment design

Current serving status:

Package and API version aligned to 0.2.0

FastAPI application factory

Typed /health endpoint

Typed /model-info endpoint

Automatic OpenAPI and Swagger documentation

API tests

API usage documentation

Versioned model-artifact contract

Model serialization and loading

Artifact compatibility and integrity validation

Model-aware readiness endpoint

/search inference endpoint

/recommend inference endpoint

Containerized deployment

The current API is a tested service foundation. It does not yet load a trainedmodel artifact or provide search and recommendation inference. Model readinessmust remain false until artifact loading and validation are implemented.

Engineering principles

Fit models only with information allowed by the evaluation protocol.

Keep the official MIND-small development split reserved until a deliberatefinal-holdout decision is made.

Use deterministic ordering and explicit random seeds.

Validate behavior with synthetic, hand-checkable tests.

Preserve machine-readable reports for important experiments.

Document limitations and negative findings alongside positive findings.

Never commit raw licensed MIND data, trained redistributable artifactswithout permission, credentials, or secrets.

Separate liveness, readiness, model loading, and inference concerns.

Publish work through reviewable feature branches and pull requests.

Phase 1 — Project foundation

Status: completed

Create the Python src/ package layout

Add isolated environment instructions

Add smoke tests

Configure Ruff formatting and linting

Add GitHub Actions continuous integration

Create the public GitHub repository

Establish feature-branch and pull-request development

Add a license, Makefile, and developer documentation

Exclude environments, caches, generated outputs, and raw data from Git

Evidence:

pyproject.toml
Makefile
.github/workflows/ci.yml
src/newslens/
tests/test_smoke.py
README.md
START_HERE.md

Phase 2 — Validated MIND-small ingestion and dataset auditing

Status: completed

Loading and validation

Parse news.tsv

Parse behaviors.tsv

Validate expected TSV schemas

Reject malformed rows and invalid click labels

Reject duplicate article and impression identifiers

Parse behavior timestamps, histories, and candidate impressions

Preserve empty histories as valid cold-start records

Validate article references

Add synthetic loader fixtures and tests

Dataset audit

Report users, articles, impressions, clicks, and non-clicks

Report missing titles and abstracts

Check candidate and history references against article metadata

Report empty histories, time coverage, and categories

Add a reproducible audit command

Audit MIND-small train and development data

Publish deterministic JSON audit reports

Document assumptions, limitations, and non-claims

Keep licensed raw data outside Git

Evidence:

src/newslens/data/mind.py
src/newslens/data/audit.py
tests/test_mind_loader.py
tests/test_audit.py
tests/test_cli.py
reports/mindsmall_train_audit.json
reports/mindsmall_dev_audit.json

Optional hardening:

Add streaming ingestion for datasets larger than memory

Add checksums for locally downloaded archives

Produce structured row-level validation-error reports

Add immutable typed domain records

Phase 3 — Evaluation-safe search and recommendation baselines

Status: completed

Chronological splitting

Sort behavior records deterministically

Keep identical timestamps within one partition

Require strict temporal separation

Test temporal-leakage invariants

Confirm the input data are not modified

Run the split on real MIND-small training records

Property

Value

Total behavior records

156,965

Training records

125,572

Validation records

31,393

Actual validation fraction

20%

Cutoff

2019-11-13 20:36:26

Final training timestamp

2019-11-13 20:36:19

First validation timestamp

2019-11-13 20:36:26

Temporal overlap

None

The official MIND-small development split remains reserved and has not beenused for model selection.

Baselines

Training-only click-count popularity recommender

TF-IDF article search

TF-IDF user-history content recommender

Training-only popularity fallback

Candidate-set restrictions

History-article exclusions

Deterministic article-ID tie-breaking

Empty-history handling

Unknown-history handling

Zero-profile and zero-signal handling

Synthetic and integration tests

Version-controlled parameters

Baseline assumptions and limitations documentation

Evidence:

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

Phase 4 — Offline ranking evaluation and model selection

Status: completed

Popularity, TF-IDF history content, and content with popularity fallback wereevaluated under one chronological protocol. The milestone also includessubgroup analysis, uncertainty estimation, paired comparison, andhigh-confidence failure inspection.

Evaluation framework

Binary NDCG@K

Mean Reciprocal Rank@K

Recall@K

Hit Rate@K

Catalog coverage

Multiple-click and no-click semantics

Rankings shorter than K

Duplicate recommendation validation

Hand-calculated metric tests

Model-independent candidate-ranking evaluator

Empty-ranking and skipped-impression accounting

Deterministic evaluation behavior

Primary model results

All systems were evaluated on the same 31,393 chronological validationimpressions at K = 10.

Metric

Popularity

TF-IDF content

Content + popularity fallback

NDCG@10

0.2853

0.3594

0.3664

MRR@10

0.2308

0.3133

0.3179

Recall@10

0.5047

0.5819

0.5955

Hit Rate@10

0.5705

0.6610

0.6762

Catalog Coverage@10

0.0402

0.0719

0.0719

Unique recommended articles

2,061

3,687

3,687

Empty rankings

0

927

0

Selected serving candidate:

tfidf_content_with_popularity_fallback

Selection rationale:

strongest recorded NDCG, MRR, Recall, and Hit Rate;

recovers every content-model abstention;

preserves content catalog coverage;

uses an explicit, deterministic, tested cold-start policy; and

is practical for the serving milestone.

This is an offline model-selection result, not evidence of online impact.

Fallback routing

Route

Impressions

Share

Content

30,466

97.05%

Popularity fallback

927

2.95%

Empty-history fallback

801

2.55%

Zero-signal fallback

126

0.40%

Unknown-history fallback

0

0.00%

Zero-profile fallback

0

0.00%

Recover all 927 content abstentions

Produce no empty fallback rankings

Preserve candidate restrictions

Preserve temporal-leakage protections

History-length analysis

Define mutually exclusive and exhaustive history groups

Assign every validation impression exactly once

Keep membership independent of validation clicks

Preserve overall metrics after segmentation

Segment

History length

Impressions

Share

NDCG@10

MRR@10

Recall@10

Hit Rate@10

Cold start

0

801

2.55%

0.2954

0.2334

0.5339

0.6017

Short

1–4

3,379

10.76%

0.3471

0.2809

0.5911

0.6324

Medium

5–9

4,994

15.91%

0.3767

0.3184

0.6148

0.6676

Long

10+

22,219

70.78%

0.3695

0.3265

0.5940

0.6875

These are descriptive subgroup findings, not causal effects of history length.

Article-category analysis

Define cohorts from clicked-article metadata

Preserve global ranking positions within cohorts

Record overlapping category membership

Apply minimum-support rules

Preserve overall and history-segment results

Key findings:

43,413 impression-category memberships occurred across 31,393 impressions.

8,074 impressions contained clicks from multiple categories.

Weather recorded the strongest supported relevance metrics.

Lifestyle recorded the broadest supported within-category coverage.

Travel recorded the weakest supported relevance metrics.

kids, middleeast, and northamerica had no clicked validation support.

Category memberships overlap, so their shares must not be summed. These aredescriptive, outcome-conditioned diagnostics.

Training-exposure analysis

Define unseen as zero chronological-training candidate exposures

Define low, medium, and high exposure bands

Preserve global ranking positions

Record overlapping band membership

Preserve overall, history, and category results

Band

Exposures

Catalog articles

Relevant impressions

NDCG@10

MRR@10

Recall@10

Hit Rate@10

Coverage@10

Unseen

0

34,450

15,003

0.3924

0.3291

0.6298

0.6655

0.0577

Low

1–9

9,648

13,038

0.2807

0.2159

0.5162

0.5512

0.0470

Medium

10–99

4,405

3,012

0.2916

0.2246

0.5136

0.5196

0.0854

High

100+

2,779

8,580

0.2740

0.2265

0.4697

0.5169

0.3127

The unseen result is compatible with lexical generalization to articles whosemetadata are available but which had no chronological-training candidateexposures. It is not evidence that low exposure causes stronger performance.

Bootstrap uncertainty

Resample evaluated impressions

Run 1,000 bootstrap samples

Use 95% percentile intervals

Use random seed 42

Record standard errors

Reproduce evaluator point estimates

Confirm deterministic repeated reports

Metric

Estimate

Lower 95%

Upper 95%

Standard error

NDCG@10

0.3664

0.3627

0.3703

0.0019

MRR@10

0.3179

0.3140

0.3218

0.0020

Recall@10

0.5955

0.5907

0.6003

0.0026

Hit Rate@10

0.6762

0.6712

0.6815

0.0027

These intervals quantify impression-sampling variability conditional on thefixed split, model, candidates, and metric definitions. They do not quantifytemporal drift, model-selection uncertainty, or online impact.

Paired comparison

Compare aligned content and fallback rankings

Resample aligned impression pairs

Report fallback-minus-content differences

Confirm all reported 95% intervals exclude zero

Metric

Content

Fallback

Difference

Paired 95% interval

NDCG@10

0.3594

0.3664

+0.0069

[0.0058, 0.0081]

MRR@10

0.3133

0.3179

+0.0046

[0.0034, 0.0058]

Recall@10

0.5819

0.5955

+0.0135

[0.0119, 0.0152]

Hit Rate@10

0.6610

0.6762

+0.0152

[0.0135, 0.0170]

The comparison provides evidence of an offline improvement under this exactprotocol. It does not establish production or causal impact.

High-confidence failure analysis

Define source-specific score thresholds

Use the 90th percentile of top scores within each source

Identify high-score top-K misses

Keep content and popularity score interpretations separate

Retain deterministic examples for inspection

Document score-confidence limitations

Property

Result

Score-eligible impressions

31,162

Top-K misses

10,164 (32.38%)

High-score misses

1,222 (3.92% of eligible impressions)

Retained examples

50

Content threshold

0.1918

Popularity threshold

1,007 training clicks

High model scores are not calibrated probabilities. These examples aredebugging evidence, not proof of real-world confidence.

Phase 4 evidence

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

Closure checklist

Evaluate models on identical validation impressions

Publish deterministic machine-readable reports

Analyze history, category, and exposure subgroups

Estimate overall uncertainty

Run a paired model comparison

Inspect high-confidence failures

Document metric and comparison limitations

Select the serving candidate

Pass the complete offline-evaluation test suite

Tag the offline evaluation milestone as v0.2.0

Keep the official development split reserved

Deliberately deferred:

Run one final evaluation on the official MIND-small development split

Before using the reserved split, freeze the model, configuration, metrics,reporting plan, and no-further-tuning policy.

Phase 5 — Rule-based hybrid fallback candidate

Status: completed for the selected rule-based system

The selected system combines TF-IDF history-content ranking withtraining-only popularity whenever content cannot produce a usable ranking.This is deterministic routing, not learned score fusion.

Define routing rules

Track recommendation source

Compare against both component models

Analyze cold-start and warm-start behavior

Quantify uncertainty

Inspect failure cases

Select the rule-based system for serving

Optional model research:

Weighted content, popularity, and recency fusion

Learning-to-rank baseline

Dense text embeddings

Neural user representation

Diversity-aware re-ranking

Calibrated confidence estimates

These experiments are optional and do not block production-ML engineeringwork with the selected candidate.

Phase 6 — Model packaging and serving

Status: in progress

6A. FastAPI foundation

Align package and API version metadata to 0.2.0

Add FastAPI, Pydantic, Uvicorn, and API-test dependencies

Implement an application factory

Define typed response schemas

Implement GET /health

Implement GET /model-info

Generate /docs and /openapi.json

Add API tests

Verify the routes manually with Uvicorn

Document current usage in docs/API.md

Document current non-claims

Evidence:

src/newslens/api/__init__.py
src/newslens/api/app.py
src/newslens/api/schemas.py
tests/test_api.py
docs/API.md

Current routes:

Route

Purpose

Expected status

GET /health

Process liveness

200

GET /model-info

Package and configured-model metadata

200

GET /docs

Swagger documentation

200

GET /openapi.json

OpenAPI schema

200

The complete suite contains 348 tests after the API foundation was added.

6B. Versioned model artifacts — next milestone

Define an artifact schema version

Define typed metadata and a manifest

Record package and dependency versions

Record model name, configuration, and training cutoff

Record vocabulary and indexed-catalog sizes

Serialize the TF-IDF content model

Serialize the popularity fallback

Store inference-required article metadata

Record file sizes and SHA-256 checksums

Validate checksums before loading

Reject missing or corrupt files

Reject unsupported schema versions

Reject incompatible package versions clearly

Test save/load round trips with synthetic fixtures

Confirm rankings survive round trips exactly

Add a reproducible artifact-build command

Document artifact security assumptions

Keep licensed data and non-redistributable artifacts outside Git

Planned files:

src/newslens/artifacts/__init__.py
src/newslens/artifacts/metadata.py
src/newslens/artifacts/manifest.py
src/newslens/artifacts/io.py
scripts/build_artifact.py
tests/test_artifacts.py
docs/ARTIFACTS.md

6C. Model-aware startup and readiness

Configure an artifact directory through an environment variable

Load and validate the artifact during FastAPI lifespan startup

Store the loaded service in application state

Keep /health independent of model state

Add GET /ready

Return readiness failure when no valid model is loaded

Return artifact metadata from /model-info

Test valid, missing, corrupt, and incompatible artifacts

Document startup failure policy

6D. Search inference

Define typed search requests and ranked results

Implement POST /search

Validate empty and oversized queries

Validate top_k

Support result exclusions

Return deterministic results

Return model and artifact versions

Return 503 when the model is unavailable

Add unit and API integration tests

6E. Recommendation inference

Define typed recommendation requests and results

Accept history article identifiers

Support optional candidate restrictions

Implement POST /recommend

Preserve cold-start fallback behavior

Return content or popularity source

Validate unknown and duplicate identifiers

Validate top_k

Return model and artifact versions

Return 503 when the model is unavailable

Add unit and API integration tests

6F. Serving documentation

Document foundation endpoints

Document current limitations

Document artifact configuration

Document liveness versus readiness

Document search examples

Document recommendation examples

Document cold-start routing in responses

Document error responses

Phase 7 — MLOps, containerization, and deployment

Status: planned

Reproducibility and artifacts

Add a complete training command

Record data, code, parameter, and artifact versions

Add artifact-validation checks to CI

Add lightweight experiment tracking

Add a model card

Containerization

Add a production Dockerfile

Use a non-root runtime user

Add a container health check

Exclude raw data and secrets

Add a CI container-build check

Test local startup from the image

Observability and performance

Add structured logs and request identifiers

Track readiness and inference failures

Measure artifact-loading time

Run controlled load tests

Report p50, p95, and p99 latency

Report throughput and error rate

Document test hardware and limitations

Deployment

Select a deployment target

Store artifacts outside the source repository

Configure secrets and environment variables

Deploy a synthetic or redistribution-safe demonstration model

Validate public health, readiness, and inference endpoints

Document cost, security, and availability limitations

Phase 8 — Controlled online-experiment design

Status: planned

Define deterministic user assignment

Define exposure and outcome events

Define experiment eligibility

Define primary and secondary metrics

Define product, latency, and reliability guardrails

Add sample-ratio-mismatch checks

Implement power and sample-size analysis

Calculate uncertainty intervals

Add experiment-analysis tests

Document stopping and decision rules

No online experiment has been conducted. This phase demonstrates sound designwithout inventing production impact.

Recommended pull-request sequence

feat/api-foundation

application factory, health, model information, schemas, tests, and docs

feat/model-artifacts

metadata, serialization, manifest, checksums, loader, and tests

feat/api-model-loading

lifespan loading, application state, and readiness

feat/api-search

search endpoint and tests

feat/api-recommendation

recommendation endpoint, fallback source, and tests

feat/containerization

Dockerfile, health check, and CI build

feat/observability

structured logs, request IDs, and latency reporting

docs/model-card

model card, reproduction guide, and serving limitations

Each pull request should contain focused tests, updated documentation, and anexplicit statement of what remains unimplemented.

Portfolio completion criteria

Data and evaluation

Validated real-world data ingestion

Reproducible dataset auditing

Leakage-safe chronological preparation

Three documented search or recommendation baselines

Explicit cold-start behavior

Hand-validated ranking metrics

Model-independent offline evaluator

Machine-readable experiment reports

Same-split model comparison

History, category, and exposure analysis

Overall uncertainty intervals

Paired comparison intervals

High-confidence failure analysis

Selected serving candidate

Software engineering

Installable Python package

Object-oriented model implementations

Command-line workflows

Automated tests and continuous integration

Formatting and linting

Feature-branch and pull-request history

Tested FastAPI foundation

Typed API schemas

Liveness and model-information endpoints

Automatic OpenAPI documentation

Versioned and validated model artifacts

Model-aware readiness

Tested search inference API

Tested recommendation inference API

Containerization

Load and latency testing

Structured observability

Public redistribution-safe demonstration deployment

Research and product discipline

Explicit evaluation protocol

Reserved final-holdout policy

Deterministic random seeds

Subgroup and failure analysis

Statistical uncertainty

Honest non-claims about causality and online impact

Model card

Complete artifact reproduction guide

Controlled-experiment design document

NewsLens is already a credible offline machine-learning portfolio project. Theremaining work will demonstrate full-lifecycle ML engineering through artifactmanagement, reliable inference, containerization, observability, anddeployment.

Immediate next action

Complete and merge feat/api-foundation. After it is merged:

git switch main
git pull --ff-only origin main
git fetch --prune origin
git status -sb
git switch -c feat/model-artifacts

On feat/model-artifacts, define and test the metadata and manifest contractsbefore implementing the serializer or loader.