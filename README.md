NewsLens

NewsLens is a leakage-aware news search and recommendation system built withthe Microsoft MIND news-recommendation dataset. It demonstrates the progressionfrom validated interaction data to reproducible recommendation baselines,formal offline evaluation, model comparison, error analysis, and tested APIserving.

Current status: Offline model selection is complete. The selectedcandidate combines TF-IDF user-history recommendations with a training-onlypopularity fallback. A paired bootstrap comparison shows improvements overthe content-only baseline across NDCG@10, MRR@10, Recall@10, and Hit Rate@10.Phase 6 serving is in progress: a tested FastAPI foundation exposes/health and /model-info; versioned model-artifact loading and inferenceendpoints remain future work. The project currently has 348 automatedtests.

Why this project

NewsLens demonstrates:

production-style Python package organization;

validated ingestion of real interaction data;

explicit temporal-leakage prevention;

deterministic search and recommendation baselines;

an explicit cold-start policy;

model-independent ranking evaluation;

subgroup, exposure, uncertainty, and failure analysis;

statistically paired model comparison;

typed API responses and automated API tests;

continuous integration with GitHub Actions; and

explicit documentation of limitations and non-claims.

System overview

flowchart TD
    A["Licensed MIND-small files"] --> B["Validated loaders and audit"]
    B --> C["Leakage-safe chronological split"]
    C --> D["TF-IDF history-content model"]
    C --> E["Training-only popularity model"]
    D --> F{"Usable content signal?"}
    E --> F
    F --> G["Candidate-set ranking"]
    G --> H["Metrics, uncertainty, and diagnostics"]
    G --> I["FastAPI serving foundation"]

The selected candidate preserves TF-IDF content rankings whenever a positivesimilarity signal exists. It routes only content abstentions to popularity.Content and popularity scores are not blended.

Key results

All systems use the same chronological split, validation impressions,candidate sets, catalog, metric implementation, and ranking cutoff.

Property

Value

Total MIND-small training records

156,965

Chronological training records

125,572

Chronological validation records

31,393

Requested validation fraction

20%

Actual validation fraction

20%

Cutoff timestamp

2019-11-13 20:36:26

Temporal overlap

None

Three-model comparison

Metric @10

Popularity

TF-IDF content

Content + fallback

NDCG

0.2853

0.3594

0.3664

MRR

0.2308

0.3133

0.3179

Recall

0.5047

0.5819

0.5955

Hit Rate

0.5705

0.6610

0.6762

Catalog Coverage

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

The content model handles 30,466 validation impressions, or 97.05%. Theremaining 927 impressions, or 2.95%, are routed to popularity. All fallbackroutes receive rankings, reducing empty rankings from 927 to zero.

Paired bootstrap comparison

The final fallback candidate and content-only baseline were compared onaligned evaluated impressions using a paired nonparametric percentilebootstrap.

Metric

Content baseline

Fallback candidate

Difference

95% CI for difference

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

Paired-comparison protocol:

evaluated impressions: 31,393;

bootstrap replicates: 1,000;

confidence level: 95%;

random seed: 42;

resampling unit: aligned evaluated impression pair; and

difference direction: candidate minus baseline.

Every paired interval excludes zero under this evaluation protocol. This isevidence of an offline metric improvement, not evidence of online or causalproduct impact.

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

Fallback recovery is 100% for the 927 routed impressions.

Diagnostic evaluation

History length

History groups are mutually exclusive and exhaustive. Recombining themreproduces the overall fallback metrics.

Segment

History articles

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

Cold-start impressions are weakest across all four relevance metrics. Thepattern among non-empty histories is metric-specific rather than uniformlymonotonic.

Clicked-article category

Category cohorts intentionally overlap when an impression contains clickedarticles from multiple categories. Therefore, category shares do not sum to100%.

Category

Relevant impressions

NDCG@10

Recall@10

Hit Rate@10

Weather

1,640

0.4764

0.7359

0.7366

Lifestyle

6,594

0.3894

0.6016

0.6175

News

10,338

0.3361

0.5804

0.6106

Sports

3,564

0.3151

0.5287

0.5485

Finance

3,464

0.2819

0.5025

0.5141

Health

2,268

0.2124

0.3978

0.4114

Travel

1,920

0.1772

0.3493

0.3563

The full supported-category table and category-specific coverage values arepublished in docs/EVALUATION.md and reports/fallback_metrics.json.

These category results are descriptive and conditioned on observed clicks.They are not causal or candidate-availability-adjusted comparisons.

Training exposure

Article exposure is calculated using only candidate appearances in thechronological training partition.

Band

Training exposures

Catalog articles

Relevant impressions

NDCG@10

Recall@10

Hit Rate@10

Unseen

0

34,450

15,003

0.3924

0.6298

0.6655

Low

1–9

9,648

13,038

0.2807

0.5162

0.5512

Medium

10–99

4,405

3,012

0.2916

0.5136

0.5196

High

100+

2,779

8,580

0.2740

0.4697

0.5169

Here, “unseen” means zero training candidate appearances. It does not meanmissing article text. Cohorts can overlap when a multi-click impression hasrelevant items from several exposure bands.

Overall uncertainty

The final candidate's overall metrics use a deterministic impression-levelnonparametric percentile bootstrap.

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

These intervals condition on the fixed dataset, split, candidate sets, fittedmodel, and metric definitions. They do not quantify temporal drift,model-selection uncertainty, or online impact.

High-confidence failure analysis

NewsLens inspects top-K misses whose top recommendation scores are highrelative to other impressions routed through the same recommendation source.Thresholds are source-specific because TF-IDF similarity and popularity clickcounts are not on a common scale.

Property

Result

Evaluated impressions

31,393

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

1,007 clicks

This diagnostic identifies confident failures for inspection. It does notclaim that scores from different recommendation sources are calibratedprobabilities.

Implemented functionality

Data ingestion and audit

validated news.tsv and behaviors.tsv loaders;

duplicate-identifier detection;

malformed-schema and invalid-label rejection;

timestamp, history, and candidate parsing;

preservation of empty histories as valid cold-start records;

reproducible train and development audit reports; and

exclusion of raw licensed data from Git.

Search and recommendation models

training-only popularity ranking;

TF-IDF article search;

TF-IDF user-history content recommendation;

content-to-popularity cold-start fallback;

deterministic tie handling; and

candidate-set and viewed-history restrictions.

Evaluation

leakage-safe chronological splitting;

binary NDCG@K, MRR@K, Recall@K, and Hit Rate@K;

catalog coverage;

model-independent candidate-ranking evaluation;

reproducible JSON reports;

history, category, and training-exposure diagnostics;

deterministic overall bootstrap intervals;

high-confidence failure inspection; and

paired bootstrap model comparison.

API foundation

FastAPI application factory;

typed Pydantic response schemas;

GET /health;

GET /model-info;

automatically generated OpenAPI schema;

interactive Swagger documentation; and

API integration tests.

The API honestly reports model_ready: false until validated artifacts areimplemented and loaded.

Quick start

Python 3.12 is recommended. NewsLens supports Python 3.11 and Python 3.12.

macOS or Linux

bash scripts/setup.sh
source .venv/bin/activate
python -m newslens --help

Windows PowerShell

powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
python -m newslens --help

Manual setup

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest

Expected validation:

All checks passed!
348 passed

Run the HTTP API

Start the local development server:

python -m uvicorn newslens.api.app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload

Available foundation endpoints:

GET /health
GET /model-info
GET /docs
GET /openapi.json

Examples:

curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info

See docs/API.md for the API contract and current non-claims.

Dataset policy

NewsLens uses MIND-small from the Microsoft MIND news-recommendation dataset:

https://msnews.github.io/

Read and accept the applicable Microsoft Research License Terms before usingthe dataset. Dataset archives and extracted raw files must not be committed tothis repository.

Expected local layout:

data/
├── MINDsmall_train/
│   ├── news.tsv
│   ├── behaviors.tsv
│   ├── entity_embedding.vec
│   └── relation_embedding.vec
└── MINDsmall_dev/
    ├── news.tsv
    ├── behaviors.tsv
    ├── entity_embedding.vec
    └── relation_embedding.vec

Only aggregate audit and evaluation reports are version controlled. Raw MINDrecords and embeddings are not redistributed.

Reproduce the dataset audit

python -m newslens audit-data \
  --data-dir data \
  --split train \
  --output reports/mindsmall_train_audit.json

python -m newslens audit-data \
  --data-dir data \
  --split dev \
  --output reports/mindsmall_dev_audit.json

Reproduce the evaluations

Popularity

python -m newslens evaluate-popularity \
  --data-dir data \
  --k 10 \
  --validation-fraction 0.20 \
  --output reports/popularity_metrics.json

TF-IDF content

python -m newslens evaluate-content \
  --data-dir data \
  --k 10 \
  --validation-fraction 0.20 \
  --max-features 50000 \
  --output reports/content_metrics.json

Content with fallback

python -m newslens evaluate-fallback \
  --data-dir data \
  --k 10 \
  --validation-fraction 0.20 \
  --max-features 50000 \
  --bootstrap-samples 1000 \
  --bootstrap-confidence-level 0.95 \
  --bootstrap-random-seed 42 \
  --failure-score-quantile 0.90 \
  --maximum-failures-per-source 25 \
  --output reports/fallback_metrics.json

TF-IDF article search example

from newslens.data import load_news
from newslens.models import TfidfArticleSearch

news = load_news("data/MINDsmall_train/news.tsv")
search = TfidfArticleSearch().fit(news)

for result in search.search("space exploration mission", top_k=5):
    print(result.news_id, result.score, result.title)

Repository structure

NewsLens/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── baselines.json
├── data/
│   └── README.md
├── docs/
│   ├── API.md
│   ├── BASELINES.md
│   ├── DECISIONS.md
│   ├── EVALUATION.md
│   ├── INTERVIEW_NOTES.md
│   └── LEARNING_LOG.md
├── notebooks/
│   └── README.md
├── reports/
│   ├── content_metrics.json
│   ├── fallback_metrics.json
│   ├── mindsmall_dev_audit.json
│   ├── mindsmall_train_audit.json
│   └── popularity_metrics.json
├── scripts/
│   ├── setup.ps1
│   └── setup.sh
├── src/
│   └── newslens/
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   └── schemas.py
│       ├── data/
│       │   ├── audit.py
│       │   └── mind.py
│       ├── evaluation/
│       │   ├── categories.py
│       │   ├── comparison.py
│       │   ├── content.py
│       │   ├── evaluator.py
│       │   ├── exposure.py
│       │   ├── failures.py
│       │   ├── fallback.py
│       │   ├── metrics.py
│       │   ├── popularity.py
│       │   ├── segments.py
│       │   ├── split.py
│       │   └── uncertainty.py
│       ├── models/
│       │   ├── content.py
│       │   ├── fallback.py
│       │   ├── popularity.py
│       │   └── tfidf.py
│       ├── __init__.py
│       ├── __main__.py
│       └── cli.py
├── tests/
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── ROADMAP.md

Virtual environments, caches, downloaded datasets, and other local artifactsare excluded from Git.

Testing and continuous integration

Run the complete suite:

python -m pytest

Run lint checks:

python -m ruff check .

Apply formatting:

python -m ruff format .

The 348-test suite covers:

malformed data and schema validation;

deterministic parsing and dataset auditing;

chronological splitting and temporal-leakage prevention;

popularity, TF-IDF search, content, and fallback models;

ranking metrics and model-independent evaluation;

subgroup and training-exposure diagnostics;

deterministic bootstrap uncertainty;

high-confidence failure analysis;

paired bootstrap model comparison;

CLI report generation;

API application-factory behavior;

typed health and model-information responses;

OpenAPI route publication; and

invalid-input and error behavior.

GitHub Actions installs the project, runs Ruff, and executes the complete testsuite for pull requests and pushes to main.

Development roadmap

Foundation — completed.

MIND ingestion and audit — completed.

Evaluation-safe baselines — completed.

Ranking evaluation — completed, including subgroup analysis,uncertainty estimation, high-confidence failure inspection, and pairedbootstrap model comparison.

Learned hybrid ranking — deferred; the evaluated rule-based fallbackremains the selected serving candidate.

Inference service — in progress; FastAPI health and model-informationendpoints are implemented and tested.

MLOps and deployment — planned.

Controlled experiments — planned.

The next milestone is a versioned, validated artifact bundle loaded onceduring application startup. See ROADMAP.md for detailedacceptance criteria.

Current limitations and non-claims

The official MIND-small development split has not been consumed for finalholdout evaluation.

The selected fallback switches strategies rather than learning a jointranker.

No learned hybrid or neural recommendation model has been trained.

TF-IDF captures lexical overlap rather than deeper semantic similarity.

User-history articles receive equal weight.

Popularity scores reflect historical exposure as well as user interest.

Popularity provides limited catalog coverage.

Article publication timestamps are unavailable in the local metadata.

Category, history, and exposure results are descriptive rather than causal.

Category and exposure cohorts can overlap and their shares cannot be summed.

Overall and paired 95% bootstrap intervals are reported, but subgroup-specificintervals are not.

A tested API foundation exists, but it does not yet load a serialized modelartifact or expose search and recommendation inference.

The API has not been containerized or publicly deployed.

No online experiment has been conducted.

Passing tests and improving offline metrics do not establish production oronline impact.

Results should not be compared directly with systems using different datasplits, candidate policies, catalogs, or metric definitions.

Development workflow

NewsLens uses small feature branches and pull requests. A feature is completeonly when:

its implementation is tested;

Ruff checks pass;

the complete test suite passes;

assumptions and limitations are documented;

reproducible outputs are recorded when applicable; and

GitHub Actions passes.

License

Code in this repository is released under the MIT License.

The Microsoft MIND dataset is governed by separate Microsoft Research LicenseTerms and is not redistributed in this repository.