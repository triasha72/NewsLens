# NewsLens

NewsLens is a leakage-aware news search and recommendation system built with
the Microsoft MIND news-recommendation dataset.

The project demonstrates the progression from validated interaction data to
reproducible recommendation baselines, formal offline ranking evaluation,
model comparison, serving, and deployment. Each phase documents its
assumptions, limitations, and non-claims.

> Current status: Phase 4 ranking evaluation is in progress. NewsLens includes
> validated MIND ingestion, reproducible dataset auditing, leakage-safe
> chronological splitting, deterministic search and recommendation baselines,
> formally tested ranking metrics, a model-independent evaluator, and a
> reproducible MIND-small popularity evaluation. The project currently has
> 120 automated tests. Formal evaluation of the content-based and cold-start
> models remains in progress.

## Why this project

NewsLens is designed to demonstrate:

- production-style Python package organization;
- validated ingestion of real interaction data;
- explicit prevention and testing of temporal leakage;
- deterministic recommendation and search baselines;
- cold-start handling;
- formally tested ranking metrics;
- model-independent offline evaluation;
- reproducible experiment commands and machine-readable reports;
- model comparison and error analysis;
- API serving, monitoring, testing, and packaging;
- CI, containerization, experiment tracking, and deployment; and
- clear communication of limitations and negative results.

## Implemented functionality

### Phase 1: Foundation

- installable Python package;
- isolated environment instructions;
- automated smoke tests;
- Ruff linting and formatting;
- GitHub Actions continuous integration; and
- feature-branch and pull-request development workflow.

### Phase 2: MIND ingestion and audit

- validated `news.tsv` loader;
- validated `behaviors.tsv` loader;
- timestamp and candidate-impression parsing;
- duplicate-identifier detection;
- malformed-schema and invalid-label rejection;
- reproducible dataset-audit command;
- train and development audit reports; and
- synthetic unit and command-line tests.

### Phase 3: Evaluation-safe baselines

- strict chronological train/validation split;
- temporal-overlap tests;
- training-only popularity recommender;
- TF-IDF article search;
- TF-IDF user-history content recommender;
- popularity fallback for cold-start users;
- version-controlled baseline parameters; and
- documented assumptions and limitations.

### Phase 4: Ranking evaluation — in progress

- hand-validated NDCG@K;
- hand-validated Mean Reciprocal Rank@K;
- hand-validated Recall@K;
- hand-validated Hit Rate@K;
- catalog-coverage measurement;
- explicit multiple-click and no-click semantics;
- duplicate-ranking and invalid-input validation;
- model-independent candidate-ranking evaluation;
- reproducible popularity-evaluation command;
- machine-readable popularity metrics;
- chronological MIND-small validation results; and
- 120 automated tests.

Formal evaluation of the content-based recommender and cold-start fallback is
the next Phase 4 milestone.

## Quick start

Python 3.12 is recommended. The package supports Python 3.11 and Python 3.12.

### macOS or Linux

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m newslens --help
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
python -m newslens --help
```

### Manual setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

Expected checks:

```text
All checks passed!
120 passed
```

## Repository structure

```text
NewsLens/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── baselines.json
├── data/
│   └── README.md
├── docs/
│   ├── BASELINES.md
│   ├── DECISIONS.md
│   ├── EVALUATION.md
│   ├── INTERVIEW_NOTES.md
│   └── LEARNING_LOG.md
├── notebooks/
│   └── README.md
├── reports/
│   ├── mindsmall_dev_audit.json
│   ├── mindsmall_train_audit.json
│   └── popularity_metrics.json
├── scripts/
│   ├── setup.ps1
│   └── setup.sh
├── src/
│   └── newslens/
│       ├── data/
│       │   ├── audit.py
│       │   └── mind.py
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── metrics.py
│       │   ├── popularity.py
│       │   └── split.py
│       ├── models/
│       │   ├── content.py
│       │   ├── fallback.py
│       │   ├── popularity.py
│       │   └── tfidf.py
│       ├── cli.py
│       └── __main__.py
├── tests/
├── LICENSE
├── pyproject.toml
├── README.md
└── ROADMAP.md
```

Downloaded data, virtual environments, caches, and other local artifacts are
excluded from Git.

## Dataset policy

NewsLens uses MIND-small from the Microsoft MIND news-recommendation dataset:

https://msnews.github.io/

Read and accept the applicable Microsoft Research License Terms before
downloading or using the dataset. Dataset archives and extracted raw files must
not be committed to this repository.

The expected local data layout is:

```text
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
```

Only aggregate audit and evaluation reports are version controlled. Raw MIND
records and embeddings are not redistributed.

## Data validation

The MIND loaders:

- verify that input files exist;
- validate the expected number of tab-separated fields;
- reject empty required identifiers;
- reject duplicate news and impression identifiers;
- parse timestamps using the documented MIND format;
- preserve empty histories as valid cold-start cases;
- parse candidate impressions into article and label pairs;
- reject labels other than `0` and `1`; and
- report malformed rows with actionable error messages.

## Reproducing the MIND-small audit

After placing the licensed files in the expected local directories, run:

```bash
python -m newslens audit-data \
  --split train \
  --output reports/mindsmall_train_audit.json
```

```bash
python -m newslens audit-data \
  --split dev \
  --output reports/mindsmall_dev_audit.json
```

Each command validates the selected split, prints its audit results, and writes
a machine-readable JSON report.

## MIND-small audit results

| Metric | Train | Development |
|---|---:|---:|
| News articles | 51,282 | 42,416 |
| Behavior records | 156,965 | 73,152 |
| Unique users | 50,000 | 50,000 |
| Candidate impressions | 5,843,444 | 2,740,998 |
| Clicks | 236,344 | 111,383 |
| Non-clicks | 5,607,100 | 2,629,615 |
| Click-through rate | 4.0446% | 4.0636% |
| Average candidates per impression | 37.2277 | 37.4699 |
| Average history length | 32.5400 | 32.2960 |
| Empty histories | 3,238 | 2,214 |
| Missing titles | 0 | 0 |
| Missing abstracts | 2,666 | 2,021 |
| Referenced articles missing metadata | 0 | 0 |

Click-through rate is defined as clicked candidate occurrences divided by all
candidate occurrences.

The complete reports are stored in:

```text
reports/
├── mindsmall_train_audit.json
└── mindsmall_dev_audit.json
```

These results establish dataset integrity and summarize the available records.
They do not establish recommendation quality.

## Leakage-safe evaluation protocol

The `MINDsmall_train` behavior records are divided into internal training and
validation partitions using a strict chronological boundary.

| Property | Value |
|---|---:|
| Total behavior records | 156,965 |
| Training records | 125,572 |
| Validation records | 31,393 |
| Requested validation fraction | 20% |
| Actual validation fraction | 20% |
| Cutoff timestamp | 2019-11-13 20:36:26 |
| Last training timestamp | 2019-11-13 20:36:19 |
| First validation timestamp | 2019-11-13 20:36:26 |
| Temporal overlap | None |

Records sharing the same timestamp remain in the same partition. Consequently,
the maximum training timestamp is strictly earlier than the minimum validation
timestamp.

The official MIND-small development split is reserved as a later final holdout.
It is not used for fitting or selecting the current models.

## Ranking metrics

NewsLens implements formally tested offline ranking metrics.

### NDCG@K

Normalized Discounted Cumulative Gain uses binary click relevance and discounts
relevant articles appearing lower in the recommendation list.

### Mean Reciprocal Rank@K

MRR measures the reciprocal position of the first clicked article appearing
within the first `K` recommendations.

### Recall@K

Recall measures the fraction of clicked candidate articles retrieved within the
first `K` positions.

### Hit Rate@K

Hit Rate records whether at least one clicked article appears within the first
`K` positions.

### Catalog Coverage@K

Catalog coverage measures the fraction of the available article catalog
recommended at least once.

The metric implementation also defines and tests:

- impressions containing multiple clicked articles;
- impressions without clicked articles;
- rankings shorter than `K`;
- duplicate recommended identifiers;
- invalid cutoff values;
- invalid or empty inputs; and
- catalog consistency.

## Model-independent evaluator

The evaluator accepts candidate rankings from different recommendation
strategies and calculates all supported metrics using consistent validation
impressions.

This separates metric computation from model implementation and allows
popularity, content-based, fallback, and future hybrid models to be compared
under the same protocol.

The evaluator reports:

- NDCG@K;
- MRR@K;
- Recall@K;
- Hit Rate@K;
- catalog coverage;
- evaluated and skipped impressions;
- empty rankings;
- unique recommended articles; and
- candidate-level coverage information.

## Training-only popularity baseline

The popularity recommender counts clicks observed only in the chronological
training partition.

For each supplied candidate set, articles are ranked by:

1. descending training click count; and
2. article identifier as a deterministic tie-breaker.

Unseen articles receive a score of zero. Validation clicks never modify fitted
scores.

## Reproducing the popularity evaluation

After placing the licensed MIND-small data in the expected local directories,
run:

```bash
python -m newslens evaluate-popularity \
  --data-dir data \
  --k 10 \
  --validation-fraction 0.20 \
  --output reports/popularity_metrics.json
```

The command:

1. validates and loads the MIND-small training data;
2. creates the strict chronological split;
3. fits popularity using only the training partition;
4. ranks each validation candidate set;
5. calculates the formal ranking metrics; and
6. writes a machine-readable JSON report.

## Formal popularity evaluation

| Metric | Result |
|---|---:|
| NDCG@10 | 0.2853 |
| MRR@10 | 0.2308 |
| Recall@10 | 0.5047 |
| Hit Rate@10 | 0.5705 |
| Catalog Coverage@10 | 0.0402 |
| Evaluated impressions | 31,393 |
| Evaluation fraction | 100% |
| Empty rankings | 0 |
| Unique recommended articles | 2,061 |
| Catalog size | 51,282 |
| Candidate occurrences | 1,264,253 |
| Unseen candidate occurrences | 383,855 |
| Unseen candidate fraction | 30.36% |
| Temporal leakage detected | No |

The popularity model places at least one clicked article in its first ten
positions for 57.05% of validation impressions and retrieves approximately
50.47% of all clicked articles within those positions.

However, the model recommends only 4.02% of the available catalog. Its rankings
are concentrated among frequently clicked articles, which demonstrates the
coverage limitation of global popularity ranking.

Approximately 30.36% of validation candidate occurrences were unseen during
training. These articles receive zero popularity scores, exposing an important
cold-start limitation for the content-based and future hybrid systems to
address.

The formal Hit Rate@10 agrees with the earlier preliminary top-ten diagnostic,
providing a useful reproducibility check.

The machine-readable report is stored at:

```text
reports/popularity_metrics.json
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for the complete protocol, metric
semantics, interpretation, limitations, and next milestones.

## TF-IDF article search

The article-search baseline combines:

- title;
- abstract;
- category; and
- subcategory.

The resulting text is represented using English TF-IDF unigrams and bigrams.
Search queries are ranked using cosine similarity.

Example:

```python
from newslens.data import load_news
from newslens.models import TfidfArticleSearch

news = load_news("data/MINDsmall_train/news.tsv")
search = TfidfArticleSearch().fit(news)

for result in search.search("space exploration mission", top_k=5):
    print(result.news_id, result.score, result.title)
```

## Content-based history recommendations

The content recommender creates a user profile by averaging the TF-IDF vectors
of articles in that user's reading history. Candidate articles are ranked by
cosine similarity to the resulting profile.

For leakage-aware experiments, the TF-IDF vocabulary and inverse-document
frequencies can be fitted only on articles referenced by the chronological
training partition. Available candidate metadata can then be transformed
without using validation interaction labels.

Formal content-based evaluation remains a Phase 4 milestone.

## Cold-start fallback

The system uses training-only popularity when:

- a user has an empty history;
- none of the history articles are recognized; or
- the history produces no usable content-similarity signal.

This is a rule-based fallback rather than a weighted hybrid model. Content and
popularity scores are not blended.

Formal evaluation of fallback frequency and ranking quality remains a Phase 4
milestone.

## Reproducible baseline configuration

The baseline parameters are recorded in:

```text
configs/baselines.json
```

The configuration documents:

- dataset and holdout policies;
- chronological split behavior;
- popularity scoring;
- TF-IDF fields and parameters;
- content-profile construction;
- cold-start triggers; and
- deterministic tie handling.

See [docs/BASELINES.md](docs/BASELINES.md) for the baseline methodology,
assumptions, diagnostics, and limitations.

## Testing and code quality

Run the complete test suite:

```bash
python -m pytest
```

Run lint checks:

```bash
python -m ruff check .
```

Apply automatic formatting:

```bash
python -m ruff format .
```

The 120-test suite covers:

- malformed TSV schemas;
- duplicate identifiers;
- invalid click labels;
- timestamp parsing;
- dataset auditing;
- command-line report generation;
- chronological splitting;
- temporal-overlap prevention;
- popularity fitting and ranking;
- TF-IDF search;
- content-profile recommendations;
- cold-start behavior;
- hand-calculated ranking metrics;
- metric boundary and invalid-input conditions;
- model-independent evaluation;
- reproducible popularity evaluation;
- CLI behavior;
- model-not-fitted errors; and
- deterministic outputs.

GitHub Actions runs linting and tests automatically for pushes and pull
requests.

## Development workflow

NewsLens is developed using small feature branches and pull requests.

A feature is considered complete only when:

1. its implementation is tested;
2. Ruff checks pass;
3. the complete test suite passes;
4. assumptions and limitations are documented;
5. reproducible outputs are recorded when applicable; and
6. GitHub Actions passes after the pull request is opened.

## Development roadmap

1. **Foundation** — completed.
2. **MIND ingestion and audit** — completed.
3. **Evaluation-safe baselines** — completed.
4. **Ranking evaluation** — in progress; metrics, evaluator, and popularity
   evaluation completed.
5. **Hybrid ranking** — planned.
6. **Inference service** — planned.
7. **MLOps and deployment** — planned.
8. **Controlled experiments** — planned.

See [ROADMAP.md](ROADMAP.md) for detailed acceptance criteria and planned
deliverables.

## Current limitations and non-claims

- Only the popularity baseline has completed formal ranking evaluation.
- Content-based and cold-start fallback evaluations remain in progress.
- A complete baseline-comparison table has not yet been published.
- The official MIND-small development split has not yet been evaluated.
- Popularity scores reflect historical exposure as well as user interest.
- Popularity provides limited catalog coverage.
- Unseen articles receive zero popularity scores.
- TF-IDF captures lexical overlap but not deeper semantic similarity.
- User-history articles currently receive equal weight.
- Article publication timestamps are unavailable in the local metadata.
- The fallback switches strategies instead of learning a joint ranker.
- No neural recommendation model has been trained.
- No uncertainty estimates or statistical-significance tests are reported.
- No user-history or category-level subgroup analysis has been completed.
- No inference API has been implemented or deployed.
- No online experiment has been conducted.
- Passing tests and offline metrics do not establish online product impact.
- Results should not be compared directly with systems using different data
  splits, candidate policies, or metric definitions.

These statements should be updated only after the corresponding work has been
implemented, tested, and documented.

## License

Code in this repository is released under the MIT License.

The Microsoft MIND dataset is governed by separate Microsoft Research License
Terms and is not redistributed in this repository.