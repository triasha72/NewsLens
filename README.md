# NewsLens

NewsLens is an in-progress, leakage-aware news search and recommendation system
built with the Microsoft MIND news-recommendation dataset.

The project demonstrates the complete progression from validated interaction
data to reproducible recommendation baselines, offline evaluation, model
serving, and deployment. Each phase documents its assumptions, limitations, and
non-claims.

> Current status: Phase 3 evaluation-safe baselines completed. NewsLens includes
> validated MIND ingestion, reproducible dataset auditing, a strict chronological
> split, training-only popularity ranking, TF-IDF search, content-based history
> recommendations, a popularity cold-start fallback, continuous integration, and
> 66 automated tests. Formal ranking evaluation and deployment remain future
> work.

## Why this project

NewsLens is designed to demonstrate:

- production-style Python package organization;
- validated ingestion of real interaction data;
- explicit prevention and testing of temporal leakage;
- deterministic recommendation and search baselines;
- cold-start handling;
- reproducible configurations and dataset reports;
- ranking evaluation with NDCG, MRR, Recall, and coverage;
- model comparison and error analysis;
- API serving, monitoring, testing, and packaging;
- CI, containerization, experiment tracking, and deployment; and
- clear communication of model limitations and negative results.

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
- version-controlled baseline parameters;
- documented assumptions and limitations; and
- 66 automated tests.

## Quick start

Python 3.12 is recommended. The package supports Python 3.11 and 3.12.

### macOS or Linux

```bash
bash scripts/setup.sh
source .venv/bin/activate
newslens
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
newslens
```

### Manual setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

Expected checks:

```text
66 passed
All checks passed!
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
│   ├── INTERVIEW_NOTES.md
│   └── LEARNING_LOG.md
├── notebooks/
│   └── README.md
├── reports/
│   ├── mindsmall_dev_audit.json
│   └── mindsmall_train_audit.json
├── scripts/
│   ├── setup.ps1
│   └── setup.sh
├── src/
│   └── newslens/
│       ├── data/
│       │   ├── audit.py
│       │   └── mind.py
│       ├── evaluation/
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

Downloaded data, virtual environments, caches, and other generated artifacts are
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

Only aggregate audit reports are version controlled. Raw MIND records and
embeddings are not redistributed.

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

Click-through rate is defined as clicked candidate impressions divided by all
candidate impressions.

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
It is not used for fitting or selecting Phase 3 models.

## Training-only popularity baseline

The popularity recommender counts clicks observed only in the chronological
training partition.

For any supplied candidate set, articles are ranked by:

1. descending training click count; and
2. article identifier as a deterministic tie-breaker.

Unseen articles receive a score of zero. Validation clicks never modify the
fitted scores.

### Preliminary popularity diagnostic

| Diagnostic | Result |
|---|---:|
| Validation impressions evaluated | 31,393 |
| Top-1 clicked-article rate | 11.48% |
| Top-10 clicked-article rate | 57.05% |
| Average best clicked-article rank | 15.72 |
| Unseen candidate occurrence rate | 30.36% |
| Temporal leakage detected | No |

These are preliminary diagnostics rather than final ranking-quality claims.
Phase 4 will introduce formally tested ranking metrics before publishing a
definitive model-comparison table.

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
of articles in that user's reading history. Candidate articles are then ranked
by cosine similarity to the profile.

For leakage-aware experiments, the TF-IDF vocabulary and inverse-document
frequencies can be fitted only on articles referenced by the chronological
training partition. Available candidate metadata can then be transformed
without using validation interaction labels.

## Cold-start fallback

The system uses training-only popularity when:

- a user has an empty history;
- none of the history articles are recognized; or
- the history produces no usable content-similarity signal.

This is a rule-based fallback rather than a weighted hybrid model. Content and
popularity scores are not blended during Phase 3.

## Reproducible baseline configuration

The Phase 3 parameters are recorded in:

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

See [docs/BASELINES.md](docs/BASELINES.md) for the full methodology, assumptions,
diagnostics, and limitations.

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

The test suite covers:

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
- invalid inputs; and
- model-not-fitted errors.

GitHub Actions runs linting and tests automatically for pushes and pull
requests.

## Development roadmap

1. **Foundation** — completed.
2. **MIND ingestion and audit** — completed.
3. **Evaluation-safe baselines** — completed.
4. **Ranking evaluation** — next.
5. **Hybrid ranking** — planned.
6. **Inference service** — planned.
7. **MLOps and deployment** — planned.
8. **Controlled experiments** — planned.

See [ROADMAP.md](ROADMAP.md) for detailed acceptance criteria and planned
deliverables.

## Current limitations and non-claims

- Formal NDCG@K, MRR, Recall@K, and coverage evaluation is not yet implemented.
- Preliminary popularity diagnostics are not a final model comparison.
- Popularity scores reflect both user interest and historical exposure.
- TF-IDF captures lexical overlap but not deeper semantic similarity.
- User-history articles currently receive equal weight.
- Article publication timestamps are unavailable in the local metadata.
- The fallback switches strategies instead of learning a joint ranker.
- No neural recommendation model has been trained.
- No uncertainty estimates or statistical significance tests are reported.
- No inference API has been implemented or deployed.
- No online experiment has been conducted.
- Passing tests and dataset audits do not establish product impact.

These statements should be updated only after the corresponding work has been
implemented, tested, and documented.

## License

Code in this repository is released under the MIT License.

The Microsoft MIND dataset is governed by separate Microsoft Research License
Terms and is not redistributed in this repository.