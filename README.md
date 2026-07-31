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
> formally tested ranking metrics, a model-independent evaluator, and
> reproducible MIND-small popularity, TF-IDF history-content, and rule-based
> cold-start fallback evaluations. It also includes exhaustive history-length
> subgroup evaluation with exact impression accounting. The project currently
> has 171 automated tests. Category-level analysis, unseen-article analysis,
> and uncertainty estimation remain in progress.

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
- reproducible popularity, content, and fallback evaluation commands;
- machine-readable popularity, content, and fallback metrics;
- same-split three-model comparison;
- explicit content abstention and cold-start accounting;
- explicit fallback routing and recovery accounting;
- exhaustive history-length segmentation with per-segment ranking metrics;
- chronological MIND-small validation results; and
- 171 automated tests.

Category-level analysis, unseen-article analysis, uncertainty estimation, and
final holdout evaluation are the next Phase 4 milestones.

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
159 passed
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
│       ├── data/
│       │   ├── audit.py
│       │   └── mind.py
│       ├── evaluation/
│       │   ├── content.py
│       │   ├── evaluator.py
│       │   ├── fallback.py
│       │   ├── metrics.py
│       │   ├── popularity.py
│       │   ├── segments.py
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

## Reproducing the content evaluation

The TF-IDF content recommender builds each user profile from the mean article
vectors in that user's history and ranks only the candidates presented in that
validation impression.

The vocabulary and inverse-document frequencies are fitted using articles
referenced by the chronological training partition. Validation interaction
labels are not used during fitting.

Run:

```bash
python -m newslens evaluate-content \
  --data-dir data \
  --k 10 \
  --validation-fraction 0.20 \
  --max-features 50000 \
  --output reports/content_metrics.json
```

## Reproducing the fallback evaluation

The fallback system preserves TF-IDF content rankings whenever positive
similarity is available. It routes only content abstentions to the popularity
model fitted on the chronological training partition.

Run:

```bash
python -m newslens evaluate-fallback \
  --data-dir data \
  --k 10 \
  --validation-fraction 0.20 \
  --max-features 50000 \
  --output reports/fallback_metrics.json
```

The fallback report also includes exhaustive `history_segments`. Segment
membership depends only on the number of articles in each impression's user
history; validation click labels do not affect assignment.

## Three-model evaluation

All three systems use the same 125,572 training records, 31,393 later
validation records, candidate sets, catalog, metric implementation, and
ranking cutoff.

| Metric @10 | Popularity | TF-IDF content | Content + fallback |
|---|---:|---:|---:|
| NDCG | 0.2853 | 0.3594 | **0.3664** |
| MRR | 0.2308 | 0.3133 | **0.3179** |
| Recall | 0.5047 | 0.5819 | **0.5955** |
| Hit Rate | 0.5705 | 0.6610 | **0.6762** |
| Catalog Coverage | 0.0402 | **0.0719** | **0.0719** |
| Unique recommended articles | 2,061 | **3,687** | **3,687** |
| Empty rankings | 0 | 927 | **0** |

Relative to content alone, fallback improves NDCG by 1.92%, MRR by 1.47%,
Recall by 2.33%, and Hit Rate by 2.30%. Catalog coverage is unchanged.

The content model produces a meaningful ranking for 30,466 validation
impressions, or 97.05%. The remaining 927 impressions, or 2.95%, are routed to
training-only popularity. Every routed impression receives a ranking, reducing
empty rankings from 927 to zero.

The fallback routes comprise 801 empty-history impressions and 126 impressions
whose candidates have zero TF-IDF similarity to the user profile. There are no
unknown-history or zero-profile cases in this validation partition.

These are internal chronological-validation results, not final holdout or
online-product results. The official MIND-small development split remains
untouched.

## History-length segment evaluation

The fallback system's validation impressions are partitioned into four
mutually exclusive, exhaustive history-length groups:

| Segment | History articles | Impressions | Share | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold start | 0 | 801 | 2.55% | 0.2954 | 0.2334 | 0.5339 | 0.6017 |
| Short history | 1–4 | 3,379 | 10.76% | 0.3471 | 0.2809 | 0.5911 | 0.6324 |
| Medium history | 5–9 | 4,994 | 15.91% | **0.3767** | 0.3184 | **0.6148** | 0.6676 |
| Long history | 10+ | 22,219 | 70.78% | 0.3695 | **0.3265** | 0.5940 | **0.6875** |

All 31,393 validation impressions are assigned exactly once, and recombining
the segment results reproduces the overall fallback metrics. Cold-start
impressions are weakest across all four relevance metrics, while usable
history generally improves ranking quality. The pattern is not uniformly
monotonic: medium histories achieve the highest NDCG and Recall, whereas long
histories achieve the highest MRR and Hit Rate.

The 801 cold-start impressions exactly match the empty-history fallback
routes. The additional 126 zero-signal fallback routes have nonempty histories
and therefore belong to the short-, medium-, or long-history groups. History
segments and fallback-reason groups answer different questions and should not
be treated as interchangeable.

Because 70.78% of impressions are in the long-history group, these subgroup
results are descriptive associations under this split, not evidence that
longer histories causally improve recommendations. Confidence intervals have
not yet been estimated.

The machine-readable reports are stored at:

```text
reports/popularity_metrics.json
reports/content_metrics.json
reports/fallback_metrics.json
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

Formal chronological evaluation is complete. The content model improves all
recorded ranking metrics over popularity while abstaining on 2.95% of
validation impressions. The content-only report intentionally retains those
abstentions as zero-scoring empty rankings.

## Cold-start fallback

The system uses training-only popularity when:

- a user has an empty history;
- none of the history articles are recognized; or
- the history produces no usable content-similarity signal.

This is a rule-based fallback rather than a weighted hybrid model. Content and
popularity scores are not blended.

Formal chronological evaluation is complete. Content handles 30,466 validation
impressions, while popularity handles the remaining 927 abstentions. The
fallback recovers all 927 and eliminates empty rankings. Relative to content
alone, it improves NDCG@10 by 1.92%, MRR@10 by 1.47%, Recall@10 by 2.33%, and
Hit Rate@10 by 2.30%, while catalog coverage remains 7.19%.

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

The 171-test suite covers:

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
- leakage-aware content evaluation;
- content abstention and cold-start accounting;
- reproducible content-evaluation CLI behavior;
- leakage-aware fallback evaluation;
- content-versus-popularity routing and fallback-reason accounting;
- exhaustive and non-overlapping history-segment definitions;
- empty-history and no-click segment accounting;
- model-independent per-segment ranking evaluation;
- fallback integration and deterministic segment JSON output;
- reproducible fallback-evaluation CLI behavior;
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
4. **Ranking evaluation** — in progress; metrics, evaluator, popularity
   evaluation, TF-IDF content evaluation, and cold-start fallback evaluation
   completed. History-length segment evaluation is complete; category-level,
   unseen-article, and uncertainty analyses remain.
5. **Hybrid ranking** — planned.
6. **Inference service** — planned.
7. **MLOps and deployment** — planned.
8. **Controlled experiments** — planned.

See [ROADMAP.md](ROADMAP.md) for detailed acceptance criteria and planned
deliverables.

## Current limitations and non-claims

- Popularity, TF-IDF content, and the rule-based cold-start fallback have
  completed formal ranking evaluation; no learned hybrid model has been
  evaluated.
- The content-only model abstains on 2.95% of validation impressions; the
  fallback fills those rankings using global training-only popularity.
- Fallback removes empty rankings but does not improve catalog coverage over
  content alone.
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
- History-length segment sizes are highly imbalanced: 70.78% of validation
  impressions belong to the long-history group.
- History-length results are descriptive associations, not causal estimates.
- No category-level subgroup analysis has been completed.
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
