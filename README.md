# NewsLens

NewsLens is an in-progress, leakage-aware news search and recommendation system.
The goal is to build the complete path from raw interaction logs to a tested
inference API while documenting evaluation assumptions and model limitations.

> Current status: Phase 2 data ingestion and audit completed. NewsLens includes
> validated MIND news and behavior loaders, a reproducible dataset-audit command,
> JSON audit reports generated from MIND-small, and 12 automated tests. No
> recommendation model has been implemented yet.

## Why this project

This project is designed to demonstrate:

- data validation and chronological train/validation splitting;
- search and recommendation baselines;
- ranking evaluation with NDCG, MRR, Recall, and coverage;
- model comparison and error analysis;
- API serving, feedback-event storage, testing, and packaging;
- CI, containerization, experiment tracking, and deployment.

## Quick start

Python 3.12 is recommended.

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
12 passed
All checks passed!
```

## Repository structure

- `src/newslens/`: application package
- `tests/`: automated unit and command-line tests
- `data/`: local dataset documentation; downloaded data are ignored
- `reports/`: reproducible dataset-audit summaries; raw data are excluded
- `notebooks/`: exploratory analysis only
- `docs/`: learning log, decisions, and interview notes
- `scripts/`: reproducible setup scripts
- `.github/workflows/ci.yml`: automated lint and test checks

## Dataset policy

NewsLens uses MIND-small from the Microsoft MIND news-recommendation dataset:

https://msnews.github.io/

Read and accept the applicable Microsoft Research License Terms before
downloading or using the dataset. Dataset archives and extracted files must not
be committed to this repository.

The repository's `.gitignore` excludes the downloaded data while allowing small,
reproducible summary reports to be version controlled.

Expected local data layout:

```text
data/
├── MINDsmall_train/
│   ├── news.tsv
│   └── behaviors.tsv
└── MINDsmall_dev/
    ├── news.tsv
    └── behaviors.tsv
```

## Data validation

The MIND data loaders:

- verify that input files exist;
- validate the expected number of TSV columns;
- reject missing required identifiers;
- reject duplicate news and impression identifiers;
- parse behavior timestamps;
- parse user histories;
- validate candidate-impression labels;
- reject labels other than `0` or `1`; and
- report malformed rows with actionable error messages.

The raw dataset remains local and is not redistributed.

## Reproducing the MIND-small audit

After placing the licensed dataset files in the expected local directories, run:

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

Each command prints the audit results and writes a machine-readable JSON report.

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

Click-through rate is calculated as clicked candidate impressions divided by all
candidate impressions.

The complete reproducible results are stored in:

```text
reports/
├── mindsmall_train_audit.json
└── mindsmall_dev_audit.json
```

These audit results establish the integrity and basic characteristics of the
loaded dataset. They do not measure recommendation quality.

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

The GitHub Actions workflow runs linting and tests automatically for pushes and
pull requests.

## Development sequence

1. Repository and environment setup.
2. MIND schema validation and data audit.
3. Chronological splitting and simple baselines.
4. Ranking metrics and error analysis.
5. Hybrid content, popularity, and recency recommender.
6. FastAPI service and feedback-event store.
7. Model artifacts, CI, Docker, tracking, and deployment.
8. Controlled experiment integration with ExperimentLab.

See [ROADMAP.md](ROADMAP.md) for acceptance criteria and future phases.

## Current limitations and non-claims

- No recommendation or ranking model has been trained.
- No ranking-quality evaluation has been performed.
- No chronological model-validation pipeline has been implemented.
- No inference API has been deployed.
- Dataset audit results do not establish recommendation quality.
- The committed reports summarize MIND-small but do not redistribute raw data.

These statements should be updated only when the corresponding work has been
implemented, tested, and documented.

## License

Code in this repository is released under the MIT License. The MIND dataset has
separate Microsoft Research License Terms and is not redistributed here.