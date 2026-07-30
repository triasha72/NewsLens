# NewsLens

NewsLens is an in-progress, leakage-aware news search and recommendation
system. The goal is to build the complete path from raw interaction logs to a
tested inference API while documenting evaluation assumptions and model
limitations.

> Current status: Phase 1 repository scaffold only. The package, tests, linting,
> and continuous-integration workflow run successfully. No recommendation model
> has been implemented yet.

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

Manual setup:

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
2 passed
All checks passed!
```

## Repository structure

- `src/newslens/`: application package
- `tests/`: automated tests
- `data/`: local dataset documentation; downloaded data are ignored
- `notebooks/`: exploratory analysis only
- `docs/`: learning log, decisions, and interview notes
- `scripts/`: reproducible setup scripts
- `.github/workflows/ci.yml`: automated lint and test checks

## Dataset policy

NewsLens will use MIND-small from the official Microsoft MIND dataset page:
https://msnews.github.io/

Read and accept the applicable Microsoft Research License Terms before
downloading the data. Dataset archives and extracted files must not be committed
to this repository.

## Development sequence

1. Repository and environment setup.
2. MIND schema validation and data audit.
3. Chronological splitting and simple baselines.
4. Ranking metrics and error analysis.
5. Hybrid content/popularity/recency recommender.
6. FastAPI service and feedback event store.
7. Model artifacts, CI, Docker, tracking, and deployment.
8. Controlled experiment integration with ExperimentLab.

See [ROADMAP.md](ROADMAP.md) for acceptance criteria and suggested commits.

## Current non-claims

- No real MIND evaluation has been completed.
- No neural or embedding model has been trained.
- No service has been deployed.
- Passing smoke tests do not establish recommendation quality.

These statements should be updated only when the corresponding work has been
implemented, tested, and documented.

## License

Code in this repository is released under the MIT License. The MIND dataset has
separate terms and is not redistributed here.
