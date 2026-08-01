# NewsLens API

NewsLens includes a tested FastAPI service foundation for exposing model
metadata and, in later milestones, news-search and recommendation inference.

## Current scope

The current API provides:

- service liveness information;
- package-version information;
- configured recommendation-model metadata;
- automatic OpenAPI documentation; and
- typed and tested JSON responses.

The current API does not load a serialized model artifact or provide search or
recommendation inference.

## Running locally

Install NewsLens with its development dependencies:

```bash
python -m pip install -e ".[dev]"