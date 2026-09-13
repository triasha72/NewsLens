# Public demo runbook

This mode exposes the NewsLens real-time search API without licensed MIND
records, trained artifacts, user data, or a PostgreSQL database. It exists to
let a reviewer inspect query understanding, candidate selection, freshness
ranking, and API observability safely.

It is not a live news service and it does not demonstrate recommendation
quality, throughput, multi-host resilience, or user impact.

## Run locally

```bash
NEWSLENS_DEMO_MODE=true python -m uvicorn newslens.api.app:app --host 127.0.0.1 --port 8000
```

Then open the API documentation at `http://127.0.0.1:8000/docs`, or send a
search request:

```bash
curl --get --data-urlencode 'q=latest AI chip' http://127.0.0.1:8000/search
```

The response names its request identifier, parsed query intent, candidate
count, result count, total search time, and each result's relevance, freshness,
popularity, and index-freshness components.

## Deployment contract

- Set `NEWSLENS_DEMO_MODE=true`.
- Do not set `NEWSLENS_REALTIME_DATABASE_URL`; the API rejects that mixed mode
  at startup so a demonstration cannot accidentally read a live article store.
- Do not set `NEWSLENS_ARTIFACT_PATH` or `NEWSLENS_RELEASE_ROOT`; neither is
  needed for search demonstration mode.
- Place the service behind a platform-managed HTTPS endpoint and keep the
  OpenAPI interface available only if the host's access policy permits it.
- Treat logs as operational data and configure the host's retention policy
  before accepting public traffic.

The bundled articles are intentionally synthetic and refreshed relative to
process start. Their text, identifiers, and popularity values are test fixtures,
not real reporting or user engagement signals.

## Deploy on Render

This repository can be deployed as a small public demonstration on Render
without copying licensed data or model artifacts to the host.

1. Create a **Web Service** from `triasha72/NewsLens` on the `main` branch.
2. Choose the **Docker** runtime. Render builds the repository's `Dockerfile`.
3. Set the health check path to `/health`.
4. Add `NEWSLENS_DEMO_MODE=true` as a runtime environment variable.
5. Confirm that `NEWSLENS_REALTIME_DATABASE_URL`, `NEWSLENS_ARTIFACT_PATH`,
   and `NEWSLENS_RELEASE_ROOT` are not configured, then create the service.

Render supplies the `PORT` variable used by the container. Locally, the
container continues to listen on port 8000 when that variable is absent.

After the deployment is marked live, verify the public URL:

```bash
curl --fail https://<service-name>.onrender.com/health
curl --get --fail --data-urlencode 'q=latest AI chip' \
  https://<service-name>.onrender.com/search
```

Use the service description or linked documentation to state that this is a
synthetic search demonstration, not a live news or personalized recommendation
service. Configure log retention before sharing the URL publicly.
