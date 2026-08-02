# NewsLens container deployment

NewsLens provides a Dockerized FastAPI recommendation service.

## Prerequisites

- Docker Desktop
- a locally generated NewsLens model artifact
- port 8000 available on the host

The licensed MIND data and generated model artifacts are excluded from Git and the Docker image.

## Build the image

```bash
docker build --tag newslens-api:0.3.0 .
```

## Export the model artifact

```bash
python -m newslens export-model \
  --data-dir data \
  --output artifacts/newslens-fallback-0.3.0 \
  --artifact-version 0.3.0 \
  --k 10 \
  --max-features 50000
```

The destination must not already exist.

## Run with Docker Compose

```bash
docker compose up --build --detach
docker compose ps
```

Verify the service:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/model-info
```

Interactive documentation is available at:

<http://127.0.0.1:8000/docs>

View service logs:

```bash
docker compose logs --follow api
```

Stop the service:

```bash
docker compose down
```

## Run the published container

Pull the multi-platform v0.3.0 image:

```bash
docker pull ghcr.io/triasha72/newslens:0.3.0
```

Run it with the locally generated artifact mounted read-only:

```bash
docker run --rm --detach \
  --name newslens-api \
  --publish 8000:8000 \
  --env NEWSLENS_ARTIFACT_PATH=/models/newslens-fallback-0.3.0 \
  --volume "$PWD/artifacts/newslens-fallback-0.3.0:/models/newslens-fallback-0.3.0:ro" \
  ghcr.io/triasha72/newslens:0.3.0
```

Verify it:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/model-info
```

Stop it:

```bash
docker stop newslens-api
```

## Liveness and readiness

- `/health` verifies that the HTTP process is running.
- `/ready` verifies that a valid model artifact was loaded.
- `/model-info` returns the configured model and artifact metadata.

A service can be live while not ready. Starting without an artifact allows `/health` to succeed while `/ready` returns HTTP `503`.

## Security note

NewsLens artifacts use joblib/pickle-compatible deserialization. Only mount artifacts produced by a trusted NewsLens training workflow. Checksums detect corruption but do not make untrusted serialized Python objects safe.