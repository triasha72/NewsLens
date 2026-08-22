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

## Kubernetes

The manifests in `deploy/kubernetes` run two non-root replicas, use the API's
separate liveness and readiness endpoints, mount the model read-only, and add a
service, CPU autoscaling, a disruption budget, and a default-deny egress policy.

The `newslens-model` claim must be bound to storage containing the files from
`artifacts/newslens-fallback-0.3.0` at the root of that volume. The repository
cannot create this data because MIND is licensed and trained artifacts are not
committed. Adjust the claim or add a storage-class-specific population job for
the target cluster, then validate and deploy:

```bash
kubectl kustomize deploy/kubernetes
kubectl apply -k deploy/kubernetes
kubectl -n newslens rollout status deployment/newslens-api
kubectl -n newslens port-forward service/newslens-api 8000:80
```

The HPA requires the Kubernetes Metrics Server. The NetworkPolicy requires a
network plugin that enforces `networking.k8s.io/v1` policies.

## Repeatable load test

Create a request using article IDs present in the mounted artifact:

```json
{
  "history_news_ids": ["N123"],
  "candidate_news_ids": ["N456", "N789"],
  "top_k": 2
}
```

Save it as `/tmp/newslens-request.json`, then run:

```bash
python scripts/load_test_api.py \
  --payload /tmp/newslens-request.json \
  --requests 1000 \
  --concurrency 20 \
  --output reports/load-test.json
```

The runner refuses to start unless `/ready` succeeds. It records every client
request plus observed throughput, success rate, and p50/p95/p99 latency. Commit
or quote a report only when its target, cluster resources, artifact, and command
are also disclosed; the repository does not include a fabricated performance
result.
