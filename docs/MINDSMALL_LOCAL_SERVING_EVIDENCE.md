# MIND-small local serving evidence

This check connects the repository's data, artifact, API, and container paths using the licensed MIND-small train and development archives. It answers a narrow question: can NewsLens validate the real files, build the released fallback artifact, and serve recommendations from that artifact on a clean local setup?

## What was checked

The ingestion audit read both archives after extraction into the ignored `data/` directory. The training split contained 156,965 behavior records and 51,282 news articles; the development split contained 73,152 behavior records and 42,416 news articles. Every news ID referenced by a behavior record had matching metadata in both splits.

The export command then built artifact version `0.3.0` from the training split. The artifact indexed 51,282 articles with a 50,000-term TF-IDF vocabulary and retained the training-only popularity fallback used for cold-start and zero-signal requests.

The artifact was loaded twice: once by a local Uvicorn process and once through the repository's Docker Compose service. In both cases, liveness returned `ok`, readiness returned `ready`, and model information reported artifact version `0.3.0`.

## Bounded load observation

Each serving path received 1,000 recommendation requests at concurrency 20 using a valid payload drawn from the local development split. The raw payload and per-request results are excluded because they contain licensed article identifiers.

| Serving path | Successes | Failures | Throughput | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Local Uvicorn | 1,000 | 0 | 936.24 requests/s | 19.48 ms | 40.33 ms | 64.58 ms |
| Docker Compose | 1,000 | 0 | 742.31 requests/s | 25.34 ms | 44.83 ms | 66.14 ms |

These are client-observed results from one Apple M4 MacBook Air with 16 GB of memory. They show that the bounded local paths worked without request failures; they are not a production capacity claim or an SLO.

## Reproduction and evidence boundaries

The archive, artifact, environment, and aggregate load details are recorded in [`reports/mindsmall_local_serving_v0_1.json`](../reports/mindsmall_local_serving_v0_1.json). The MIND-small files, generated artifact, request payload, and raw load results stay outside version control.

This original check did not use Kubernetes. A later Docker Desktop deployment
connected the same artifact to the repository manifests; its separate scope
and results are recorded in
[`KUBERNETES_LOCAL_EVIDENCE.md`](KUBERNETES_LOCAL_EVIDENCE.md).
