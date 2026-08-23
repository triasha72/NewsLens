# Local Kubernetes deployment evidence

NewsLens was deployed to the single-node Kubernetes cluster included with Docker Desktop to test the repository manifests against a real API server, scheduler, volume provisioner, kubelet, probes, and service. This closes the local deployment gap without claiming cloud or multi-node production readiness.

## What the deployment exposed

Docker Desktop's local-path provisioner rejected the base `ReadOnlyMany` claim because it supports only single-node write access. The Docker Desktop overlay changes that claim to `ReadWriteOnce`; it leaves the production storage contract intact and is safe here because both replicas run on one node.

Kubernetes 1.36 also refused the container's textual `newslens` user while enforcing `runAsNonRoot`. The deployment now declares the image's existing numeric UID and GID, `10001`, so the kubelet can verify the non-root identity before starting it.

The generated artifact was copied to the root of the bound claim and its files were verified before deployment. Both replicas loaded artifact version `0.3.0`, passed startup and readiness probes, and served through the cluster service. After one ready pod was deliberately removed, the Deployment created a replacement and returned to two ready replicas.

## Bounded service observation

The port-forwarded service received 1,000 recommendation requests at concurrency 20. All 1,000 succeeded, with client-observed throughput of 535.81 requests per second and latency of 33.21 ms at p50, 65.20 ms at p95, and 74.92 ms at p99.

The request used licensed article identifiers from MIND-small, so its payload and raw per-request records are not committed. Aggregate results and the exact environment are stored in [`reports/mindsmall_kubernetes_local_v0_1.json`](../reports/mindsmall_kubernetes_local_v0_1.json).

## Evidence boundaries

This run verifies a two-replica rollout, artifact-backed readiness, a ClusterIP service, a bound local claim, a pod disruption budget, the HPA resource, and numeric non-root execution on one Docker Desktop node. Docker Desktop did not provide the Metrics API, so CPU-driven HPA scaling was not exercised. Its Kind networking also does not establish that the NetworkPolicy is enforced, and the run does not test multi-node volume attachment, node failure, ingress, TLS, or cloud load balancing.
