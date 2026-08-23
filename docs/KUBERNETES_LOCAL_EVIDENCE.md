# Local Kubernetes deployment evidence

NewsLens was deployed to the single-node Kubernetes cluster included with Docker Desktop to test the repository manifests against a real API server, scheduler, volume provisioner, kubelet, probes, and service. This closes the local deployment gap without claiming cloud or multi-node production readiness.

## What the deployment exposed

Docker Desktop's local-path provisioner rejected the base `ReadOnlyMany` claim because it supports only single-node write access. The Docker Desktop overlay changes that claim to `ReadWriteOnce`; it leaves the production storage contract intact and is safe here because both replicas run on one node.

Kubernetes 1.36 also refused the container's textual `newslens` user while enforcing `runAsNonRoot`. The deployment now declares the image's existing numeric UID and GID, `10001`, so the kubelet can verify the non-root identity before starting it.

The generated artifact was copied to the root of the bound claim and its files were verified before deployment. Both replicas loaded artifact version `0.3.0`, passed startup and readiness probes, and served through the cluster service. After one ready pod was deliberately removed, the Deployment created a replacement and returned to two ready replicas.

## Bounded service observation

The port-forwarded service received 1,000 recommendation requests at concurrency 20. All 1,000 succeeded, with client-observed throughput of 535.81 requests per second and latency of 33.21 ms at p50, 65.20 ms at p95, and 74.92 ms at p99.

The request used licensed article identifiers from MIND-small, so its payload and raw per-request records are not committed. Aggregate results and the exact environment are stored in [`reports/mindsmall_kubernetes_local_v0_1.json`](../reports/mindsmall_kubernetes_local_v0_1.json).

## Autoscaling, ingress, policy, and disruption follow-up

Metrics Server `v0.9.0` was added to the local cluster so the HPA could act on
real CPU measurements. Docker Desktop's kubelet certificate did not contain its
node IP as a subject alternative name, so this disposable cluster used the
documented local-only `--kubelet-insecure-tls` exception.

An in-cluster k6 workload held 100 virtual users for two minutes. All 50,182
requests succeeded. CPU reached 389% of the configured request, the HPA raised
NewsLens from 2 to its maximum of 10 ready replicas, and the Deployment returned
to 2 replicas after the five-minute stabilization window.

The Docker Desktop overlay now includes an nginx Ingress. A separate bounded
run through `http://localhost` completed 1,000 of 1,000 requests successfully,
with p95 latency of 50.59 ms.

The Eviction API accepted removal of one ready replica while the disruption
budget required one to remain available. The Deployment replaced the pod and
returned to two ready replicas. NetworkPolicy enforcement was checked by
connecting to the Kubernetes API service: a selected NewsLens pod was blocked,
while an otherwise equivalent unselected control pod connected; an allowed
client still reached the NewsLens readiness endpoint.

The follow-up environment and aggregate measurements are recorded in
[`reports/mindsmall_kubernetes_local_v0_2.json`](../reports/mindsmall_kubernetes_local_v0_2.json).

## Evidence boundaries

Together these runs verify artifact-backed rollout and readiness, service and
Ingress traffic, CPU-driven scale-up and scale-down, PDB-aware eviction, egress
policy enforcement, and numeric non-root execution on one Docker Desktop node.
They do not test multi-node volume attachment, complete node failure, public
DNS, TLS, or a cloud load balancer. The Metrics Server TLS exception is evidence
for this disposable local cluster only and must not be copied into production.
