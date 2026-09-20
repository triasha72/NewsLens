# NewsLens multi-host evidence runbook

This run closes the deployment-topology gap. It must run with the API on one
host and Kafka/PostgreSQL on a separate state host. A local Compose run is not
accepted as multi-host evidence.

## Preconditions

- Both hosts run the same Git commit and pinned Docker images.
- The app host can reach the state host on the Kafka, PostgreSQL, and search
  API ports required by the selected Compose manifests.
- NTP is enabled on both hosts.
- Capture the private hostnames, instance types, image digests, and commit SHA
  in the run manifest. Do not commit credentials or private IPs.

## Run

On the state host:

```bash
cd NewsLens
sudo docker compose --env-file .env -f deploy/ec2/state-host.compose.yaml up -d
sudo docker compose --env-file .env -f deploy/ec2/state-host.compose.yaml ps
```

On the app host, after setting the state-host address in `.env`:

```bash
cd NewsLens
sudo docker compose --env-file .env -f deploy/ec2/app-host.compose.yaml up -d
sudo docker compose --env-file .env -f deploy/ec2/app-host.compose.yaml ps
PYTHONPATH=src python scripts/realtime/run_soak_evidence.py \
  --duration-minutes 60 --events 100000 --concurrency 20 \
  --environment aws-ec2-two-host \
  --application-hosts 1 \
  --output-dir reports/multihost-$(date -u +%Y%m%dT%H%M%SZ)
```

The receipt must include a separate topology manifest showing the state host,
app host, commit, image digests, and service health. Record failures by
restarting the consumer and state services during separate runs; do not fold
unplanned failures into a passing run.

## Acceptance

Publish the raw load, recovery, DLQ, readiness, topology, and checksum files.
The gap is closed only when the readiness receipt reports the multi-host
topology, 100,000 events, freshness and recovery values, duplicate handling,
DLQ behavior, and no unexplained event loss. If Kafka/PostgreSQL are a single
stateful pair, label the result as multi-host application evidence, not highly
available stateful-service evidence.
