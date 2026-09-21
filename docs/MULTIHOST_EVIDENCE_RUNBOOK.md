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

The deployment keeps the PostgreSQL password separate from its URL-encoded
connection-string form. On each host, after setting
`NEWSLENS_POSTGRES_PASSWORD`, derive the encoded form without printing either
value:

```bash
python3 - <<'PY'
from pathlib import Path
from urllib.parse import quote

env_file = Path(".env")
lines = env_file.read_text().splitlines()
password = next(line.split("=", 1)[1] for line in lines if line.startswith("NEWSLENS_POSTGRES_PASSWORD="))
lines = [line for line in lines if not line.startswith("NEWSLENS_POSTGRES_PASSWORD_URLENCODED=")]
lines.append(f"NEWSLENS_POSTGRES_PASSWORD_URLENCODED={quote(password, safe='')}")
env_file.write_text("\n".join(lines) + "\n")
PY
```

## Run

Run stages on the host that owns the operation. Replace
`<state-host-private-dns>` and `<app-host-private-dns>` at execution time;
never place host addresses or `.env` content in Git.

On the app host, run load against the state tier and record both consumers:

```bash
PYTHONPATH=src python scripts/realtime/benchmark_ingestion.py \
  --ingestion-url http://<state-host-private-dns>:8080 \
  --search-url http://<state-host-private-dns>:8000 \
  --consumer-url http://<state-host-private-dns>:8081 \
  --consumer-url http://127.0.0.1:8082 \
  --execution-host-role application --environment aws-ec2-two-host \
  --endpoint-role ingestion=state-host --endpoint-role search=state-host \
  --endpoint-role consumer-1=state-host --endpoint-role consumer-2=application-host \
  --output reports/multihost/load.json
```

On the state host, run recovery and DLQ with its local Compose project. Pass
the state and app consumer endpoints explicitly, saving `recovery.json` and
`dlq.json` in the same evidence directory:

```bash
PYTHONPATH=src python scripts/realtime/failure_recovery.py \
  --compose-file deploy/ec2/state-host.compose.yaml \
  --ingestion-url http://127.0.0.1:8080 --search-url http://127.0.0.1:8000 \
  --execution-host-role state --environment aws-ec2-two-host \
  --endpoint-role ingestion=state-host --endpoint-role search=state-host \
  --single-consumer-only \
  --output reports/multihost/recovery.json

PYTHONPATH=src python scripts/realtime/verify_dlq.py \
  --compose-file deploy/ec2/state-host.compose.yaml \
  --consumer-url http://127.0.0.1:8081 \
  --consumer-url http://<app-host-private-dns>:8082 \
  --execution-host-role state --environment aws-ec2-two-host \
  --endpoint-role consumer-1=state-host --endpoint-role consumer-2=application-host \
  --output reports/multihost/dlq.json
```

Copy the three redacted JSON reports to an operator-controlled evidence
directory. Do not use Git as the transfer mechanism and do not copy `.env`
files. Build and verify the receipt there together with a redacted
`topology.json`:

```bash
PYTHONPATH=src python scripts/realtime/build_multihost_receipt.py \
  --load reports/multihost/load.json --recovery reports/multihost/recovery.json \
  --dlq reports/multihost/dlq.json --topology reports/multihost/topology.json \
  --output-dir reports/multihost

PYTHONPATH=src python scripts/realtime/build_multihost_receipt.py \
  --verify reports/multihost/multihost_checksums_v0_1.json
```

The topology manifest must set `stateful_services_highly_available` to
`false`. This is multi-host application evidence, not an HA deployment.

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
```

The receipt must include a separate topology manifest showing the state host,
app host, commit, image digests, and service health. Record failures by
restarting the consumer and state services during separate runs; do not fold
unplanned failures into a passing run.

## Acceptance

Publish the redacted load, recovery, DLQ, topology, receipt, and checksum
files. This closes the multi-host application-evidence gap when the receipt
records the two-host topology, measured freshness and recovery values,
duplicate handling, DLQ behavior, and no unexplained event loss. The current
one-broker, one-PostgreSQL setup correctly keeps the separate high-availability
gate blocked. It must be described as multi-host application evidence, not as
a highly available production deployment.

The two-host run proves consumer-1 failover while consumer-2 stays active on
the application host. A deliberately coordinated all-consumer outage is a
separate exercise; do not infer that result from this receipt.
