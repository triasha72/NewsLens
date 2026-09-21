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
  --execution-host-role application --output reports/multihost/load.json
```

On the state host, run recovery and DLQ with its local Compose project. Pass
the state and app consumer endpoints explicitly, saving `recovery.json` and
`dlq.json` in the same evidence directory. Then build the receipt from the
three raw reports and a redacted `topology.json`:

```bash
PYTHONPATH=src python scripts/realtime/build_multihost_receipt.py \
  --load reports/multihost/load.json --recovery reports/multihost/recovery.json \
  --dlq reports/multihost/dlq.json --topology reports/multihost/topology.json \
  --output-dir reports/multihost
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
PYTHONPATH=src python scripts/realtime/run_soak_evidence.py \
  --duration-minutes 60 --events 100000 --concurrency 20 \
  --environment aws-ec2-two-host \
  --application-hosts 2 \
  --output-dir reports/multihost-$(date -u +%Y%m%dT%H%M%SZ)
```

The receipt must include a separate topology manifest showing the state host,
app host, commit, image digests, and service health. Record failures by
restarting the consumer and state services during separate runs; do not fold
unplanned failures into a passing run.

## Acceptance

Publish the raw load, recovery, DLQ, readiness, topology, and checksum files.
This closes the multi-host application-evidence gap when the receipt records
the two-host topology, 100,000 events, freshness and recovery values, duplicate
handling, DLQ behavior, and no unexplained event loss. The current one-broker,
one-PostgreSQL setup will correctly keep the separate high-availability gate
blocked. It must be described as multi-host application evidence, not as a
highly available production deployment.
