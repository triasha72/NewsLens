# Real-time search operations

This runbook exercises the NewsLens event path on one machine. It uses the same
commands for success, load, and failure evidence so a report can be reproduced
instead of copied from an unrelated environment.

## Start and verify

Requirements are Docker Engine with Compose v2, `curl`, Bash, and Python 3.11 or
3.12. The stack uses ports 8000, 8080, 8081, 8082, and 9090.

```bash
./scripts/realtime/start_stack.sh
docker compose -f deploy/realtime/compose.yaml ps
```

The start script builds both applications, waits for Compose health checks, and
then verifies the producer, both consumers, and the search store. Prometheus is
available at <http://127.0.0.1:9090>.

Publish one event:

```bash
curl --fail-with-body \
  -X POST http://127.0.0.1:8080/events \
  -H 'Content-Type: application/json' \
  -d '{
    "event_id": "manual-event-1",
    "article_id": "manual-article-1",
    "title": "Apple launches an AI chip today",
    "category": "technology",
    "published_at": "2026-08-23T12:00:00Z",
    "produced_at": "2026-08-23T12:00:01Z",
    "body": "A manual end-to-end probe."
  }'

curl --get --fail-with-body \
  --data-urlencode 'q=latest Apple AI chip' \
  --data-urlencode 'top_k=5' \
  http://127.0.0.1:8000/search
```

Publishing the same request again should increment the duplicate metric without
creating a second article mutation:

```bash
curl http://127.0.0.1:8081/metrics
curl http://127.0.0.1:8082/metrics
```

## Quality contract

The query and ranking fixture does not need Docker:

```bash
PYTHONPATH=src python scripts/evaluate_realtime_search.py
```

It writes `reports/realtime_search_evaluation_v0_1.json`, including each expected
and observed intent field plus relevance-only and freshness-aware ranking metrics.

## Load and freshness evidence

After the stack is ready:

```bash
python scripts/realtime/benchmark_ingestion.py \
  --events 500 \
  --concurrency 20 \
  --freshness-samples 25
```

The report records publish throughput, failure rate, p50/p95/p99 acknowledgement
latency, and sampled produced-to-indexed p50/p95/p99. Quote a result only with its
event count, concurrency, machine, Compose topology, and report limitations.

## Failure and recovery evidence

The recovery exercise stops services and must run only against the local Compose
project:

```bash
python scripts/realtime/failure_recovery.py
python scripts/realtime/verify_ordering.py
```

It first stops one consumer and proves the other can index a probe. It then stops
both consumers, publishes a backlog event, restores one consumer, and measures
the time until that event is searchable. A `finally` block starts both consumers
again even when an assertion fails.

The ordering probe publishes two distinct events for one article, with the older
event deliberately arriving second. It waits for both events to be processed and
confirms the newer title remains searchable.

Useful inspection commands:

```bash
docker compose -f deploy/realtime/compose.yaml logs --tail=200 consumer-1 consumer-2
docker compose -f deploy/realtime/compose.yaml exec kafka \
  /opt/kafka/bin/kafka-consumer-groups.sh \
  --bootstrap-server kafka:9092 \
  --group newslens-indexers \
  --describe
```

## DLQ inspection and replay

Inspect dead-letter records before replaying them:

```bash
docker compose -f deploy/realtime/compose.yaml exec kafka \
  /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server kafka:9092 \
  --topic news-events-dlq \
  --from-beginning \
  --max-messages 20
```

The original bytes are base64 encoded because malformed JSON must itself remain
representable in the DLQ envelope. There is intentionally no automatic replay:
an operator must identify the cause, decode and repair the event when appropriate,
assign a new stable event ID, and publish it through `POST /events`.

## Shutdown and data reset

Stop processes while retaining the PostgreSQL volume:

```bash
./scripts/realtime/stop_stack.sh
```

To delete the local real-time database as well, explicitly target this Compose
project and its volumes:

```bash
docker compose -f deploy/realtime/compose.yaml down --volumes
```

That last command is destructive and removes only the named Compose stack's
local PostgreSQL volume. It does not affect the licensed MIND archives or the
separate recommendation artifact.
