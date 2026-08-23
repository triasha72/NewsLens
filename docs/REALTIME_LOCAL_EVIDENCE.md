# Real-time search local evidence

This record separates behavior observed in a live stack from behavior covered
only by unit tests or design analysis. It was produced on August 23, 2026 after
building the repository's own images and starting the complete Compose topology.

## Environment

| Property | Value |
|---|---|
| Runtime | Docker Desktop 29.7.2, Linux containers on Apple silicon |
| Docker allocation | 10 CPUs, 8.32 GB memory |
| Kafka | Apache Kafka 3.9.1, one broker, three `news-events` partitions |
| Consumers | Two Go 1.23 services in `newslens-indexers` |
| Search store | PostgreSQL 17 |
| Search API | Python 3.12 and FastAPI |
| Metrics | Prometheus 3.5.0 |

The commands and report generators are documented in
[`REALTIME_OPERATIONS.md`](REALTIME_OPERATIONS.md). Raw machine-readable results
are stored in `reports/`.

## End-to-end load and freshness

The workload published 500 events with concurrency 20, sampled 25 articles for
produced-to-indexed freshness, and replayed 25 stable event IDs.

| Observation | Result |
|---|---:|
| Accepted publish requests | 500 / 500 |
| Publish failure rate | 0% |
| Publish throughput | 1,225 events/s |
| Publish latency p50 / p95 / p99 | 15.15 / 20.02 / 44.12 ms |
| Produced-to-indexed p50 / p95 / p99 | 62.51 / 78.76 / 82.07 ms |
| Duplicate probes recognized | 25 / 25 |

The first run exposed a one-second Kafka producer batch delay. A second run
showed that synchronous per-event offset commits moved a 500-event burst's p95
freshness above 18 seconds. The final configuration uses a 10 ms producer batch
timeout and one-second periodic consumer commits; the table reports only the
post-fix rerun.

This is one single-machine development workload, not a production capacity or
service-level claim. The freshness sample records each article's own
`produced_at` to PostgreSQL `indexed_at` duration; search polling confirms the
article is visible but is not added to that stored duration.

Evidence: [`realtime_load_v0_1.json`](../reports/realtime_load_v0_1.json).

## Consumer failure and backlog recovery

The failure probe first read the active group assignment, chose an article key
owned by consumer 1, and stopped that consumer. Kafka reassigned its partition
and the event became searchable in 5.67 seconds. The probe then stopped both
consumers, published an event into the backlog, and measured 2.43 seconds from
starting one consumer until the event was searchable.

This proves local process reassignment and retained-backlog recovery. It does not
exercise broker, PostgreSQL, host, or multi-zone failure.

Evidence: [`realtime_recovery_v0_1.json`](../reports/realtime_recovery_v0_1.json).

## Dead-letter behavior

A malformed non-JSON record was inserted directly into `news-events`. A consumer
counted one invalid record, published one dead-letter record, and preserved the
original bytes in a valid base64 envelope with the source position and reason.
Database retry exhaustion is covered by the Go worker test; it was not induced in
the live PostgreSQL service.

Evidence: [`realtime_dlq_v0_1.json`](../reports/realtime_dlq_v0_1.json).

## Late-event ordering

Two distinct events updated one article, with the older `produced_at` event sent
second. Both entered the event ledger, while the searchable title remained the
newer version. This verifies the PostgreSQL conditional upsert in the live stack;
its correctness still depends on trustworthy producer timestamps.

Evidence: [`realtime_ordering_v0_1.json`](../reports/realtime_ordering_v0_1.json).

## Query and ranking contract

A deterministic fixture checks 18 category, entity, and freshness fields across
six labeled queries. All 18 match. On the same ten-article fixture,
freshness-aware ranking raises NDCG@10 from 0.869 to 1.000 and
freshness-weighted NDCG@10 from 0.634 to 0.878 relative to the relevance-only
configuration.

This small hand-labeled fixture shows that the implemented rules behave as
specified. It is not an independently judged corpus and cannot support a broad
search-quality claim.

Evidence:
[`realtime_search_evaluation_v0_1.json`](../reports/realtime_search_evaluation_v0_1.json).

## What this evidence does not establish

- Kafka broker replication or loss behavior;
- PostgreSQL high availability or point-in-time recovery;
- multi-host, multi-zone, or managed-cloud operation;
- sustained capacity beyond this short workload;
- a production alerting and DLQ replay process; or
- relevance quality on live or independently judged search traffic.
