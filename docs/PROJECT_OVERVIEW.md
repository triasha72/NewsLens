# NewsLens project overview

## The problem

Recommendation scores can be overstated by temporal leakage, while a good
offline model can still fail on cold starts, duplicate events, stale articles,
or a stopped consumer.

## What I built

I built a leakage-safe MIND recommendation workflow with chronological splits,
uncertainty analysis, and deterministic fallback routing. Separately, I built
a Go, Kafka, PostgreSQL, and FastAPI article path with idempotency, dead-letter
handling, observability, and repeatable recovery tests.

## What the evidence says

The content-plus-fallback recommender reached NDCG@10 0.3664 on the frozen
chronological protocol. In a 500-event local real-time run, publish p99 was
44.12 ms and sampled produced-to-indexed p95 was 78.76 ms; duplicate events and
a stopped-consumer recovery path were also exercised. These are offline and
single-machine observations, not user-lift or cloud-scale claims.

## Reproduce the software checks

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

The MIND dataset requires separate access under its license. Reproduction,
dataset layout, and system workflows are linked from the [README](../README.md).

## Next validation

The remaining question is user impact: an online experiment with a fixed
horizon, guardrails, and sample-ratio checks would be needed before claiming
that offline ranking gains improve user outcomes.
