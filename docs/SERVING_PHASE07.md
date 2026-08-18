# Phase 07: Production Serving and Performance

## Purpose

Phase 07 converts the frozen NewsLens recommendation stack into a
production-oriented serving system without changing the ranking decisions
selected in earlier phases.

## Frozen recommendation policy

Phase 07 starts from:

- frozen Phase-03 hard-negative two-tower;
- Phase-04 FAISS IndexFlatIP retrieval;
- Phase-06 MMR lambda = 0.80;
- retrieval depth = 100;
- final recommendation depth = 10;
- popularity fallback where previously defined.

The rejected Phase-05 learned ranker remains excluded.

## Phase 07A: MMR implementation optimization

Phase-06E identified the correctness-first Python MMR implementation as the
dominant post-embedding latency bottleneck.

Frozen Phase-06E reference measurements:

- FAISS top-100 p95: approximately 0.464 ms;
- reference MMR top-100 to top-10 p95: approximately 195.871 ms;
- MMR post-embedding path p95: approximately 194.110 ms.

Phase 07A may optimize implementation only.

It must not modify:

- lambda;
- relevance score semantics;
- candidate retrieval depth;
- final recommendation depth;
- article embeddings;
- tie-breaking behavior;
- Phase-06 policy selection.

## Reference implementation

`maximal_marginal_relevance` remains the behavioral oracle.

The optimized candidate implementation is evaluated against the reference
rather than replacing it before parity is established.

## Exact ranking-parity requirement

The optimized implementation must return the same ordered recommendation IDs
as the reference implementation.

Parity must hold for:

- deterministic unit examples;
- exact-score ties;
- near-score ties;
- randomized normalized embeddings;
- multiple candidate-set sizes;
- multiple lambda values;
- the 512 frozen Phase-06E FAISS top-100 query pools.

Any ranking mismatch fails the optimization gate.

## Performance methodology

The Phase-07 benchmark compares the reference and optimized implementations
on identical candidate pools.

Both implementations receive:

- identical candidate IDs;
- identical relevance scores;
- identical normalized article vectors;
- identical lambda;
- identical top-k.

Warm-up occurs before measurement.

Latency is measured independently per request and summarized with:

- mean;
- p50;
- p95;
- p99.

## Phase 07A promotion gate

The optimized implementation is eligible for serving only if:

1. exact ordered top-10 parity is 100%;
2. there are zero parity mismatches across the frozen 512-query benchmark;
3. p95 MMR latency improves by at least 20x relative to the Phase-06
   correctness-first reference;
4. optimized top-100 to top-10 MMR p95 is no greater than 10 ms;
5. all existing NewsLens tests pass.

If the implementation misses the performance target, preserve the result and
continue systems optimization without changing the frozen ranking policy.

## Later Phase-07 stages

After MMR optimization:

- serving API and request contracts;
- latency/error SLOs;
- Docker packaging;
- metrics and observability;
- load and concurrency testing;
- Kubernetes deployment;
- readiness/liveness probes;
- resource requests and limits;
- autoscaling policy;
- controlled online A/B experiment design.

Phase 07 does not claim that offline MIND metrics are online business impact.
