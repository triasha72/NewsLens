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

## Phase 07A-2: Frozen real-workload parity benchmark

The optimization benchmark reuses the Phase-06E workload definition.

The workload is reconstructed deterministically using:

- the same Phase-06 chronological benchmark partition;
- query count = 512;
- seed = 42;
- exact FAISS IndexFlatIP;
- FAISS threads = 1;
- retrieval depth = 100;
- final depth = 10;
- MMR lambda = 0.80;
- frozen two-tower temperature = 0.07;
- complete user-history exclusion.

Candidate pools are generated once and shared by both MMR implementations.

The reference and optimized implementations therefore receive exactly the
same:

- candidate IDs;
- relevance scores;
- candidate embeddings;
- lambda;
- top-k.

The reference implementation remains the behavioral oracle.

### Timing methodology

The first 50 candidate pools are used for warm-up.

For measured queries, reference and optimized execution order alternates by
query index to reduce systematic order/cache bias.

Each of the 512 frozen candidate pools contributes one measured latency
observation per implementation.

The benchmark reports:

- reference mean / p50 / p95 / p99;
- optimized mean / p50 / p95 / p99;
- speedup relative to the measured reference;
- speedup relative to the frozen Phase-06E reference p95;
- ordered top-10 parity count;
- parity mismatch count.

### Promotion gate

The gate remains the previously frozen Phase-07A rule:

- 512 / 512 ordered top-10 parity;
- zero parity mismatches;
- at least 20x p95 speedup relative to the frozen Phase-06 reference;
- optimized p95 <= 10 ms.

The measured reference timing in Phase 07A-2 is diagnostic.

The preregistered performance gate is evaluated against the frozen Phase-06E
reference p95, so normal benchmark noise cannot move the target after results
are observed.

No ranking policy or MMR hyperparameter is changed based on this benchmark.

## Phase 07A result

The optimized vectorized MMR implementation passed the frozen promotion gate.

Frozen workload:

- 512 FAISS top-100 candidate pools;
- 51,200 candidate occurrences;
- 15,700 unique candidate articles;
- zero short candidate pools;
- zero duplicate candidates;
- zero history-exclusion violations.

Exact behavioral parity:

- ordered top-10 parity: 512 / 512;
- parity mismatches: 0.

Optimized MMR latency:

- mean: approximately 4.018 ms;
- p50: approximately 4.025 ms;
- p95: approximately 4.096 ms;
- p99: approximately 4.226 ms.

The frozen Phase-06 reference p95 was approximately 195.871 ms, so the
predeclared gate-relative speedup is approximately 47.82x.

The same-run Phase-07 reference p95 was approximately 6.122 ms, giving a
same-run MMR-core speedup of approximately 1.49x.

These two quantities are intentionally reported separately. The 47.82x
number compares against the frozen Phase-06 benchmark and is used only for
the preregistered promotion gate. The 1.49x value is the direct same-run
algorithm comparison after candidate-pool materialization is held constant.

The optimized implementation is promoted as the Phase-07 serving MMR
implementation because exact recommendation parity and all frozen performance
gates passed.
