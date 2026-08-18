# Phase 04: Vector Retrieval and FAISS

## Purpose

Phase 03 froze the selected recommendation representation:

- v0.2 same-impression hard-negative two-tower;
- 64-dimensional L2-normalized retrieval embeddings;
- popularity fallback for empty-history users.

Phase 04 does not retrain the recommendation model.

Instead, it converts the frozen neural representation into an explicit
candidate-retrieval subsystem and evaluates the systems tradeoffs associated
with exact and approximate nearest-neighbor search.

## Frozen Phase-03 inputs

The following are fixed throughout Phase 04:

- two-tower checkpoint:
  `1fd73a1236c8a84e29ac7ec7e94089e95b1bdd42aa828378dfaf04e816760cd8`;
- article text representation;
- article tower weights;
- user tower weights;
- retrieval embedding dimension: 64;
- maximum user-history length: 20;
- chronological cutoff: `2019-11-13T20:36:26`;
- seed: 42.

Phase 04 does not tune:

- two-tower architecture;
- embedding dimension;
- article feature representation;
- training epochs;
- hard-negative objective;
- learning rate;
- history length.

## Similarity metric

The Phase-03 article and user embeddings are L2-normalized.

Retrieval therefore uses inner product.

For normalized vectors, inner product is equivalent to cosine-similarity
ranking.

## Phase 04 stages

### Phase 04A — exact retrieval oracle

Build a dependency-light NumPy exact inner-product retriever.

This implementation defines the retrieval contract and acts as the correctness
oracle for subsequent FAISS experiments.

### Phase 04B — exact FAISS

Build `IndexFlatIP` over the identical frozen article embedding catalog.

The purpose is to establish parity between FAISS and the NumPy exact oracle
before introducing approximation.

### Phase 04C — approximate retrieval

Build an HNSW inner-product index.

Fixed structural configuration:

- M = 32
- efConstruction = 200

Serving-time search breadth is evaluated for:

- efSearch = 16
- efSearch = 32
- efSearch = 64
- efSearch = 128

The quality metric is retrieval recall against exact search rather than click
labels.

### Phase 04D — production retrieval behavior

Validate:

- history exclusions;
- bounded over-retrieval;
- stable article-ID mapping;
- index serialization;
- index reload parity;
- artifact hashes.

### Phase 04E — systems benchmark

Compare:

- NumPy exact retrieval;
- FAISS IndexFlatIP;
- FAISS HNSW.

Measure:

- p50 query latency;
- p95 query latency;
- p99 query latency;
- mean query latency;
- batch throughput;
- index build time;
- serialization size;
- reload time;
- ANN Recall@10;
- ANN Recall@50;
- ANN Recall@100.

### Phase 04F — freeze

Select the Phase-04 serving configuration.

HNSW is selected only if:

1. Recall@100 against exact search is at least 0.99; and
2. request-level p95 latency is at least 20% lower than IndexFlatIP.

Otherwise IndexFlatIP is selected for the current catalog scale.

This rule intentionally permits exact retrieval to win when approximation is
not operationally justified.

## Information boundary

`MINDsmall_dev` is not used for Phase-04 configuration selection.

Chronological validation histories from `MINDsmall_train` are used only to
construct realistic query vectors.

ANN quality is measured against the exact retrieval oracle.

No global-catalog relevance claims are made from MIND impression labels.

## Non-goals

Phase 04 does not implement:

- learned second-stage ranking;
- diversity reranking;
- A/B testing;
- Kubernetes deployment;
- TorchRec;
- distributed vector indexing.

Those are handled by later roadmap stages.

## Phase 04F: Final retrieval selection

Phase 04 freezes the candidate-retrieval subsystem built on the frozen
Phase-03 hard-negative two-tower representation.

### Frozen representation

- article count: 51,282
- embedding dimension: 64
- similarity: inner product over L2-normalized vectors
- embedding SHA-256:
  `5aae2f910298fc078d7085fd8e4f63e703610def3cfa37ab07c2562d2320f607`
- two-tower checkpoint SHA-256:
  `1fd73a1236c8a84e29ac7ec7e94089e95b1bdd42aa828378dfaf04e816760cd8`

### Exact FAISS validation

IndexFlatIP Recall@100 against the NumPy exact oracle:

- 1.000000

Exact-parity gate:

- True

### HNSW candidate

- M: 32
- efConstruction: 200
- selected efSearch: 128
- HNSW Recall@100: 0.932373

### Request latency

IndexFlatIP:

- p50: 0.284229 ms
- p95: 0.340206 ms
- p99: 0.384837 ms

HNSW:

- p50: 0.146146 ms
- p95: 0.177087 ms
- p99: 0.204493 ms

### Selected serving index

**faiss_flat**

IndexFlatIP retained because HNSW did not simultaneously meet both the quality and material p95-latency requirements.

The selection followed the predeclared rule:

- HNSW Recall@100 must be at least 0.99; and
- HNSW p95 request latency must be at least 20% lower than IndexFlatIP.

Otherwise exact IndexFlatIP is retained at the current catalog scale.

No Phase-03 model parameters were modified during Phase 04.

Phase 04 is frozen after this result.
