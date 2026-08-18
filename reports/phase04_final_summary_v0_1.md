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
