## Phase 06F: Final diversity and exposure result

Phase 06 evaluated deterministic MMR reranking over the frozen
Phase-03 and Phase-04 recommendation stack.

### Selected offline policy

**MMR lambda = 0.80**

The policy was selected mechanically using the preregistered
relevance-retention, diversity, and exposure constraints.

### Logged-candidate benchmark

| Metric | Baseline | MMR 0.80 |
|---|---:|---:|
| NDCG@10 | 0.382602 | 0.382500 |
| Recall@10 | 0.632203 | 0.632120 |
| Mean ILD | 0.154710 | 0.155073 |
| Exposure Gini | 0.995782 | 0.995777 |

The selected policy remained inside the frozen relevance
non-inferiority budget.

### Global FAISS top-100 benchmark

The global benchmark makes no relevance-quality claim because
arbitrary FAISS candidates do not have valid logged click labels.

| Metric | Relevance top-10 | MMR 0.80 |
|---|---:|---:|
| Mean ILD | 0.071482 | 0.072748 |
| Mean unique categories | 2.263672 | 2.298828 |
| Mean unique subcategories | 3.601562 | 3.654297 |
| Exposure Gini | 0.969502 | 0.969573 |
| Catalog coverage | 0.051578 | 0.051480 |

MMR changed
70.51%
of top-10 rankings while retaining mean top-10 set overlap of
97.79%.

Global semantic diversity improved.

Global aggregate exposure concentration did not improve.

### Systems result

Post-user-embedding p95:

- relevance-only path: 0.357 ms
- current MMR path: 194.110 ms

The current deterministic Python MMR implementation is not yet
considered production-serving ready.

Phase 06 did not preregister a latency promotion threshold, so this
is recorded as a systems limitation rather than a post-hoc policy
rejection.

### Final interpretation

Phase 06 freezes MMR lambda=0.80 as the selected offline diversity
policy.

The logged-candidate experiment preserved relevance inside the
predeclared non-inferiority budget.

The global FAISS benchmark showed a larger semantic-diversity effect,
but did not improve aggregate exposure concentration or catalog
coverage.

Phase 07 will optimize the same frozen MMR policy under exact ranking
parity and establish serving performance, observability,
containerization, Kubernetes deployment, and online experiment design.
