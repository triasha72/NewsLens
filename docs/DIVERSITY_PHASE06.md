# Phase 06: Diversity, Exposure, and Item-Side Representation

## Purpose

Phase 06 studies whether deterministic post-retrieval reranking can improve
recommendation diversity and item-side exposure balance while preserving the
relevance quality of the selected NewsLens recommendation system.

## Selected input system

Phase 05 rejected the learned second-stage ranker.

Therefore Phase 06 starts from the selected production candidate:

- frozen Phase-03 hard-negative two-tower;
- popularity fallback for histories without usable representation;
- frozen Phase-04 FAISS IndexFlatIP retrieval backend.

The rejected Phase-05 learned ranker is not used.

## Fairness scope

MIND does not provide protected demographic attributes sufficient for
demographic user-fairness evaluation.

Phase 06 therefore makes no claims about demographic fairness.

The fairness-related analysis is restricted to item-side exposure and
representation, including:

- category representation;
- subcategory representation;
- popularity-group exposure;
- catalog concentration;
- long-tail exposure;
- opportunity-normalized item-group exposure.

## Information boundary

MINDsmall_train remains the development dataset.

The Phase-05 chronological validation interval has already been observed and
is therefore treated as a development benchmark rather than a pristine
holdout.

MINDsmall_dev has previously been exposed in an earlier project phase and is
not treated as a pristine holdout.

No Phase-06 claim will describe either partition as unseen or pristine.

## Offline relevance boundary

MIND relevance labels exist only for articles shown in logged impressions.

Therefore relevance comparisons are performed only over logged impression
candidate sets.

Global FAISS retrieval experiments measure serving behavior and diversity but
do not assign relevance labels to unlogged catalog articles.

## Baseline

The relevance baseline is the frozen Phase-03 two-tower candidate ordering
with popularity fallback.

## Candidate reranker

Phase 06 evaluates deterministic Maximal Marginal Relevance (MMR).

For a candidate article i:

MMR(i) =
lambda * relevance(i)
-
(1 - lambda) * max_similarity_to_selected(i)

where:

- relevance is the frozen two-tower score;
- similarity is cosine similarity between frozen normalized article
  embeddings.

Because Phase-04 article embeddings are L2-normalized, cosine similarity and
inner product are equivalent.

## Predeclared MMR values

The only evaluated lambda values are:

- 1.00
- 0.95
- 0.90
- 0.85
- 0.80

lambda = 1.00 is an implementation-parity control.

No additional lambda values are added after results are observed.

## Relevance metrics

At k = 10:

- NDCG@10
- MRR@10
- Recall@10
- Hit Rate@10
- catalog coverage@10

Paired relevance comparisons use:

- 1,000 bootstrap samples;
- 95% confidence intervals;
- seed 42;
- impression-level aligned resampling.

## Diversity metrics

At k = 10:

- intra-list diversity;
- unique categories per recommendation list;
- unique subcategories per recommendation list;
- category entropy;
- subcategory entropy.

## Aggregate exposure metrics

Across recommendation lists:

- unique exposed articles;
- catalog coverage;
- exposure Gini coefficient;
- top-1-percent item exposure share;
- top-10-percent item exposure share;
- category exposure entropy;
- subcategory exposure entropy.

## Popularity-group exposure

Item popularity is computed only from the original Phase-03 training
partition.

Articles are assigned to exposure groups using their training-period
candidate-exposure counts:

- head: top 20%;
- mid: next 30%;
- tail: bottom 50%.

The audit reports:

- recommendation exposure share by group;
- candidate-opportunity share by group;
- exposure-minus-opportunity gap;
- long-tail recommendation share.

These metrics represent item-side exposure balance and do not imply
demographic fairness.

## Selection rule

A non-baseline MMR policy is eligible only when all of the following hold:

1. NDCG@10 is at least 98% of baseline NDCG@10;
2. Recall@10 is at least 98% of baseline Recall@10;
3. its paired NDCG@10 confidence interval does not establish a material
   relevance regression larger than the 2% retention budget;
4. mean intra-list diversity is higher than baseline;
5. exposure Gini is lower than baseline.

Among eligible policies, select the one with the highest mean intra-list
diversity.

Ties are broken by:

1. higher NDCG@10;
2. lower exposure Gini;
3. larger lambda.

If no non-baseline policy is eligible, retain the original two-tower
ordering.

## Interpretation

Phase 06 is a constrained relevance-diversity experiment.

Diversity improvements are not considered successful when they require
relevance degradation outside the preregistered quality budget.

## Non-goals

Phase 06 does not:

- retrain the two-tower;
- change Phase-04 retrieval;
- revive the rejected Phase-05 ranker;
- claim demographic fairness;
- use protected user attributes;
- optimize directly against MINDsmall_dev;
- assign relevance labels to arbitrary FAISS-retrieved catalog articles.

Phase 07 handles production serving, Kubernetes, observability, and A/B-test
design.

## Phase 06C score-scale operationalization

The relevance term used inside MMR is the frozen Phase-03 inference score
returned by `TwoTowerNetwork.score_candidates`.

For normalized user and article embeddings this is:

`inner_product / frozen_temperature`

The frozen Phase-03 temperature is 0.07.

Raw inner product and temperature-scaled score produce identical pure
two-tower ordering, which is why the Phase-06B baseline remains unchanged.
They are not equivalent when mixed with the MMR diversity penalty.

Therefore all non-baseline Phase-06 MMR policies use the frozen
temperature-scaled score.

No per-impression min-max normalization, standardization, calibration, or
other post-hoc score rescaling is performed.

### Relevance-budget confidence-interval rule

The 2% NDCG retention budget is operationalized as a non-inferiority margin.

For baseline NDCG `B`, the allowed absolute loss is:

`0.02 * B`

A candidate satisfies the confidence-interval guardrail only when the lower
bound of its paired candidate-minus-baseline NDCG interval is greater than or
equal to:

`-0.02 * B`

This requires the complete 95% interval to remain inside the preregistered
2% relevance-loss budget.

This interpretation is frozen before observing the Phase-06C MMR sweep.

## Phase 06E: Global retrieval and diversity-serving benchmark

Phase 06E evaluates the already selected Phase-06D MMR policy in the
global retrieval path.

The selected policy remains:

- MMR lambda = 0.80.

Phase 06E does not reopen Phase-06 policy selection.

### Serving path

The benchmark evaluates:

1. frozen Phase-03 user embedding;
2. frozen Phase-04 FAISS IndexFlatIP retrieval;
3. global top-100 retrieval with history exclusion;
4. Phase-06 MMR lambda=0.80 reranking;
5. final top-10 recommendations.

The comparison baseline uses the top 10 articles from the same exact FAISS
top-100 candidate pool in pure relevance-score order.

### Fixed benchmark configuration

- query population: Phase-06 chronological development benchmark;
- deterministic nonempty-history sample;
- query count: 512;
- random seed: 42;
- FAISS threads: 1;
- retrieval depth: 100;
- final recommendation depth: 10;
- MMR lambda: 0.80;
- MMR relevance score:
  frozen two-tower inner product divided by temperature;
- frozen temperature: 0.07;
- warm-up queries: 50.

### Metrics

Post-user-embedding system latency:

- FAISS top-100 retrieval p50 / p95 / p99;
- MMR top-100 to top-10 reranking p50 / p95 / p99;
- relevance-only post-embedding request p50 / p95 / p99;
- MMR post-embedding request p50 / p95 / p99.

Recommendation-composition metrics:

- mean intra-list diversity;
- mean unique categories;
- mean unique subcategories;
- category entropy;
- subcategory entropy;
- unique exposed articles;
- catalog coverage;
- exposure Gini;
- top-1-percent exposure share;
- top-10-percent exposure share;
- fraction of top-10 rankings changed;
- mean top-10 set overlap.

### Quality boundary

MIND click labels are unavailable for arbitrary global FAISS-retrieved
catalog articles.

Therefore Phase 06E makes no relevance-quality claim for global retrieval
candidates.

The benchmark is restricted to systems behavior, recommendation composition,
diversity, and exposure.

Phase-06D policy selection is not changed based on Phase-06E results.

Full request serving, production optimization, Kubernetes, observability,
and online A/B-test design remain Phase-07 work.

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
