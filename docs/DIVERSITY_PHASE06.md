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
