# Phase 05: Learned Second-Stage Ranking

## Purpose

Phase 04 froze FAISS IndexFlatIP as the candidate-retrieval backend.

Phase 05 adds a learned second-stage scorer that reranks candidate articles
using the frozen Phase-03 representation plus leakage-safe behavioral and
metadata features.

## Serving architecture

User history
-> frozen Phase-03 user tower
-> Phase-04 FAISS IndexFlatIP retrieval
-> top-N candidate set
-> Phase-05 learned ranker
-> final top-k ranking

## Offline evaluation boundary

MIND supplies click labels only for articles that appeared in each logged
impression.

Therefore Phase 05 evaluates ranking quality on logged impression candidate
sets.

Global FAISS-retrieved articles that were not shown in a MIND impression are
not treated as negatives.

No global-catalog relevance claim is made.

## Leakage boundary

The Phase-03 two-tower was trained using the original chronological training
partition.

To avoid fitting the second-stage model on in-sample two-tower scores, Phase 05
uses the already held-out Phase-03 validation window as its development pool.

That development pool is split chronologically again:

- first 80%: Phase-05 ranker training;
- final 20%: Phase-05 ranker validation.

The frozen Phase-03 two-tower therefore has not trained on either Phase-05
partition.

Popularity features are fitted only from the original Phase-03 training
partition, which precedes the entire Phase-05 development window.

MINDsmall_dev is not used.

## Fixed feature set

1. two_tower_score
2. mean_history_similarity
3. max_history_similarity
4. log1p_popularity_clicks
5. log1p_popularity_exposures
6. popularity_ctr
7. category_match_any
8. subcategory_match_any
9. category_history_fraction
10. subcategory_history_fraction
11. usable_history_length

No original impression position is used as a feature.

## Training rows

All clicked candidates from eligible ranker-training impressions are retained.

For each impression, the highest-scoring nonclicked candidates under the
frozen two-tower are retained as hard negatives.

Maximum hard negatives:

- 5 per clicked candidate.

These are logged nonclicks and are not interpreted as verified dislikes.

## Frozen model

Model:

`sklearn.ensemble.HistGradientBoostingClassifier`

Configuration:

- learning_rate = 0.05
- max_iter = 200
- max_leaf_nodes = 31
- min_samples_leaf = 50
- l2_regularization = 0.001
- early_stopping = False
- random_state = 42

No hyperparameter sweep is performed.

## Primary Phase-05 comparison

Baseline:

- frozen Phase-03 hard-negative two-tower ranking;
- popularity fallback for users without usable history.

Candidate:

- learned second-stage ranker;
- same popularity fallback.

Metrics:

- NDCG@10
- MRR@10
- Recall@10
- Hit Rate@10
- catalog coverage@10

Paired bootstrap:

- 1,000 samples
- 95% confidence
- seed 42
- impression-level paired resampling

## Selection rule

Select the learned ranker only if:

1. the paired NDCG@10 interval is entirely above zero; and
2. MRR@10, Recall@10, and Hit Rate@10 are not significantly worse
   (their candidate-minus-baseline intervals are not entirely below zero).

Otherwise retain the frozen two-tower ordering.

The result is preserved even if the learned ranker is not selected.

## Non-goals

Phase 05 does not:

- retrain the two-tower;
- alter the Phase-04 FAISS index;
- tune HNSW;
- perform diversity reranking;
- use MINDsmall_dev;
- claim relevance for unlogged global-catalog candidates.

Phase 06 handles diversity, exposure, and fairness.
