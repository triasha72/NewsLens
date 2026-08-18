## Phase 05F: Final learned-ranking decision

Phase 05 evaluated a learned second-stage ranker over the frozen
Phase-03 two-tower representation.

### Information boundary

- ranker training impressions:
  25,114
- ranker validation impressions:
  6,279
- Phase-05 cutoff:
  `2019-11-14T14:57:43`
- MINDsmall_dev used:
  False

### Training

The candidate ranker used the frozen 11-feature contract and
HistGradientBoosting configuration.

Training rows:

- 220,833

Clicked rows:

- 38,240

Selected hard nonclick rows:

- 182,593

### Ranking quality

| Model | NDCG@10 | MRR@10 | Recall@10 | Hit@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Phase-03 two-tower + popularity | 0.382602 | 0.324512 | 0.632203 | 0.704889 | 0.031512 |
| Phase-05 learned ranker | 0.275035 | 0.227320 | 0.478553 | 0.535117 | 0.023088 |

### Paired candidate-minus-baseline differences

- NDCG@10:
  -0.107567
  [-0.117060,
   -0.098008]
- MRR@10:
  -0.097193
  [-0.107988,
   -0.086430]
- Recall@10:
  -0.153650
  [-0.167281,
   -0.139853]
- Hit Rate@10:
  -0.169772
  [-0.183632,
   -0.154798]

### Selection

**phase03_two_tower_popularity**

The learned second-stage ranker is rejected.

All four paired ranking metrics deteriorated, and the preregistered
selection guardrails failed.

The candidate changed
96.58%
of validation rankings while also reducing catalog coverage.

### Serving decision

The rejected Phase-05 ranker is not promoted into the serving path.

The selected system remains:

- frozen Phase-03 hard-negative two-tower;
- popularity fallback for unusable histories;
- Phase-04 `faiss_flat` retrieval backend.

No Phase-05 hyperparameters were changed after observing validation.

No MINDsmall_dev result was used.

Phase 05 is frozen as a negative learned-reranking result.
