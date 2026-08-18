# Phase 02: Support-Gated Hybrid Selection

## Objective

Determine whether the collaborative BPR signal from Phase 01 provides
incremental ranking value when used as an auxiliary residual on top of the
content-first NewsLens recommender.

All architecture and hyperparameter selection in this phase uses only the
internal chronological split of `MINDsmall_train`.

The official `MINDsmall_dev` validation set is reserved as a final holdout.

## Starting point

Phase 01 established that standalone BPR is substantially weaker than the
existing Content + popularity fallback recommender.

Phase 02 therefore treats BPR as an optional candidate-level residual rather
than a standalone routing destination.

## Hybrid formulation

Content scores are normalized per impression and remain the primary ranking
signal.

For BPR-supported user-item pairs:

`hybrid_score = normalized_content_score + alpha * (normalized_bpr_score - 0.5)`

Unsupported candidates receive no collaborative penalty:

`hybrid_score = normalized_content_score`

Content cold-start cases continue to use the existing popularity fallback.

## Coarse weight sweep

The following collaborative weights were evaluated:

- 0.00
- 0.05
- 0.10
- 0.20
- 0.30

All runs used:

- the same chronological validation set;
- k = 10;
- BPR embedding dimension = 64;
- BPR epochs = 10;
- BPR batch size = 2,048;
- BPR learning rate = 0.01;
- BPR weight decay = 1e-6;
- three negatives per positive;
- random seed = 42.

Results:

| Alpha | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.366361 | 0.317947 | 0.595474 | 0.676234 |
| 0.05 | 0.366508 | 0.318047 | 0.595854 | 0.676552 |
| 0.10 | 0.366554 | 0.318114 | 0.595918 | 0.676680 |
| 0.20 | **0.366885** | 0.318445 | **0.596224** | **0.677030** |
| 0.30 | 0.366760 | **0.318534** | 0.595471 | 0.676648 |

Alpha 0.20 produced the highest internal-validation NDCG@10.

Its aggregate paired-bootstrap NDCG interval versus Content + fallback still
crossed zero, so the ungated hybrid was not promoted directly.

## Support diagnostics

Deployable diagnostics used only serving-time information:

- whether the user has a BPR embedding;
- candidate BPR support count;
- candidate BPR support fraction;
- history length.

No clicked-item labels were used to define deployment gates.

The strongest broad support segment was:

`at least k=10 BPR-supported candidates`

This segment contained 13,133 impressions and showed:

- hybrid NDCG@10 = 0.224850;
- fallback NDCG@10 = 0.223030;
- difference = +0.001820;
- 95% paired-bootstrap CI = [+0.000636, +0.002950].

This result motivated one frozen serving-time gate.

## Frozen support-gated policy

The selected Phase 02 architecture is:

- collaborative weight: **0.20**
- minimum BPR-supported candidates: **10**
- ranking cutoff: **10**

Policy:

1. If the user is represented by BPR and at least 10 current candidate
   articles are represented by BPR, use the content + BPR residual ranking.
2. Otherwise, preserve Content + popularity fallback behavior.

The gate depends only on information available at recommendation time.

## Full internal-validation result

The frozen gated hybrid was evaluated on all 31,393 internal chronological
validation impressions.

| Model | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|
| Content + fallback | 0.366361 | 0.317947 | 0.595474 | 0.676234 |
| Ungated hybrid | 0.366885 | 0.318445 | 0.596224 | 0.677030 |
| Support-gated hybrid | **0.367122** | **0.318536** | **0.596702** | **0.677540** |

Support gate:

- open impressions: 13,133
- closed impressions: 18,260
- gate-open fraction: 41.83%
- gated rankings changed relative to fallback: 12,000

Paired bootstrap, gated hybrid minus Content + fallback:

- NDCG@10: +0.000762
- 95% CI: [+0.000338, +0.001289]

- MRR@10: +0.000589
- 95% CI: [+0.000112, +0.001139]

- Recall@10: +0.001228
- 95% CI: [+0.000088, +0.002462]

- Hit Rate@10: +0.001306
- 95% CI includes zero at the boundary.

The internal-validation decision therefore promotes the gated architecture as
a candidate for final holdout evaluation.

## Frozen final-holdout configuration

No further tuning will be performed before or after opening the official
MIND-small dev holdout.

### BPR

- embedding dimension: 64
- epochs: 10
- batch size: 2,048
- learning rate: 0.01
- weight decay: 1e-6
- negatives per positive: 3
- seed: 42

### Hybrid

- collaborative weight: 0.20
- minimum supported candidates: 10
- k: 10

### Statistical evaluation

- paired bootstrap replicates: 1,000
- confidence level: 95%
- seed: 42

## Holdout rule

For the final evaluation:

1. Retrain the frozen component models using all permitted MIND-small
   training data.
2. Evaluate Content + fallback and the frozen gated hybrid on
   `MINDsmall_dev`.
3. Run the predefined paired-bootstrap comparison.
4. Record the result exactly once.
5. Do not change alpha, support threshold, BPR hyperparameters, or architecture
   based on the holdout result.

If the holdout does not confirm the internal-validation improvement, that
negative generalization result will be preserved rather than tuned away.

## Final MIND-small dev holdout

The frozen Phase 02 architecture was evaluated exactly once on the official
`MINDsmall_dev` holdout after both the architecture and evaluator had been
committed.

### Reproducibility

- architecture freeze commit: `4d0b493`
- frozen holdout evaluator commit: `da331c6`
- MINDsmall_dev ZIP SHA-256:
  `d6ce515dcaa6b6d47ddf0a326eebc8a31b84735ae410285c9882ca2a06eec669`
- training behaviors: 156,965
- holdout behaviors: 73,152
- training articles: 51,282
- holdout articles: 42,416
- dev-only articles: 13,956
- BPR training triples after full-train refit: 683,919

No model architecture, support threshold, fusion weight, or BPR
hyperparameter was changed after opening the holdout.

### Holdout metrics

| Model | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|
| Content + fallback | 0.379386 | 0.323244 | 0.624221 | 0.700432 |
| Frozen gated hybrid | 0.379426 | 0.323254 | 0.624350 | 0.700582 |

Paired bootstrap, frozen gated hybrid minus Content + fallback:

- NDCG@10: +0.000040
- 95% CI: [-0.000097, +0.000161]

- MRR@10: +0.000010
- 95% CI: [-0.000122, +0.000122]

- Recall@10: +0.000129
- 95% CI: [-0.000195, +0.000431]

- Hit Rate@10: +0.000150
- 95% CI: [-0.000205, +0.000506]

All four point estimates remain in the positive direction, but every 95%
paired-bootstrap interval contains zero.

The final holdout therefore reproduces the direction of the internal result
but does not provide evidence for a stable aggregate improvement over
Content + popularity fallback.

### Support shift

The selected support gate opened on:

- 13,133 / 31,393 internal-validation impressions: 41.83%
- 4,569 / 73,152 final-holdout impressions: 6.25%

This substantial change in gate availability shows that a collaborative
policy based on user/item ID support can be sensitive to temporal changes in
the recommendation population.

The reduced opportunity to apply the collaborative residual is an important
limitation of the Phase 02 architecture.

### Phase 02 conclusion

Phase 02 does not promote the support-gated BPR hybrid as a proven replacement
for Content + popularity fallback.

The experiments establish instead that:

1. standalone BPR is substantially weaker than the content-first baseline;
2. collaborative scores can contain complementary ranking information in
   sufficiently supported regions;
3. simple linear fusion can expose a small positive collaborative signal;
4. a serving-time support gate improves the internal-validation result;
5. the positive direction persists on the untouched holdout, but the effect
   is very small and its paired confidence intervals include zero; and
6. BPR user/item coverage changes substantially under temporal distribution
   shift.

No post-holdout tuning will be performed.

The next phase should replace ID-only matrix factorization with a learned
two-tower architecture capable of combining user-history and article
representations and providing better support for previously unseen or
temporally new articles.
