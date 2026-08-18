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
