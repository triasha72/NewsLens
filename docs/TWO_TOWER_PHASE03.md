# Phase 03: Native PyTorch Two-Tower Recommender

## Motivation

Phase 01 showed that standalone BPR matrix factorization performs substantially
worse than the content-first recommender.

Phase 02 showed that BPR can provide a small complementary ranking signal in
well-supported regions, but the serving-time support gate changed from 41.83%
of internal-validation impressions to only 6.25% of the later MIND-small dev
benchmark.

This indicates that ID-only user and article embeddings are brittle under
temporal user/item support shift.

## Hypothesis

A content-aware two-tower model can learn personalized similarity while
avoiding the strongest coverage limitation of ID-only matrix factorization.

The article tower will construct embeddings from article content features
rather than article IDs.

The user tower will construct a user representation from encoded history
articles rather than from a persistent user-ID embedding.

This should allow:

- previously unseen users with usable histories to receive recommendations;
- temporally new articles to receive learned representations from content;
- candidate embeddings to be precomputed for later ANN retrieval;
- the model to support a future FAISS retrieval phase.

## Initial architecture

### Article tower

Dense article content features

→ linear projection

→ GELU

→ linear projection

→ L2-normalized embedding

### User tower

History article content features

→ shared article tower

→ masked mean pooling

→ learned projection

→ L2-normalized user embedding

### Training objective

The first training implementation will use positive clicked articles with
in-batch negatives and cross-entropy over the user-to-article similarity
matrix.

Additional explicit same-impression negatives may be evaluated later as a
bounded ablation.

## Information boundary

Phase 03 development and hyperparameter selection will use only chronological
splits derived from `MINDsmall_train`.

`MINDsmall_dev` has already been evaluated during Phase 02 and is therefore
not considered an untouched holdout for Phase 03.

It must not be used for:

- architecture selection;
- hyperparameter tuning;
- early-stopping decisions;
- ablation selection;
- feature-selection decisions.

If used after the Phase 03 architecture is frozen, it will be described only
as a previously exposed external benchmark.

## Initial experimental plan

1. Implement native PyTorch article and user towers.
2. Build leakage-safe dense article text features.
3. Build chronological positive training examples.
4. Train with in-batch negatives.
5. Evaluate on the existing chronological validation protocol.
6. Compare against Content + popularity fallback.
7. Add temporal/support diagnostics.
8. Run bounded architecture ablations only if justified.
9. Freeze the selected architecture.
10. Prepare representations for the later ANN/FAISS phase.

## Non-goals

Phase 03 does not yet implement:

- FAISS or approximate-nearest-neighbor serving;
- TorchRec;
- distributed training;
- learned second-stage ranking;
- production A/B testing.

Those remain separate roadmap stages.

## Phase 03B: Train-only article feature coverage

The first article representation uses:

- TF-IDF vocabulary and IDF fitted only from chronologically permitted
  training articles;
- 50,000 maximum TF-IDF features;
- 256-component randomized TruncatedSVD;
- deterministic seed 42;
- L2-normalized dense output vectors.

Articles outside the fitting set are transformed using the frozen
train-fitted TF-IDF/SVD basis.

### Chronological coverage result

The feature basis was fitted using 47,367 articles and then used to transform
the complete 51,282-article MIND-small train catalog.

Results:

- catalog articles with nonzero dense features: 51,282 / 51,282 = 100%;
- validation candidate-occurrence support: 100%;
- validation clicked-item support: 100%;
- validation impressions with at least k=10 supported candidates:
  31,393 / 31,393 = 100%;
- nonempty histories with at least one usable history article:
  30,592 / 30,592 = 100%.

The most important temporal generalization check concerns articles that were
not permitted to influence TF-IDF or SVD fitting.

Validation contained:

- 3,404 unique non-fit candidate articles;
- 375,506 non-fit candidate occurrences;
- 17,949 clicked non-fit article occurrences.

Feature support was 100% for both non-fit candidate occurrences and non-fit
clicked occurrences.

### Interpretation

This removes the principal article-representation coverage limitation found
with ID-only BPR.

The result establishes representation availability, not ranking quality.

A 256-dimensional train-fitted text representation being available for an
article does not imply that the learned two-tower ranking will outperform the
existing Content + popularity fallback baseline.

Phase 03C therefore proceeds to a fixed-objective neural training pipeline
using chronological positive interactions and in-batch negatives.

The SVD explained-variance ratio is recorded for reproducibility but is not
used as a model-selection criterion.

## Phase 03C: Neural training pipeline

The first native PyTorch training objective uses:

- train-only TF-IDF + 256-dimensional SVD article features;
- user representations derived only from impression-time history;
- maximum history length of 20 most recent usable articles;
- shared article tower;
- masked mean history pooling;
- 128-dimensional hidden layer;
- 64-dimensional retrieval embedding;
- temperature 0.07;
- dropout 0.10;
- unique in-batch clicked articles as contrastive classes;
- AdamW optimization;
- deterministic seed 42.

Duplicate positive article IDs inside a batch are deduplicated so that two
users clicking the same article do not incorrectly treat duplicate copies of
the same article as negatives.

### Training-data accounting

On the chronological training partition:

- input impressions: 125,572
- positive click occurrences: 187,856
- usable history-to-click examples: 184,282
- skipped positive occurrences: 3,574
- positive articles missing content features: 0
- histories without usable content features: 0
- eligible impressions: 123,135
- histories truncated to the most recent 20 articles: 59,343
- unique positive articles: 6,294

Approximately 98.1% of positive click occurrences can therefore participate
in the neural training objective.

The remaining examples are primarily unavoidable empty-history cases rather
than representation failures.

### Smoke training

A deterministic CPU smoke test used 4,096 training examples for two epochs.

| Epoch | Average loss | In-batch top-1 |
|---:|---:|---:|
| 1 | 4.012369 | 14.0381% |
| 2 | 3.883381 | 13.8184% |

All 4,096 examples were processed in every epoch and no batch was skipped for
having fewer than two unique positive articles.

The decrease in optimization loss establishes that the training pipeline is
functioning. In-batch top-1 accuracy is retained only as an optimization
diagnostic and is not used for architecture selection.

The next step is one full fixed-configuration training run followed by
chronological ranking evaluation. No hyperparameters are changed based on the
smoke run.

## Phase 03C: Full fixed-configuration training

After the successful smoke test, the predefined configuration was trained on
all usable examples from the chronological training partition.

No architecture or optimization hyperparameter was changed based on the
smoke result.

### Training configuration

- article input dimension: 256
- hidden dimension: 128
- retrieval embedding dimension: 64
- maximum history length: 20
- dropout: 0.10
- temperature: 0.07
- epochs: 3
- batch size: 128
- learning rate: 0.001
- weight decay: 0.00001
- gradient clipping norm: 5.0
- device: CPU
- seed: 42

### Training-data accounting

- chronological training impressions: 125,572
- chronological validation impressions: 31,393
- positive click occurrences: 187,856
- usable history-to-click examples: 184,282
- eligible impressions: 123,135
- empty-history impressions: 2,437
- positive articles missing features: 0
- histories without usable features: 0
- skipped positive occurrences: 3,574
- histories truncated to the most recent 20 articles: 59,343
- unique positive articles: 6,294

### Optimization result

| Epoch | Average loss | In-batch top-1 | Examples seen | Skipped |
|---:|---:|---:|---:|---:|
| 1 | 4.383181 | 5.2235% | 184,282 | 0 |
| 2 | 4.330023 | 6.0527% | 184,282 | 0 |
| 3 | 4.312826 | 6.2817% | 184,282 | 0 |

The optimization loss decreases monotonically while the in-batch top-1
diagnostic improves across the full training run.

Training loss is not used as the recommendation-selection criterion. The
three-epoch checkpoint is taken forward unchanged to chronological ranking
evaluation.

Checkpoint SHA-256:

`11fd97c2b8a826de0c8d29c38b2a736935516c7d493ac3e87e460d384db3363c`

`MINDsmall_dev` was not used.

### Evaluation parity rule

Training uses at most the 20 most recent feature-supported history articles.
Chronological inference must apply the same history rule.

This parity requirement is part of the already-defined training protocol and
is not a post-training hyperparameter change.

## Phase 03D: First chronological ranking evaluation

The frozen three-epoch two-tower checkpoint was evaluated on the same 31,393
chronological validation impressions used by previous internal experiments.

The evaluation protocol was committed before the ranking result was observed.

### Results

| Model | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 | Empty |
|---|---:|---:|---:|---:|---:|
| Content + popularity fallback | 0.366361 | 0.317947 | 0.595474 | 0.676234 | 0 |
| Raw two-tower | 0.357428 | 0.305928 | 0.589788 | 0.668907 | 801 |
| Two-tower + popularity fallback | 0.364966 | 0.311884 | 0.603411 | 0.684261 | 0 |

Paired bootstrap, two-tower + popularity minus Content + fallback:

- NDCG@10: -0.001395
- 95% CI: [-0.004475, +0.001718]

- MRR@10: -0.006063
- 95% CI: [-0.009697, -0.002579]

- Recall@10: +0.007937
- 95% CI: [+0.003696, +0.011967]

- Hit Rate@10: +0.008027
- 95% CI: [+0.003280, +0.012391]

### Interpretation

The first two-tower model improves top-10 retrieval breadth but not top-rank
ordering.

Recall and Hit Rate improve with paired intervals entirely above zero, while
MRR decreases with its interval entirely below zero. NDCG is slightly lower
and statistically inconclusive.

This pattern indicates that the learned representation contains useful
retrieval signal but the in-batch-negative training objective is insufficiently
aligned with candidate-level ordering.

Raw two-tower rankings are available for 30,592 / 31,393 validation
impressions (97.45%). The remaining 801 impressions have empty histories.
There are no feature-support abstentions among nonempty histories.

The representation-coverage objective of Phase 03 is therefore satisfied.

### Next bounded experiment

Only one training-objective ablation will be attempted next:

**same-impression hard negatives**

The article representation, history construction, network architecture, and
all existing optimization hyperparameters remain fixed.

This ablation tests whether negatives actually shown alongside clicked
articles improve top-rank ordering without sacrificing the two-tower's
observed Recall/Hit advantage.

## Phase 03E: Same-impression hard-negative ablation

### Motivation

Phase 03D shows a clear retrieval-versus-ordering tradeoff.

Compared with Content + popularity fallback, the first two-tower model with
popularity fallback produces:

- NDCG@10 difference: -0.001395,
  95% CI [-0.004475, +0.001718];
- MRR@10 difference: -0.006063,
  95% CI [-0.009697, -0.002579];
- Recall@10 difference: +0.007937,
  95% CI [+0.003696, +0.011967];
- Hit Rate@10 difference: +0.008027,
  95% CI [+0.003280, +0.012391].

The model therefore retrieves clicked articles into the top 10 more often,
but ranks those clicked articles less effectively near the top of the list.

### Hypothesis

The original in-batch objective mostly contrasts a user's clicked article
against clicked articles from unrelated impressions.

This is useful for representation learning and retrieval, but does not
directly teach the model to distinguish the clicked article from articles
that were actually shown alongside it.

One bounded objective ablation will therefore add the unclicked candidates
from the same impression as hard negatives.

### Frozen components

The following remain unchanged:

- TF-IDF maximum features: 50,000
- SVD components: 256
- maximum history length: 20
- hidden dimension: 128
- embedding dimension: 64
- dropout: 0.10
- temperature: 0.07
- epochs: 3
- batch size: 128
- learning rate: 0.001
- weight decay: 0.00001
- gradient clipping norm: 5.0
- seed: 42
- chronological train/validation split

### Hard-negative policy

For each trainable click:

1. preserve the original in-batch positive-article candidate set;
2. include every feature-supported candidate labeled unclicked in that same
   impression as an additional row-specific negative;
3. do not sample or tune a hard-negative count;
4. do not assign additional weighting to the hard-negative loss.

An unclicked candidate is treated only as an implicit training negative. It
is not interpreted as a verified user dislike.

### Decision rule

The primary question is whether candidate-aware negatives repair the
top-ordering weakness observed in v0.1.

The ablation is considered useful if v0.2 improves MRR/NDCG relative to v0.1
without materially eliminating the Recall/Hit advantage over the
Content + popularity fallback baseline.

No additional objective or neural hyperparameter sweep will follow this
ablation.

### Phase 03E implementation validation

The same-impression hard-negative implementation was validated before the
full ablation run.

Two correctness safeguards were added.

First, distinct clicked articles belonging to the same impression are masked
from one another in the shared in-batch positive pool. They therefore cannot
be accidentally treated as negatives for the same user context.

A deterministic unit-level check produced:

- row 1 valid-positive mask: `[True, False, True]`
- row 2 valid-positive mask: `[False, True, True]`
- unrelated row mask: `[True, True, True]`

This confirms that known co-clicked articles are excluded as false negatives
while unrelated in-batch clicked articles remain valid contrastive negatives.

Second, training reports serialize aggregate accounting only. The complete
184,282-example interaction collection is not included in report JSON.

### Corrected Phase 03E smoke run

The corrected hard-negative objective was trained on a deterministic
4,096-example prefix for two epochs.

| Epoch | Loss | In-batch top-1 | Examples | Hard negatives | Skipped batches |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.449878 | 13.9893% | 4,096 | 189,285 | 0 |
| 2 | 4.341978 | 14.3555% | 4,096 | 189,285 | 0 |

The loss remains finite and decreases while the diagnostic top-1 accuracy
increases.

The full chronological training construction contains:

- 184,282 usable examples;
- 184,282 examples with at least one same-impression hard negative;
- 0 examples without hard negatives;
- 8,420,452 attached hard-negative occurrences;
- 16,661 unique hard-negative articles;
- 6,294 unique positive articles;
- 0 positive articles missing content features;
- 0 nonempty histories missing content features.

The smoke run therefore clears the Phase 03E implementation gate.

No architecture or optimizer hyperparameter is changed before the full
hard-negative ablation.

## Phase 03E: Full hard-negative evaluation

The frozen hard-negative objective was trained for the predefined three
epochs on all 184,282 usable chronological training examples.

Each epoch processed 8,420,452 attached same-impression hard-negative
occurrences and skipped no batches.

### Optimization result

| Epoch | Loss | Diagnostic top-1 |
|---:|---:|---:|
| 1 | 4.732982 | 3.5592% |
| 2 | 4.688528 | 4.0145% |
| 3 | 4.673004 | 4.2457% |

Checkpoint SHA-256:

`1fd73a1236c8a84e29ac7ec7e94089e95b1bdd42aa828378dfaf04e816760cd8`

### Chronological validation result

| Model | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|
| Content + popularity fallback | 0.366361 | 0.317947 | 0.595474 | 0.676234 |
| Raw hard-negative two-tower | 0.365065 | 0.313482 | 0.600528 | 0.681235 |
| Hard-negative two-tower + popularity fallback | 0.372602 | 0.319438 | 0.614151 | 0.696588 |

Paired bootstrap, hard-negative two-tower + popularity fallback minus
Content + popularity fallback:

- NDCG@10: +0.006242,
  95% CI [+0.003161, +0.009405];
- MRR@10: +0.001491,
  95% CI [-0.002282, +0.005179];
- Recall@10: +0.018677,
  95% CI [+0.014384, +0.023024];
- Hit Rate@10: +0.020355,
  95% CI [+0.015544, +0.024943].

The paired intervals for NDCG, Recall, and Hit Rate are entirely above zero.
The MRR point estimate is positive but its interval includes zero.

### Interpretation

The same-impression hard-negative objective repairs the retrieval-versus-
ordering tradeoff observed in the first two-tower experiment.

Compared with the original v0.1 two-tower + popularity model, v0.2 improves
the aggregate point estimates by approximately:

- NDCG@10: +0.007636
- MRR@10: +0.007554
- Recall@10: +0.010740
- Hit Rate@10: +0.012327

A direct paired v0.2-versus-v0.1 bootstrap comparison is performed next before
the Phase 03 model is frozen.

No additional two-tower architecture or hyperparameter tuning will be
performed.

## Phase 03F: Direct v0.2 versus v0.1 comparison

The final Phase 03 experiment directly compares the original two-tower
checkpoint against the same-impression hard-negative checkpoint using the
same 31,393 chronological validation impressions.

Configuration parity was verified before comparison:

- network configuration: identical;
- article-feature configuration: identical;
- maximum history length: identical.

The two versions therefore differ in the intended training-objective
intervention rather than network capacity or feature representation.

### Final paired result

| Model | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|
| v0.1 two-tower + popularity | 0.364966 | 0.311884 | 0.603411 | 0.684261 |
| v0.2 hard-negative two-tower + popularity | 0.372602 | 0.319438 | 0.614151 | 0.696588 |

Paired bootstrap, v0.2 minus v0.1:

- NDCG@10: +0.007636,
  95% CI [+0.006211, +0.009164];
- MRR@10: +0.007554,
  95% CI [+0.005480, +0.009474];
- Recall@10: +0.010740,
  95% CI [+0.008091, +0.013348];
- Hit Rate@10: +0.012328,
  95% CI [+0.009301, +0.015450].

All four paired intervals are entirely above zero.

The two versions produce different top-k rankings on 28,026 of the 31,393
validation impressions.

### Diversity / coverage observation

The hard-negative model improves ranking metrics while slightly reducing
catalog exposure:

- v0.1 catalog coverage@10: 0.076206;
- v0.2 catalog coverage@10: 0.074900;
- v0.1 unique recommended articles: 3,908;
- v0.2 unique recommended articles: 3,841.

This tradeoff is recorded rather than optimized during Phase 03. Exposure and
diversity behavior are handled in a later dedicated phase.

### Phase 03 model selection

**Selected model: v0.2 same-impression hard-negative two-tower with
popularity fallback for empty-history users.**

The selection is based on:

1. complete train-fitted content representation coverage on chronological
   validation candidates;
2. statistically supported improvements over v0.1 on NDCG, MRR, Recall, and
   Hit Rate;
3. statistically supported improvements over Content + popularity fallback
   on NDCG, Recall, and Hit Rate;
4. deterministic training/evaluation protocol and artifact SHA verification;
5. no architecture or hyperparameter changes after the bounded objective
   ablation.

No additional Phase 03 tuning will be performed.

Phase 04 moves the frozen v0.2 retrieval representation into an explicit
vector-index candidate-retrieval system.
