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
