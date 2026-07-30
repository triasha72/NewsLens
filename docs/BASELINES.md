# Phase 3: Evaluation-safe baselines

Phase 3 introduces deterministic search and recommendation baselines while
preserving strict separation between training and validation interactions.

## Evaluation protocol

The `MINDsmall_train` behavior records are divided into internal training and
validation partitions using a strict chronological boundary.

| Property | Value |
|---|---:|
| Total behavior records | 156,965 |
| Training records | 125,572 |
| Validation records | 31,393 |
| Requested validation fraction | 20% |
| Actual validation fraction | 20% |
| Cutoff | 2019-11-13 20:36:26 |
| Last training timestamp | 2019-11-13 20:36:19 |
| First validation timestamp | 2019-11-13 20:36:26 |
| Temporal overlap | None |

Records sharing an identical timestamp remain in the same partition. Therefore,
the maximum training timestamp is strictly earlier than the minimum validation
timestamp.

The official MIND-small development split is reserved as a later final holdout.
It is not used for model fitting or baseline selection during Phase 3.

## Implemented baselines

### Training-only popularity

Articles are ranked by click counts observed in the chronological training
partition. Validation clicks never contribute to the fitted popularity scores.
Ties are resolved deterministically using the article identifier.

This provides a simple and interpretable reference model, but it is affected by
historical exposure bias.

### TF-IDF article search

Article title, abstract, category, and subcategory are combined into a text
document. Unigrams and bigrams are represented with TF-IDF, and queries are
ranked using cosine similarity.

The search baseline is deterministic and does not use interaction labels.

### Content-based history recommender

A user profile is created by averaging the TF-IDF vectors of articles in that
user's reading history. Candidate articles are ranked by cosine similarity to
the resulting profile.

The TF-IDF vocabulary and inverse-document-frequency statistics can be fitted
only on articles referenced by the chronological training partition. Available
candidate metadata can then be transformed without using validation clicks.

### Popularity cold-start fallback

The content recommender falls back to training-only popularity when:

- the user history is empty;
- no history articles are available in the content index; or
- the history produces no usable content-similarity signal.

The fallback is rule-based. It does not blend content and popularity scores.
Weighted hybrid ranking remains future work.

## Preliminary popularity diagnostic

The popularity baseline was fitted only on the chronological training partition
and used to rank candidate articles from the later validation partition.

| Diagnostic | Result |
|---|---:|
| Validation impressions evaluated | 31,393 |
| Top-1 clicked-article rate | 11.48% |
| Top-10 clicked-article rate | 57.05% |
| Average best clicked-article rank | 15.72 |
| Unseen candidate occurrence rate | 30.36% |
| Temporal leakage detected | No |

These values are preliminary diagnostics. Phase 4 will introduce tested and
formally defined NDCG@K, MRR, Recall@K, and catalog-coverage metrics before
publishing a definitive model comparison.

## Reproducibility

The version-controlled baseline parameters are stored in:

```text
configs/baselines.json