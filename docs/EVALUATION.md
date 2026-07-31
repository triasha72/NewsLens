# NewsLens ranking evaluation

## Status

Phase 4 ranking evaluation is in progress.

NewsLens currently includes formally tested ranking metrics, a
model-independent candidate-ranking evaluator, and reproducible MIND-small
evaluations of:

- a training-only popularity baseline; and
- a TF-IDF user-history content baseline.

Both baselines are evaluated on the same strict chronological validation
partition. Evaluation of the rule-based popularity fallback, subgroup
behavior, and uncertainty remains in progress.

## Evaluation protocol

The 156,965 behavior records in `MINDsmall_train` are divided using a strict
chronological 80/20 split.

| Property | Value |
|---|---:|
| Total behavior records | 156,965 |
| Training records | 125,572 |
| Validation records | 31,393 |
| Requested validation fraction | 20% |
| Actual validation fraction | 20% |
| Cutoff timestamp | 2019-11-13 20:36:26 |
| Last training timestamp | 2019-11-13 20:36:19 |
| First validation timestamp | 2019-11-13 20:36:26 |
| Temporal overlap | None |

Records sharing the same timestamp remain in the same partition. The maximum
training timestamp is therefore strictly earlier than the minimum validation
timestamp.

The official MIND-small development split remains reserved for later final
holdout evaluation. It is not used for fitting, selecting, or comparing the
current baselines.

## Leakage controls

### Popularity baseline

Popularity scores use clicks observed only in the earlier training partition.
Validation clicks never modify fitted article scores.

### TF-IDF content baseline

The TF-IDF vocabulary and inverse-document frequencies are fitted using only
articles referenced by chronological training interactions.

Available catalog metadata can be transformed after fitting because article
text does not expose validation click labels. Validation histories and
candidate sets are used only to construct later evaluation examples.

### Shared evaluation

Both models use:

- the same chronological boundary;
- the same 31,393 validation impressions;
- the same impression-level candidate sets;
- the same clicked-item relevance labels;
- the same catalog of 51,282 articles;
- the same metric implementation; and
- the same ranking cutoff of `K = 10`.

## Metric semantics

NewsLens evaluates binary click relevance using:

- NDCG@K;
- Mean Reciprocal Rank@K;
- Recall@K;
- Hit Rate@K; and
- Catalog Coverage@K.

Impressions containing multiple clicked candidates treat every clicked
candidate as relevant. Rankings shorter than `K` remain valid.

The current MIND-small validation partition contains at least one click in
every evaluated impression, so no impression is skipped for lacking a clicked
candidate.

An empty model ranking remains in the evaluation and contributes zero to
NDCG, MRR, Recall, and Hit Rate. This prevents content metrics from being
inflated by silently excluding cold-start or zero-signal cases.

## Reproduction commands

After placing the licensed MIND-small files under `data/`, run the popularity
evaluation:

```bash
python -m newslens evaluate-popularity \
  --data-dir data \
  --k 10 \
  --validation-fraction 0.20 \
  --output reports/popularity_metrics.json
```

Run the content evaluation:

```bash
python -m newslens evaluate-content \
  --data-dir data \
  --k 10 \
  --validation-fraction 0.20 \
  --max-features 50000 \
  --output reports/content_metrics.json
```

Both commands validate and load the local data, build the chronological split,
fit the selected model, evaluate every validation impression, print the
result, and write a deterministic JSON report.

Repeated runs with identical parameters produced byte-identical reports.

## Baseline comparison

| Metric @10 | Popularity | TF-IDF content | Relative change |
|---|---:|---:|---:|
| NDCG | 0.2853 | 0.3594 | +26.0% |
| MRR | 0.2308 | 0.3133 | +35.7% |
| Recall | 0.5047 | 0.5819 | +15.3% |
| Hit Rate | 0.5705 | 0.6610 | +15.9% |
| Catalog Coverage | 0.0402 | 0.0719 | +78.9% |
| Evaluated impressions | 31,393 | 31,393 | — |
| Evaluation fraction | 100% | 100% | — |
| Empty rankings | 0 | 927 | — |
| Unique recommended articles | 2,061 | 3,687 | +78.9% |
| Catalog size | 51,282 | 51,282 | — |
| Candidate occurrences | 1,264,253 | 1,264,253 | — |
| Temporal leakage detected | No | No | — |

Relative change is calculated as `(content - popularity) / popularity` for
the applicable metric. These values describe this specific internal
chronological validation protocol.

## Popularity findings

The popularity model places at least one clicked article in its first ten
positions for 57.05% of validation impressions and retrieves approximately
50.47% of clicked candidates within those positions.

It recommends 2,061 unique articles, or 4.02% of the available catalog. Its
rankings are therefore concentrated among frequently clicked articles.

Approximately 30.36% of validation candidate occurrences were unseen during
training. These articles receive zero popularity scores.

Popularity always produces a ranking, including for unseen candidates, because
zero-score articles are ordered using the deterministic article-identifier
tie-breaker.

## TF-IDF content findings

The TF-IDF history model improves every recorded ranking-quality metric over
popularity:

- MRR@10 increases by 35.7%;
- NDCG@10 increases by 26.0%;
- Hit Rate@10 increases by 15.9%;
- Recall@10 increases by 15.3%; and
- Catalog Coverage@10 increases by 78.9%.

The model recommends 3,687 unique articles and produces a meaningful content
ranking for 30,466 impressions, or 97.05% of the validation set.

### Content evaluation accounting

| Property | Result |
|---|---:|
| Training records | 125,572 |
| Validation records | 31,393 |
| Vocabulary-referenced articles | 47,367 |
| Indexed catalog articles | 51,282 |
| TF-IDF vocabulary size | 50,000 |
| Content-ranked impressions | 30,466 (97.05%) |
| Empty-history impressions | 801 (2.55%) |
| Unknown-history impressions | 0 |
| Zero-profile impressions | 0 |
| Zero-signal impressions | 126 (0.40%) |
| Total abstentions | 927 (2.95%) |

The 927 abstentions are the sum of 801 empty-history cases and 126 cases where
no candidate has positive TF-IDF similarity to the user profile. Every
abstention is retained as an empty ranking in the common evaluator.

This accounting distinguishes a model-quality improvement from a coverage
tradeoff: content ranking is stronger when usable history signal exists, but a
content-only system cannot serve every impression.

## Interpretation

The same-split comparison supports three conclusions:

1. User-history text contains useful ranking signal beyond global click
   frequency under this validation protocol.
2. TF-IDF content ranking exposes substantially more of the article catalog
   than global popularity.
3. A separate fallback remains necessary because the content-only model
   abstains on empty-history and zero-similarity impressions.

The results motivate evaluating the existing training-only popularity fallback
as the next baseline. That experiment should preserve the content ranking when
usable signal exists and route only abstaining impressions to popularity.

## Reproducibility artifacts

Machine-readable reports:

```text
reports/popularity_metrics.json
reports/content_metrics.json
```

Evaluation implementation and tests:

```text
src/newslens/evaluation/metrics.py
src/newslens/evaluation/evaluator.py
src/newslens/evaluation/popularity.py
src/newslens/evaluation/content.py
tests/test_metrics.py
tests/test_evaluator.py
tests/test_popularity_evaluation.py
tests/test_content_evaluation.py
tests/test_cli.py
```

## Remaining work

- evaluate the rule-based popularity fallback;
- compare warm-start and cold-start segments;
- measure unseen-article performance directly;
- analyze results by history length and article category;
- inspect high-confidence failures;
- add uncertainty estimates or confidence intervals;
- publish a complete error analysis; and
- use the official development split only after model selection is complete.

## Limitations and non-claims

- These are offline internal-validation results, not online-product results.
- The official MIND-small development holdout has not been evaluated.
- The comparison covers popularity and TF-IDF content, not the fallback or a
  learned hybrid model.
- TF-IDF measures lexical overlap rather than deeper semantic similarity.
- Every history article currently contributes equal weight to the user profile.
- Empty histories provide no content representation.
- Article publication timestamps are unavailable in the local metadata.
- Popularity reflects historical exposure as well as user preference.
- No confidence intervals or statistical-significance tests are reported.
- No category-level or history-length subgroup analysis is reported.
- No neural recommendation model has been trained.
- No online experiment has been conducted.
- Results should not be compared directly with systems using different data
  splits, candidate policies, catalogs, or metric definitions.

Passing tests and improving offline metrics do not establish production or
online impact.
