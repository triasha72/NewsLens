# Phase 01: Collaborative Filtering Baseline

## Objective

Evaluate whether classical implicit-feedback collaborative filtering adds a
useful personalization signal to NewsLens under a leakage-safe chronological
MIND-small evaluation protocol.

This phase implements Bayesian Personalized Ranking (BPR) matrix
factorization as a classical collaborative-filtering baseline. The purpose is
not to replace the existing content recommender unless the evidence supports
that conclusion.

## Information boundary

All model development and selection use only the internal chronological split
of `MINDsmall_train`.

- Training impressions: 125,572
- Validation impressions: 31,393
- Cutoff: `2019-11-13T20:36:26`
- Training boundary: `event_timestamp < cutoff`
- Validation boundary: `event_timestamp >= cutoff`

The official MIND-small development split is not used for Phase 01 model
selection or tuning.

## Implementation

The collaborative model uses implicit-feedback BPR matrix factorization with:

- learned user embeddings;
- learned article embeddings;
- pairwise positive-versus-negative ranking loss;
- deterministic seeds;
- same-impression negative sampling;
- chronological training-data extraction from DuckDB.

The selected configuration uses:

- embedding dimension: 64
- epochs: 10
- batch size: 2,048
- learning rate: 0.01
- weight decay: 1e-6
- negatives per positive: 3
- random seed: 42
- evaluation cutoff: 10

## Negative-sampling ablation

Three negative-sampling configurations were evaluated.

| Negatives per positive | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 |
|---:|---:|---:|---:|---:|
| 1 | 0.1140 | 0.1067 | 0.1804 | 0.2352 |
| 3 | **0.1289** | **0.1194** | **0.2046** | **0.2621** |
| 5 | 0.1242 | 0.1144 | 0.2012 | 0.2588 |

Three negatives per positive were selected because they produced the best
ranking-quality metrics among the evaluated configurations.

The five-negative configuration improved representation coverage further but
reduced ranking quality, demonstrating a coverage-versus-ranking-quality
tradeoff.

## Same-protocol model comparison

All systems were evaluated on the same 31,393 chronological validation
impressions.

| Model | NDCG@10 | MRR@10 | Recall@10 | Hit Rate@10 | Coverage@10 | Empty rankings |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.2853 | 0.2308 | 0.5047 | 0.5705 | 0.0402 | 0 |
| TF-IDF content | 0.3594 | 0.3133 | 0.5819 | 0.6610 | 0.0719 | 927 |
| Content + popularity fallback | **0.3664** | **0.3179** | **0.5955** | **0.6762** | **0.0719** | **0** |
| BPR, 3 negatives | 0.1289 | 0.1194 | 0.2046 | 0.2621 | 0.0256 | 8,002 |

Standalone BPR therefore does not replace the existing content/fallback
recommender.

## Paired bootstrap comparison

Paired nonparametric bootstrap comparisons use:

- 1,000 replicates;
- aligned validation impressions;
- 95% percentile confidence intervals;
- random seed 42;
- difference direction: BPR minus baseline.

Against Content + popularity fallback, the selected BPR model produced:

- NDCG@10 difference: -0.23747
- 95% CI: [-0.24216, -0.23302]

The interval excludes zero. The same conclusion holds for MRR@10,
Recall@10, and Hit Rate@10, and for comparisons against TF-IDF and
popularity.

## Coverage diagnosis

The selected BPR model learned:

- 45,848 users;
- 9,886 articles;
- 541,860 BPR training triples.

Validation support diagnostics showed:

- cold-start user fraction: 16.58%;
- unknown candidate occurrence fraction: 56.56%;
- unknown clicked-item occurrence fraction: 61.58%;
- impressions with no BPR-supported candidates: 3,472;
- empty BPR rankings: 8,002.

This establishes severe temporal representation limitations for the
user-ID/item-ID collaborative model.

## Support-aware subgroup analysis

BPR improves substantially when evaluation is restricted to examples where
the clicked items are already represented by the model. However, that
condition uses future click labels and is therefore an oracle diagnostic, not
a deployable routing rule.

This analysis also initially gave BPR a smaller effective candidate universe,
because unsupported candidates were not scored.

## Matched-candidate control

To remove that candidate-universe confound, both BPR and Content + popularity
fallback were evaluated on exactly the same BPR-supported candidate sets.

For 7,721 impressions where the user and clicked items were represented:

- BPR NDCG@10: 0.4260
- Content + fallback NDCG@10: 0.4732
- BPR minus fallback: -0.0472
- 95% CI: [-0.0554, -0.0392]

For the stricter 5,069-impression subset with at least top-k supported
candidates:

- BPR NDCG@10: 0.2840
- Content + fallback NDCG@10: 0.3467
- BPR minus fallback: -0.0627
- 95% CI: [-0.0737, -0.0532]

Every matched-candidate comparison also favored Content + fallback on MRR,
Recall, and Hit Rate.

## Conclusion

The Phase 01 evidence indicates two independent limitations of standalone
BPR in this setting:

1. temporal user/item representation coverage is limited; and
2. even after candidate support is controlled, the current BPR ranking signal
   is weaker than the content/fallback system.

BPR is therefore retained as a classical collaborative-filtering baseline,
not promoted to the primary recommender.

## Phase 02 implication

Phase 02 should test a content-dominant hybrid rather than route whole
impressions to BPR.

For each candidate:

- content similarity remains the primary signal;
- popularity can remain available for fallback/context;
- a BPR score is added only when both user and article embeddings exist;
- BPR support is represented explicitly rather than treating unsupported
  candidates as meaningful zero scores.

The Phase 02 research question is whether collaborative scores provide
incremental value when fused with content, not whether standalone BPR is
stronger.

## Limitations and non-claims

- The official MIND-small development split remains untouched.
- Phase 01 does not claim BPR improves the production recommender.
- The support-aware clicked-item subgroups are oracle diagnostics and cannot
  be used as deployment gates.
- The hyperparameter study is intentionally bounded rather than exhaustive.
- No claim is made that all matrix-factorization formulations would behave
  identically.
- Collaborative serving integration is outside the scope of Phase 01.
