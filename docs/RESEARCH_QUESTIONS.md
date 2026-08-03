# Research questions

This document collects questions raised by the current NewsLens results. It is a working research record rather than a feature wish list.

## Why chronological evaluation matters

A random interaction split can let later behavior influence a model that is evaluated on earlier recommendations. NewsLens therefore uses a strict chronological boundary and keeps identical timestamps within one partition.

Questions still open:

- How sensitive are conclusions to the exact cutoff?
- Do rolling temporal windows tell a different story from one fixed split?
- How quickly does news preference drift over hours or days?
- Which features would genuinely have been available at recommendation time?

## Why cold start remains interesting

Popularity fallback removes empty rankings, but cold-start impressions remain the weakest history segment.

Current observation:

| Segment | Impressions | NDCG@10 | Recall@10 | Hit Rate@10 |
|---|---:|---:|---:|---:|
| Cold start | 801 | 0.2954 | 0.5339 | 0.6017 |
| Short history | 3,379 | 0.3471 | 0.5911 | 0.6324 |
| Medium history | 4,994 | 0.3767 | 0.6148 | 0.6676 |
| Long history | 22,219 | 0.3695 | 0.5940 | 0.6875 |

Questions still open:

- Would session context or coarse category preference help without requiring a long user history?
- Does recency-weighted popularity outperform global training-period popularity?
- How much of cold-start performance is constrained by candidate construction?
- Can the system distinguish “no history” from “history unavailable”?

## A counterintuitive exposure result

Articles with zero candidate appearances in the chronological training partition perform better than low-, medium-, and high-exposure bands under the current analysis. “Unseen” here means unseen in training candidates, not missing article text.

Questions still open:

- Are unseen articles easier because their text is more distinctive?
- Does publication recency confound the exposure bands?
- Are high-exposure articles associated with broader, harder candidate sets?
- Would matching on category, time, and candidate-set size change the pattern?

## Lexical similarity versus semantics

TF-IDF is transparent and efficient, but it mainly captures shared terms.

Questions still open:

- Which top-ranked failures are caused by lexical overlap without semantic relevance?
- Which misses are semantically related but use different vocabulary?
- Does BM25 close part of the gap before dense embeddings are needed?
- Is the gain from a semantic encoder worth its memory, latency, and artifact-size cost?

## History representation

The current content profile gives history articles equal weight.

Questions still open:

- Should recent articles receive more weight?
- Should repeated categories or entities be damped to preserve diversity?
- Does a user profile benefit from separating short-term session intent from longer-term preference?
- How should unknown or deleted history articles affect confidence?

## Fallback routing

The current router uses content when the history produces a positive TF-IDF signal and otherwise uses training-only popularity.

Questions still open:

- Is top-score margin a useful abstention signal?
- How should route confidence be interpreted when score scales differ?
- Can a learned router improve quality without leaking validation outcomes?
- Which failure types are introduced by fallback even while aggregate metrics improve?

## Robustness of the reported improvement

The current paired bootstrap intervals favor content plus fallback over content alone on the fixed validation impressions.

Questions still open:

- Does the difference persist on the untouched development split?
- How stable are results across ranking cutoffs?
- Do subgroup intervals reveal regressions hidden by the overall estimate?
- How sensitive are intervals to the resampling unit and dependence between impressions?

## Serving behavior

NewsLens can reproduce rankings from a versioned artifact and serve them through a containerized API.

Questions still open:

- What are p50, p95, and p99 latency across candidate-set sizes?
- What is the memory cost of the vectorizer, index, and metadata?
- How should readiness behave during artifact replacement?
- What signals should trigger an alert rather than only a log entry?
- How does concurrent load affect deterministic ranking and tail latency?

## What would justify a more complex model?

A more complex model is useful only if it answers a limitation that simpler baselines expose.

Before adding one, record:

1. the failure mode it is intended to address;
2. the unchanged evaluation protocol;
3. the expected quality or operational benefit;
4. the additional training, serving, and maintenance cost;
5. the subgroup risks; and
6. the result that would cause the idea to be rejected.
