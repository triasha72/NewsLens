## High-score ranking failure inspection

NewsLens performs route-aware inspection of top-k ranking failures. Because
TF-IDF similarity scores and popularity click counts have different meanings
and scales, high-score thresholds are calculated independently for each
recommendation source.

The analysis uses the 90th percentile of positive top scores within each
source. A retained failure is an evaluated impression where no clicked article
appears in the top 10 and the top score meets or exceeds its source-specific
threshold.

| Source | Eligible impressions | Top-10 misses | High-score misses | Retained examples | 90th-percentile threshold |
|---|---:|---:|---:|---:|---:|
| TF-IDF content | 30,466 | 9,844 | 1,178 | 25 | 0.1918 |
| Training-only popularity | 696 | 317 | 44 | 25 | 1,007 |

Across all 31,393 evaluated validation impressions, 10,164 were top-10 misses.
Of the 31,162 impressions with a positive top score, 1,222 were high-score
misses. The report retains 50 deterministic examples, with at most 25 from
each recommendation source.

Each retained example records:

- the impression and recommendation source;
- clicked and recommended article IDs;
- article titles, categories, and subcategories;
- ranked scores and the source-specific threshold;
- user-history and candidate-set sizes;
- the absolute and relative top-two score margin; and
- a deterministic margin classification.

Margin classes describe ranking separation rather than calibrated confidence:

- `near_tie`: relative margin no greater than 5%;
- `intermediate_margin`: relative margin above 5% and below 25%;
- `high_margin`: relative margin at least 25%; and
- `single_result`: no second ranked score is available.

A high score does not imply a calibrated probability of relevance. These
examples are diagnostic cases for qualitative inspection, not evidence that
the numerical scores have comparable probabilistic meaning across routes.

The original ranking metrics, bootstrap uncertainty, history-segment results,
article-category results, and training-exposure results remain unchanged after
adding article-context enrichment.