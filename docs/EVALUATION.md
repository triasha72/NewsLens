# NewsLens ranking evaluation

## Status

Phase 4 ranking evaluation is in progress.

NewsLens currently includes tested ranking metrics, a model-independent
candidate-ranking evaluator, and a reproducible evaluation of the
training-only popularity baseline on MIND-small.

## Evaluation protocol

The 156,965 behavior records in `MINDsmall_train` are divided using a strict
chronological 80/20 split.

| Property | Value |
|---|---:|
| Training records | 125,572 |
| Validation records | 31,393 |
| Requested validation fraction | 20% |
| Actual validation fraction | 20% |
| Cutoff timestamp | 2019-11-13 20:36:26 |
| Temporal overlap | None |

The popularity model is fitted exclusively on the earlier training partition.
Validation clicks are never used to calculate article popularity.

The official MIND-small development split remains reserved for later final
holdout evaluation.

## Reproduction command

After placing the licensed MIND-small files under `data/`, run:

```bash
python -m newslens evaluate-popularity \
  --data-dir data \
  --k 10 \
  --validation-fraction 0.20 \
  --output reports/popularity_metrics.json