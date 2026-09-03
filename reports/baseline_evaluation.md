# Classical Baseline Evaluation

Character TF-IDF (3-5 grams) and one logistic-regression classifier per target. The vectorizer was fitted only on train text. Hyperparameters and class weighting were selected only with validation macro-F1; test predictions were produced after selection.

## Test results

| Target | Macro-F1 | Weighted-F1 | Accuracy | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| category | 0.2721 | 0.3740 | 0.3837 | 0.1549 | 0.7972 |
| sentiment | 0.6373 | 0.6794 | 0.6977 | 0.1106 | 0.4576 |
| priority | 0.5177 | 0.5701 | 0.6047 | 0.0561 | 0.4983 |

## Category

Selected configuration: `C=0.5`, `class_weight=balanced`. Validation macro-F1: **0.2875**.

Validation-selected low-confidence review threshold: **0.30** (coverage 0.304, selective accuracy 0.327).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Abuse/Harassment | 0.0000 | 0.0000 | 0.0000 | 3 |
| Account/Login | 0.2500 | 0.2222 | 0.2353 | 9 |
| Delivery/Order | 0.2414 | 0.8400 | 0.3750 | 25 |
| Info/Query | 1.0000 | 0.0667 | 0.1250 | 15 |
| Other | 0.7179 | 0.5490 | 0.6222 | 51 |
| Payment | 0.1364 | 0.3750 | 0.2000 | 8 |
| Refund/Return | 0.5000 | 0.2500 | 0.3333 | 4 |
| Technical/App Bug | 0.7692 | 0.1754 | 0.2857 | 57 |

## Sentiment

Selected configuration: `C=2.0`, `class_weight=balanced`. Validation macro-F1: **0.6370**.

Validation-selected low-confidence review threshold: **0.52** (coverage 0.707, selective accuracy 0.812).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.7011 | 0.8356 | 0.7625 | 73 |
| Neutral | 0.6712 | 0.7424 | 0.7050 | 66 |
| Positive | 0.8333 | 0.3030 | 0.4444 | 33 |

## Priority

Selected configuration: `C=2.0`, `class_weight=balanced`. Validation macro-F1: **0.5278**.

Validation-selected low-confidence review threshold: **0.72** (coverage 0.265, selective accuracy 0.812).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| High | 0.7143 | 0.1389 | 0.2326 | 36 |
| Low | 0.7143 | 0.6349 | 0.6723 | 63 |
| Medium | 0.5413 | 0.8082 | 0.6484 | 73 |

Weighted Cohen's kappa: **0.4544**; mean absolute ordinal error: **0.4186**; High→Low severe-error rate: **0.0833**.

## Limitations

- The grouped split intentionally favors leakage isolation over ideal class balance.
- `Abuse/Harassment` has very little training support, so its estimate is highly uncertain.
- Probabilities are native logistic-regression probabilities and are not post-hoc calibrated.
- This report is a classical baseline, not evidence that transformer training will improve results.
