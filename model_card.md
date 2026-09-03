# Model Card: BrandPulse-BD v1

## Model summary

The selected application model is `char-tfidf-logreg-v1`: a train-only-fitted character TF-IDF representation (3-5 grams) with separate logistic-regression classifiers for category, sentiment, and priority. It is text-first and does not use company or platform metadata as predictive features.

The model was selected over the completed compact character Transformer because it achieved higher fixed-test metrics for all three targets. XLM-R and BanglaBERT weight downloads were blocked, so no result is claimed for them.

## Intended use

The model supports exploratory analysis and human-assisted triage of Bangla, Banglish, English, and code-mixed public customer feedback. It is not suitable for unattended routing, punitive action, safety decisions, or claims about individual customers. Every predicted `High` priority and every low-confidence output must be reviewable by a person.

## Training and evaluation data

- Train: 691 real supplied labeled records
- Validation: 181 real supplied labeled records
- Test: 172 real supplied labeled records
- Split seed: 42
- Leakage groups: connected components over normalized-identical text and exact source URL
- Cross-split source URL, normalized text, and connected-group overlap: 0
- Synthetic examples in test: 0

The test set was not used for vectorizer fitting, hyperparameter selection, class-weight selection, or confidence-threshold selection.

## Test metrics

| Target | Macro-F1 | Weighted-F1 | Accuracy |
|---|---:|---:|---:|
| Category | 0.2721 | 0.3740 | 0.3837 |
| Sentiment | 0.6373 | 0.6794 | 0.6977 |
| Priority | 0.5177 | 0.5701 | 0.6047 |

Priority weighted Cohen's kappa is 0.4544, and the High-to-Low severe-error rate is 0.0833. Category `Abuse/Harassment` recall is 0.0 on three test examples.

## Confidence and human review

The model returns native logistic-regression probabilities, not invented confidence values. Validation-selected low-confidence thresholds are:

- Category: 0.30
- Sentiment: 0.52
- Priority: 0.72

The category threshold could not achieve high selective accuracy and should not be interpreted as a safety guarantee. Operational review must also trigger for predicted `High`, unsupported/empty/very short input, and explicitly configured inconsistent head combinations.

## Performance characteristics

On the measured local CPU path, single-row inference has approximately 0.615 ms median and 0.826 ms P95 latency across 100 runs. Batch throughput on 172 records is approximately 7,950 rows/s. The serialized model is approximately 1.032 MB. These measurements are hardware- and process-specific, not deployment guarantees.

## Limitations and risks

- Category performance is too weak for unattended automation.
- Training support is extremely small for `Abuse/Harassment`, `Refund/Return`, and some grouped subpopulations.
- Long, multi-intent, sarcastic, implicit, or contrastive comments are frequent error sources.
- Label ambiguity is visible in the review sample; supplied labels are not assumed infallible.
- Latin-script classification is a character heuristic and does not prove that text is English.
- Company/source patterns can create shortcuts and may not generalize to unseen companies.
- Relative timestamps have no trustworthy collection reference, so the model and dashboard must not imply temporal trends.
- Probabilities have measured calibration error and have not been post-hoc calibrated.

## Artifacts and reproducibility

- Model: `artifacts/baseline/baseline_model.joblib`
- Training code: `src/models/baseline.py`
- Machine-readable metrics: `reports/baseline_metrics.json`
- Comparative/subgroup report: `reports/model_evaluation.md`
- Error analysis: `reports/error_analysis.md`
- Data documentation: `data_card.md`

Reproduce with the fixed audit, preparation, and baseline commands documented in the project README when available.
