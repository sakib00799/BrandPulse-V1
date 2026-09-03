# BrandPulse-BD Data Card

## Dataset summary

BrandPulse-BD contains public customer comments intended for supervised prediction of category, sentiment, and operational priority. Dataset version 1 has 1,044 supervised records from 8 companies and 4 source platforms.

Companies observed: Banglalink, Daraz, Grameenphone, Nagad, Pathao, Robi, bKash, foodpanda.

Platforms observed: Play Store, YouTube, playstore, youtube.

## Sources and collection

The supplied CSV files contain comment text, company, platform, source URL, and relative creation-time strings. The precise collection process, collection timestamps, licenses, and platform permissions were not supplied and remain documentation gaps.

## Version 1 columns

- `id`: supplied record identifier
- `text_raw`: original text from a well-formed raw record when it can be joined unambiguously; otherwise labeled-source text
- `text_labeled`: text supplied with human labels
- `text_cleaned`: separately supplied preprocessed text, retained for comparison
- `company`, `source_platform`, `source_url`: supplied provenance metadata
- `created_at_raw`: original relative or absolute date string
- `created_at_parsed`: null because no trustworthy reference collection timestamp was found
- `category`, `sentiment`, `priority`: supplied supervised labels
- `source_record_role`: provenance marker for the selected supervised row

## Transformations and exclusions

No source file was modified. The labeled file was used as the supervised row set. Well-formed raw records were reconciled by unique ID, while both raw and labeled text were preserved when they differed. Preprocessed text was retained in a separate column. Ten malformed raw records were excluded from version 1 because their fields could not be reconstructed safely; they are documented in `reports/label_review_queue.csv` and `reports/data_quality.json`. Relative dates were not converted.

## Label definitions

Observed category labels: Abuse/Harassment, Account/Login, Delivery/Order, Info/Query, Other, Payment, Refund/Return, Technical/App Bug.

Observed sentiment labels: Negative, Neutral, and Positive.

Observed priority labels: Low, Medium, and High. Priority is operationally ordered, but the supplied annotation policy was not provided and must be clarified before strong business interpretations.

## Quality and limitations

- `Abuse/Harassment` is an extremely rare class and has only 8 records.
- Raw CSV malformation and cross-file text differences require traceable human review.
- Relative timestamps cannot support time trends without a collection reference date.
- Language/script grouping is heuristic; Latin-script text is not guaranteed to be English.
- The dataset may contain duplicated, spam-like, noisy, sarcastic, or context-dependent comments.
- Labels are treated as supplied annotations, not objective ground truth.

## Privacy and redistribution

Comments and source URLs may contain personal or account-related information. Before publication, perform a dedicated PII review and release only a small anonymized sample unless licensing and platform terms clearly permit full redistribution. Never commit credentials or private annotations.

## Intended and out-of-scope use

The intended use is portfolio research and human-assisted customer-feedback triage. Predictions must not be used for punitive decisions or fully automated high-impact actions. High-priority and low-confidence cases require human review.
