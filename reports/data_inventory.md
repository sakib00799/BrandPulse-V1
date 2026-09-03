# BrandPulse-BD Data Inventory

Generated during read-only workspace discovery on 2026-09-03. Original dataset files were not modified, renamed, or moved.

## Inventory

| File | Records | Columns | Probable role | Encoding | Delimiter | Parsing notes |
|---|---:|---|---|---|---|---|
| `tickets_raw.csv` | 1,054 logical data records: 1,044 well-formed and 10 malformed | `id`, `text`, `company`, `source_platform`, `source_url`, `created_at` | Raw, unlabeled source | UTF-8 with BOM | Comma | Ten records have 7-12 fields instead of 6 because delimiter/quoting is malformed. A normal dataframe parser fails at the first affected record. |
| `tickets_labeled.csv` | 1,044 | Raw columns plus `category`, `sentiment`, `priority` | Primary supervised labeled source | UTF-8 with BOM | Comma | Parses cleanly; no missing values or duplicate IDs found in the initial structural check. |
| `tickets_preprocessed.csv` | 1,044 | Same nine columns as the labeled file | Cleaned labeled derivative | UTF-8 with BOM | Comma | Parses cleanly. IDs, metadata, and labels match `tickets_labeled.csv`; text differs in 909 records. |

CSV record counts exclude the single header record. Multiline quoted fields, if any, count as one logical record.

## Source integrity fingerprints

| File | SHA-256 |
|---|---|
| `tickets_raw.csv` | `f61b3fc80f817c4de9fb61794c3600cdaa3440a65a819591db63f7d7a6c187ee` |
| `tickets_labeled.csv` | `6bf454f69fbc2389d86b07737f0f7095cd512aa40164463fedd3e51848610fb6` |
| `tickets_preprocessed.csv` | `83a71e581bc9d95dc8ab5420190b279c2fb856d61e6a0dc513380adcc67620f1` |

These hashes establish the source versions inspected and can be used later to confirm that generated processing did not alter them.

## Initial reconciliation findings

- The 1,044 well-formed raw records align by ID with all 1,044 labeled records.
- Ten malformed raw records are absent from the labeled and preprocessed files. Their IDs are: `19`, `334`, `363`, `387`, `400`, `404`, `407`, `419`, `420`, and `423`.
- For the 1,044 aligned records, metadata fields match between raw and labeled files. Two `text` values differ and must be preserved and reviewed rather than silently resolved.
- `tickets_preprocessed.csv` changes 909 text values and leaves 135 unchanged relative to `tickets_labeled.csv`. All non-text columns match.
- No missing values, repeated header rows, or duplicate IDs were found in either cleanly parsed labeled file during this initial structural check.

## Initial label distribution

The labeled and preprocessed files have identical labels.

### Category

| Label | Count |
|---|---:|
| Other | 348 |
| Technical/App Bug | 237 |
| Delivery/Order | 123 |
| Info/Query | 118 |
| Payment | 107 |
| Account/Login | 65 |
| Refund/Return | 38 |
| Abuse/Harassment | 8 |

### Sentiment

| Label | Count |
|---|---:|
| Neutral | 416 |
| Negative | 399 |
| Positive | 229 |

### Priority

| Label | Count |
|---|---:|
| Low | 455 |
| Medium | 399 |
| High | 190 |

## Risks and ambiguities to resolve in the audit

1. The ten malformed raw records cannot be interpreted safely by shifting fields automatically; they require explicit reconciliation and review.
2. The category distribution is highly imbalanced: `Abuse/Harassment` has only 8 examples, compared with 348 for `Other`.
3. The two raw-versus-labeled text differences need a traceable comparison to determine whether they are legitimate cleanup or accidental changes.
4. The exact preprocessing that produced the 909 changed texts is not yet documented and must not automatically become the transformer input.
5. `created_at` includes relative values such as `8 months ago`. Unless a trustworthy collection timestamp is found, the audit will retain these as `created_at_raw` and leave parsed dates null.
6. Filename-based roles are only provisional. The reproducible audit will validate provenance, duplicates, label consistency, URLs, text characteristics, and cross-file conflicts.

## Ordered implementation plan

1. Build a reproducible audit command and generate the complete audit, quality JSON, review queue, figures, reconciled processed dataset, and data card.
2. Implement tested minimal normalization and construct leakage-safe grouped train/validation/test manifests.
3. Train and evaluate character TF-IDF baselines without using test information for model selection.
4. Run the controlled transformer experiments and complete calibration, subgroup evaluation, and error analysis.
5. Build the FastAPI/SQLAlchemy service and human-review workflow with automated tests.
6. Build the Next.js dashboard, then add Docker, CI, documentation, and end-to-end verification.
