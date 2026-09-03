# Model Evaluation

All test metrics use the fixed, grouped real-label holdout. Hyperparameter, weighting, seed, and review-threshold selection used train/validation only.

## Model comparison

| Model | Target | Macro-F1 | Weighted-F1 | Accuracy |
|---|---|---|---|---|
| char_tfidf_logistic_regression | category | 0.2721 | 0.3740 | 0.3837 |
| char_tfidf_logistic_regression | sentiment | 0.6373 | 0.6794 | 0.6977 |
| char_tfidf_logistic_regression | priority | 0.5177 | 0.5701 | 0.6047 |
| compact_character_transformer | category | 0.1493 | 0.2432 | 0.2791 |
| compact_character_transformer | sentiment | 0.4271 | 0.4984 | 0.5581 |
| compact_character_transformer | priority | 0.4483 | 0.4907 | 0.4709 |

The character TF-IDF logistic-regression baseline is selected for the application because it outperforms the locally trainable compact Transformer on all three targets. This selection is based on the complete evaluation, not a claim that classical models are generally superior.

The required pretrained XLM-R and BanglaBERT downloads reached their metadata/tokenizer endpoints, but their weight CDN returned no body. They were not trained and no score is claimed for either model.

## Selected-model operational metrics

Model size: **1.032 MB**. Single-record CPU latency: median **0.615 ms**, P95 **0.826 ms**. Batch throughput: **7949.8 rows/s**.

## Subgroup evaluation

Subgroup macro-F1 averages only classes present inside that subgroup, so values are not directly interchangeable with whole-test macro-F1. Small-support rows are highly uncertain.

### Category

**company**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| Banglalink | 19 | 0.4737 | 0.1778 |
| Grameenphone | 10 | 0.6000 | 0.1600 |
| Nagad | 1 | 0.0000 | 0.0000 |
| Pathao | 117 | 0.3333 | 0.2571 |
| Robi | 7 | 0.4286 | 0.2000 |
| bKash | 2 | 0.5000 | 0.3333 |
| foodpanda | 16 | 0.5000 | 0.4061 |

**source_platform_normalized**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| play store | 7 | 0.2857 | 0.1000 |
| playstore | 152 | 0.3816 | 0.2914 |
| youtube | 13 | 0.4615 | 0.1250 |

**script_group**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| bangla_script | 6 | 1.0000 | 1.0000 |
| code_mixed | 1 | 1.0000 | 1.0000 |
| latin_script_proxy | 165 | 0.3576 | 0.2492 |

**text_length**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| long | 128 | 0.3281 | 0.2746 |
| short_or_equal_train_median | 44 | 0.5455 | 0.2583 |

**category_frequency**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| common_category_train_ge_50 | 156 | 0.4038 | 0.2381 |
| minority_category_train_lt_50 | 16 | 0.1875 | 0.1527 |

### Sentiment

**company**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| Banglalink | 19 | 0.7895 | 0.6989 |
| Grameenphone | 10 | 0.6000 | 0.4889 |
| Nagad | 1 | 1.0000 | 1.0000 |
| Pathao | 117 | 0.7179 | 0.6297 |
| Robi | 7 | 0.5714 | 0.5714 |
| bKash | 2 | 0.5000 | 0.3333 |
| foodpanda | 16 | 0.5625 | 0.3918 |

**source_platform_normalized**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| play store | 7 | 0.5714 | 0.4524 |
| playstore | 152 | 0.7105 | 0.6666 |
| youtube | 13 | 0.6154 | 0.5064 |

**script_group**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| bangla_script | 6 | 0.5000 | 0.3556 |
| code_mixed | 1 | 1.0000 | 1.0000 |
| latin_script_proxy | 165 | 0.7030 | 0.6317 |

**text_length**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| long | 128 | 0.6953 | 0.6003 |
| short_or_equal_train_median | 44 | 0.7045 | 0.6643 |

**category_frequency**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| common_category_train_ge_50 | 156 | 0.6987 | 0.6449 |
| minority_category_train_lt_50 | 16 | 0.6875 | 0.4796 |

### Priority

**company**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| Banglalink | 19 | 0.6842 | 0.2708 |
| Grameenphone | 10 | 0.8000 | 0.4444 |
| Nagad | 1 | 0.0000 | 0.0000 |
| Pathao | 117 | 0.5812 | 0.4626 |
| Robi | 7 | 0.5714 | 0.3636 |
| bKash | 2 | 1.0000 | 1.0000 |
| foodpanda | 16 | 0.5625 | 0.5466 |

**source_platform_normalized**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| play store | 7 | 0.4286 | 0.3000 |
| playstore | 152 | 0.6053 | 0.5181 |
| youtube | 13 | 0.6923 | 0.4091 |

**script_group**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| bangla_script | 6 | 1.0000 | 1.0000 |
| code_mixed | 1 | 1.0000 | 1.0000 |
| latin_script_proxy | 165 | 0.5879 | 0.5032 |

**text_length**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| long | 128 | 0.5781 | 0.4779 |
| short_or_equal_train_median | 44 | 0.6818 | 0.5471 |

**category_frequency**

| Group | Support | Accuracy | Macro-F1 (present) |
|---|---|---|---|
| common_category_train_ge_50 | 156 | 0.6474 | 0.5132 |
| minority_category_train_lt_50 | 16 | 0.1875 | 0.1111 |

## Confidence and review

Native logistic-regression probabilities are used; they are not described as perfectly calibrated. Reliability diagrams, ECE, Brier scores, and selective performance are in `reports/baseline_metrics.json` and `reports/figures/`. Review thresholds were chosen on validation only. Every predicted `High` priority remains reviewable regardless of confidence.

## Error-analysis sample

A deterministic sample of false predictions is stored in `reports/error_analysis_sample.csv`. Manual findings and the taxonomy are in `reports/error_analysis.md`.

## Limitations

- Category classification is weak, especially for rare classes; the selected model is a portfolio prototype, not production-ready automation.
- The grouped test distribution differs from train because large connected source/duplicate groups must remain isolated.
- Platform names contain capitalization variants; the subgroup report normalizes them only for analysis.
- Script grouping is character-based and Latin-script is not equivalent to verified English.
- Three-seed statistics exist for the compact Transformer; the baseline currently has one fixed seed.
