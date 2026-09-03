# Manual Error Analysis

This analysis uses the deterministic 24-row sample in `reports/error_analysis_sample.csv` (eight false predictions per target, sampled with fixed seeds). The taxonomy is descriptive; it does not overwrite any supplied label.

## Taxonomy

| Error type | Evidence from reviewed sample | Interpretation |
|---|---|---|
| Label ambiguity or possible annotation inconsistency | ID 396 says support is helpful but is labeled `Info/Query`; IDs 903 and 891 contain complaints but are labeled `Neutral`. | Some apparent model errors may partly reflect unclear or inconsistent labels. These IDs should be reviewed, not automatically relabeled. |
| Multiple issues in one comment | IDs 805, 809, 861, and 803 combine app behavior, rider behavior, pricing/payment, cancellation, and support complaints. | A single-label category cannot fully represent multi-intent feedback. |
| Mixed or contrastive sentiment | ID 763 praises offers but reports a network problem; ID 861 praises the app while criticizing service; ID 884 starts with praise and ends with a one-star complaint. | Bag-of-character evidence can emphasize the positive clause and miss the final or dominant negative judgment. |
| Category boundary ambiguity | IDs 879 and 853 describe riders accepting then cancelling; ground truth is `Technical/App Bug`, while the baseline predicts `Delivery/Order`. | The annotation boundary between platform behavior and service/order operations is not self-evident. |
| Implicit operational priority | ID 386 reports money being deducted; ID 897 reports an app crash during payment. | Priority requires an annotation policy about financial risk and service impact, not text polarity alone. |
| Company/source shortcut risk | Several sampled errors are long Pathao Play Store reviews. | Large source groups and company-specific terminology may dominate learned character patterns. |
| Sparse or generic text | IDs 346 and 396 are short generic praise statements. | There is little lexical evidence for fine-grained categories, and category confidences are below 0.19. |
| Minority-class scarcity | Category test metrics show zero recall for `Abuse/Harassment`; training has only three examples in the grouped split. | This class is not learnable reliably from the current split and needs more reviewed real examples. |

## Observations by target

### Category

All eight reviewed category errors have low confidence (approximately 0.18-0.35). Most should enter the confidence-based review queue. Four Pathao examples labeled `Technical/App Bug` contain delivery, rider, price, cancellation, or support issues, revealing genuine multi-intent/category-boundary problems rather than a clean lexical distinction.

### Sentiment

The reviewed failures frequently contain explicit contrast: positive service/app language alongside a complaint. IDs 903 and 891 look negative on their face but carry `Neutral` labels, so they are candidates for label review. This sample does not establish that those labels are wrong; context or an annotation rule may explain them.

### Priority

The baseline often maps low- and medium-priority cases into each other. High-priority IDs 897 and 870 are predicted `Medium`; their confidence values are below the validation-selected priority threshold of 0.72, so the low-confidence rule catches both. The separate rule that flags every predicted `High` cannot catch false-negative High cases by itself.

## Recommended data actions

1. Write explicit annotation rules for multi-intent category selection and financial/payment priority.
2. Review the cited ambiguous IDs and the existing label-review queue with two annotators where practical.
3. Collect additional real `Abuse/Harassment` and `Refund/Return` examples without adding them to the locked test set.
4. Consider multi-label category annotation as a future schema change; do not retroactively convert labels without a versioned review process.
5. Keep confidence and High-priority human review mandatory. Current category quality is not sufficient for unattended routing.
