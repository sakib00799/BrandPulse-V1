# Transformer Experiment

## Completed local experiment

A compact character-level Transformer was trained from scratch as an offline fallback. It uses one shared two-layer trainable encoder and separate category, sentiment, and priority heads. Its vocabulary was fitted on train text only, target losses were equally weighted, and inverse-square-root class weighting was selected over unweighted loss using validation mean macro-F1 (`0.4043` versus `0.3935`).

Three final seeds were run. The production seed was selected using validation mean macro-F1 only.

| Target | Production-seed test macro-F1 | Three-seed mean | Three-seed standard deviation |
|---|---:|---:|---:|
| Category | 0.1493 | 0.1558 | 0.0073 |
| Sentiment | 0.4271 | 0.4232 | 0.0319 |
| Priority | 0.4483 | 0.4498 | 0.0333 |

These results are below the TF-IDF baseline on every target. The compact Transformer is retained as a reproducible experiment, not selected for serving.

## Pretrained-model status

`xlm-roberta-base` and `csebuetnlp/banglabert` were both attempted. Their metadata and tokenizer endpoints responded, but the redirected Hugging Face weight CDN returned no model body and left zero-byte incomplete files. The stalled processes were terminated without fabricating results. Neither pretrained model has an evaluation score.

The implementation for the intended pretrained shared-encoder experiment remains in `src/models/multitask_transformer.py`. It can be rerun once weights are locally available or the download path works.

## Limitations

- A from-scratch character model on 691 training records cannot substitute for a pretrained multilingual encoder.
- Three seeds reduce but do not eliminate uncertainty from the small, grouped dataset.
- The locked test split was used only for reporting after each seed's best epoch was selected on validation.
