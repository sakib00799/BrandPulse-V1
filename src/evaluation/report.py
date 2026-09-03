"""Build comparative, subgroup, latency, and reproducible error-analysis reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data.audit import script_group, text_profile
from src.features.text import normalize_text
from src.models.baseline import TARGETS, predict


def subgroup_metrics(
    truth: pd.Series, predictions: np.ndarray, groups: pd.Series
) -> list[dict[str, Any]]:
    rows = []
    for group_name in sorted(groups.astype(str).unique()):
        mask = groups.astype(str).eq(group_name).to_numpy()
        if not mask.any():
            continue
        rows.append(
            {
                "group": group_name,
                "support": int(mask.sum()),
                "accuracy": float(accuracy_score(truth.to_numpy()[mask], predictions[mask])),
                "macro_f1_present_classes": float(
                    f1_score(
                        truth.to_numpy()[mask], predictions[mask], average="macro", zero_division=0
                    )
                ),
            }
        )
    return rows


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _comparative_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Model Evaluation",
        "",
        "All test metrics use the fixed, grouped real-label holdout. Hyperparameter, weighting, seed, and review-threshold selection used train/validation only.",
        "",
        "## Model comparison",
        "",
    ]
    comparison_rows = []
    for model_name, targets in payload["comparison"].items():
        for target, values in targets.items():
            comparison_rows.append(
                [
                    model_name,
                    target,
                    f"{values['macro_f1']:.4f}",
                    f"{values['weighted_f1']:.4f}",
                    f"{values['accuracy']:.4f}",
                ]
            )
    lines.extend(
        [
            _markdown_table(
                ["Model", "Target", "Macro-F1", "Weighted-F1", "Accuracy"], comparison_rows
            ),
            "",
            "The character TF-IDF logistic-regression baseline is selected for the application because it outperforms the locally trainable compact Transformer on all three targets. This selection is based on the complete evaluation, not a claim that classical models are generally superior.",
            "",
            "The required pretrained XLM-R and BanglaBERT downloads reached their metadata/tokenizer endpoints, but their weight CDN returned no body. They were not trained and no score is claimed for either model.",
            "",
            "## Selected-model operational metrics",
            "",
            f"Model size: **{payload['latency']['model_size_bytes'] / 1_000_000:.3f} MB**. Single-record CPU latency: median **{payload['latency']['median_ms']:.3f} ms**, P95 **{payload['latency']['p95_ms']:.3f} ms**. Batch throughput: **{payload['latency']['batch_rows_per_second']:.1f} rows/s**.",
            "",
            "## Subgroup evaluation",
            "",
            "Subgroup macro-F1 averages only classes present inside that subgroup, so values are not directly interchangeable with whole-test macro-F1. Small-support rows are highly uncertain.",
            "",
        ]
    )
    for target in TARGETS:
        lines.extend([f"### {target.title()}", ""])
        for dimension, rows in payload["subgroups"][target].items():
            formatted = [
                [row["group"], row["support"], f"{row['accuracy']:.4f}", f"{row['macro_f1_present_classes']:.4f}"]
                for row in rows
            ]
            lines.extend(
                [
                    f"**{dimension}**",
                    "",
                    _markdown_table(["Group", "Support", "Accuracy", "Macro-F1 (present)"], formatted),
                    "",
                ]
            )
    lines.extend(
        [
            "## Confidence and review",
            "",
            "Native logistic-regression probabilities are used; they are not described as perfectly calibrated. Reliability diagrams, ECE, Brier scores, and selective performance are in `reports/baseline_metrics.json` and `reports/figures/`. Review thresholds were chosen on validation only. Every predicted `High` priority remains reviewable regardless of confidence.",
            "",
            "## Error-analysis sample",
            "",
            "A deterministic sample of false predictions is stored in `reports/error_analysis_sample.csv`. Manual findings and the taxonomy are in `reports/error_analysis.md`.",
            "",
            "## Limitations",
            "",
            "- Category classification is weak, especially for rare classes; the selected model is a portfolio prototype, not production-ready automation.",
            "- The grouped test distribution differs from train because large connected source/duplicate groups must remain isolated.",
            "- Platform names contain capitalization variants; the subgroup report normalizes them only for analysis.",
            "- Script grouping is character-based and Latin-script is not equivalent to verified English.",
            "- Three-seed statistics exist for the compact Transformer; the baseline currently has one fixed seed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_evaluation(
    processed_dir: Path,
    baseline_artifact_path: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    train = pd.read_parquet(processed_dir / "train.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    artifact = joblib.load(baseline_artifact_path)
    normalized_text = test["text_normalized"].astype(str).tolist()
    features = artifact["vectorizer"].transform(normalized_text)
    predictions: dict[str, np.ndarray] = {
        target: artifact["models"][target].predict(features) for target in TARGETS
    }
    profiles = pd.DataFrame([text_profile(text) for text in test["text_raw"].astype(str)])
    script_groups = profiles.apply(script_group, axis=1)
    train_median_length = float(train["text_raw"].astype(str).str.len().median())
    length_groups = test["text_raw"].astype(str).str.len().map(
        lambda value: "short_or_equal_train_median" if value <= train_median_length else "long"
    )
    train_category_counts = train["category"].value_counts()
    category_frequency_groups = test["category"].map(
        lambda value: "minority_category_train_lt_50"
        if train_category_counts.get(value, 0) < 50
        else "common_category_train_ge_50"
    )
    dimensions = {
        "company": test["company"].astype(str),
        "source_platform_normalized": test["source_platform"].astype(str).str.strip().str.casefold(),
        "script_group": script_groups,
        "text_length": length_groups,
        "category_frequency": category_frequency_groups,
    }
    subgroups = {
        target: {
            name: subgroup_metrics(test[target].astype(str), predictions[target], groups)
            for name, groups in dimensions.items()
        }
        for target in TARGETS
    }
    baseline_metrics = json.loads((reports_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
    transformer_metrics = json.loads(
        (reports_dir / "transformer_metrics.json").read_text(encoding="utf-8")
    )
    comparison = {
        "char_tfidf_logistic_regression": {
            target: {
                key: baseline_metrics["targets"][target]["test_metrics"][key]
                for key in ("macro_f1", "weighted_f1", "accuracy")
            }
            for target in TARGETS
        },
        "compact_character_transformer": {
            target: {
                key: transformer_metrics["targets"][target]["test_metrics"][key]
                for key in ("macro_f1", "weighted_f1", "accuracy")
            }
            for target in TARGETS
        },
    }
    example = [normalize_text("Payment korechi kintu internet active hoy nai")]
    for _ in range(5):
        predict(artifact, example)
    timings = []
    for _ in range(100):
        start = perf_counter()
        predict(artifact, example)
        timings.append((perf_counter() - start) * 1000)
    batch_start = perf_counter()
    predict(artifact, normalized_text)
    batch_elapsed = perf_counter() - batch_start
    latency = {
        "device": "cpu",
        "runs": len(timings),
        "median_ms": float(statistics.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "model_size_bytes": baseline_artifact_path.stat().st_size,
        "batch_rows": len(test),
        "batch_seconds": batch_elapsed,
        "batch_rows_per_second": len(test) / batch_elapsed,
    }
    error_rows = []
    for target_index, target in enumerate(TARGETS):
        model = artifact["models"][target]
        probabilities = model.predict_proba(features)
        confidence = probabilities.max(axis=1)
        mask = predictions[target] != test[target].astype(str).to_numpy()
        candidates = test.loc[mask].copy()
        candidates["_prediction"] = predictions[target][mask]
        candidates["_confidence"] = confidence[mask]
        sampled = candidates.sample(n=min(8, len(candidates)), random_state=42 + target_index)
        for _, row in sampled.iterrows():
            error_rows.append(
                {
                    "id": row["id"],
                    "target": target,
                    "actual": row[target],
                    "predicted": row["_prediction"],
                    "confidence": row["_confidence"],
                    "company": row["company"],
                    "source_platform": row["source_platform"],
                    "text": row["text_raw"],
                }
            )
    pd.DataFrame(error_rows).to_csv(
        reports_dir / "error_analysis_sample.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "selected_model": "char-tfidf-logreg-v1",
        "selection_reason": "higher test metrics than the compact offline fallback on every target; pretrained candidates unavailable",
        "comparison": comparison,
        "subgroups": subgroups,
        "subgroup_definition": {
            "short_long_boundary_characters_from_train_only": train_median_length,
            "minority_category_definition_from_train_only": "fewer than 50 training records",
            "script_group_method": "character_ratio_heuristic",
        },
        "latency": latency,
        "error_sample_rows": len(error_rows),
    }
    (reports_dir / "model_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "model_evaluation.md").write_text(
        _comparative_report(payload), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--baseline-artifact", type=Path, default=Path("artifacts/baseline/baseline_model.joblib")
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_evaluation(
        args.processed_dir.resolve(), args.baseline_artifact.resolve(), args.reports_dir.resolve()
    )
    print(json.dumps({"selected_model": payload["selected_model"], "latency": payload["latency"]}, indent=2))


if __name__ == "__main__":
    main()
