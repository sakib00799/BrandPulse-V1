"""Train and evaluate character TF-IDF logistic-regression baselines."""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from src.evaluation.metrics import choose_review_threshold, classification_metrics

LOGGER = logging.getLogger("brandpulse.models.baseline")
TARGETS = ("category", "sentiment", "priority")
PRIORITY_ORDER = ["Low", "Medium", "High"]


def _candidate_parameters() -> list[dict[str, Any]]:
    return [
        {"C": c_value, "class_weight": class_weight}
        for class_weight in (None, "balanced")
        for c_value in (0.5, 1.0, 2.0)
    ]


def _fit_classifier(
    features: Any, labels: pd.Series, params: dict[str, Any], seed: int
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(params["C"]),
        class_weight=params["class_weight"],
        solver="lbfgs",
        max_iter=2_000,
        random_state=seed,
    )
    return model.fit(features, labels.astype(str))


def _plot_confusion(target: str, metrics: dict[str, Any], output_dir: Path) -> str:
    labels = metrics["labels"]
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=35)
    axis.set_title(f"Baseline {target} confusion matrix")
    figure.tight_layout()
    path = output_dir / f"baseline_{target}_confusion_matrix.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path.name


def _plot_reliability(
    target: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    output_dir: Path,
) -> str:
    predictions = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, 11)
    xs, ys, sizes = [], [], []
    for index in range(10):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == 9 else confidence < edges[index + 1]
        )
        if mask.any():
            xs.append(float(confidence[mask].mean()))
            ys.append(float(correct[mask].mean()))
            sizes.append(int(mask.sum()))
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.plot(xs, ys, marker="o", color="#0f766e", label="Observed")
    for x_value, y_value, size in zip(xs, ys, sizes):
        axis.annotate(str(size), (x_value, y_value), fontsize=8)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        title=f"Baseline {target} reliability",
    )
    axis.legend()
    figure.tight_layout()
    path = output_dir / f"baseline_{target}_reliability.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path.name


def _metrics_table(results: dict[str, Any]) -> str:
    lines = [
        "| Target | Macro-F1 | Weighted-F1 | Accuracy | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        values = results[target]["test_metrics"]
        lines.append(
            f"| {target} | {values['macro_f1']:.4f} | {values['weighted_f1']:.4f} | "
            f"{values['accuracy']:.4f} | {values['expected_calibration_error']:.4f} | "
            f"{values['multiclass_brier_score']:.4f} |"
        )
    return "\n".join(lines)


def _per_class_table(target: str, metrics: dict[str, Any]) -> str:
    lines = [
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    report = metrics["classification_report"]
    for label in metrics["labels"]:
        values = report[label]
        lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1-score']:.4f} | {int(values['support'])} |"
        )
    return "\n".join(lines)


def _build_report(results: dict[str, Any]) -> str:
    lines = [
        "# Classical Baseline Evaluation",
        "",
        "Character TF-IDF (3-5 grams) and one logistic-regression classifier per target. The vectorizer was fitted only on train text. Hyperparameters and class weighting were selected only with validation macro-F1; test predictions were produced after selection.",
        "",
        "## Test results",
        "",
        _metrics_table(results),
        "",
    ]
    for target in TARGETS:
        details = results[target]
        metrics = details["test_metrics"]
        lines.extend(
            [
                f"## {target.title()}",
                "",
                f"Selected configuration: `C={details['selected_params']['C']}`, `class_weight={details['selected_params']['class_weight']}`. Validation macro-F1: **{details['validation_macro_f1']:.4f}**.",
                "",
                f"Validation-selected low-confidence review threshold: **{details['review_threshold']['threshold']:.2f}** (coverage {details['review_threshold']['validation_coverage']:.3f}, selective accuracy {details['review_threshold']['validation_accuracy']:.3f}).",
                "",
                _per_class_table(target, metrics),
                "",
            ]
        )
        if target == "priority":
            lines.extend(
                [
                    f"Weighted Cohen's kappa: **{metrics['weighted_cohens_kappa']:.4f}**; mean absolute ordinal error: **{metrics['mean_absolute_ordinal_error']:.4f}**; High→Low severe-error rate: **{metrics['severe_high_as_low_rate']:.4f}**.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Limitations",
            "",
            "- The grouped split intentionally favors leakage isolation over ideal class balance.",
            "- `Abuse/Harassment` has very little training support, so its estimate is highly uncertain.",
            "- Probabilities are native logistic-regression probabilities and are not post-hoc calibrated.",
            "- This report is a classical baseline, not evidence that transformer training will improve results.",
            "",
        ]
    )
    return "\n".join(lines)


def train_baselines(
    processed_dir: Path,
    artifact_dir: Path,
    reports_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    train = pd.read_parquet(processed_dir / "train.parquet")
    validation = pd.read_parquet(processed_dir / "validation.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame["split"].nunique() != 1 or frame["split"].iloc[0] != name:
            raise ValueError(f"{name}.parquet has an invalid split marker")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_features = vectorizer.fit_transform(train["text_normalized"].astype(str))
    validation_features = vectorizer.transform(validation["text_normalized"].astype(str))
    test_features = vectorizer.transform(test["text_normalized"].astype(str))
    results: dict[str, Any] = {}
    models: dict[str, LogisticRegression] = {}
    total_start = perf_counter()
    for target in TARGETS:
        candidates = []
        selected_model: LogisticRegression | None = None
        selected_score = -1.0
        selected_params: dict[str, Any] | None = None
        for params in _candidate_parameters():
            model = _fit_classifier(train_features, train[target], params, seed)
            validation_predictions = model.predict(validation_features)
            score = float(
                f1_score(validation[target], validation_predictions, average="macro", zero_division=0)
            )
            candidates.append({**params, "validation_macro_f1": score})
            if score > selected_score:
                selected_score = score
                selected_model = model
                selected_params = params
        assert selected_model is not None and selected_params is not None
        validation_predictions = selected_model.predict(validation_features)
        validation_probabilities = selected_model.predict_proba(validation_features)
        review_threshold = choose_review_threshold(
            validation[target].astype(str).to_numpy(),
            validation_predictions,
            validation_probabilities,
        )
        inference_start = perf_counter()
        test_probabilities = selected_model.predict_proba(test_features)
        test_predictions = selected_model.classes_[np.argmax(test_probabilities, axis=1)]
        elapsed = perf_counter() - inference_start
        test_metrics = classification_metrics(
            test[target].astype(str).to_numpy(),
            test_predictions,
            test_probabilities,
            selected_model.classes_,
            target,
        )
        figures = [
            _plot_confusion(target, test_metrics, figures_dir),
            _plot_reliability(
                target,
                test[target].astype(str).to_numpy(),
                test_probabilities,
                selected_model.classes_,
                figures_dir,
            ),
        ]
        results[target] = {
            "candidate_validation_results": candidates,
            "selected_params": selected_params,
            "validation_macro_f1": selected_score,
            "review_threshold": review_threshold,
            "test_metrics": test_metrics,
            "test_inference_seconds": elapsed,
            "test_rows_per_second": len(test) / elapsed if elapsed else None,
            "figures": figures,
        }
        models[target] = selected_model
    model_version = "char-tfidf-logreg-v1"
    metadata = {
        "model_version": model_version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "feature_count": len(vectorizer.get_feature_names_out()),
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": [3, 5],
            "min_df": 2,
            "max_features": 100_000,
            "sublinear_tf": True,
            "fitted_on": "train_only",
        },
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "total_training_and_evaluation_seconds": perf_counter() - total_start,
    }
    artifact = {
        "model_version": model_version,
        "vectorizer": vectorizer,
        "models": models,
        "review_thresholds": {
            target: results[target]["review_threshold"]["threshold"] for target in TARGETS
        },
        "normalization_input_column": "text_normalized",
        "metadata": metadata,
    }
    model_path = artifact_dir / "baseline_model.joblib"
    metadata["model_size_bytes"] = 0
    joblib.dump(artifact, model_path)
    metadata["model_size_bytes"] = model_path.stat().st_size
    joblib.dump(artifact, model_path)
    final_size = model_path.stat().st_size
    if final_size != metadata["model_size_bytes"]:
        metadata["model_size_bytes"] = final_size
        joblib.dump(artifact, model_path)
    payload = {"metadata": metadata, "targets": results}
    (reports_dir / "baseline_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "baseline_evaluation.md").write_text(
        _build_report(results), encoding="utf-8"
    )
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def predict(artifact: dict[str, Any], texts: list[str]) -> list[dict[str, Any]]:
    features = artifact["vectorizer"].transform(texts)
    output = [dict() for _ in texts]
    for target, model in artifact["models"].items():
        probabilities = model.predict_proba(features)
        indices = np.argmax(probabilities, axis=1)
        for row, index in enumerate(indices):
            output[row][target] = {
                "label": str(model.classes_[index]),
                "confidence": float(probabilities[row, index]),
            }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    payload = train_baselines(
        args.processed_dir.resolve(), args.artifact_dir.resolve(), args.reports_dir.resolve(), args.seed
    )
    for target, values in payload["targets"].items():
        LOGGER.info("%s test macro-F1: %.4f", target, values["test_metrics"]["macro_f1"])


if __name__ == "__main__":
    main()
