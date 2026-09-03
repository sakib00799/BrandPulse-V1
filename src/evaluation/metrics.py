"""Metrics shared by classical and transformer experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, bins: int = 10
) -> float:
    predictions = classes[np.argmax(probabilities, axis=1)]
    confidences = probabilities.max(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences >= lower) & (
            confidences <= upper if index == bins - 1 else confidences < upper
        )
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return result


def multiclass_brier_score(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> float:
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = np.zeros_like(probabilities, dtype=float)
    for row, label in enumerate(y_true):
        encoded[row, class_to_index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))


def selective_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Iterable[float] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> list[dict[str, float | int | None]]:
    confidence = probabilities.max(axis=1)
    rows: list[dict[str, float | int | None]] = []
    for threshold in thresholds:
        mask = confidence >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(mask.mean()),
                "accepted_rows": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true[mask], predictions[mask])) if mask.any() else None,
                "macro_f1": float(
                    f1_score(y_true[mask], predictions[mask], average="macro", zero_division=0)
                )
                if mask.any()
                else None,
            }
        )
    return rows


def choose_review_threshold(
    y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Choose a confidence cutoff on validation only using a documented utility."""

    candidates = selective_metrics(
        y_true, predictions, probabilities, thresholds=np.round(np.arange(0.30, 0.91, 0.02), 2)
    )
    usable = [row for row in candidates if row["coverage"] >= 0.20 and row["accuracy"] is not None]
    if not usable:
        return {"threshold": 0.0, "validation_coverage": 1.0, "validation_accuracy": 0.0}
    meeting_target = [row for row in usable if float(row["accuracy"]) >= 0.80]
    if meeting_target:
        selected = max(meeting_target, key=lambda row: (float(row["coverage"]), -float(row["threshold"])))
    else:
        selected = max(
            usable,
            key=lambda row: float(row["accuracy"]) + 0.25 * float(row["coverage"]),
        )
    return {
        "threshold": float(selected["threshold"]),
        "validation_coverage": float(selected["coverage"]),
        "validation_accuracy": float(selected["accuracy"]),
    }


def classification_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    target: str,
) -> dict[str, object]:
    labels = list(classes)
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities, classes
        ),
        "multiclass_brier_score": multiclass_brier_score(y_true, probabilities, classes),
        "selective_performance": selective_metrics(y_true, predictions, probabilities),
    }
    if target == "priority":
        order = {"Low": 0, "Medium": 1, "High": 2}
        true_ordered = np.array([order[value] for value in y_true])
        predicted_ordered = np.array([order[value] for value in predictions])
        result["weighted_cohens_kappa"] = float(
            cohen_kappa_score(true_ordered, predicted_ordered, weights="quadratic")
        )
        result["mean_absolute_ordinal_error"] = float(
            np.mean(np.abs(true_ordered - predicted_ordered))
        )
        result["severe_high_as_low_rate"] = float(
            ((y_true == "High") & (predictions == "Low")).sum() / max(1, (y_true == "High").sum())
        )
    return result
