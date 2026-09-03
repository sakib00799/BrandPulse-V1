import numpy as np

from src.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    multiclass_brier_score,
)


def test_perfect_probabilities_have_zero_calibration_and_brier_error() -> None:
    classes = np.array(["a", "b"])
    truth = np.array(["a", "b"])
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(truth, probabilities, classes) == 0.0
    assert multiclass_brier_score(truth, probabilities, classes) == 0.0


def test_priority_metrics_include_ordinal_severity() -> None:
    classes = np.array(["High", "Low", "Medium"])
    truth = np.array(["High", "Low", "Medium"])
    predictions = np.array(["Low", "Low", "Medium"])
    probabilities = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    result = classification_metrics(truth, predictions, probabilities, classes, "priority")
    assert result["severe_high_as_low_rate"] == 1.0
    assert "weighted_cohens_kappa" in result
