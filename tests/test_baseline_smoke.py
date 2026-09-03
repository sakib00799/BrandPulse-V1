import joblib
import pytest

from src.models.baseline import predict
from src.features.text import normalize_text


def test_saved_baseline_loads_and_predicts() -> None:
    artifact_path = "artifacts/baseline/baseline_model.joblib"
    if not __import__("pathlib").Path(artifact_path).is_file():
        pytest.skip("Local generated baseline artifact is not present")
    artifact = joblib.load(artifact_path)
    result = predict(artifact, [normalize_text("Payment korechi kintu internet active hoy nai")])
    assert len(result) == 1
    assert set(result[0]) == {"category", "sentiment", "priority"}
    for output in result[0].values():
        assert output["label"]
        assert 0.0 <= output["confidence"] <= 1.0
