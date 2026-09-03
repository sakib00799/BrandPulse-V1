from pathlib import Path

import joblib
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from services.api.app.config import Settings
from services.api.app.main import create_app


def build_client(tmp_path: Path) -> TestClient:
    texts = ["payment failed", "delivery late", "good service", "login problem"]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3)).fit(texts)
    features = vectorizer.transform(texts)
    labels = {
        "category": ["Payment", "Delivery/Order", "Other", "Account/Login"],
        "sentiment": ["Negative", "Negative", "Positive", "Neutral"],
        "priority": ["High", "Medium", "Low", "Medium"],
    }
    models = {
        target: LogisticRegression(solver="lbfgs").fit(features, values)
        for target, values in labels.items()
    }
    model_path = tmp_path / "test-model.joblib"
    joblib.dump(
        {
            "model_version": "char-tfidf-logreg-v1",
            "vectorizer": vectorizer,
            "models": models,
            "review_thresholds": {"category": 0.3, "sentiment": 0.5, "priority": 0.5},
            "metadata": {"model_version": "char-tfidf-logreg-v1"},
        },
        model_path,
    )
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        model_path=model_path,
        seed_data_path=None,
        reports_dir=Path("reports"),
        cors_origins=("http://localhost:3000",),
    )
    return TestClient(create_app(settings))


def test_health_and_prediction_contract(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        response = client.post(
            "/predict",
            json={
                "text": "Payment korechi kintu internet active hoy nai",
                "company": "Grameenphone",
                "source_platform": "youtube",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model_version"] == "char-tfidf-logreg-v1"
        assert set(body) == {
            "category",
            "sentiment",
            "priority",
            "needs_human_review",
            "review_reasons",
            "model_version",
        }
        assert 0 <= body["priority"]["confidence"] <= 1


def test_empty_input_is_flagged_and_batch_is_bounded(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 200
        assert response.json()["needs_human_review"] is True
        assert "empty_input" in response.json()["review_reasons"]
        assert client.post("/batch-predict", json={"items": []}).status_code == 422


def test_feedback_is_stored_without_retraining(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/feedback",
            json={
                "text": "service bhalo na",
                "original_sentiment": "Neutral",
                "corrected_sentiment": "Negative",
                "reviewer_note": "Explicit complaint",
            },
        )
        assert response.status_code == 201
        assert response.json()["stored"] is True
        assert response.json()["automatic_retraining_triggered"] is False


def test_empty_comment_and_trend_collections(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        comments = client.get("/comments").json()
        assert comments["total"] == 0
        trends = client.get("/analytics/trends").json()
        assert trends["available"] is False
        assert trends["series"] == []
        performance = client.get("/analytics/performance")
        assert performance.status_code == 200
        assert performance.json()["selected_model"] == "char-tfidf-logreg-v1"
