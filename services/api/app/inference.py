"""Baseline inference with explicit human-review policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.features.text import normalize_text


class InferenceService:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.artifact = joblib.load(model_path)
        self.model_version = str(self.artifact["model_version"])
        self.thresholds = self.artifact["review_thresholds"]

    def _known_ngram_ratio(self, normalized_text: str) -> float:
        analyzer = self.artifact["vectorizer"].build_analyzer()
        ngrams = analyzer(normalized_text)
        if not ngrams:
            return 0.0
        vocabulary = self.artifact["vectorizer"].vocabulary_
        return sum(ngram in vocabulary for ngram in ngrams) / len(ngrams)

    def predict_many(self, texts: list[str]) -> list[dict[str, Any]]:
        normalized = [normalize_text(text) for text in texts]
        features = self.artifact["vectorizer"].transform(normalized)
        raw_predictions = [dict() for _ in texts]
        for target, model in self.artifact["models"].items():
            probabilities = model.predict_proba(features)
            indices = np.argmax(probabilities, axis=1)
            for row, index in enumerate(indices):
                raw_predictions[row][target] = {
                    "label": str(model.classes_[index]),
                    "confidence": float(probabilities[row, index]),
                }
        output = []
        for original, normalized_text, item in zip(texts, normalized, raw_predictions):
            reasons = []
            if not original.strip():
                reasons.append("empty_input")
            elif len(original.strip()) < 3:
                reasons.append("extremely_short_input")
            if normalized_text and self._known_ngram_ratio(normalized_text) < 0.05:
                reasons.append("possible_out_of_distribution_input")
            for target in ("category", "sentiment", "priority"):
                if item[target]["confidence"] < float(self.thresholds[target]):
                    reasons.append(f"low_{target}_confidence")
            if item["priority"]["label"] == "High":
                reasons.append("predicted_high_priority")
            if item["sentiment"]["label"] == "Positive" and item["priority"]["label"] == "High":
                reasons.append("operationally_inconsistent_positive_high_combination")
            output.append(
                {
                    **item,
                    "needs_human_review": bool(reasons),
                    "review_reasons": reasons,
                    "model_version": self.model_version,
                }
            )
        return output

    def predict_one(self, text: str) -> dict[str, Any]:
        return self.predict_many([text])[0]

    def model_info(self) -> dict[str, Any]:
        metadata = dict(self.artifact["metadata"])
        metadata["review_thresholds"] = self.thresholds
        metadata["model_path"] = str(self.model_path)
        return metadata
