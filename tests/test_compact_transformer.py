from pathlib import Path

import torch
import pytest

from src.models.compact_transformer import (
    CharacterVocabulary,
    CompactConfig,
    CompactMultiTaskTransformer,
    predict_texts,
)


def test_character_vocabulary_is_train_fitted_and_has_unknown_fallback() -> None:
    vocabulary = CharacterVocabulary.fit(["বাংলা", "hello"], minimum_frequency=1)
    encoded = vocabulary.encode("বাংলা🙂", max_length=20)
    assert encoded[0] == vocabulary.tokens[CharacterVocabulary.CLS]
    assert encoded[-1] == vocabulary.tokens[CharacterVocabulary.UNK]


def test_compact_model_has_three_heads() -> None:
    config = CompactConfig(embedding_size=16, attention_heads=4, feedforward_size=32, encoder_layers=1)
    model = CompactMultiTaskTransformer(
        20, {"category": 3, "sentiment": 3, "priority": 3}, config
    )
    inputs = torch.tensor([[2, 3, 0], [2, 4, 5]])
    mask = inputs != 0
    output = model(inputs, mask)
    assert output["category"].shape == (2, 3)
    assert set(output) == {"category", "sentiment", "priority"}


def test_saved_compact_artifact_loads_on_cpu() -> None:
    artifact_dir = Path("artifacts/transformer/compact_v1")
    if not artifact_dir.is_dir():
        pytest.skip("Local generated compact-transformer artifact is not present")
    result = predict_texts(artifact_dir, ["ইন্টারনেট কাজ করছে না"])
    assert set(result[0]) == {"category", "sentiment", "priority"}
