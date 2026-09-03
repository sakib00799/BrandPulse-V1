import pandas as pd

from src.data.split import SplitConfig, build_leakage_groups, grouped_multitarget_split, leakage_check
from src.features.text import normalize_text


def test_normalization_preserves_sentiment_signals_and_replaces_identifiers() -> None:
    value = normalize_text("  ভালো  না! 😞 Call 01712-345678 or a@b.com  ")
    assert "ভালো না! 😞" in value
    assert "<PHONE>" in value
    assert "<EMAIL>" in value


def _sample_frame() -> pd.DataFrame:
    rows = []
    categories = ["Other", "Payment", "Delivery/Order"]
    sentiments = ["Neutral", "Negative", "Positive"]
    priorities = ["Low", "Medium", "High"]
    for index in range(30):
        rows.append(
            {
                "id": str(index),
                "text_raw": f"unique comment {index}",
                "source_url": f"https://example.com/{index // 2}",
                "category": categories[index % 3],
                "sentiment": sentiments[index % 3],
                "priority": priorities[index % 3],
            }
        )
    rows[1]["text_raw"] = rows[0]["text_raw"]
    return pd.DataFrame(rows)


def test_grouped_split_is_deterministic_and_has_no_overlap() -> None:
    frame = _sample_frame()
    frame["leakage_group"] = build_leakage_groups(frame)
    config = SplitConfig(seed=7, search_attempts=20)
    first = grouped_multitarget_split(frame, config)
    second = grouped_multitarget_split(frame, config)
    assert first.equals(second)
    frame["split"] = first
    assert leakage_check(frame)["passed"]
    assert set(first) == {"train", "validation", "test"}
