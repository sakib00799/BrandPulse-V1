from pathlib import Path

import pandas as pd

from src.data.audit import duplicate_signature, minimal_compare_normalize, run_audit, valid_url


def test_minimal_compare_normalize_preserves_negation_and_emoji() -> None:
    assert minimal_compare_normalize("  ভালো   না 😞 ") == "ভালো না 😞"


def test_duplicate_signature_normalizes_formatting() -> None:
    assert duplicate_signature("Hello,   WORLD!") == duplicate_signature("hello world")


def test_url_validation() -> None:
    assert valid_url("https://example.com/post/1")
    assert not valid_url("not-a-url")


def test_audit_does_not_modify_source_and_preserves_text_versions(tmp_path: Path) -> None:
    source = tmp_path / "source"
    reports = tmp_path / "reports"
    processed = tmp_path / "processed"
    source.mkdir()
    raw = "id,text,company,source_platform,source_url,created_at\n1,Raw text,ACME,youtube,https://example.com/1,2 days ago\n"
    labeled = "id,text,company,source_platform,source_url,created_at,category,sentiment,priority\n1,Labeled text,ACME,youtube,https://example.com/1,2 days ago,Other,Neutral,Low\n"
    cleaned = "id,text,company,source_platform,source_url,created_at,category,sentiment,priority\n1,Clean text,ACME,youtube,https://example.com/1,2 days ago,Other,Neutral,Low\n"
    (source / "tickets_raw.csv").write_text(raw, encoding="utf-8-sig")
    (source / "tickets_labeled.csv").write_text(labeled, encoding="utf-8-sig")
    (source / "tickets_preprocessed.csv").write_text(cleaned, encoding="utf-8-sig")
    before = (source / "tickets_raw.csv").read_bytes()

    quality = run_audit(source, reports, processed)

    assert (source / "tickets_raw.csv").read_bytes() == before
    result = pd.read_csv(processed / "dataset_version_1.csv", encoding="utf-8-sig")
    assert result.loc[0, "text_raw"] == "Raw text"
    assert result.loc[0, "text_labeled"] == "Labeled text"
    assert result.loc[0, "text_cleaned"] == "Clean text"
    assert quality["reconciliation"]["raw_vs_labeled_text_conflicts"] == 1
    assert pd.isna(result.loc[0, "created_at_parsed"])
