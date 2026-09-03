"""Reproducible, read-only audit and reconciliation for BrandPulse-BD data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger("brandpulse.data.audit")
DATA_SUFFIXES = {".csv", ".tsv"}
BASE_COLUMNS = ["id", "text", "company", "source_platform", "source_url", "created_at"]
LABEL_COLUMNS = ["category", "sentiment", "priority"]
EXPECTED_LABELS = {
    "category": {
        "Other",
        "Technical/App Bug",
        "Delivery/Order",
        "Info/Query",
        "Payment",
        "Account/Login",
        "Refund/Return",
        "Abuse/Harassment",
    },
    "sentiment": {"Negative", "Neutral", "Positive"},
    "priority": {"Low", "Medium", "High"},
}
RELATIVE_DATE_RE = re.compile(
    r"^\s*(?:a|an|\d+(?:\.\d+)?)\s+(?:minute|hour|day|week|month|year)s?\s+ago\s*$",
    re.IGNORECASE,
)
RELATIVE_LIKE_DATE_RE = re.compile(r"(?:\bago|^\s*just\s+now\s*$)", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
CONTROL_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]")


@dataclass
class FileAudit:
    filename: str
    sha256: str
    encoding: str
    delimiter: str
    columns: list[str]
    logical_records: int
    well_formed_records: int
    malformed_records: int
    malformed_record_numbers: list[int]
    repeated_header_records: int
    role: str
    parsing_warnings: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_encoding(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_delimiter(text: str, suffix: str) -> str:
    if suffix.lower() == ".tsv":
        return "\t"
    try:
        return csv.Sniffer().sniff(text[:65_536], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def infer_role(path: Path, columns: Iterable[str]) -> str:
    column_set = set(columns)
    stem = path.stem.casefold()
    if "preprocess" in stem or "clean" in stem:
        return "cleaned_labeled" if set(LABEL_COLUMNS) <= column_set else "cleaned_unlabeled"
    if set(LABEL_COLUMNS) <= column_set:
        return "labeled"
    if set(BASE_COLUMNS) <= column_set:
        return "raw_unlabeled"
    return "unknown"


def read_delimited(path: Path) -> tuple[pd.DataFrame, list[list[str]], FileAudit]:
    raw = path.read_bytes()
    encoding = detect_encoding(raw)
    text = raw.decode(encoding)
    delimiter = detect_delimiter(text, path.suffix)
    with path.open("r", encoding=encoding, newline="") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path} is empty")
    columns = [value.strip() for value in rows[0]]
    data_rows = rows[1:]
    good_rows = [row for row in data_rows if len(row) == len(columns)]
    malformed_numbers = [
        number for number, row in enumerate(data_rows, start=2) if len(row) != len(columns)
    ]
    repeated_headers = sum(
        all(value.strip() == columns[index] for index, value in enumerate(row))
        for row in good_rows
    )
    frame = pd.DataFrame(good_rows, columns=columns, dtype=str)
    warnings: list[str] = []
    if malformed_numbers:
        warnings.append(
            f"{len(malformed_numbers)} record(s) have a field count different from the header"
        )
    if repeated_headers:
        warnings.append(f"{repeated_headers} repeated header record(s)")
    audit = FileAudit(
        filename=path.name,
        sha256=sha256_file(path),
        encoding=encoding,
        delimiter="TAB" if delimiter == "\t" else delimiter,
        columns=columns,
        logical_records=len(data_rows),
        well_formed_records=len(good_rows),
        malformed_records=len(malformed_numbers),
        malformed_record_numbers=malformed_numbers,
        repeated_header_records=repeated_headers,
        role=infer_role(path, columns),
        parsing_warnings=warnings,
    )
    malformed = [row for row in data_rows if len(row) != len(columns)]
    return frame, malformed, audit


def minimal_compare_normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    value = CONTROL_RE.sub("", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def duplicate_signature(value: str) -> str:
    normalized = minimal_compare_normalize(value)
    normalized = URL_RE.sub(" <url> ", normalized)
    normalized = "".join(char for char in normalized if char.isalnum() or char.isspace())
    return re.sub(r"\s+", " ", normalized).strip()


def valid_url(value: str) -> bool:
    try:
        parsed = urlparse(str(value).strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def text_profile(text: str) -> dict[str, Any]:
    text = str(text)
    bangla = sum("\u0980" <= char <= "\u09FF" for char in text)
    latin = sum(("a" <= char.lower() <= "z") for char in text)
    letters = sum(char.isalpha() for char in text)
    punctuation = sum(unicodedata.category(char).startswith("P") for char in text)
    emoji = sum(unicodedata.category(char) in {"So", "Sk"} for char in text)
    return {
        "characters": len(text),
        "tokens": len(TOKEN_RE.findall(text)),
        "bangla_characters": bangla,
        "latin_characters": latin,
        "bangla_letter_ratio": bangla / letters if letters else 0.0,
        "latin_letter_ratio": latin / letters if letters else 0.0,
        "contains_emoji": emoji > 0,
        "contains_url": bool(URL_RE.search(text)),
        "contains_number": any(char.isdigit() for char in text),
        "punctuation_characters": punctuation,
    }


def script_group(profile: pd.Series) -> str:
    bangla = float(profile["bangla_letter_ratio"])
    latin = float(profile["latin_letter_ratio"])
    if bangla >= 0.20 and latin >= 0.20:
        return "code_mixed"
    if bangla > latin and bangla > 0:
        return "bangla_script"
    if latin > 0:
        return "latin_script_proxy"
    return "other_or_no_letters"


def top_tokens(texts: Iterable[str], n: int = 10) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(token.casefold() for token in TOKEN_RE.findall(str(text)))
    return counts.most_common(n)


def top_character_ngrams(texts: Iterable[str], n: int = 10) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for text in texts:
        normalized = minimal_compare_normalize(str(text))
        counts.update(normalized[index : index + 3] for index in range(max(0, len(normalized) - 2)))
    return counts.most_common(n)


def table_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("|", "\\|").replace("\n", " ") for column in frame.columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def add_review(
    queue: list[dict[str, str]],
    source_file: str,
    record_id: str,
    target: str,
    original_value: str,
    problem: str,
    reason: str,
) -> None:
    queue.append(
        {
            "source_file": source_file,
            "record_id": record_id,
            "target": target,
            "original_value": original_value,
            "suspected_problem": problem,
            "suggested_review_reason": reason,
        }
    )


def audit_labels(frame: pd.DataFrame, filename: str, queue: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in LABEL_COLUMNS:
        if target not in frame:
            continue
        values = frame[target].astype(str)
        counts = values.value_counts(dropna=False).to_dict()
        folded: dict[str, set[str]] = {}
        for value in values.unique():
            folded.setdefault(value.strip().casefold(), set()).add(value)
        variants = {key: sorted(items) for key, items in folded.items() if len(items) > 1}
        invalid = sorted(set(values) - EXPECTED_LABELS[target])
        for _, row in frame[~values.isin(EXPECTED_LABELS[target])].iterrows():
            add_review(
                queue,
                filename,
                str(row.get("id", "")),
                target,
                str(row[target]),
                "missing_or_unrecognized_label",
                f"Value is outside the configured {target} vocabulary; do not remap automatically.",
            )
        result[target] = {
            "counts": {str(key): int(value) for key, value in counts.items()},
            "capitalization_or_spelling_variants": variants,
            "invalid_values": invalid,
            "rare_classes_under_10": {str(k): int(v) for k, v in counts.items() if int(v) < 10},
        }
    return result


def add_conflict_reviews(frame: pd.DataFrame, filename: str, queue: list[dict[str, str]]) -> dict[str, int]:
    duplicate_ids = frame[frame.duplicated("id", keep=False)] if "id" in frame else pd.DataFrame()
    id_conflicts = 0
    if not duplicate_ids.empty:
        for record_id, group in duplicate_ids.groupby("id", dropna=False):
            comparable = [column for column in ["text", *LABEL_COLUMNS] if column in group]
            if any(group[column].nunique(dropna=False) > 1 for column in comparable):
                id_conflicts += 1
                add_review(
                    queue,
                    filename,
                    str(record_id),
                    "record",
                    "duplicate ID",
                    "same_id_conflicting_values",
                    "Rows sharing this ID disagree on text or labels.",
                )
    text_label_conflicts = 0
    if "text" in frame and all(column in frame for column in LABEL_COLUMNS):
        normalized = frame.assign(_text_key=frame["text"].map(duplicate_signature))
        for text_key, group in normalized.groupby("_text_key", dropna=False):
            if not text_key or len(group) < 2:
                continue
            conflicting = [column for column in LABEL_COLUMNS if group[column].nunique() > 1]
            if conflicting:
                text_label_conflicts += 1
                for _, row in group.iterrows():
                    add_review(
                        queue,
                        filename,
                        str(row.get("id", "")),
                        ",".join(conflicting),
                        str({column: row[column] for column in conflicting}),
                        "same_text_conflicting_labels",
                        "Normalized-identical text has conflicting supervised labels.",
                    )
    return {
        "duplicate_id_rows": int(len(duplicate_ids)),
        "conflicting_id_groups": id_conflicts,
        "conflicting_text_groups": text_label_conflicts,
    }


def add_sentiment_heuristic_reviews(
    frame: pd.DataFrame, filename: str, queue: list[dict[str, str]]
) -> int:
    if not {"id", "text", "sentiment"} <= set(frame.columns):
        return 0
    negative_markers = re.compile(
        r"(?:\b(?:bad|worst|fraud|scam|problem|issue|bekar|kharap|baje|চোর|খারাপ|সমস্যা|প্রতার)\w*\b)",
        re.IGNORECASE,
    )
    flagged = 0
    for _, row in frame.iterrows():
        if row["sentiment"] in {"Neutral", "Positive"} and negative_markers.search(str(row["text"])):
            flagged += 1
            add_review(
                queue,
                filename,
                str(row["id"]),
                "sentiment",
                str(row["sentiment"]),
                "possible_sentiment_contradiction",
                "A conservative keyword heuristic found negative language. Human review is required; the label is not changed.",
            )
    return flagged


def save_figures(frame: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    if "text" in frame:
        lengths = frame["text"].astype(str).str.len()
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.hist(lengths, bins=30, color="#2563eb", edgecolor="white")
        axis.set(title="Comment length distribution", xlabel="Characters", ylabel="Records")
        fig.tight_layout()
        target = figures_dir / "text_length_characters.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        created.append(target.name)
    for target_column in LABEL_COLUMNS:
        if target_column not in frame:
            continue
        counts = frame[target_column].value_counts().sort_values(ascending=True)
        fig, axis = plt.subplots(figsize=(8, max(3.5, len(counts) * 0.48)))
        counts.plot.barh(ax=axis, color="#0f766e")
        axis.set(title=f"{target_column.title()} distribution", xlabel="Records", ylabel="")
        fig.tight_layout()
        target = figures_dir / f"{target_column}_distribution.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        created.append(target.name)
    return created


def reconcile(
    raw: pd.DataFrame | None,
    labeled: pd.DataFrame,
    cleaned: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, str]]]:
    reviews: list[dict[str, str]] = []
    labeled_copy = labeled.copy()
    for column in [*BASE_COLUMNS, *LABEL_COLUMNS]:
        if column not in labeled_copy:
            labeled_copy[column] = ""
    labeled_copy = labeled_copy[[*BASE_COLUMNS, *LABEL_COLUMNS]]
    raw_by_id = raw.set_index("id", drop=False) if raw is not None and "id" in raw else None
    clean_by_id = cleaned.set_index("id", drop=False) if cleaned is not None and "id" in cleaned else None
    rows: list[dict[str, Any]] = []
    matched_raw = 0
    metadata_conflicts = 0
    text_conflicts = 0
    for _, labeled_row in labeled_copy.iterrows():
        record_id = str(labeled_row["id"])
        raw_row = None
        if raw_by_id is not None and record_id in raw_by_id.index:
            candidate = raw_by_id.loc[record_id]
            if isinstance(candidate, pd.DataFrame):
                add_review(
                    reviews,
                    "tickets_raw.csv",
                    record_id,
                    "record",
                    "duplicate raw ID",
                    "ambiguous_raw_join",
                    "Multiple raw rows share this ID; no automatic one-to-many merge is allowed.",
                )
            else:
                raw_row = candidate
                matched_raw += 1
        raw_text = str(raw_row["text"]) if raw_row is not None else str(labeled_row["text"])
        if raw_row is not None:
            differing_metadata = [
                column
                for column in ["company", "source_platform", "source_url", "created_at"]
                if str(raw_row[column]) != str(labeled_row[column])
            ]
            if differing_metadata:
                metadata_conflicts += 1
                add_review(
                    reviews,
                    "cross_file_reconciliation",
                    record_id,
                    ",".join(differing_metadata),
                    "raw and labeled values differ",
                    "metadata_conflict",
                    "Cross-file metadata differs; preserve both sources and resolve manually.",
                )
            if raw_text != str(labeled_row["text"]):
                text_conflicts += 1
                add_review(
                    reviews,
                    "cross_file_reconciliation",
                    record_id,
                    "text",
                    str(labeled_row["text"]),
                    "raw_and_labeled_text_differ",
                    "Review whether the labeled text change is intentional; both versions are preserved.",
                )
        cleaned_text = ""
        if clean_by_id is not None and record_id in clean_by_id.index:
            candidate = clean_by_id.loc[record_id]
            if not isinstance(candidate, pd.DataFrame):
                cleaned_text = str(candidate["text"])
        output = {column: str(labeled_row[column]) for column in ["id", "company", "source_platform", "source_url"]}
        output.update(
            {
                "created_at_raw": str(labeled_row["created_at"]),
                "created_at_parsed": pd.NA,
                "text_raw": raw_text,
                "text_labeled": str(labeled_row["text"]),
                "text_cleaned": cleaned_text,
                "category": str(labeled_row["category"]),
                "sentiment": str(labeled_row["sentiment"]),
                "priority": str(labeled_row["priority"]),
                "source_record_role": "labeled_supervised",
            }
        )
        rows.append(output)
    processed = pd.DataFrame(rows)
    raw_ids = set(raw["id"].astype(str)) if raw is not None and "id" in raw else set()
    labeled_ids = set(labeled_copy["id"].astype(str))
    summary = {
        "labeled_rows": int(len(labeled_copy)),
        "matched_to_well_formed_raw": matched_raw,
        "unmatched_labeled_ids": sorted(labeled_ids - raw_ids),
        "well_formed_raw_ids_absent_from_labeled": sorted(raw_ids - labeled_ids),
        "metadata_conflicts": metadata_conflicts,
        "raw_vs_labeled_text_conflicts": text_conflicts,
        "cleaned_text_available": int(processed["text_cleaned"].ne("").sum()),
        "cleaned_text_changed_from_labeled": int(
            (processed["text_cleaned"].ne("" ) & processed["text_cleaned"].ne(processed["text_labeled"])).sum()
        ),
    }
    return processed, summary, reviews


def build_report(quality: dict[str, Any], primary: pd.DataFrame) -> str:
    lines = [
        "# BrandPulse-BD Data Audit",
        "",
        "This report was generated by `python -m src.data.audit`. Counts are computed from the inspected files; no source dataset was modified.",
        "",
        "## File structure",
        "",
    ]
    inventory = pd.DataFrame(
        [
            {
                "file": item["filename"],
                "logical records": item["logical_records"],
                "well formed": item["well_formed_records"],
                "malformed": item["malformed_records"],
                "role": item["role"],
                "encoding": item["encoding"],
            }
            for item in quality["files"]
        ]
    )
    lines.extend([table_markdown(inventory), "", "## Schema and record quality", ""])
    schema_rows = []
    for filename, details in quality["schema_quality"].items():
        schema_rows.append(
            {
                "file": filename,
                "duplicate ID rows": details["duplicate_id_rows"],
                "exact duplicate text rows": details["exact_duplicate_text_rows"],
                "normalized duplicate text rows": details["normalized_duplicate_text_rows"],
                "invalid URLs": details["invalid_urls"],
                "empty text": details["empty_text"],
                "short text (<3 chars)": details["extremely_short_text"],
            }
        )
    lines.extend([table_markdown(pd.DataFrame(schema_rows)), "", "## Reconciliation", ""])
    lines.append(table_markdown(pd.DataFrame([quality["reconciliation"]])))
    lines.extend(["", "The ten malformed raw rows are not silently repaired or added to the supervised dataset. They are listed in the review queue.", ""])
    lines.extend(["## Label distributions", ""])
    for target, details in quality["labels"].items():
        label_table = pd.DataFrame(
            [{"label": key, "count": value} for key, value in details["counts"].items()]
        )
        lines.extend([f"### {target.title()}", "", table_markdown(label_table), ""])
    lines.extend(["## Text profile", ""])
    text_summary = pd.DataFrame([quality["text_profile"]])
    lines.extend([table_markdown(text_summary), ""])
    lines.append("Latin-script is a proxy, not a verified English-language classification. Code-mixed status is based on observed Bangla and Latin character ratios.")
    lines.extend(["", "### Script groups", ""])
    script_table = pd.DataFrame(
        [{"group": key, "count": value} for key, value in quality["script_groups"].items()]
    )
    lines.extend([table_markdown(script_table), "", "### Top unigrams and character trigrams by category", ""])
    for category, values in quality["top_features_by_category"].items():
        lines.extend(
            [
                f"- **{category}** — unigrams: "
                + ", ".join(f"`{token}` ({count})" for token, count in values["unigrams"])
                + "; trigrams: "
                + ", ".join(f"`{token}` ({count})" for token, count in values["character_trigrams"])
            ]
        )
    lines.extend(
        [
            "",
            "## Date quality",
            "",
            f"Relative-like date values: **{quality['dates']['relative_like_values']}** of **{quality['dates']['nonempty_values']}** non-empty values. Of those, **{quality['dates']['recognized_relative_values']}** follow a recognized quantity/unit pattern.",
            "",
            "No reliable collection timestamp was found. The processed dataset preserves `created_at_raw` and leaves `created_at_parsed` null; time-trend analysis is not supported.",
            "",
            "## Label review queue",
            "",
            f"The generated review queue contains **{quality['review_queue_rows']}** issue rows. These are review suggestions only; no label was changed.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend([f"- `reports/figures/{name}`" for name in quality["figures"]])
    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- Malformed CSV fields are reported but not automatically reconstructed.",
            "- Normalized hashing detects formatting-level near duplicates; semantic embedding similarity is deferred because it is optional and would require an additional model.",
            "- Sentiment contradiction flags use a conservative keyword heuristic and require human review.",
            "- Public-comment status does not by itself establish redistribution rights.",
            "",
        ]
    )
    return "\n".join(lines)


def write_data_card(path: Path, quality: dict[str, Any], processed: pd.DataFrame) -> None:
    companies = sorted(processed["company"].dropna().astype(str).unique())
    platforms = sorted(processed["source_platform"].dropna().astype(str).unique())
    content = f"""# BrandPulse-BD Data Card

## Dataset summary

BrandPulse-BD contains public customer comments intended for supervised prediction of category, sentiment, and operational priority. Dataset version 1 has {len(processed):,} supervised records from {len(companies)} companies and {len(platforms)} source platforms.

Companies observed: {", ".join(companies)}.

Platforms observed: {", ".join(platforms)}.

## Sources and collection

The supplied CSV files contain comment text, company, platform, source URL, and relative creation-time strings. The precise collection process, collection timestamps, licenses, and platform permissions were not supplied and remain documentation gaps.

## Version 1 columns

- `id`: supplied record identifier
- `text_raw`: original text from a well-formed raw record when it can be joined unambiguously; otherwise labeled-source text
- `text_labeled`: text supplied with human labels
- `text_cleaned`: separately supplied preprocessed text, retained for comparison
- `company`, `source_platform`, `source_url`: supplied provenance metadata
- `created_at_raw`: original relative or absolute date string
- `created_at_parsed`: null because no trustworthy reference collection timestamp was found
- `category`, `sentiment`, `priority`: supplied supervised labels
- `source_record_role`: provenance marker for the selected supervised row

## Transformations and exclusions

No source file was modified. The labeled file was used as the supervised row set. Well-formed raw records were reconciled by unique ID, while both raw and labeled text were preserved when they differed. Preprocessed text was retained in a separate column. Ten malformed raw records were excluded from version 1 because their fields could not be reconstructed safely; they are documented in `reports/label_review_queue.csv` and `reports/data_quality.json`. Relative dates were not converted.

## Label definitions

Observed category labels: {", ".join(sorted(EXPECTED_LABELS['category']))}.

Observed sentiment labels: Negative, Neutral, and Positive.

Observed priority labels: Low, Medium, and High. Priority is operationally ordered, but the supplied annotation policy was not provided and must be clarified before strong business interpretations.

## Quality and limitations

- `Abuse/Harassment` is an extremely rare class and has only {quality['labels']['category']['counts'].get('Abuse/Harassment', 0)} records.
- Raw CSV malformation and cross-file text differences require traceable human review.
- Relative timestamps cannot support time trends without a collection reference date.
- Language/script grouping is heuristic; Latin-script text is not guaranteed to be English.
- The dataset may contain duplicated, spam-like, noisy, sarcastic, or context-dependent comments.
- Labels are treated as supplied annotations, not objective ground truth.

## Privacy and redistribution

Comments and source URLs may contain personal or account-related information. Before publication, perform a dedicated PII review and release only a small anonymized sample unless licensing and platform terms clearly permit full redistribution. Never commit credentials or private annotations.

## Intended and out-of-scope use

The intended use is portfolio research and human-assisted customer-feedback triage. Predictions must not be used for punitive decisions or fully automated high-impact actions. High-priority and low-confidence cases require human review.
"""
    path.write_text(content, encoding="utf-8")


def run_audit(input_dir: Path, output_dir: Path, processed_dir: Path) -> dict[str, Any]:
    paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in DATA_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No CSV or TSV files found directly under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    frames: dict[str, pd.DataFrame] = {}
    malformed_by_file: dict[str, list[list[str]]] = {}
    file_audits: list[FileAudit] = []
    for path in paths:
        frame, malformed, file_audit = read_delimited(path)
        frames[path.name] = frame
        malformed_by_file[path.name] = malformed
        file_audits.append(file_audit)
        LOGGER.info("Audited %s: %d well-formed rows", path.name, len(frame))
    initial_hashes = {item.filename: item.sha256 for item in file_audits}
    labeled_candidates = [item for item in file_audits if item.role == "labeled"]
    if not labeled_candidates:
        raise ValueError("No labeled dataset containing category, sentiment, and priority was found")
    labeled_meta = sorted(labeled_candidates, key=lambda item: ("labeled" not in item.filename.casefold(), item.filename))[0]
    cleaned_candidates = [item for item in file_audits if item.role == "cleaned_labeled"]
    raw_candidates = [item for item in file_audits if item.role == "raw_unlabeled"]
    labeled = frames[labeled_meta.filename]
    cleaned = frames[cleaned_candidates[0].filename] if cleaned_candidates else None
    raw = frames[raw_candidates[0].filename] if raw_candidates else None
    review_queue: list[dict[str, str]] = []
    for file_meta in file_audits:
        for malformed in malformed_by_file[file_meta.filename]:
            record_id = malformed[0].strip() if malformed else ""
            add_review(
                review_queue,
                file_meta.filename,
                record_id,
                "record",
                json.dumps(malformed, ensure_ascii=False),
                "malformed_field_count",
                f"Record has {len(malformed)} fields but header has {len(file_meta.columns)}; do not shift fields automatically.",
            )
    schema_quality: dict[str, Any] = {}
    conflict_quality: dict[str, Any] = {}
    for filename, frame in frames.items():
        missing_columns = [column for column in BASE_COLUMNS if column not in frame]
        empty_counts = {
            column: int(frame[column].astype(str).str.strip().eq("").sum())
            for column in frame.columns
        }
        repeated_header_mask = pd.Series(False, index=frame.index)
        if len(frame.columns):
            repeated_header_mask = frame.apply(
                lambda row: all(str(row[column]).strip() == column for column in frame.columns), axis=1
            )
        exact_dupes = int(frame["text"].duplicated(keep=False).sum()) if "text" in frame else 0
        normalized_dupes = (
            int(frame["text"].map(duplicate_signature).duplicated(keep=False).sum())
            if "text" in frame
            else 0
        )
        invalid_urls = (
            int((~frame["source_url"].astype(str).map(valid_url)).sum())
            if "source_url" in frame
            else 0
        )
        duplicate_url_rows = (
            int(frame["source_url"].duplicated(keep=False).sum())
            if "source_url" in frame
            else 0
        )
        schema_quality[filename] = {
            "missing_required_columns": missing_columns,
            "missing_or_empty_by_column": empty_counts,
            "empty_text": empty_counts.get("text", 0),
            "extremely_short_text": int(frame["text"].astype(str).str.strip().str.len().lt(3).sum()) if "text" in frame else 0,
            "repeated_header_rows": int(repeated_header_mask.sum()),
            "duplicate_id_rows": int(frame["id"].duplicated(keep=False).sum()) if "id" in frame else 0,
            "exact_duplicate_text_rows": exact_dupes,
            "normalized_duplicate_text_rows": normalized_dupes,
            "invalid_urls": invalid_urls,
            "duplicate_source_url_rows": duplicate_url_rows,
        }
        conflict_quality[filename] = add_conflict_reviews(frame, filename, review_queue)
    label_quality = audit_labels(labeled, labeled_meta.filename, review_queue)
    heuristic_flags = add_sentiment_heuristic_reviews(labeled, labeled_meta.filename, review_queue)
    processed, reconciliation, reconciliation_reviews = reconcile(raw, labeled, cleaned)
    review_queue.extend(reconciliation_reviews)
    profiles = pd.DataFrame([text_profile(text) for text in labeled["text"].astype(str)])
    profiles["script_group"] = profiles.apply(script_group, axis=1)
    text_summary = {
        "character_length_min": int(profiles["characters"].min()),
        "character_length_median": float(profiles["characters"].median()),
        "character_length_p95": float(profiles["characters"].quantile(0.95)),
        "character_length_max": int(profiles["characters"].max()),
        "token_length_median": float(profiles["tokens"].median()),
        "contains_emoji": int(profiles["contains_emoji"].sum()),
        "contains_url": int(profiles["contains_url"].sum()),
        "contains_number": int(profiles["contains_number"].sum()),
        "punctuation_characters": int(profiles["punctuation_characters"].sum()),
    }
    feature_summary: dict[str, Any] = {}
    for category, group in labeled.groupby("category", dropna=False):
        feature_summary[str(category)] = {
            "unigrams": top_tokens(group["text"].astype(str)),
            "character_trigrams": top_character_ngrams(group["text"].astype(str)),
        }
    company_label_vocabularies = {
        str(company): {
            target: sorted(group[target].astype(str).unique().tolist())
            for target in LABEL_COLUMNS
        }
        for company, group in labeled.groupby("company", dropna=False)
    }
    source_dates = labeled["created_at"].astype(str)
    figures = save_figures(labeled, figures_dir)
    quality: dict[str, Any] = {
        "audit_version": "1.0.0",
        "source_files_unchanged": True,
        "files": [asdict(item) for item in file_audits],
        "primary_labeled_file": labeled_meta.filename,
        "schema_quality": schema_quality,
        "conflicts": conflict_quality,
        "labels": label_quality,
        "company_specific_label_vocabularies": company_label_vocabularies,
        "possible_sentiment_contradictions": heuristic_flags,
        "reconciliation": reconciliation,
        "text_profile": text_summary,
        "script_groups": {str(k): int(v) for k, v in profiles["script_group"].value_counts().to_dict().items()},
        "top_features_by_category": feature_summary,
        "dates": {
            "nonempty_values": int(source_dates.str.strip().ne("").sum()),
            "relative_like_values": int(
                source_dates.map(lambda value: bool(RELATIVE_LIKE_DATE_RE.search(value))).sum()
            ),
            "recognized_relative_values": int(
                source_dates.map(lambda value: bool(RELATIVE_DATE_RE.match(value))).sum()
            ),
            "unrecognized_relative_values": sorted(
                source_dates[
                    source_dates.map(lambda value: bool(RELATIVE_LIKE_DATE_RE.search(value)))
                    & ~source_dates.map(lambda value: bool(RELATIVE_DATE_RE.match(value)))
                ].unique().tolist()
            ),
            "created_at_parsed_policy": "null_without_trustworthy_collection_timestamp",
        },
        "figures": figures,
        "review_queue_rows": len(review_queue),
        "transformations": [
            "Selected the explicit labeled dataset as the supervised row universe.",
            "Joined only unique, well-formed raw rows by exact ID.",
            "Preserved raw, labeled, and supplied-cleaned text in separate columns.",
            "Renamed created_at to created_at_raw in generated data and left created_at_parsed null.",
            "Excluded malformed raw-only rows from supervised version 1 and queued them for review.",
            "Did not alter labels or source files.",
        ],
    }
    final_hashes = {path.name: sha256_file(path) for path in paths}
    quality["source_files_unchanged"] = initial_hashes == final_hashes
    quality["source_hashes_before"] = initial_hashes
    quality["source_hashes_after"] = final_hashes
    if not quality["source_files_unchanged"]:
        raise RuntimeError("A source dataset changed while the audit was running")
    review_columns = [
        "source_file",
        "record_id",
        "target",
        "original_value",
        "suspected_problem",
        "suggested_review_reason",
    ]
    pd.DataFrame(review_queue, columns=review_columns).to_csv(
        output_dir / "label_review_queue.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "data_audit.md").write_text(build_report(quality, labeled), encoding="utf-8")
    processed.to_csv(processed_dir / "dataset_version_1.csv", index=False, encoding="utf-8-sig")
    processed.to_parquet(processed_dir / "dataset_version_1.parquet", index=False)
    write_data_card(input_dir / "data_card.md", quality, processed)
    return quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    quality = run_audit(args.input_dir.resolve(), args.output_dir.resolve(), args.processed_dir.resolve())
    LOGGER.info(
        "Audit complete: %d source files, %d review queue rows",
        len(quality["files"]),
        quality["review_queue_rows"],
    )


if __name__ == "__main__":
    main()
