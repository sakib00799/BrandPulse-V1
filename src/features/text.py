"""Minimal multilingual normalization that preserves sentiment-bearing signals."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

INVISIBLE_AND_CONTROL_RE = re.compile(
    r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F"
    r"\u00AD\u061C\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]"
)
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?88)?01[3-9](?:[\s-]?\d){8}(?!\w)")
ORDER_ID_RE = re.compile(
    r"(?i)\b(?:order|invoice|account|acct|a/c|transaction|trx|tx|ref)\s*(?:id|no|number|#)?\s*[:#-]?\s*[A-Z0-9-]{5,}\b"
)
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationConfig:
    """Options are explicit so preprocessing behavior can be versioned."""

    replace_urls: bool = True
    replace_emails: bool = True
    replace_phone_numbers: bool = True
    replace_order_ids: bool = True


def normalize_text(text: str, config: NormalizationConfig | None = None) -> str:
    """Normalize safely while retaining punctuation, emoji, case, and negation."""

    config = config or NormalizationConfig()
    value = unicodedata.normalize("NFC", str(text))
    value = INVISIBLE_AND_CONTROL_RE.sub("", value)
    if config.replace_urls:
        value = URL_RE.sub("<URL>", value)
    if config.replace_emails:
        value = EMAIL_RE.sub("<EMAIL>", value)
    if config.replace_phone_numbers:
        value = PHONE_RE.sub("<PHONE>", value)
    if config.replace_order_ids:
        value = ORDER_ID_RE.sub("<ORDER_ID>", value)
    return WHITESPACE_RE.sub(" ", value).strip()
