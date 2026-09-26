"""Phase 1 -- Normalization.

Reuses the normalization already validated in the EDA report. Normalization is
applied ONCE, at load time, to every record in S1/S2/S3 (train and test); the
normalized fields are persisted alongside the raw ones so downstream modules never
re-normalize inside a hot loop.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_LEADING_NUMBER_RE = re.compile(r"(\d+[\w/-]*)")


def normalize(text: str, unicode_form: str = "NFKC", lowercase: bool = True,
              strip_punct: bool = True) -> str:
    """Unicode-normalize, lowercase and strip punctuation from ``text``."""
    text = str(text).strip()
    if lowercase:
        text = text.lower()
    text = unicodedata.normalize(unicode_form, text)
    if strip_punct:
        text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def extract_house_number(address: str) -> str | None:
    """Return the first numeric token in an (already normalized) address, or None."""
    if not address:
        return None
    match = _LEADING_NUMBER_RE.search(address)
    return match.group(1) if match else None


def extract_name_prefix(name: str, length: int = 4) -> str:
    """First ``length`` chars of the normalized name with spaces removed."""
    if not name:
        return ""
    compact = name.replace(" ", "")
    return compact[:length]


# Maps the logical field names used in config.yaml (tier1_lexical.fields,
# tier2_semantic.fields) to the actual normalized dataframe columns produced by
# normalize_dataframe() below.
FIELD_COLUMN_MAP = {
    "name": "norm_name",
    "name_address": "name_address",
}


def field_column(field: str) -> str:
    return FIELD_COLUMN_MAP.get(field, field)


def normalize_dataframe(
    df: pd.DataFrame, config: dict, keep_raw: bool = True
) -> pd.DataFrame:
    """Add normalized columns to a source dataframe (S1/S2/S3).

    Adds: norm_name, norm_address, name_address, house_number, name_prefix.
    Set ``keep_raw=False`` for blocking phases that do not need the original
    free-text columns after normalization.
    """
    norm_cfg = config.get("normalization", {})
    unicode_form = norm_cfg.get("unicode_form", "NFKC")
    lowercase = norm_cfg.get("lowercase", True)
    strip_punct = norm_cfg.get("strip_punct", True)

    out = df.copy()
    out["norm_name"] = out["business_name"].map(
        lambda x: normalize(x, unicode_form, lowercase, strip_punct)
    )
    out["norm_address"] = out["business_address"].map(
        lambda x: normalize(x, unicode_form, lowercase, strip_punct)
    )
    out["name_address"] = (out["norm_name"] + " " + out["norm_address"]).str.strip()
    out["house_number"] = out["norm_address"].map(extract_house_number)
    out["name_prefix"] = out["norm_name"].map(extract_name_prefix)
    # country is treated as an open-set string; only whitespace/case-normalized,
    # never mapped to a fixed enum (France, and any other unseen value, must pass
    # through untouched apart from this light normalization).
    out["norm_country"] = out["country"].map(lambda x: normalize(x, unicode_form, True, False))
    if not keep_raw:
        out = out.drop(
            columns=["business_name", "business_address", "norm_address", "name_address", "name_prefix"]
        )
    return out
