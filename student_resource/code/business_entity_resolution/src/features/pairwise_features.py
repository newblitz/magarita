"""Phase 7 -- Pairwise feature engineering.

Computes the full feature vector for every row in the fused candidate set.
Several features (MinHash Jaccard, embedding cosine) are recomputed for EVERY
candidate pair regardless of which tier originally surfaced it -- a pair found
only by Tier 0 still needs its lexical/semantic scores filled in, or the
classifier will overfit to ``source_channel`` as a provenance shortcut instead
of learning from the actual similarity signal.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.normalize import extract_house_number

_QGRAM_SIZE_DEFAULT = 4
_POSTAL_RE = re.compile(r"\b\d{4,6}\b")
_NUMERIC_TOKEN_RE = re.compile(r"\d+")


def _qgram_set(text: str, q: int) -> set[str]:
    text = text or ""
    if len(text) < q:
        return {text} if text else set()
    return {text[i : i + q] for i in range(len(text) - q + 1)}


def _char_ngram_cosine(a: str, b: str, q: int = _QGRAM_SIZE_DEFAULT) -> float:
    ga, gb = _qgram_set(a, q), _qgram_set(b, q)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    return inter / ((len(ga) ** 0.5) * (len(gb) ** 0.5)) if ga and gb else 0.0


def _minhash_jaccard_exact(a: str, b: str, q: int = _QGRAM_SIZE_DEFAULT) -> float:
    """Exact q-gram Jaccard, used as the always-present lexical feature (Tier 1's
    MinHash is an *approximation* of this same quantity; recomputing it exactly
    here keeps the feature well-defined for every pair, not only shortlisted ones)."""
    ga, gb = _qgram_set(a, q), _qgram_set(b, q)
    if not ga and not gb:
        return 1.0
    union = len(ga | gb)
    return (len(ga & gb) / union) if union else 0.0


def _token_set(text: str) -> set[str]:
    return set((text or "").split())


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta and not tb:
        return 1.0
    union = len(ta | tb)
    return (len(ta & tb) / union) if union else 0.0


def _token_overlap(a: str, b: str) -> int:
    return len(_token_set(a) & _token_set(b))


def _numeric_tokens(text: str) -> set[str]:
    return set(_NUMERIC_TOKEN_RE.findall(text or ""))


def _postal_code(text: str) -> str | None:
    matches = _POSTAL_RE.findall(text or "")
    return matches[-1] if matches else None


def compute_features(candidate_row: dict, s1_record: dict, s2s3_record: dict) -> dict:
    """Compute the full feature dict for one (S1, S2/S3) candidate pair.

    ``candidate_row`` carries fusion-stage metadata (source_channel, jaccard_score,
    cosine_score, n_channels). ``s1_record``/``s2s3_record`` carry the normalized
    fields produced by :func:`normalize.normalize_dataframe`.
    """
    name1, name2 = s1_record.get("norm_name", ""), s2s3_record.get("norm_name", "")
    addr1, addr2 = s1_record.get("norm_address", ""), s2s3_record.get("norm_address", "")
    country1, country2 = s1_record.get("norm_country", ""), s2s3_record.get("norm_country", "")

    # --- Name features ---
    name_exact_match = int(name1 == name2 and name1 != "")
    name_lev_sim = Levenshtein.normalized_similarity(name1, name2) if (name1 or name2) else 1.0
    name_jw_sim = JaroWinkler.normalized_similarity(name1, name2) if (name1 or name2) else 1.0
    name_char_ngram_cosine = _char_ngram_cosine(name1, name2)
    name_minhash_jaccard = _minhash_jaccard_exact(name1, name2)
    name_token_jaccard = _token_jaccard(name1, name2)
    name_token_overlap = _token_overlap(name1, name2)
    name_prefix_agreement = int(
        s1_record.get("name_prefix", "") == s2s3_record.get("name_prefix", "")
        and s1_record.get("name_prefix", "") != ""
    )
    name_len_diff = abs(len(name1) - len(name2))

    # --- Address features ---
    addr_exact_match = int(addr1 == addr2 and addr1 != "")
    addr_char_sim = Levenshtein.normalized_similarity(addr1, addr2) if (addr1 or addr2) else 1.0
    addr_token_jaccard = _token_jaccard(addr1, addr2)
    hn1 = s1_record.get("house_number") or extract_house_number(addr1)
    hn2 = s2s3_record.get("house_number") or extract_house_number(addr2)
    house_number_agreement = int(bool(hn1) and bool(hn2) and hn1 == hn2)
    postal1, postal2 = _postal_code(addr1), _postal_code(addr2)
    postal_agreement = int(bool(postal1) and bool(postal2) and postal1 == postal2)
    numeric_token_overlap = len(_numeric_tokens(addr1) & _numeric_tokens(addr2))
    addr_len_diff = abs(len(addr1) - len(addr2))
    address_missing_s2s3 = int(addr2 == "")

    # --- Retrieval-channel features ---
    embedding_cosine = candidate_row.get("cosine_score")
    if embedding_cosine is None or (isinstance(embedding_cosine, float) and np.isnan(embedding_cosine)):
        embedding_cosine = 0.0
    minhash_jaccard_retrieval = candidate_row.get("jaccard_score")
    if minhash_jaccard_retrieval is None or (
        isinstance(minhash_jaccard_retrieval, float) and np.isnan(minhash_jaccard_retrieval)
    ):
        minhash_jaccard_retrieval = name_minhash_jaccard
    source_channel = candidate_row.get("source_channel", "unknown")
    n_channels_agreeing = candidate_row.get("n_channels", 1)
    is_high_conf = int("high_conf" in str(source_channel))

    # --- Metadata ---
    country_match = int(country1 == country2 and country1 != "")
    source_indicator = 1 if str(s2s3_record.get("entity_id", "")).startswith("S2-") else 0
    name_len1, name_len2 = len(name1), len(name2)
    addr_len1, addr_len2 = len(addr1), len(addr2)

    # --- Interactions ---
    name_sim = name_jw_sim
    addr_sim = addr_char_sim
    name_addr_sim_product = name_sim * addr_sim
    house_number_match_name_sim = house_number_agreement * name_sim
    exact_name_and_partial_address = name_exact_match * addr_token_jaccard
    high_conf_interaction = is_high_conf * name_sim

    return {
        "source1_entity_id": candidate_row.get("source1_entity_id"),
        "candidate_entity_id": candidate_row.get("candidate_entity_id"),
        # name
        "name_exact_match": name_exact_match,
        "name_levenshtein_sim": name_lev_sim,
        "name_jaro_winkler_sim": name_jw_sim,
        "name_char_ngram_cosine": name_char_ngram_cosine,
        "name_minhash_jaccard": name_minhash_jaccard,
        "name_token_jaccard": name_token_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_prefix_agreement": name_prefix_agreement,
        "name_len_diff": name_len_diff,
        # address
        "addr_exact_match": addr_exact_match,
        "addr_char_sim": addr_char_sim,
        "addr_token_jaccard": addr_token_jaccard,
        "house_number_agreement": house_number_agreement,
        "postal_agreement": postal_agreement,
        "numeric_token_overlap": numeric_token_overlap,
        "addr_len_diff": addr_len_diff,
        "address_missing_s2s3": address_missing_s2s3,
        # retrieval channel
        "embedding_cosine": float(embedding_cosine),
        "minhash_jaccard_retrieval": float(minhash_jaccard_retrieval),
        "source_channel": source_channel,
        "n_channels_agreeing": n_channels_agreeing,
        "is_high_conf": is_high_conf,
        # metadata
        "country_match": country_match,
        "source_indicator": source_indicator,
        "name_len1": name_len1,
        "name_len2": name_len2,
        "addr_len1": addr_len1,
        "addr_len2": addr_len2,
        # interactions
        "name_addr_sim_product": name_addr_sim_product,
        "house_number_match_name_sim": house_number_match_name_sim,
        "exact_name_and_partial_address": exact_name_and_partial_address,
        "high_conf_interaction": high_conf_interaction,
    }


FEATURE_COLUMNS = [
    "name_exact_match",
    "name_levenshtein_sim",
    "name_jaro_winkler_sim",
    "name_char_ngram_cosine",
    "name_minhash_jaccard",
    "name_token_jaccard",
    "name_token_overlap",
    "name_prefix_agreement",
    "name_len_diff",
    "addr_exact_match",
    "addr_char_sim",
    "addr_token_jaccard",
    "house_number_agreement",
    "postal_agreement",
    "numeric_token_overlap",
    "addr_len_diff",
    "address_missing_s2s3",
    "embedding_cosine",
    "minhash_jaccard_retrieval",
    "n_channels_agreeing",
    "is_high_conf",
    "country_match",
    "source_indicator",
    "name_len1",
    "name_len2",
    "addr_len1",
    "addr_len2",
    "name_addr_sim_product",
    "house_number_match_name_sim",
    "exact_name_and_partial_address",
    "high_conf_interaction",
]
CATEGORICAL_COLUMNS = ["source_channel"]


def build_feature_matrix(
    fused_df: pd.DataFrame, s1_df: pd.DataFrame, other_df: pd.DataFrame
) -> pd.DataFrame:
    """Vectorization note: joins the fused candidate frame against the S1 and
    S2/S3 record tables once (dict lookups keyed by entity_id), then computes
    features via a single zip-based loop -- not ``DataFrame.iterrows()``, which
    the EDA report flags as the dominant bottleneck at multi-million-row scale.
    """
    s1_lookup = s1_df.set_index("entity_id").to_dict(orient="index")
    other_lookup = other_df.set_index("entity_id").to_dict(orient="index")

    rows = []
    for rec in fused_df.to_dict(orient="records"):
        s1_id = rec["source1_entity_id"]
        cand_id = rec["candidate_entity_id"]
        s1_record = s1_lookup.get(s1_id)
        s2s3_record = other_lookup.get(cand_id)
        if s1_record is None or s2s3_record is None:
            continue
        s1_record = {**s1_record, "entity_id": s1_id}
        s2s3_record = {**s2s3_record, "entity_id": cand_id}
        rows.append(compute_features(rec, s1_record, s2s3_record))

    return pd.DataFrame(rows)
