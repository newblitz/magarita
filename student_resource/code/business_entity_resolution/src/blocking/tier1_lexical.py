"""Phase 3 -- Tier 1 lexical retrieval (MinHash/LSH, the LSBlock lexical channel).

For each of ``name`` and ``name_address`` (config: tier1_lexical.fields):
1. Character q-grams (config: qgram_size) per record.
2. MinHash signature (config: minhash_permutations) over the q-gram set, via
   ``datasketch`` (MIT licensed).
3. An LSH index (banding, config: lsh_bands), built PER COUNTRY PARTITION, PER
   FIELD -- i.e. one LSH index per (country, field) pair, over the union of
   S2/S3 signatures for that country.
4. For every S1 record in that partition, query the LSH index for a shortlist,
   then compute *exact* Jaccard similarity only within that shortlist.

No acceptance threshold is applied here -- every shortlisted pair is returned with
its raw Jaccard score; thresholding happens in Phase 5/6 using a threshold learned
from labeled data.
"""

from __future__ import annotations

import gc
from typing import Iterable

import pandas as pd
from datasketch import MinHash, MinHashLSH

from src.normalize import field_column


def _qgrams(text: str, q: int) -> set[str]:
    text = text or ""
    if len(text) < q:
        return {text} if text else set()
    return {text[i : i + q] for i in range(len(text) - q + 1)}


def _build_minhash(qgrams: set[str], num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for g in qgrams:
        mh.update(g.encode("utf8"))
    return mh


def _rows_per_band(num_perm: int, bands: int) -> int:
    rows = max(1, num_perm // bands)
    return rows


def tier1_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    country_value: str,
    config: dict,
) -> pd.DataFrame:
    """Returns columns: source1_entity_id, candidate_entity_id, field, jaccard_score,
    source_channel='lexical'.

    Callable independently per country partition, including a country value never
    seen at training time (open-set requirement) -- there is no country-specific
    branching here, only a string-equality filter on ``norm_country``.
    """
    cfg = config["tier1_lexical"]
    q = cfg["qgram_size"]
    num_perm = cfg["minhash_permutations"]
    bands = cfg["lsh_bands"]
    fields = cfg["fields"]
    rows_per_band = _rows_per_band(num_perm, bands)

    s1_part = s1_df[s1_df["norm_country"] == country_value]
    s2_part = s2_df[s2_df["norm_country"] == country_value] if s2_df is not None else s2_df.iloc[0:0]
    s3_part = s3_df[s3_df["norm_country"] == country_value] if s3_df is not None else s3_df.iloc[0:0]

    other_parts = []
    if s2_part is not None and len(s2_part):
        other_parts.append(s2_part)
    if s3_part is not None and len(s3_part):
        other_parts.append(s3_part)
    if not other_parts or len(s1_part) == 0:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "jaccard_score", "source_channel"]
        )
    other_df = pd.concat(other_parts, ignore_index=True)

    result_frames = []
    for field in fields:
        lsh = MinHashLSH(num_perm=num_perm, params=(bands, rows_per_band))
        column = field_column(field)
        other_qgrams: dict[str, set[str]] = {}
        other_minhash: dict[str, MinHash] = {}
        for eid, text in zip(other_df["entity_id"], other_df[column]):
            qg = _qgrams(text, q)
            if not qg:
                continue
            mh = _build_minhash(qg, num_perm)
            other_qgrams[eid] = qg
            other_minhash[eid] = mh
            lsh.insert(eid, mh)

        if not other_minhash:
            # Nothing to query — drop these (possibly large) dicts early.
            del lsh, other_minhash, other_qgrams
            gc.collect()
            continue

        rows = []
        for eid, text in zip(s1_part["entity_id"], s1_part[column]):
            qg = _qgrams(text, q)
            if not qg:
                continue
            mh = _build_minhash(qg, num_perm)
            shortlist = lsh.query(mh)
            for cand_id in shortlist:
                cand_qg = other_qgrams[cand_id]
                inter = len(qg & cand_qg)
                union = len(qg | cand_qg)
                jaccard = inter / union if union else 0.0
                rows.append((eid, cand_id, field, jaccard))

        # Free the large LSH index and MinHash dicts before the next field.
        del lsh, other_minhash, other_qgrams
        gc.collect()

        if rows:
            result_frames.append(
                pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "field", "jaccard_score"])
            )


    if not result_frames:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "jaccard_score", "source_channel"]
        )

    out = pd.concat(result_frames, ignore_index=True)
    out["source_channel"] = "lexical"
    return out


def tier1_candidates_all_countries(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Convenience wrapper: run :func:`tier1_candidates` over every country present
    in ``s1_df`` (open-set -- whatever string values are actually present).

    Processes one country at a time and calls ``gc.collect()`` between iterations
    so that each country's MinHash/LSH structures are freed before the next one
    is built -- critical for 13 GB RAM with large per-country partitions.
    """
    countries: Iterable[str] = s1_df["norm_country"].unique()
    frames = []
    for c in countries:
        frame = tier1_candidates(s1_df, s2_df, s3_df, c, config)
        if len(frame):
            frames.append(frame)
        # Trigger GC to release cyclic garbage from this country's processing.
        gc.collect()
    if not frames:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "jaccard_score", "source_channel"]
        )
    return pd.concat(frames, ignore_index=True)
