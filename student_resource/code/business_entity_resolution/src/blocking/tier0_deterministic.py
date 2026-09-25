"""Phase 2 -- Tier 0 deterministic block.

Straight port of the EDA's Block A: (country, house_number, exact normalized name).
Implemented as a hash-join (pandas merge on the composite key), not a nested loop,
so it runs in minutes even at 2M+ S1 rows. Its role in the fused pipeline is to be
tagged ``source_channel = high_conf`` and fed to the classifier as ~98%-precise
evidence, not thresholded away.
"""

from __future__ import annotations

import pandas as pd


def _build_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["norm_country"].fillna("")
        + "\u0001"
        + df["house_number"].fillna("")
        + "\u0001"
        + df["norm_name"].fillna("")
    )


def tier0_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """Returns columns: source1_entity_id, candidate_entity_id, source_channel='high_conf'.

    Key: (country, house_number, exact_normalized_name). Expects ``s1_df``/``s2_df``/
    ``s3_df`` to already carry the normalized columns from :func:`normalize.normalize_dataframe`.
    """
    frames = []
    for other_df in (s2_df, s3_df):
        if other_df is None or len(other_df) == 0:
            continue
        s1_keyed = s1_df.assign(_key=_build_key(s1_df))
        other_keyed = other_df.assign(_key=_build_key(other_df))
        # Only keys with a non-empty house number and non-empty name are meaningful;
        # an empty key would spuriously join every missing-field record together.
        valid = (
            (s1_keyed["house_number"].fillna("") != "")
            & (s1_keyed["norm_name"].fillna("") != "")
        )
        s1_valid = s1_keyed.loc[valid, ["entity_id", "_key"]]
        other_valid = other_keyed.loc[
            (other_keyed["house_number"].fillna("") != "") & (other_keyed["norm_name"].fillna("") != ""),
            ["entity_id", "_key"],
        ]
        merged = s1_valid.merge(other_valid, on="_key", suffixes=("_s1", "_other"))
        if len(merged) == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "source1_entity_id": merged["entity_id_s1"],
                    "candidate_entity_id": merged["entity_id_other"],
                    "source_channel": "high_conf",
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "source_channel"])

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
    return result.reset_index(drop=True)
