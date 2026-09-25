"""Phase 6 -- Candidate fusion.

Fuses Tier 0 (unfiltered, ~98% precise by construction), Tier 1 (lexical,
filtered by its learned threshold) and Tier 2 (semantic, filtered by its
learned threshold) into the literal set written to ``candidate_pairs.tsv``.
This must be regenerated from this exact function in the same pipeline run
that feeds the classifier -- never hand-patched or produced by a separate
"close enough" script.
"""

from __future__ import annotations

import pandas as pd

from src.utils.io import CANDIDATE_COLUMNS, write_id_list_tsv


def _filter_tier1(tier1_df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    if tier1_df is None or tier1_df.empty:
        return tier1_df
    keep = tier1_df.apply(
        lambda r: r["jaccard_score"] >= thresholds.get(r["field"], thresholds.get("default", 1.0)),
        axis=1,
    )
    return tier1_df[keep]


def _filter_tier2(tier2_df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    if tier2_df is None or tier2_df.empty:
        return tier2_df
    keep = tier2_df.apply(
        lambda r: r["cosine_score"] >= thresholds.get(r["field"], thresholds.get("default", 1.0)),
        axis=1,
    )
    return tier2_df[keep]


def fuse(
    tier0_df: pd.DataFrame,
    tier1_df: pd.DataFrame,
    tier2_df: pd.DataFrame,
    learned_thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """1. Filter tier1/tier2 by their learned thresholds; tier0 is NOT filtered.
    2. Union all three on (source1_entity_id, candidate_entity_id).
    3. Merge duplicated pairs into one row; ``source_channel`` reflects ALL
       contributing tiers (e.g. "lexical+semantic"); raw scores kept as
       separate columns (jaccard_score, cosine_score), never collapsed.
    4. Deduplicate strictly.

    Returns a dataframe with columns: source1_entity_id, candidate_entity_id,
    source_channel, jaccard_score, cosine_score, n_channels.
    """
    tier1_thresholds = learned_thresholds.get("lexical", {})
    tier2_thresholds = learned_thresholds.get("semantic", {})

    t1 = _filter_tier1(tier1_df, tier1_thresholds)
    t2 = _filter_tier2(tier2_df, tier2_thresholds)

    pieces = []
    if tier0_df is not None and len(tier0_df):
        p = tier0_df[["source1_entity_id", "candidate_entity_id"]].copy()
        p["channel_high_conf"] = True
        pieces.append(p)
    if t1 is not None and len(t1):
        # Collapse per-field lexical scores to the best (max) Jaccard per pair.
        agg = t1.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)[
            "jaccard_score"
        ].max()
        agg["channel_lexical"] = True
        pieces.append(agg)
    if t2 is not None and len(t2):
        agg = t2.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)[
            "cosine_score"
        ].max()
        agg["channel_semantic"] = True
        pieces.append(agg)

    if not pieces:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "source_channel",
                "jaccard_score",
                "cosine_score",
                "n_channels",
            ]
        )

    fused = pieces[0]
    for p in pieces[1:]:
        fused = fused.merge(p, on=["source1_entity_id", "candidate_entity_id"], how="outer")

    for col in ("channel_high_conf", "channel_lexical", "channel_semantic"):
        if col not in fused.columns:
            fused[col] = False
        fused[col] = fused[col].fillna(False)
    for col in ("jaccard_score", "cosine_score"):
        if col not in fused.columns:
            fused[col] = pd.NA

    def _channel_label(row) -> str:
        parts = []
        if row["channel_high_conf"]:
            parts.append("high_conf")
        if row["channel_lexical"]:
            parts.append("lexical")
        if row["channel_semantic"]:
            parts.append("semantic")
        return "+".join(parts) if parts else "unknown"

    fused["source_channel"] = fused.apply(_channel_label, axis=1)
    fused["n_channels"] = (
        fused["channel_high_conf"].astype(int)
        + fused["channel_lexical"].astype(int)
        + fused["channel_semantic"].astype(int)
    )

    fused = fused.drop(columns=["channel_high_conf", "channel_lexical", "channel_semantic"])
    fused = fused.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
    return fused.reset_index(drop=True)


def write_candidate_pairs_tsv(
    fused_df: pd.DataFrame, all_s1_ids, out_path: str
) -> None:
    """Write output/candidate_pairs.tsv in the exact required schema."""
    id_lists: dict[str, set[str]] = {}
    for s1, cid in zip(fused_df["source1_entity_id"], fused_df["candidate_entity_id"]):
        id_lists.setdefault(s1, set()).add(cid)
    write_id_list_tsv(id_lists, all_s1_ids, out_path, id_col=CANDIDATE_COLUMNS[1])
