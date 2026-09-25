"""Phase 10 -- Inference pipeline.

Runs Phases 1-4, 6, 7 on dataset/test/*, applies the trained classifier and
calibrated thresholds (produced by ``run_train.py --phase train_full``), and
writes:

    output/candidate_pairs.tsv   -- literally the Phase 6 output on test data.
    output/matching_results.tsv  -- for every test_source1.tsv entity, the
                                     comma-joined list of candidate ids scoring
                                     above the calibrated threshold for their
                                     source_channel tier, empty if none.

After running, validate with:

    python3 utils/validate_submission.py \\
        --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv \\
        --test-dir dataset/test
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import pandas as pd

from src.blocking.fuse_candidates import fuse, write_candidate_pairs_tsv
from src.blocking.tier0_deterministic import tier0_candidates
from src.blocking.tier1_lexical import tier1_candidates_all_countries
from src.features.pairwise_features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_feature_matrix
from src.normalize import normalize_dataframe
from src.utils.io import (
    RESULT_COLUMNS,
    load_config,
    read_source_tsv,
    resolve_path,
    write_id_list_tsv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_inference")


def load_test_data(config: dict) -> dict[str, pd.DataFrame]:
    paths = config["paths"]
    s1 = read_source_tsv(paths["test_source1"])
    s2 = read_source_tsv(paths["test_source2"])
    s3 = read_source_tsv(paths["test_source3"])
    log.info("Loaded test S1=%d S2=%d S3=%d", len(s1), len(s2), len(s3))
    s1 = normalize_dataframe(s1, config)
    s2 = normalize_dataframe(s2, config)
    s3 = normalize_dataframe(s3, config)
    return {"s1": s1, "s2": s2, "s3": s3}


def _load_artifacts(config: dict) -> dict:
    artifacts_dir = resolve_path(config["paths"]["artifacts_dir"])
    with open(artifacts_dir / "learned_retrieval_thresholds.json", encoding="utf-8") as f:
        retrieval_thresholds = json.load(f)
    with open(artifacts_dir / "calibration.json", encoding="utf-8") as f:
        calibration = json.load(f)
    model = joblib.load(artifacts_dir / "classifier.joblib")
    return {"retrieval_thresholds": retrieval_thresholds, "calibration": calibration, "model": model}


def build_test_candidates(data: dict, config: dict, retrieval_thresholds: dict) -> pd.DataFrame:
    tier0_df = tier0_candidates(data["s1"], data["s2"], data["s3"])
    tier1_df = tier1_candidates_all_countries(data["s1"], data["s2"], data["s3"], config)

    try:
        from src.blocking.tier2_semantic import tier2_candidates_all_countries

        tier2_df = tier2_candidates_all_countries(data["s1"], data["s2"], data["s3"], config)
    except ImportError as exc:
        log.warning("Tier2 skipped at inference: optional dependency missing (%s).", exc)
        tier2_df = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "cosine_score", "source_channel"]
        )

    fused_df = fuse(tier0_df, tier1_df, tier2_df, retrieval_thresholds)
    log.info("Test fused candidates: %d pairs", len(fused_df))
    return fused_df


def score_candidates(fused_df: pd.DataFrame, data: dict, model) -> pd.DataFrame:
    feat_s2 = build_feature_matrix(fused_df, data["s1"], data["s2"])
    feat_s3 = build_feature_matrix(fused_df, data["s1"], data["s3"])
    feature_df = pd.concat([feat_s2, feat_s3], ignore_index=True)
    feature_df = feature_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])

    if feature_df.empty:
        feature_df["score"] = []
        return feature_df

    X = feature_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype("category")
    feature_df["score"] = model.predict_proba(X)[:, 1]
    return feature_df


def apply_thresholds(scored_df: pd.DataFrame, calibration: dict) -> dict[str, set[str]]:
    global_threshold = calibration["global_threshold"]
    per_tier = calibration.get("per_tier_thresholds", {})
    use_per_tier = calibration.get("use_per_tier", False)

    if use_per_tier:
        threshold_series = scored_df["source_channel"].map(per_tier).fillna(global_threshold)
    else:
        threshold_series = pd.Series(global_threshold, index=scored_df.index)

    keep = scored_df["score"] >= threshold_series
    kept = scored_df[keep]

    id_lists: dict[str, set[str]] = {}
    for s1, cid in zip(kept["source1_entity_id"], kept["candidate_entity_id"]):
        id_lists.setdefault(s1, set()).add(cid)
    return id_lists


def run_inference(config: dict) -> None:
    data = load_test_data(config)
    artifacts = _load_artifacts(config)

    fused_df = build_test_candidates(data, config, artifacts["retrieval_thresholds"])
    all_s1_ids = data["s1"]["entity_id"].tolist()

    out_dir = config["paths"]["output_dir"]
    write_candidate_pairs_tsv(fused_df, all_s1_ids, f"{out_dir}candidate_pairs.tsv")
    log.info("Wrote %scandidate_pairs.tsv", out_dir)

    scored_df = score_candidates(fused_df, data, artifacts["model"])
    matched_ids = apply_thresholds(scored_df, artifacts["calibration"])

    write_id_list_tsv(matched_ids, all_s1_ids, f"{out_dir}matching_results.tsv", id_col=RESULT_COLUMNS[1])
    log.info("Wrote %smatching_results.tsv", out_dir)

    # By construction, matched ids must be a subset of candidate ids for every S1
    # (BUILDING.md Phase 10) -- verify here rather than relying only on the
    # external validator.
    cand_ids: dict[str, set[str]] = {}
    for s1, cid in zip(fused_df["source1_entity_id"], fused_df["candidate_entity_id"]):
        cand_ids.setdefault(s1, set()).add(cid)
    violations = [s1 for s1, mids in matched_ids.items() if mids - cand_ids.get(s1, set())]
    if violations:
        log.warning(
            "%d S1 entities have matched IDs not present in candidate_pairs.tsv "
            "(pipeline drift) -- e.g. %s",
            len(violations), violations[:5],
        )
    else:
        log.info("Verified: matching_results.tsv IDs are a subset of candidate_pairs.tsv for every S1.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run business-entity-resolution inference on the test set.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_inference(config)


if __name__ == "__main__":
    main()
