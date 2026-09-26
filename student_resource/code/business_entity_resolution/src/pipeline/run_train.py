"""Orchestrates Phases 1-9 on training data.

Run each phase independently and check its own definition-of-done before moving
on (BUILDING.md §12):

    python3 -m src.pipeline.run_train --phase tier0_only
    python3 -m src.pipeline.run_train --phase tier1_measure
    python3 -m src.pipeline.run_train --phase tier2_measure
    python3 -m src.pipeline.run_train --phase fuse
    python3 -m src.pipeline.run_train --phase train_full

Add ``--sample N`` to restrict the number of S1 rows for fast local iteration
(Magellan's "downsample before experimenting" habit -- architecture report §10.1).
Must be run with the current working directory anywhere; paths resolve via
``src/utils/io.py::resolve_path`` against the project root regardless of cwd.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from src.blocking.fuse_candidates import fuse, write_candidate_pairs_tsv
from src.blocking.tier0_deterministic import tier0_candidates
from src.blocking.tier0_deterministic import write_tier0_checkpoint
from src.blocking.tier1_lexical import tier1_candidates_all_countries
from src.features.pairwise_features import build_feature_matrix
from src.model.calibrate_thresholds import calibrate_thresholds
from src.model.train_classifier import train_classifier
from src.normalize import normalize_dataframe
from src.thresholds.learn_threshold import learn_threshold
from src.utils.io import (
    ground_truth_to_pairs,
    load_config,
    read_ground_truth_tsv,
    read_source_tsv,
    resolve_path,
)
from src.utils.mem_monitor import log_mem
from src.utils.validation_split import split_s1_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_train")


def load_train_data(config: dict, sample: int | None = None) -> dict[str, pd.DataFrame]:
    paths = config["paths"]
    log_mem("before loading train data")
    s1 = read_source_tsv(paths["train_source1"], nrows=sample)
    log_mem("after S1 load")
    s2 = read_source_tsv(paths["train_source2"])
    log_mem("after S2 load")
    s3 = read_source_tsv(paths["train_source3"])
    log_mem("after S3 load")
    gt = read_ground_truth_tsv(paths["train_ground_truth"])

    if sample is not None:
        keep_ids = set(s1["entity_id"])
        gt = gt[gt["source1_entity_id"].isin(keep_ids)]

    log.info("Loaded train S1=%d S2=%d S3=%d ground_truth=%d", len(s1), len(s2), len(s3), len(gt))

    s1 = normalize_dataframe(s1, config)
    s2 = normalize_dataframe(s2, config)
    s3 = normalize_dataframe(s3, config)
    # normalize_dataframe returns a copy; the original raw DataFrames are now
    # unreachable — force GC to reclaim them before blocking begins.
    gc.collect()
    log_mem("after normalization (GC'd raw frames)")
    return {"s1": s1, "s2": s2, "s3": s3, "gt": gt}


def _ground_truth_pairs(gt: pd.DataFrame) -> set[tuple[str, str]]:
    pairs_df = ground_truth_to_pairs(gt)
    return set(zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]))


def _measure_recall(candidate_pairs: set[tuple[str, str]], gt_pairs: set[tuple[str, str]]) -> dict:
    if not gt_pairs:
        return {"recall": float("nan"), "n_true": 0, "n_candidates": len(candidate_pairs)}
    hit = len(candidate_pairs & gt_pairs)
    return {
        "recall": hit / len(gt_pairs),
        "n_true": len(gt_pairs),
        "n_hit": hit,
        "n_candidates": len(candidate_pairs),
    }


def _artifacts_dir(config: dict) -> Path:
    out = resolve_path(config["paths"]["artifacts_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def phase_tier0_only(data: dict, config: dict) -> pd.DataFrame:
    t0 = time.time()
    tier0_df = tier0_candidates(data["s1"], data["s2"], data["s3"])
    gt_pairs = _ground_truth_pairs(data["gt"])
    cand_pairs = set(zip(tier0_df["source1_entity_id"], tier0_df["candidate_entity_id"]))
    stats = _measure_recall(cand_pairs, gt_pairs)
    precision = len(cand_pairs & gt_pairs) / len(cand_pairs) if cand_pairs else float("nan")
    log.info(
        "Tier0: %d candidates in %.1fs | recall=%.4f precision=%.4f (EDA reference: ~97.8-97.9%% "
        "precision, ~14.7-15.2%% recall)",
        len(tier0_df), time.time() - t0, stats["recall"], precision,
    )
    return tier0_df


def phase_tier0_checkpoint(data: dict, config: dict) -> None:
    """Run Tier 0's disk-checkpointed implementation in this process."""
    output_dir = _artifacts_dir(config) / "tier0"
    manifest = write_tier0_checkpoint(data["s1"], data["s2"], data["s3"], output_dir)
    log.info("Tier0 checkpoint manifest: %s", manifest)


def phase_tier1_measure(data: dict, config: dict) -> pd.DataFrame:
    t0 = time.time()
    tier1_df = tier1_candidates_all_countries(data["s1"], data["s2"], data["s3"], config)
    gt_pairs = _ground_truth_pairs(data["gt"])
    cand_pairs = set(zip(tier1_df["source1_entity_id"], tier1_df["candidate_entity_id"]))
    stats = _measure_recall(cand_pairs, gt_pairs)
    elapsed = time.time() - t0
    per_100k = elapsed / max(1, len(data["s1"]) / 100_000)
    log.info(
        "Tier1 (no threshold): %d shortlisted pairs in %.1fs (%.1fs/100K S1 rows) | "
        "recall-ceiling=%.4f",
        len(tier1_df), elapsed, per_100k, stats["recall"],
    )
    return tier1_df


def phase_tier2_measure(data: dict, config: dict) -> pd.DataFrame:
    try:
        from src.blocking.tier2_semantic import tier2_candidates_all_countries
    except ImportError as exc:
        log.warning(
            "Tier2 skipped: optional dependency missing (%s). Install "
            "sentence-transformers + faiss-cpu (see requirements.txt) to run it.",
            exc,
        )
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "field", "cosine_score", "source_channel"]
        )

    t0 = time.time()
    tier2_df = tier2_candidates_all_countries(data["s1"], data["s2"], data["s3"], config)
    gt_pairs = _ground_truth_pairs(data["gt"])
    cand_pairs = set(zip(tier2_df["source1_entity_id"], tier2_df["candidate_entity_id"]))
    stats = _measure_recall(cand_pairs, gt_pairs)
    log.info(
        "Tier2 (no threshold, top_k=%d): %d shortlisted pairs in %.1fs | recall-ceiling=%.4f",
        config["tier2_semantic"]["top_k"], len(tier2_df), time.time() - t0, stats["recall"],
    )
    return tier2_df


def phase_fuse(data: dict, config: dict) -> tuple[pd.DataFrame, dict]:
    tier0_df = phase_tier0_only(data, config)
    tier1_df = phase_tier1_measure(data, config)
    tier2_df = phase_tier2_measure(data, config)

    learned_thresholds: dict[str, dict[str, float]] = {"lexical": {}, "semantic": {}}
    if len(tier1_df):
        for field in tier1_df["field"].unique():
            field_df = tier1_df[tier1_df["field"] == field]
            result = learn_threshold(field_df, data["gt"], "jaccard_score", config)
            learned_thresholds["lexical"][field] = result.threshold
            log.info(
                "Learned lexical threshold field=%s: t=%.4f precision=%.3f recall=%.3f (holdout)",
                field, result.threshold, result.precision_at_threshold, result.recall_at_threshold,
            )
    if len(tier2_df):
        for field in tier2_df["field"].unique():
            field_df = tier2_df[tier2_df["field"] == field]
            result = learn_threshold(field_df, data["gt"], "cosine_score", config)
            learned_thresholds["semantic"][field] = result.threshold
            log.info(
                "Learned semantic threshold field=%s: t=%.4f precision=%.3f recall=%.3f (holdout)",
                field, result.threshold, result.precision_at_threshold, result.recall_at_threshold,
            )

    fused_df = fuse(tier0_df, tier1_df, tier2_df, learned_thresholds)
    gt_pairs = _ground_truth_pairs(data["gt"])
    cand_pairs = set(zip(fused_df["source1_entity_id"], fused_df["candidate_entity_id"]))
    stats = _measure_recall(cand_pairs, gt_pairs)
    log.info(
        "Fused candidates: %d pairs | recall=%.4f (target: beat Tier0-only ceiling, "
        "well below the old ~1.34B deterministic-union volume)",
        len(fused_df), stats["recall"],
    )

    artifacts_dir = _artifacts_dir(config)
    with open(artifacts_dir / "learned_retrieval_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(learned_thresholds, f, indent=2)
    fused_df.to_parquet(artifacts_dir / "train_fused_candidates.parquet", index=False)
    write_candidate_pairs_tsv(
        fused_df, data["s1"]["entity_id"].tolist(), str(artifacts_dir / "train_candidate_pairs.tsv")
    )
    return fused_df, learned_thresholds


def phase_train_full(data: dict, config: dict) -> None:
    fused_df, _ = phase_fuse(data, config)
    log_mem("after fuse")

    # Build feature matrices for S2 and S3 sequentially — never hold both
    # fully-materialised frames in RAM at the same time.
    log_mem("before feature matrix (S2)")
    feat_parts = [build_feature_matrix(fused_df, data["s1"], data["s2"])]
    gc.collect()
    log_mem("after feature matrix (S2)")
    feat_parts.append(build_feature_matrix(fused_df, data["s1"], data["s3"]))
    gc.collect()
    log_mem("after feature matrix (S3)")

    feature_df = pd.concat(feat_parts, ignore_index=True)
    del feat_parts
    gc.collect()
    feature_df = feature_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
    log.info("Feature matrix: %d rows, %d columns", len(feature_df), feature_df.shape[1])
    log_mem("after feature dedup")

    gt_pairs = _ground_truth_pairs(data["gt"])
    val_fraction = config["calibration"]["val_fraction"]
    seed = config["calibration"]["random_seed"]
    train_ids, val_ids = split_s1_ids(data["s1"]["entity_id"].tolist(), val_fraction, seed)

    result = train_classifier(feature_df, gt_pairs, config, train_ids, val_ids)
    log.info("Top feature importances:\n%s", result.feature_importance.head(15).to_string(index=False))

    s1_country_lookup = dict(zip(data["s1"]["entity_id"], data["s1"]["country"]))
    all_val_s1_ids = [s1 for s1 in data["s1"]["entity_id"].tolist() if s1 in val_ids]
    cal = calibrate_thresholds(result.val_predictions, all_val_s1_ids, config, s1_country_lookup)
    log.info(
        "Calibration: global_threshold=%.4f global_macro_f0.5=%.4f | per_tier_used=%s "
        "per_tier_macro_f0.5=%.4f",
        cal.global_threshold, cal.global_macro_f05, cal.use_per_tier, cal.per_tier_macro_f05,
    )
    log.info("Macro F0.5 by country (validation): %s", cal.by_country_f05)

    artifacts_dir = _artifacts_dir(config)
    joblib.dump(result.model, artifacts_dir / "classifier.joblib")
    with open(artifacts_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "global_threshold": cal.global_threshold,
                "per_tier_thresholds": cal.per_tier_thresholds,
                "use_per_tier": cal.use_per_tier,
                "global_macro_f05": cal.global_macro_f05,
                "per_tier_macro_f05": cal.per_tier_macro_f05,
                "by_country_f05": cal.by_country_f05,
            },
            f,
            indent=2,
        )
    log.info("Saved classifier + calibration to %s", artifacts_dir)


PHASES = {
    "tier0_only": phase_tier0_only,
    "tier0_checkpoint": phase_tier0_checkpoint,
    "tier1_measure": phase_tier1_measure,
    "tier2_measure": phase_tier2_measure,
    "fuse": phase_fuse,
    "train_full": phase_train_full,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run business-entity-resolution training phases.")
    parser.add_argument("--phase", choices=list(PHASES.keys()), required=True)
    parser.add_argument("--sample", type=int, default=None, help="Restrict S1 rows for fast iteration.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    sample = args.sample if args.sample is not None else config.get("dev", {}).get("sample_s1")
    data = load_train_data(config, sample=sample)
    PHASES[args.phase](data, config)


if __name__ == "__main__":
    main()
