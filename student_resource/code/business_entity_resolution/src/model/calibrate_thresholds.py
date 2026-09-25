"""Phase 9 -- Threshold calibration for F0.5.

1. Score the validation split with the trained classifier (already done in
   train_classifier.TrainResult.val_predictions).
2. Compute macro F0.5 exactly as the competition does.
3. Sweep candidate probability thresholds and pick the one maximizing macro F0.5.
4. If calibration.per_tier_thresholds is true, repeat the sweep separately per
   source_channel bucket, then confirm empirically it beats one global threshold
   before keeping the extra complexity.
5. Break out the calibrated pipeline's macro F0.5 by country on the validation
   split (France included, even with no training labels for it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.metrics import macro_f_beta, macro_f_beta_by_group


@dataclass
class CalibrationResult:
    global_threshold: float
    global_macro_f05: float
    per_tier_thresholds: dict[str, float]
    per_tier_macro_f05: float
    use_per_tier: bool
    by_country_f05: dict[str, float]


def _predictions_at_threshold(
    val_predictions: pd.DataFrame, threshold_lookup
) -> dict[str, set[str]]:
    """``threshold_lookup`` is either a single float, or a callable
    (row) -> float returning the per-row threshold to apply."""
    if callable(threshold_lookup):
        keep_mask = val_predictions.apply(
            lambda r: r["score"] >= threshold_lookup(r), axis=1
        )
    else:
        keep_mask = val_predictions["score"] >= threshold_lookup

    kept = val_predictions[keep_mask]
    out: dict[str, set[str]] = {}
    for s1, cid in zip(kept["source1_entity_id"], kept["candidate_entity_id"]):
        out.setdefault(s1, set()).add(cid)
    return out


def _truth_from_labels(val_predictions: pd.DataFrame) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    positives = val_predictions[val_predictions["label"] == 1]
    for s1, cid in zip(positives["source1_entity_id"], positives["candidate_entity_id"]):
        truth.setdefault(s1, set()).add(cid)
    return truth


def _sweep_global_threshold(val_predictions: pd.DataFrame, all_s1_ids) -> tuple[float, float]:
    truth = _truth_from_labels(val_predictions)
    candidates = np.unique(val_predictions["score"].to_numpy())
    if len(candidates) > 200:
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))
    best_t, best_f = 0.5, -1.0
    for t in candidates:
        preds = _predictions_at_threshold(val_predictions, float(t))
        score = macro_f_beta(preds, truth, all_s1_ids, beta=0.5)
        if score > best_f:
            best_f, best_t = score, float(t)
    return best_t, best_f


def calibrate_thresholds(
    val_predictions: pd.DataFrame,
    all_s1_ids,
    config: dict,
    s1_country_lookup: dict[str, str] | None = None,
) -> CalibrationResult:
    cal_cfg = config["calibration"]
    global_threshold, global_f05 = _sweep_global_threshold(val_predictions, all_s1_ids)

    per_tier_thresholds = {}
    per_tier_f05 = global_f05
    use_per_tier = False

    if cal_cfg.get("per_tier_thresholds", True):
        tiers = val_predictions["source_channel"].unique().tolist()
        for tier in tiers:
            tier_df = val_predictions[val_predictions["source_channel"] == tier]
            if tier_df.empty or tier_df["label"].nunique() < 2:
                per_tier_thresholds[tier] = global_threshold
                continue
            t, _ = _sweep_global_threshold(tier_df, all_s1_ids)
            per_tier_thresholds[tier] = t

        per_tier_lookup = lambda row: per_tier_thresholds.get(row["source_channel"], global_threshold)
        preds = _predictions_at_threshold(val_predictions, per_tier_lookup)
        truth = _truth_from_labels(val_predictions)
        candidate_f05 = macro_f_beta(preds, truth, all_s1_ids, beta=0.5)

        # Only adopt per-tier thresholds if they actually beat the single global
        # threshold -- measured, not assumed (BUILDING.md Phase 9 step 4).
        if candidate_f05 > global_f05:
            per_tier_f05 = candidate_f05
            use_per_tier = True
        else:
            per_tier_f05 = global_f05

    final_lookup = (
        (lambda row: per_tier_thresholds.get(row["source_channel"], global_threshold))
        if use_per_tier
        else global_threshold
    )
    final_preds = _predictions_at_threshold(val_predictions, final_lookup)
    truth = _truth_from_labels(val_predictions)

    by_country_f05 = {}
    if s1_country_lookup:
        by_country_f05 = macro_f_beta_by_group(final_preds, truth, s1_country_lookup, beta=0.5)

    return CalibrationResult(
        global_threshold=global_threshold,
        global_macro_f05=global_f05,
        per_tier_thresholds=per_tier_thresholds,
        per_tier_macro_f05=per_tier_f05,
        use_per_tier=use_per_tier,
        by_country_f05=by_country_f05,
    )
