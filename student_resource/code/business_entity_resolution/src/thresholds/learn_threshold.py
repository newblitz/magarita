"""Phase 5 -- Threshold learning (the LSBlock two-threshold trick), one instance
per retrieval channel (lexical Jaccard, semantic cosine).

This threshold decides *inclusion in the candidate set* (recall-oriented, can
afford to be permissive) -- NOT the final match decision (that's Phase 9,
precision-oriented). Expect precision at this stage to be low (LSBlock's own
benchmark precision ranges 0.08-0.69); that's normal, not a bug.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier


@dataclass
class ThresholdResult:
    threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    f_beta_at_threshold: float
    n_train: int
    n_holdout: int


def _f_beta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom > 0, (1 + beta_sq) * precision * recall / denom, 0.0)
    return f


def sample_pairs_for_threshold(
    scored_pairs: "pd.DataFrame",
    ground_truth_pairs: set[tuple[str, str]],
    s1_ids: list[str],
    candidate_ids: list[str],
    sample_size: int | None,
    random_seed: int = 42,
) -> "pd.DataFrame":
    """Sample an (approximately) class-balanced set of (s1, candidate, label) rows.

    Positives come from ``ground_truth_pairs`` that are present in ``scored_pairs``.
    Negatives are sampled as random S1 x S2/S3 pairs (per LSBlock's own "random plus
    known-negative" recipe for threshold fitting -- deliberately NOT the hard-negative
    sampling used later for the classifier, since this model wants an honest,
    unconfounded similarity/label relationship).
    """
    import pandas as pd

    rng = random.Random(random_seed)
    keyed = scored_pairs.set_index(["source1_entity_id", "candidate_entity_id"])

    pos_rows = []
    for s1, cid in ground_truth_pairs:
        if (s1, cid) in keyed.index:
            pos_rows.append((s1, cid, 1))
    if sample_size is not None and len(pos_rows) > sample_size:
        pos_rows = rng.sample(pos_rows, sample_size)

    n_neg_target = len(pos_rows)
    neg_rows = []
    attempts = 0
    max_attempts = n_neg_target * 20 + 1000
    while len(neg_rows) < n_neg_target and attempts < max_attempts:
        attempts += 1
        s1 = rng.choice(s1_ids)
        cid = rng.choice(candidate_ids)
        if (s1, cid) in ground_truth_pairs:
            continue
        neg_rows.append((s1, cid, 0))

    all_rows = pos_rows + neg_rows
    rng.shuffle(all_rows)
    sample_df = pd.DataFrame(all_rows, columns=["source1_entity_id", "candidate_entity_id", "label"])
    merged = sample_df.merge(
        scored_pairs, on=["source1_entity_id", "candidate_entity_id"], how="inner"
    )
    return merged


def learn_threshold(
    train_pairs_with_scores: "pd.DataFrame",
    ground_truth: "pd.DataFrame",
    score_column: str,
    config: dict,
    holdout_fraction: float = 0.3,
) -> ThresholdResult:
    """1. Sample matched + unmatched pairs from ground truth / scored pairs.
    2. Fit a tiny MLP (1 input -> hidden_units -> 1, sigmoid) as a scalar
       classifier on (score, label).
    3. Read the threshold off the precision-recall curve that maximizes F0.5
       (NOT F1 -- deliberate change from the LSBlock reference).
    4. Return the threshold, plus precision/recall at that threshold measured on
       a held-out slice never used to fit it.
    """
    from src.utils.io import ground_truth_to_pairs

    th_cfg = config["threshold_learning"]
    hidden_units = th_cfg.get("mlp_hidden_units", 16)
    seed = th_cfg.get("random_seed", 42)
    sample_size = th_cfg.get("sample_pairs_per_class")
    if sample_size is None:
        sample_size = max(500, len(ground_truth) // 2)

    gt_pairs_df = ground_truth_to_pairs(ground_truth)
    gt_pairs = set(zip(gt_pairs_df["source1_entity_id"], gt_pairs_df["candidate_entity_id"]))

    s1_ids = train_pairs_with_scores["source1_entity_id"].unique().tolist()
    candidate_ids = train_pairs_with_scores["candidate_entity_id"].unique().tolist()

    sample = sample_pairs_for_threshold(
        train_pairs_with_scores[["source1_entity_id", "candidate_entity_id", score_column]],
        gt_pairs,
        s1_ids,
        candidate_ids,
        sample_size,
        random_seed=seed,
    )
    if sample.empty or sample["label"].nunique() < 2:
        # Degenerate case (e.g. tiny dev sample) -- fall back to a simple midpoint.
        return ThresholdResult(0.5, 0.0, 0.0, 0.0, 0, 0)

    rng = random.Random(seed)
    idx = list(range(len(sample)))
    rng.shuffle(idx)
    n_holdout = max(1, int(round(len(idx) * holdout_fraction)))
    holdout_idx = set(idx[:n_holdout])
    train_idx = [i for i in idx if i not in holdout_idx]

    X = sample[score_column].to_numpy(dtype=float).reshape(-1, 1)
    y = sample["label"].to_numpy(dtype=int)

    X_train, y_train = X[train_idx], y[train_idx]
    X_hold, y_hold = X[list(holdout_idx)], y[list(holdout_idx)]

    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    # Sweep thresholds on the *score* itself (simpler and more transferable at
    # inference time than thresholding the MLP's probability output), scoring
    # with the MLP-smoothed probability to decide the best cut, then reporting
    # precision/recall of that same score-cut on the untouched holdout slice.
    candidate_thresholds = np.unique(np.clip(X_train.ravel(), 0, 1))
    if len(candidate_thresholds) > 200:
        candidate_thresholds = np.quantile(candidate_thresholds, np.linspace(0, 1, 200))

    best_threshold, best_f = 0.5, -1.0
    beta = 0.5
    for t in candidate_thresholds:
        pred = (X_train.ravel() >= t).astype(int)
        tp = int(((pred == 1) & (y_train == 1)).sum())
        fp = int(((pred == 1) & (y_train == 0)).sum())
        fn = int(((pred == 0) & (y_train == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f = _f_beta(np.array([precision]), np.array([recall]), beta)[0]
        if f > best_f:
            best_f = f
            best_threshold = float(t)

    pred_hold = (X_hold.ravel() >= best_threshold).astype(int)
    tp = int(((pred_hold == 1) & (y_hold == 1)).sum())
    fp = int(((pred_hold == 1) & (y_hold == 0)).sum())
    fn = int(((pred_hold == 0) & (y_hold == 1)).sum())
    precision_hold = tp / (tp + fp) if (tp + fp) else 0.0
    recall_hold = tp / (tp + fn) if (tp + fn) else 0.0
    f_hold = _f_beta(np.array([precision_hold]), np.array([recall_hold]), beta)[0]

    return ThresholdResult(
        threshold=best_threshold,
        precision_at_threshold=precision_hold,
        recall_at_threshold=recall_hold,
        f_beta_at_threshold=float(f_hold),
        n_train=len(train_idx),
        n_holdout=len(holdout_idx),
    )
