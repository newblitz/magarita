"""Phase 8 -- Classifier training.

1. Split by source1_entity_id (utils/validation_split.py) -- never split by pair.
2. Hard negatives, not random negatives: positives = pairs in
   train_ground_truth.tsv; negatives = fused candidate pairs NOT in the ground
   truth. No supplemental random unrelated S1/S2/S3 pairs -- random negatives
   are too easy and make the classifier look better than it is.
3. Train the configured GBM (config: classifier.library) on the Phase 7 feature
   matrix.
4. Log feature importance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.features.pairwise_features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


@dataclass
class TrainResult:
    model: object
    feature_importance: pd.DataFrame
    train_ids: set
    val_ids: set
    val_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


def label_pairs(feature_df: pd.DataFrame, ground_truth_pairs: set[tuple[str, str]]) -> pd.Series:
    return feature_df.apply(
        lambda r: int((r["source1_entity_id"], r["candidate_entity_id"]) in ground_truth_pairs),
        axis=1,
    )


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype("category")
    return out


def train_classifier(
    feature_df: pd.DataFrame,
    ground_truth_pairs: set[tuple[str, str]],
    config: dict,
    train_ids: set[str],
    val_ids: set[str],
) -> TrainResult:
    """Train the configured GBM on the labeled, S1-split feature matrix."""
    clf_cfg = config["classifier"]
    library = clf_cfg.get("library", "lightgbm")

    labeled = feature_df.copy()
    labeled["label"] = label_pairs(labeled, ground_truth_pairs)

    train_df = labeled[labeled["source1_entity_id"].isin(train_ids)]
    val_df = labeled[labeled["source1_entity_id"].isin(val_ids)]

    X_train = _encode_categoricals(train_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    y_train = train_df["label"]
    X_val = _encode_categoricals(val_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS])
    y_val = val_df["label"]

    if library != "lightgbm":
        raise NotImplementedError(
            f"classifier.library={library!r} not wired up; this pipeline implements "
            "lightgbm (see config.yaml comment for the license/choice rationale)."
        )

    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        max_depth=clf_cfg.get("max_depth", -1),
        num_leaves=clf_cfg.get("num_leaves", 63),
        learning_rate=clf_cfg.get("learning_rate", 0.05),
        n_estimators=clf_cfg.get("n_estimators", 500),
        random_state=clf_cfg.get("random_seed", 42),
    )
    callbacks = []
    eval_set = None
    if len(X_val) and y_val.nunique() > 1:
        eval_set = [(X_val, y_val)]
        callbacks.append(
            lgb.early_stopping(clf_cfg.get("early_stopping_rounds", 50), verbose=False)
        )
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=callbacks if eval_set else None,
    )

    importance = pd.DataFrame(
        {
            "feature": model.booster_.feature_name(),
            "importance": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    val_predictions = val_df[["source1_entity_id", "candidate_entity_id", "source_channel"]].copy()
    if len(X_val):
        val_predictions["label"] = y_val.to_numpy()
        val_predictions["score"] = model.predict_proba(X_val)[:, 1]
    else:
        val_predictions["label"] = []
        val_predictions["score"] = []

    return TrainResult(
        model=model,
        feature_importance=importance,
        train_ids=train_ids,
        val_ids=val_ids,
        val_predictions=val_predictions,
    )
