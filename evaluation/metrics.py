"""
evaluation/metrics.py
---------------------
Model evaluation utilities covering:
  - ROC-AUC, Average Precision, False Positive Rate
  - Cross-validation for unsupervised anomaly detectors
  - Per-class breakdown report
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

from detection.detector import MultiClassGaussianAnomalyDetector
import config


# ── Core evaluation ────────────────────────────────────────────────────────

def evaluate(
    result_df: pd.DataFrame,
    label_col: str,
    score_col: str = "anomaly_score",
) -> Dict[str, float]:
    """
    Compute ROC-AUC, Average Precision, and FPR at the detector's threshold
    for a labelled result DataFrame.

    Parameters
    ----------
    result_df : Output of detector.predict() with ground-truth labels added.
    label_col : Column name of binary ground-truth labels (1 = anomaly).
    score_col : Column name of anomaly scores.

    Returns
    -------
    dict with keys: roc_auc, average_precision, fpr, tpr_at_fpr, threshold
    """
    y_true  = result_df[label_col].values
    y_score = result_df[score_col].values
    y_pred  = result_df["is_anomaly"].astype(int).values

    roc_auc = roc_auc_score(y_true, y_score)
    ap      = average_precision_score(y_true, y_score)

    # FPR / TPR at the detector's operating threshold
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)

    return {
        "roc_auc":           round(roc_auc, 4),
        "average_precision": round(ap, 4),
        "fpr":               round(fpr, 4),
        "tpr":               round(tpr, 4),
        "precision":         round(tp / max(tp + fp, 1), 4),
        "recall":            round(tpr, 4),
        "f1":                round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "n_anomalies_pred":  int(y_pred.sum()),
        "n_anomalies_true":  int(y_true.sum()),
    }


# ── Cross-validation ───────────────────────────────────────────────────────

def cross_validate_detector(
    df: pd.DataFrame,
    label_col: str,
    class_col: str,
    feature_cols: List[str],
    n_splits: int = None,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Stratified k-fold cross-validation for the anomaly detector.

    Stratification is on `label_col` to ensure anomaly representation
    in every fold's test set.

    Parameters
    ----------
    df           : Full labelled DataFrame.
    label_col    : Binary ground-truth label column.
    class_col    : Category column passed to detector.
    feature_cols : Feature columns passed to detector.
    n_splits     : Number of CV folds (defaults to config.CV_N_SPLITS).
    random_state : Random seed (defaults to config.CV_RANDOM_STATE).

    Returns
    -------
    DataFrame with one row per fold and columns for each metric.
    """
    n_splits     = n_splits     or config.CV_N_SPLITS
    random_state = random_state or config.CV_RANDOM_STATE

    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y       = df[label_col].values
    records = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(df, y), 1):
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        # Fit only on normal records to match unsupervised assumptions
        normal_train = train_df[train_df[label_col] == 0]

        detector = MultiClassGaussianAnomalyDetector(
            contamination=config.DETECTOR_CONTAMINATION,
            min_samples=config.DETECTOR_MIN_SAMPLES,
        )
        detector.fit(normal_train, class_col=class_col, feature_cols=feature_cols)

        result = detector.predict(test_df, class_col=class_col)
        result[label_col] = test_df[label_col].values

        metrics = evaluate(result, label_col=label_col)
        metrics["fold"] = fold
        records.append(metrics)

        print(
            f"  Fold {fold}/{n_splits} — "
            f"AUC={metrics['roc_auc']:.3f}  "
            f"AP={metrics['average_precision']:.3f}  "
            f"FPR={metrics['fpr']:.3f}"
        )

    cv_df = pd.DataFrame(records).set_index("fold")

    # Summary row
    summary = cv_df.mean().rename("mean").to_frame().T
    summary.index = ["mean"]
    std_row = cv_df.std().rename("std").to_frame().T
    std_row.index = ["std"]

    return pd.concat([cv_df, summary, std_row])


# ── Per-class breakdown ────────────────────────────────────────────────────

def per_class_report(
    result_df: pd.DataFrame,
    label_col: str,
    class_col: str,
    score_col: str = "anomaly_score",
) -> pd.DataFrame:
    """
    Compute ROC-AUC and Average Precision for each class independently.

    Useful for diagnosing which categories drive poor overall AP.

    Returns
    -------
    DataFrame indexed by class with columns: n, n_anomalies, roc_auc,
    average_precision, fpr, tpr, f1.
    """
    records = []

    for cls, grp in result_df.groupby(class_col):
        y_true  = grp[label_col].values
        y_score = grp[score_col].values
        y_pred  = grp["is_anomaly"].astype(int).values

        n_pos = y_true.sum()
        if n_pos == 0 or n_pos == len(y_true):
            # Can't compute AUC without both classes
            records.append({"class": cls, "n": len(grp), "n_anomalies": int(n_pos),
                            "roc_auc": np.nan, "average_precision": np.nan,
                            "fpr": np.nan, "tpr": np.nan, "f1": np.nan})
            continue

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        records.append({
            "class":             cls,
            "n":                 len(grp),
            "n_anomalies":       int(n_pos),
            "anomaly_rate":      round(n_pos / len(grp), 3),
            "roc_auc":           round(roc_auc_score(y_true, y_score), 4),
            "average_precision": round(average_precision_score(y_true, y_score), 4),
            "fpr":               round(fp / max(fp + tn, 1), 4),
            "tpr":               round(tp / max(tp + fn, 1), 4),
            "f1":                round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        })

    return pd.DataFrame(records).set_index("class").sort_values("average_precision")
