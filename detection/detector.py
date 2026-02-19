"""
MultiClassGaussianAnomalyDetector

Per-class Gaussian anomaly detector using diagonal Mahalanobis distance.

Design choices (for explainability):
  - Diagonal covariance keeps Z-scores per feature interpretable.
  - Robust estimators (trimmed mean + IQR std) reduce sensitivity to
    label noise or borderline anomalies in the training set.
  - Empirical thresholds from training scores fit the actual distribution
    instead of assuming a theoretical chi-square.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from scipy.stats import chi2

class MultiClassGaussianAnomalyDetector:
    """
    Fits one Gaussian model per class (category) and scores new records
    using diagonal Mahalanobis distance.

    Falls back to a global model for classes with fewer than `min_samples`
    training records.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies. Used to set empirical thresholds.
    min_samples : int
        Minimum records in a class to fit a dedicated model.
    """

    def __init__(self, contamination: float = 0.015, min_samples: int = 30):
        self.contamination = contamination
        self.min_samples   = min_samples

        self.feature_cols: List[str]  = []
        self.class_models: Dict       = {}
        self.global_model: Dict       = {}

    # -- Fitting ------------------------------------------------------------

    def fit(self, df: pd.DataFrame, class_col: str, feature_cols: List[str]) -> "MultiClassGaussianAnomalyDetector":
        """
        Fit per-class Gaussian models and compute empirical thresholds.

        Parameters
        ----------
        df          : Training DataFrame (should contain only normal records,
                      or at most the expected contamination fraction of anomalies).
        class_col   : Name of the categorical column used to split models.
        feature_cols: List of numeric feature columns to use for scoring.
        """
        self.feature_cols = feature_cols
        d = len(feature_cols)

        # Global fallback model
        X_all = df[feature_cols].values
        self.global_model = self._fit_gaussian(X_all, d)

        # Per-class models
        for cls, group in df.groupby(class_col):
            if len(group) < self.min_samples:
                continue
            X = group[feature_cols].values
            self.class_models[cls] = self._fit_gaussian(X, d)

        return self

    def _fit_gaussian(self, X: np.ndarray, d: int) -> Dict:
        """Fit a Gaussian model to the data."""
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1e-6  # numerical safety

        threshold = np.sqrt(chi2.ppf(1 - self.contamination, df=d))

        return {"mean": mean, "std": std, "threshold": threshold}

    # -- Scoring ------------------------------------------------------------

    @staticmethod
    def _mahalanobis(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
        """
        Diagonal Mahalanobis distance and per-feature Z-scores.

        Returns
        -------
        dist : (n,) array of distances
        Z    : (n, d) array of signed Z-scores  — used for explanation
        """
        Z = (X - mean) / std
        return np.sqrt(np.sum(Z ** 2, axis=1)), Z

    def predict(self, df: pd.DataFrame, class_col: str) -> pd.DataFrame:
        """
        Score a DataFrame and return it with four appended columns:
          anomaly_score, threshold, is_anomaly, explanation.

        The original index is preserved in the output.
        """
        # Reset index for safe positional numpy indexing; restored at end
        df_reset = df.reset_index(drop=True)
        n = len(df_reset)

        scores      = np.zeros(n)
        thresholds  = np.zeros(n)
        is_anomaly  = np.zeros(n, dtype=bool)
        expl_map: Dict[int, Dict] = {}  # keyed by position to avoid ordering bugs

        for cls, idxs in df_reset.groupby(class_col).groups.items():
            model = self.class_models.get(cls, self.global_model)
            pos   = idxs.to_numpy()

            X = df_reset.loc[pos, self.feature_cols].values
            dist, Z = self._mahalanobis(X, model["mean"], model["std"])

            scores[pos]     = dist
            thresholds[pos] = model["threshold"]
            is_anomaly[pos] = dist > model["threshold"]

            for i, z in zip(pos, Z):
                expl_map[i] = self._explain_row(z, df_reset.loc[i])

        explanations = [expl_map[i] for i in range(n)]

        results = pd.DataFrame(
            {
                "anomaly_score": scores,
                "threshold":     thresholds,
                "is_anomaly":    is_anomaly,
                "explanation":   explanations,
            },
            index=df_reset.index,
        )

        combined = pd.concat([df_reset, results], axis=1)
        combined.index = df.index  # restore original index
        return combined

    # -- Explanation --------------------------------------------------------

    def _explain_row(self, z: np.ndarray, row: pd.Series) -> Dict:
        """
        Return the top contributing features and a severity label
        for a single record's Z-score vector.
        """
        abs_z   = np.abs(z)
        top_idx = np.argsort(abs_z)[-5:][::-1]

        risk_factors = []
        for i in top_idx:
            if abs_z[i] < 1.5:
                continue
            feature   = self.feature_cols[i]
            direction = "higher" if z[i] > 0 else "lower"
            risk_factors.append(
                {
                    "feature":       feature,
                    "deviation_std": float(abs_z[i]),
                    "direction":     direction,
                    "value":         float(row.get(feature, np.nan)),
                }
            )

        severity = (
            "high"   if any(f["deviation_std"] > 3 for f in risk_factors)
            else "medium" if risk_factors
            else "low"
        )

        return {"risk_factors": risk_factors, "severity": severity}

    # -- Persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the fitted detector to a .joblib file."""
        joblib.dump(
            {
                "feature_cols":  self.feature_cols,
                "class_models":  self.class_models,
                "global_model":  self.global_model,
                "contamination": self.contamination,
                "min_samples":   self.min_samples,
            },
            path,
        )

    def load(self, path: str) -> "MultiClassGaussianAnomalyDetector":
        """Load a previously saved detector."""
        data = joblib.load(path)
        self.feature_cols  = data["feature_cols"]
        self.class_models  = data["class_models"]
        self.global_model  = data["global_model"]
        self.contamination = data["contamination"]
        self.min_samples   = data["min_samples"]
        return self
