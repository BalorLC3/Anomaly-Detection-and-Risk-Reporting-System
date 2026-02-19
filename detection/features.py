"""
detection/features.py
---------------------
GaussianAnomalyPreprocessor

Feature engineering pipeline optimised for the Gaussian-based anomaly
detector (diagonal Mahalanobis distance).

Features produced
-----------------
  log_amount            - log1p-transformed transaction amount
  amount_zscore         - per-customer z-score of the amount
  hour_sin / hour_cos   - cyclic encoding of transaction hour
  hour_diff             - deviation from the customer's average hour
  is_weekend            - binary flag for Saturday/Sunday
  category_freq         - global frequency of the transaction category
  category_match        - 1 if category matches customer's preferred category
  log_customer_tx_count - log1p of customer's historical transaction count
  customer_avg_amount   - customer's mean historical transaction amount

All features are scaled with RobustScaler (median/IQR) — consistent with
the robust Gaussian estimators in the detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class GaussianAnomalyPreprocessor:
    """
    Fit on training data, transform train and test independently
    to avoid target leakage.

    Expected input columns (BASE_COLS):
        nameOrig, amount, category, hour, day_of_week, month

    Usage
    -----
    prep = GaussianAnomalyPreprocessor()
    prep.fit(X_train)
    X_train_gauss = prep.transform(X_train)
    X_test_gauss  = prep.transform(X_test)
    """

    BASE_COLS = [
        "nameOrig",
        "amount",
        "category",
        "hour",
        "day_of_week",
        "month",
    ]

    GAUSSIAN_FEATURES = [
        "log_amount",
        "amount_zscore",
        "hour_sin",
        "hour_cos",
        "hour_diff",
        "is_weekend",
        "category_freq",
        "category_match",
        "log_customer_tx_count",
        "customer_avg_amount",
    ]

    def __init__(self):
        self.scaler          = RobustScaler()
        self.category_freq_  = None   # pd.Series  — fit-time only
        self.customer_stats_ = None   # pd.DataFrame — fit-time only

    # -- Fit ----------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "GaussianAnomalyPreprocessor":
        """
        Learn category frequencies and per-customer statistics from the
        training set, then fit the RobustScaler on the resulting features.

        Parameters
        ----------
        df : Training DataFrame containing BASE_COLS.
        """
        df = df[self.BASE_COLS].copy()

        # Global category frequency (computed once on train)
        self.category_freq_ = df["category"].value_counts(normalize=True)

        # Per-customer statistics — train only, prevents leakage
        customer_stats = df.groupby("nameOrig").agg(
            customer_avg_amount   =("amount",   "mean"),
            customer_std_amount   =("amount",   "std"),
            customer_tx_count     =("amount",   "count"),
            customer_avg_hour     =("hour",     "mean"),
            customer_pref_category=("category", lambda x: x.mode().iloc[0]),
        )
        customer_stats["customer_std_amount"] = (
            customer_stats["customer_std_amount"].fillna(0.0)
        )
        self.customer_stats_ = customer_stats

        X = self._build_features(df)
        self.scaler.fit(X[self.GAUSSIAN_FEATURES])
        return self

    # -- Transform ----------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted pipeline to a DataFrame.

        Returns a scaled DataFrame with columns == GAUSSIAN_FEATURES,
        index preserved from the input.
        """
        if self.category_freq_ is None or self.customer_stats_ is None:
            raise RuntimeError(
                "Preprocessor must be fitted before calling transform()."
            )

        df    = df[self.BASE_COLS].copy()
        X     = self._build_features(df)
        X_scaled = self.scaler.transform(X[self.GAUSSIAN_FEATURES])

        return pd.DataFrame(
            X_scaled,
            columns=self.GAUSSIAN_FEATURES,
            index=df.index,
        )

    # -- Internal feature engineering ---------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all raw (unscaled) features from BASE_COLS."""

        df = df.merge(self.customer_stats_, on="nameOrig", how="left")

        # Cold-start fill for unseen customers
        df["customer_avg_amount"]    = df["customer_avg_amount"].fillna(df["amount"].mean())
        df["customer_std_amount"]    = df["customer_std_amount"].fillna(0.0)
        df["customer_tx_count"]      = df["customer_tx_count"].fillna(1)
        df["customer_avg_hour"]      = df["customer_avg_hour"].fillna(12.0)
        df["customer_pref_category"] = df["customer_pref_category"].fillna(
            self.category_freq_.idxmax()
        )

        # Amount
        df["log_amount"]    = np.log1p(df["amount"])
        df["amount_zscore"] = (
            (df["amount"] - df["customer_avg_amount"])
            / (df["customer_std_amount"] + 0.01)
        )

        # Cyclic time
        df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
        df["hour_diff"] = np.abs(df["hour"] - df["customer_avg_hour"])
        df["is_weekend"] = (df["day_of_week"] >= 6).astype(int)

        # Category
        df["category_freq"]  = df["category"].map(self.category_freq_).fillna(0)
        df["category_match"] = (df["category"] == df["customer_pref_category"]).astype(int)

        # Customer history
        df["log_customer_tx_count"] = np.log1p(df["customer_tx_count"])

        return df
