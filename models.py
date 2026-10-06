"""ML Model Classes and Drift Monitoring for QuantLab Trading Bot.

Contains the ensemble model architecture, probability calibration support,
feature drift detection, and staleness monitoring.
"""

import numpy as np
from datetime import datetime, timezone
import logging

logger = logging.getLogger("quantlab")


class StackingEnsembleModel:
    """
    Stacking Ensemble combining HistGradientBoosting and LightGBM.
    Supports both meta-learner stacking (e.g. LogisticRegression) and weighted blending.
    """
    def __init__(self, hgb_model, lgb_model, weights=(0.50, 0.50), meta_learner=None):
        self.hgb = hgb_model
        self.lgb = lgb_model
        self.weights = weights
        self.meta_learner = meta_learner

    def fit(self, X, y):
        self.hgb.fit(X, y)
        self.lgb.fit(X, y)
        if self.meta_learner is not None:
            p_hgb = self.hgb.predict_proba(X)[:, 1]
            p_lgb = self.lgb.predict_proba(X)[:, 1]
            meta_X = np.column_stack([p_hgb, p_lgb])
            self.meta_learner.fit(meta_X, y)
        return self

    def predict_proba(self, X):
        p_hgb = self.hgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]
        meta_clf = getattr(self, "meta_learner", None)
        if meta_clf is not None and hasattr(meta_clf, "predict_proba"):
            meta_X = np.column_stack([p_hgb, p_lgb])
            return meta_clf.predict_proba(meta_X)
        weights = getattr(self, "weights", (0.50, 0.50))
        p_ens = (p_hgb * weights[0]) + (p_lgb * weights[1])
        return np.column_stack([1.0 - p_ens, p_ens])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


class CalibratedEnsemble:
    """
    Wraps an ensemble model with Isotonic Regression probability calibration.
    Calibrates raw predict_proba outputs against empirical win rates
    without relying on deprecated scikit-learn CalibratedClassifierCV(cv='prefit').
    """
    def __init__(self, base_model):
        self.base_model = base_model
        from sklearn.isotonic import IsotonicRegression
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, X_cal, y_cal):
        raw_probs = self.base_model.predict_proba(X_cal)[:, 1]
        self.calibrator.fit(raw_probs, y_cal)
        return self

    def predict_proba(self, X):
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        cal_probs = np.clip(self.calibrator.predict(raw_probs), 0.0, 1.0)
        return np.column_stack([1.0 - cal_probs, cal_probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


def check_model_staleness(model_bundle, max_age_days=14):
    """
    Checks if a trained model bundle is older than max_age_days.
    Returns: (is_stale: bool, age_days: float, status_msg: str)
    """
    created_at_str = model_bundle.get("created_at")
    if not created_at_str:
        return False, 0.0, "Model creation date unknown"

    try:
        created_dt = datetime.fromisoformat(created_at_str)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_days = (now - created_dt).total_seconds() / 86400.0
        is_stale = age_days > max_age_days
        msg = f"Model age: {age_days:.1f} days ({'STALE - retraining recommended' if is_stale else 'FRESH'})"
        return is_stale, round(age_days, 1), msg
    except Exception as e:
        return False, 0.0, f"Could not determine age: {e}"


def check_feature_drift(live_feats, model_bundle, threshold_std=3.5):
    """
    Detects feature distribution drift by comparing live feature values
    against training dataset means and standard deviations.
    Returns: list of (column_name, z_score) tuples for drifted features.
    """
    stats = model_bundle.get("feature_stats", {})
    if not stats or live_feats is None or live_feats.empty:
        return []

    drifted = []
    for col in live_feats.columns:
        if col in stats:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            val = float(live_feats[col].iloc[-1])
            z_score = abs(val - mean) / (std + 1e-9)
            if z_score > threshold_std:
                drifted.append((col, round(z_score, 1)))

    return drifted
