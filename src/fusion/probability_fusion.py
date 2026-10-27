"""Validation-only logistic fusion of complementary failure signals."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = (
    "neural_probability",
    "symbolic_rule_score",
    "problog_failure_risk",
    "predicted_rul",
)

_SEVERITY_WEIGHTS = (
    ("critical", 1.0),
    ("overload", 0.9),
    ("degradation", 0.8),
    ("starvation", 0.8),
    ("elevated", 0.7),
    ("high", 0.65),
    ("low", 0.6),
    ("rising", 0.45),
    ("falling", 0.45),
    ("persistent", 0.45),
    ("pattern", 0.5),
    ("variability", 0.4),
    ("moving-average", 0.45),
)


def symbolic_rule_score(insights: Iterable[str]) -> float:
    """Convert symbolic insights into a bounded severity score.

    The strongest alarm drives the score; additional independent alarms only
    add a small increment so the fusion model cannot be dominated by a long
    list of correlated rule names.
    """
    weights: List[float] = []
    for insight in insights or []:
        text = str(insight).lower()
        for keyword, weight in _SEVERITY_WEIGHTS:
            if keyword in text:
                weights.append(weight)
                break
    if not weights:
        return 0.0
    weights.sort(reverse=True)
    return float(min(1.0, weights[0] + 0.1 * (len(weights) - 1)))


class ProbabilityFusion:
    """Logistic regression trained exclusively on validation-split signals."""

    def __init__(self, random_state: int = 42,
                 feature_order: Optional[Iterable[str]] = None,
                 logger: Optional[logging.Logger] = None):
        self.random_state = int(random_state)
        self.feature_order = list(feature_order or DEFAULT_FEATURES)
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight="balanced", max_iter=1000,
            random_state=self.random_state,
        )
        self.is_fitted = False

    @staticmethod
    def build_matrix(neural_probability: np.ndarray,
                     symbolic_rule_score: np.ndarray,
                     problog_failure_risk: np.ndarray,
                     predicted_rul: Optional[np.ndarray] = None,
                     feature_order: Optional[Iterable[str]] = None) -> np.ndarray:
        """Align the four fusion inputs into a 2D feature matrix."""
        columns = {
            "neural_probability": np.asarray(
                neural_probability, dtype=float).ravel(),
            "symbolic_rule_score": np.asarray(
                symbolic_rule_score, dtype=float).ravel(),
            "problog_failure_risk": np.asarray(
                problog_failure_risk, dtype=float).ravel(),
        }
        if predicted_rul is None:
            predicted_rul = np.zeros_like(columns["neural_probability"])
        columns["predicted_rul"] = np.asarray(
            predicted_rul, dtype=float).ravel()
        order = list(feature_order or DEFAULT_FEATURES)
        lengths = {name: len(columns[name]) for name in order}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Fusion inputs have inconsistent lengths: {lengths}")
        return np.column_stack([columns[name] for name in order])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbabilityFusion":
        """Fit scaling and logistic regression on validation signals only."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("Fusion X must be 2D with one row per label.")
        if len(np.unique(y)) < 2:
            raise ValueError("Fusion requires both classes in the validation split.")
        scaled = self.scaler.fit_transform(X)
        self.model.fit(scaled, y)
        self.is_fitted = True
        self.logger.info("Probability fusion fitted on %d validation windows.", len(y))
        return self

    def positive_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fused failure probabilities."""
        if not self.is_fitted:
            raise RuntimeError("ProbabilityFusion must be fitted before prediction.")
        scaled = self.scaler.transform(np.asarray(X, dtype=float))
        probabilities = self.model.predict_proba(scaled)
        classes = list(self.model.classes_)
        if 1 not in classes:
            return np.zeros(len(probabilities), dtype=float)
        return probabilities[:, classes.index(1)].ravel()

    def coefficients(self) -> Dict[str, float]:
        """Expose standardised coefficients for auditability."""
        if not self.is_fitted:
            return {}
        values = np.asarray(self.model.coef_).ravel()
        return {
            name: float(value)
            for name, value in zip(self.feature_order, values)
        }

    def save(self, path: str) -> str:
        """Persist the fusion model with joblib."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str,
             logger: Optional[logging.Logger] = None) -> "ProbabilityFusion":
        """Load a persisted fusion model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fusion model not found at {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Object at {path} is not a ProbabilityFusion")
        if logger is not None:
            instance.logger = logger
        return instance


def evaluate_probability_fusion(y_true: np.ndarray,
                                scores: np.ndarray) -> Dict[str, Any]:
    """Small metric bundle for fused probabilities."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    true = np.asarray(y_true, dtype=int).ravel()
    probabilities = np.asarray(scores, dtype=float).ravel()
    return {
        "brier_score": float(brier_score_loss(true, probabilities)),
        "roc_auc": float(roc_auc_score(true, probabilities)),
        "average_precision": float(average_precision_score(true, probabilities)),
        "mean_predicted_probability": float(probabilities.mean()),
        "positive_rate": float(true.mean()) if len(true) else 0.0,
    }
