# src/experiments/condition_adaptation.py
"""Label-free adaptation helpers for cross-condition C-MAPSS evaluation.

C-MAPSS FD002/FD004 alternate between several operating conditions.  A model
trained on a single-condition source therefore sees two shifts at once: the
sensor scale changes per condition and the score distribution changes with it.

This module implements two *unsupervised* corrections:

* :class:`ConditionNormalizer` clusters the target operating settings, fits a
  per-condition mean/standard deviation on the target features and returns
  condition-relative z-scores.  No target label is used.  On the source side,
  FD001 has a single condition, so its ordinary standardisation is already the
  condition-relative transform the model was trained on.
* :func:`select_alert_rate_threshold` chooses a target threshold that matches
  the alert rate observed on the source validation split.  It uses target
  scores only, never target labels.

Both operations are transductive: they inspect the unlabelled target features.
Reports must say so explicitly; this is not a zero-shot protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class ConditionNormalizer:
    """Per-condition z-score fitted on unlabelled target windows."""

    n_conditions: int
    random_state: int
    labels: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    threshold: float = 1e-8
    op_scaler: Optional[StandardScaler] = None
    kmeans: Optional[KMeans] = None

    @classmethod
    def fit(cls, op_settings: Any, X: Any, n_conditions: int,
            random_state: int = 42) -> 'ConditionNormalizer':
        """Fit condition clusters and per-condition feature statistics.

        ``op_settings`` has shape ``(windows, 3)`` and ``X`` has shape
        ``(windows, window_size, features)``.  Only these arrays are used;
        labels are deliberately absent from the signature.
        """
        settings = np.asarray(op_settings, dtype=np.float64)
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError(
                "ConditionNormalizer expects X with shape "
                "(windows, window_size, features)."
            )
        if settings.ndim != 2 or len(settings) != len(data):
            raise ValueError(
                "op_settings must have shape (windows, 3) and match X."
            )
        if len(data) == 0:
            raise ValueError("Cannot fit a condition normalizer on no windows.")
        if not np.isfinite(settings).all() or not np.isfinite(data).all():
            raise ValueError(
                "Condition normalization requires finite target features and "
                "operating settings."
            )

        requested = int(n_conditions)
        if requested < 1:
            raise ValueError("n_conditions must be an integer >= 1.")
        requested = min(requested, len(data))

        unique_settings = np.unique(settings, axis=0)
        effective = min(requested, len(unique_settings))
        if effective <= 1:
            labels = np.zeros(len(data), dtype=np.int64)
            op_scaler = None
            kmeans = None
            effective = 1
        else:
            op_scaler = StandardScaler()
            op_scaled = op_scaler.fit_transform(settings)
            kmeans = KMeans(
                n_clusters=effective, random_state=int(random_state),
                n_init=10,
            )
            labels = kmeans.fit_predict(op_scaled).astype(np.int64)

        features = data.shape[-1]
        means = np.zeros((effective, features), dtype=np.float64)
        scales = np.ones((effective, features), dtype=np.float64)
        for condition in range(effective):
            rows = data[labels == condition].reshape(-1, features)
            if len(rows) == 0:
                continue
            means[condition] = rows.mean(axis=0)
            deviation = rows.std(axis=0)
            deviation[~np.isfinite(deviation)] = 1.0
            deviation[deviation < 1e-8] = 1.0
            scales[condition] = deviation

        return cls(
            n_conditions=effective,
            random_state=int(random_state),
            labels=labels,
            means=means,
            scales=scales,
            op_scaler=op_scaler,
            kmeans=kmeans,
        )

    def assign(self, op_settings: Any) -> np.ndarray:
        """Return the condition cluster of each window."""
        settings = np.asarray(op_settings, dtype=np.float64)
        if self.kmeans is None or self.op_scaler is None:
            return np.zeros(len(settings), dtype=np.int64)
        return self.kmeans.predict(
            self.op_scaler.transform(settings)
        ).astype(np.int64)

    def transform(self, X: Any, op_settings: Any) -> np.ndarray:
        """Standardise each window with its own condition statistics."""
        data = np.asarray(X, dtype=np.float64)
        labels = self.assign(op_settings)
        if len(labels) != len(data):
            raise ValueError("op_settings and X must contain the same windows.")
        output = np.empty_like(data, dtype=np.float32)
        for condition in range(self.n_conditions):
            mask = labels == condition
            if not np.any(mask):
                continue
            output[mask] = (
                (data[mask] - self.means[condition])
                / self.scales[condition]
            ).astype(np.float32)
        return output

    @property
    def condition_counts(self) -> List[int]:
        """Return the number of windows assigned to each condition."""
        return np.bincount(
            self.labels, minlength=self.n_conditions
        ).astype(int).tolist()


def select_alert_rate_threshold(scores: Any,
                                target_alert_rate: float) -> float:
    """Return a threshold whose predicted alert rate matches the target.

    ``target_alert_rate`` is normally the alert rate observed on the source
    validation split.  Target labels are not accepted, by design.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("Cannot select a threshold from empty scores.")
    if not np.isfinite(values).all():
        raise ValueError("Scores contain NaN or infinite values.")
    rate = float(target_alert_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("target_alert_rate must be in [0, 1].")
    if rate == 0.0:
        return float(np.max(values) + 1e-12)
    if rate == 1.0:
        return float(np.min(values))
    # Classifier scores are often tied.  Choose the threshold whose achieved
    # rate is closest to the requested rate instead of a raw sorted index,
    # which can land on a large tie plateau and over-alert.
    ordered = np.sort(values)
    unique, first = np.unique(ordered, return_index=True)
    rates = (len(ordered) - first) / len(ordered)
    distances = np.abs(rates - rate)
    candidates = np.flatnonzero(distances == distances.min())
    index = int(candidates[-1])
    return float(unique[index])


def achieved_alert_rate(scores: Any, threshold: float) -> float:
    """Return the share of windows at or above ``threshold``."""
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("Cannot compute an alert rate from empty scores.")
    return float(np.mean(values >= float(threshold)))
