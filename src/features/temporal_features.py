"""Explicit temporal features for the LightGBM complementary classifier.

The original RandomForest path consumes a flattened sensor window.  LightGBM
additionally receives interpretable trend, variability, persistence and
moving-average features.  Every statistic is computed from the raw window;
percentile thresholds used by the persistence features are fitted on the
training split only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np

FEATURE_VERSION = 1
STAT_NAMES = (
    "first",
    "last",
    "mean",
    "std",
    "min",
    "max",
    "slope",
    "relative_change",
    "ma_mean",
    "ma_std",
    "ma_delta",
    "persistence_above",
    "persistence_below",
)


def _as_windows(windows: np.ndarray, n_features: Optional[int] = None) -> np.ndarray:
    """Validate a ``(samples, window_size, features)`` array."""
    array = np.asarray(windows, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(
            f"Temporal features expect 3D windows, got shape {array.shape}."
        )
    if array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError("Temporal windows must have at least one cycle and one sensor.")
    if not np.isfinite(array).all():
        raise ValueError(
            "Temporal feature extraction received NaN or infinite values. "
            "Apply the configured data_quality policy first."
        )
    if n_features is not None and array.shape[2] != n_features:
        raise ValueError(
            f"Expected {n_features} sensors, got {array.shape[2]}."
        )
    return array


def _slope(series: np.ndarray) -> float:
    """Least-squares slope over normalised window time."""
    length = len(series)
    if length < 2:
        return 0.0
    time = np.linspace(0.0, 1.0, length, dtype=np.float64)
    centred = time - time.mean()
    denominator = float(np.dot(centred, centred))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centred, series - series.mean()) / denominator)


def _trailing_count(series: np.ndarray, threshold: float,
                    above: bool) -> int:
    """Count consecutive values above/below a threshold from the window end."""
    count = 0
    for value in series[::-1]:
        active = value > threshold if above else value < threshold
        if not active:
            break
        count += 1
    return count


class TemporalFeatureExtractor:
    """Build interpretable temporal features and optionally keep the raw window.

    Parameters
    ----------
    feature_names:
        Sensor names in the same order as the last window axis.
    include_raw_window:
        Append every raw sensor value, grouped by sensor and then by cycle.
    moving_average_cycles:
        Length of the trailing moving average used by the MA statistics.
    percentile_low / percentile_high:
        Training-only percentiles used by persistence features.
    """

    def __init__(self, feature_names: Iterable[str],
                 include_raw_window: bool = True,
                 moving_average_cycles: int = 5,
                 percentile_low: float = 25.0,
                 percentile_high: float = 75.0):
        self.feature_names = [str(name) for name in feature_names]
        if not self.feature_names:
            raise ValueError("feature_names must not be empty.")
        self.include_raw_window = bool(include_raw_window)
        self.moving_average_cycles = max(1, int(moving_average_cycles))
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        if not 0.0 <= self.percentile_low <= 100.0:
            raise ValueError("percentile_low must be in [0, 100].")
        if not 0.0 <= self.percentile_high <= 100.0:
            raise ValueError("percentile_high must be in [0, 100].")
        if self.percentile_low >= self.percentile_high:
            raise ValueError("percentile_low must be smaller than percentile_high.")
        self.window_size: Optional[int] = None
        self.low_thresholds: Dict[str, float] = {}
        self.high_thresholds: Dict[str, float] = {}
        self.feature_names_out: List[str] = []
        self._fitted = False

    def _stat_names(self) -> List[str]:
        return [f"{sensor}__{stat}" for sensor in self.feature_names
                for stat in STAT_NAMES]

    def _raw_names(self) -> List[str]:
        if not self.include_raw_window or self.window_size is None:
            return []
        return [f"raw__{sensor}__{cycle:02d}"
                for sensor in self.feature_names
                for cycle in range(self.window_size)]

    def _refresh_output_names(self) -> None:
        self.feature_names_out = self._stat_names() + self._raw_names()

    def fit(self, windows: np.ndarray) -> "TemporalFeatureExtractor":
        """Fit percentile thresholds using training windows only."""
        array = _as_windows(windows)
        self.window_size = int(array.shape[1])
        values = array.reshape(-1, array.shape[2])
        self.low_thresholds = {
            name: float(np.percentile(values[:, index], self.percentile_low))
            for index, name in enumerate(self.feature_names)
        }
        self.high_thresholds = {
            name: float(np.percentile(values[:, index], self.percentile_high))
            for index, name in enumerate(self.feature_names)
        }
        self._refresh_output_names()
        self._fitted = True
        return self

    def _extract_stats(self, window: np.ndarray) -> List[float]:
        features: List[float] = []
        ma_length = min(self.moving_average_cycles, len(window))
        for index, sensor in enumerate(self.feature_names):
            series = np.asarray(window[:, index], dtype=np.float64)
            first = float(series[0])
            last = float(series[-1])
            mean = float(series.mean())
            std = float(series.std())
            minimum = float(series.min())
            maximum = float(series.max())
            slope = _slope(series)
            relative_change = float(
                (last - first) / (abs(first) + 1e-8)
            )
            ma_mean = float(series[-ma_length:].mean())
            ma_std = float(series[-ma_length:].std())
            ma_delta = float(
                ma_mean - series[:ma_length].mean()
            )
            persistence_above = _trailing_count(
                series, self.high_thresholds[sensor], above=True
            )
            persistence_below = _trailing_count(
                series, self.low_thresholds[sensor], above=False
            )
            features.extend([
                first, last, mean, std, minimum, maximum, slope,
                relative_change, ma_mean, ma_std, ma_delta,
                float(persistence_above), float(persistence_below),
            ])
        return features

    def _extract_raw(self, window: np.ndarray) -> List[float]:
        if not self.include_raw_window:
            return []
        return [
            float(value)
            for sensor_index in range(window.shape[1])
            for value in window[:, sensor_index]
        ]

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Return the temporal feature matrix for raw windows."""
        if not self._fitted:
            raise RuntimeError("TemporalFeatureExtractor must be fitted first.")
        array = _as_windows(windows, n_features=len(self.feature_names))
        if array.shape[1] != self.window_size:
            raise ValueError(
                f"Expected window_size={self.window_size}, got {array.shape[1]}."
            )
        rows = []
        for window in array:
            row = self._extract_stats(window)
            row.extend(self._extract_raw(window))
            rows.append(row)
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape[1] != len(self.feature_names_out):
            raise RuntimeError(
                "Temporal feature metadata is inconsistent with extracted features."
            )
        return matrix

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Fit the training thresholds then transform the same windows."""
        return self.fit(windows).transform(windows)

    def metadata(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description of the fitted extractor."""
        return {
            "version": FEATURE_VERSION,
            "feature_names": list(self.feature_names),
            "feature_names_out": list(self.feature_names_out),
            "window_size": self.window_size,
            "include_raw_window": self.include_raw_window,
            "moving_average_cycles": self.moving_average_cycles,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "low_thresholds": dict(self.low_thresholds),
            "high_thresholds": dict(self.high_thresholds),
            "statistics": list(STAT_NAMES),
            "raw_order": "sensor-major, then cycle",
        }

    def save_metadata(self, path: str) -> str:
        """Persist the training thresholds and output feature names."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as file:
            json.dump(self.metadata(), file, indent=2)
        return path

    def save(self, path: str) -> str:
        """Persist the fitted extractor with joblib."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str) -> "TemporalFeatureExtractor":
        """Load a fitted extractor from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Temporal feature extractor not found at {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Object at {path} is not a TemporalFeatureExtractor")
        return instance
