# src/reasoning/temporal.py
"""Temporal feature extraction and training-only threshold estimation."""

from typing import Any, Dict, List, Optional

import numpy as np


def _persistence_count(values: np.ndarray, threshold: Optional[float],
                       above: bool) -> int:
    """Count consecutive end-of-window cycles above/below a threshold."""
    if threshold is None:
        return 0
    count = 0
    for value in values[::-1]:
        if (above and value > threshold) or ((not above) and value < threshold):
            count += 1
        else:
            break
    return count


def window_temporal_features(window: np.ndarray, feature_names: List[str],
                             high_thresholds: Optional[Dict[str, float]] = None,
                             low_thresholds: Optional[Dict[str, float]] = None,
                             moving_average_cycles: int = 5) -> Dict[str, Any]:
    """Compute slope, relative change, moving average, std and persistence."""
    values = np.asarray(window, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(
            f"Expected window shape (n, {len(feature_names)}), got {values.shape}"
        )
    n_cycles = values.shape[0]
    x = np.arange(n_cycles, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered ** 2)) or 1.0
    slopes = (x_centered[:, None] * (values - values.mean(axis=0))).sum(axis=0) / denominator
    first = values[0]
    last = values[-1]
    scale = np.maximum(np.abs(first), 1e-6)
    relative = (last - first) / scale
    ma_window = max(1, min(int(moving_average_cycles), n_cycles))
    moving_average = values[-ma_window:].mean(axis=0)
    std = values.std(axis=0)

    high_thresholds = high_thresholds or {}
    low_thresholds = low_thresholds or {}
    features = {
        'slope': {},
        'relative_change': {},
        'moving_average': {},
        'std': {},
        'persistence_above': {},
        'persistence_below': {},
    }
    for index, name in enumerate(feature_names):
        features['slope'][name] = float(slopes[index])
        features['relative_change'][name] = float(relative[index])
        features['moving_average'][name] = float(moving_average[index])
        features['std'][name] = float(std[index])
        features['persistence_above'][name] = _persistence_count(
            values[:, index], high_thresholds.get(name), above=True
        )
        features['persistence_below'][name] = _persistence_count(
            values[:, index], low_thresholds.get(name), above=False
        )
    return features


def compute_temporal_thresholds(windows: np.ndarray, feature_names: List[str],
                                high_thresholds: Dict[str, float],
                                low_thresholds: Dict[str, float],
                                percentile: float = 75.0,
                                moving_average_cycles: int = 5) -> Dict[str, Dict[str, float]]:
    """Estimate temporal thresholds from training windows only.

    Slope, relative-change and standard-deviation thresholds use the requested
    percentile of absolute values. Persistence is the percentile of the
    consecutive-cycle counts, with a floor of 3 and a ceiling of the window
    length.
    """
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D windows, got shape {windows.shape}")
    if windows.shape[2] != len(feature_names):
        raise ValueError("Feature count does not match feature_names.")
    if len(windows) == 0:
        raise ValueError("Cannot compute temporal thresholds from zero windows.")

    n_cycles = windows.shape[1]
    x = np.arange(n_cycles, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered ** 2)) or 1.0
    slopes = (
        x_centered[None, :, None]
        * (windows - windows.mean(axis=1, keepdims=True))
    ).sum(axis=1) / denominator
    first = windows[:, 0, :]
    last = windows[:, -1, :]
    relative = (last - first) / np.maximum(np.abs(first), 1e-6)
    ma_window = max(1, min(int(moving_average_cycles), n_cycles))
    std = windows.std(axis=1)

    thresholds: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(feature_names):
        above_counts = np.array([
            _persistence_count(window[:, index], high_thresholds.get(name), True)
            for window in windows
        ])
        below_counts = np.array([
            _persistence_count(window[:, index], low_thresholds.get(name), False)
            for window in windows
        ])
        persistence_counts = np.maximum(above_counts, below_counts)
        persistence = int(np.percentile(persistence_counts, percentile))
        thresholds[name] = {
            'slope_abs': float(np.percentile(np.abs(slopes[:, index]), percentile)),
            'relative_change_abs': float(
                np.percentile(np.abs(relative[:, index]), percentile)
            ),
            'std': float(np.percentile(np.abs(std[:, index]), percentile)),
            'persistence': int(max(3, min(n_cycles, persistence))),
            'moving_average_cycles': int(ma_window),
            'training_windows': int(len(windows)),
        }
    return thresholds
