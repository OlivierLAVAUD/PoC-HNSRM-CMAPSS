# src/monitoring/drift.py
"""Detect distribution drift between training data and incoming windows.

The reference statistics are computed on the training windows only: per-sensor
mean and standard deviation, the 5/25/50/75/95 percentiles and the observed
minimum and maximum. Incoming windows are compared on three axes:

- the share of values falling outside the training min/max bounds;
- the share of NaN/Inf values;
- the standardized mean shift and the standard-deviation ratio, which explain
  *how* the distribution moved and are reported for interpretation.

A drift never blocks inference. ``ok`` means the incoming windows look like the
training distribution, ``warning`` asks for monitoring, and ``critical`` asks
for a human review of the affected windows.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

PERCENTILES: Sequence[int] = (5, 25, 50, 75, 95)
REFERENCE_KEYS: Sequence[str] = (
    'mean', 'std', 'min', 'max', 'p5', 'p25', 'p50', 'p75', 'p95',
)


def validate_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate the drift settings and return normalised values."""
    config = config or {}
    warning = float(config.get('warning_threshold', 0.05))
    critical = float(config.get('critical_threshold', 0.15))
    min_windows = int(config.get('min_windows', 30))
    if warning < 0.0 or critical < 0.0:
        raise ValueError("monitoring.drift thresholds must be >= 0.")
    if warning >= critical:
        raise ValueError(
            "monitoring.drift.warning_threshold must be strictly smaller "
            "than critical_threshold."
        )
    if min_windows < 1:
        raise ValueError("monitoring.drift.min_windows must be >= 1.")
    return {
        'warning_threshold': warning,
        'critical_threshold': critical,
        'min_windows': min_windows,
    }


def drift_level(score: float, warning_threshold: float,
                critical_threshold: float) -> str:
    """Map a drift score to ``ok``, ``warning`` or ``critical``."""
    if float(score) >= float(critical_threshold):
        return 'critical'
    if float(score) >= float(warning_threshold):
        return 'warning'
    return 'ok'


def _distribution(values: np.ndarray) -> Dict[str, Any]:
    """Return a compact distribution summary for one array of values."""
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    total = int(len(array))
    if not len(finite):
        return {
            'count': 0, 'total_values': total,
            'non_finite_proportion': 1.0 if total else 0.0,
            'mean': None, 'std': None, 'min': None, 'max': None,
            **{f'p{percentile}': None for percentile in PERCENTILES},
        }
    return {
        'count': int(len(finite)),
        'total_values': total,
        'non_finite_proportion': float(1.0 - len(finite) / total)
        if total else 0.0,
        'mean': float(finite.mean()),
        'std': float(finite.std()),
        'min': float(finite.min()),
        'max': float(finite.max()),
        **{
            f'p{percentile}': float(np.percentile(finite, percentile))
            for percentile in PERCENTILES
        },
    }


def reference_statistics(X_train: np.ndarray,
                         feature_names: Sequence[str]) -> Dict[str, Any]:
    """Compute per-sensor reference statistics on training windows only."""
    windows = np.asarray(X_train, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D training windows, got {windows.shape}.")
    if windows.shape[2] != len(feature_names):
        raise ValueError("feature_names does not match the window width.")
    if len(windows) == 0:
        raise ValueError("Cannot build drift reference from zero windows.")
    flattened = windows.reshape(-1, windows.shape[-1])
    features = {}
    for index, name in enumerate(feature_names):
        values = flattened[:, index]
        stats = _distribution(values)
        features[name] = {
            key: stats[key] for key in REFERENCE_KEYS
        }
        features[name]['non_finite_proportion'] = stats['non_finite_proportion']
    return {
        'source': 'training_split_windows',
        'windows': int(len(windows)),
        'window_size': int(windows.shape[1]),
        'values_per_feature': int(flattened.shape[0]),
        'feature_names': list(feature_names),
        'features': features,
        'created_at': datetime.now().isoformat(),
    }


def compare_features(reference: Dict[str, Any], X_incoming: np.ndarray,
                     feature_names: Sequence[str]) -> Dict[str, Any]:
    """Compare incoming windows with the reference statistics per sensor."""
    windows = np.asarray(X_incoming, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D incoming windows, got {windows.shape}.")
    if windows.shape[2] != len(feature_names):
        raise ValueError("feature_names does not match the window width.")
    reference_features = reference.get('features', {})
    missing = [name for name in feature_names if name not in reference_features]
    if missing:
        raise ValueError(f"Reference statistics missing for: {missing}.")

    flattened = windows.reshape(-1, windows.shape[-1])
    comparison = {}
    for index, name in enumerate(feature_names):
        values = flattened[:, index]
        stats = _distribution(values)
        expected = reference_features[name]
        finite = values[np.isfinite(values)]
        if len(finite):
            lower = float(expected['min'])
            upper = float(expected['max'])
            out_of_bounds = float(
                np.mean((finite < lower) | (finite > upper))
            )
        else:
            out_of_bounds = 1.0
        reference_std = float(expected['std'] or 0.0)
        mean_shift = (
            float(stats['mean'] - expected['mean'])
            if stats['mean'] is not None and expected['mean'] is not None
            else None
        )
        comparison[name] = {
            'incoming': stats,
            'reference_mean': float(expected['mean']),
            'reference_std': reference_std,
            'reference_bounds': [float(expected['min']), float(expected['max'])],
            'mean_shift': mean_shift,
            'mean_shift_z': (
                None if mean_shift is None or reference_std == 0.0
                else float(mean_shift / reference_std)
            ),
            'std_ratio': (
                None if not reference_std or stats['std'] is None
                else float(stats['std'] / reference_std)
            ),
            'out_of_bounds_proportion': out_of_bounds,
            'non_finite_proportion': float(stats['non_finite_proportion']),
        }
    return comparison


def _signal_summary(values: Optional[Iterable[float]]) -> Optional[Dict[str, Any]]:
    """Summarise a predicted-signal distribution for the report."""
    if values is None:
        return None
    array = np.asarray(list(values), dtype=float).ravel()
    if not len(array):
        return None
    summary = _distribution(array)
    return {
        key: summary[key]
        for key in ('count', 'mean', 'std', 'min', 'max',
                    'p5', 'p25', 'p50', 'p75', 'p95')
    }


def analyse_drift(X_train: Optional[np.ndarray], X_incoming: np.ndarray,
                  feature_names: Sequence[str],
                  config: Optional[Dict[str, Any]] = None,
                  label: str = 'incoming',
                  signals: Optional[Dict[str, Optional[Iterable[float]]]] = None,
                  reference: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """Compare incoming windows with the training distribution.

    Returns a report with an ``ok`` / ``warning`` / ``critical`` level. The
    overall score is the largest per-sensor share of out-of-bounds or
    non-finite values, so a single badly shifted sensor is not averaged away.
    """
    settings = validate_config(config)
    if reference is None:
        if X_train is None:
            raise ValueError(
                "analyse_drift needs either training windows or a saved "
                "reference to compare against."
            )
        reference = reference_statistics(X_train, feature_names)
    incoming_windows = np.asarray(X_incoming, dtype=np.float32)
    comparison = compare_features(reference, incoming_windows, feature_names)

    per_feature_scores = {
        name: max(
            values['out_of_bounds_proportion'],
            values['non_finite_proportion'],
        )
        for name, values in comparison.items()
    }
    score = float(max(per_feature_scores.values())) if per_feature_scores else 0.0
    level = drift_level(
        score, settings['warning_threshold'], settings['critical_threshold']
    )
    warning_features = [
        name for name, value in per_feature_scores.items()
        if value >= settings['warning_threshold']
    ]
    critical_features = [
        name for name, value in per_feature_scores.items()
        if value >= settings['critical_threshold']
    ]

    windows = int(len(incoming_windows))
    insufficient = windows < settings['min_windows']
    status = 'insufficient_data' if insufficient else level
    if insufficient:
        level = 'ok'

    non_finite_proportion = float(np.mean(
        ~np.isfinite(incoming_windows)
    )) if incoming_windows.size else 0.0
    hardest = sorted(
        per_feature_scores.items(), key=lambda item: item[1], reverse=True
    )[:3]

    if insufficient:
        recommendation = (
            f"Only {windows} incoming windows were provided; at least "
            f"{settings['min_windows']} are needed before drift can be "
            "assessed. No action taken."
        )
    elif level == 'critical':
        recommendation = (
            "Do not block inference. Route the affected windows to the human "
            "review queue, re-check the sensors "
            f"({', '.join(critical_features)}) and plan a recalibration."
        )
    elif level == 'warning':
        recommendation = (
            "Monitor the affected sensors "
            f"({', '.join(warning_features)}); widen the human review band if "
            "the shift persists over the next cycles."
        )
    else:
        recommendation = (
            "Incoming windows match the training distribution; no action."
        )

    return {
        'status': status,
        'level': level,
        'score': score,
        'thresholds': {
            'warning': settings['warning_threshold'],
            'critical': settings['critical_threshold'],
            'min_windows': settings['min_windows'],
        },
        'reference': {
            'windows': reference.get('windows'),
            'window_size': reference.get('window_size'),
            'source': reference.get('source'),
        },
        'incoming': {
            'label': label,
            'windows': windows,
            'non_finite_proportion': non_finite_proportion,
            'features': comparison,
        },
        'feature_scores': per_feature_scores,
        'most_drifted_features': [
            {'feature': name, 'score': float(value)} for name, value in hardest
        ],
        'warning_features': warning_features,
        'critical_features': critical_features,
        'signals': {
            name: _signal_summary(values)
            for name, values in (signals or {}).items()
        },
        'hitl_recommended': bool(level in ('warning', 'critical')),
        'recommendation': recommendation,
        'timestamp': datetime.now().isoformat(),
    }


def save_drift_report(report: Dict[str, Any], path: str) -> str:
    """Persist a drift report as JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(report, file, indent=2)
    return path


def save_reference(reference: Dict[str, Any], path: str) -> str:
    """Persist the training reference statistics for later inference runs."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(reference, file, indent=2)
    return path


def load_reference(path: str) -> Dict[str, Any]:
    """Load training reference statistics produced by a pipeline run."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Drift reference not found at {path}")
    with open(path) as file:
        return json.load(file)


def check_inference_drift(X_incoming: np.ndarray,
                          feature_names: Sequence[str],
                          results_dir: str,
                          config: Optional[Dict[str, Any]] = None,
                          label: str = 'inference',
                          signals: Optional[Dict[str, Optional[Iterable[float]]]]
                          = None) -> Optional[Dict[str, Any]]:
    """Run drift detection from a saved reference, or return ``None``."""
    reference_path = os.path.join(results_dir, 'drift_reference.json')
    if not os.path.exists(reference_path):
        return None
    reference = load_reference(reference_path)
    return analyse_drift(
        None, X_incoming, feature_names, config,
        label=label, signals=signals, reference=reference,
    )
