# src/features/physical_representation.py
"""Physical, condition-aware representations for C-MAPSS windows.

The representation is fitted on training windows only and persisted with its
parameters, feature names and a hash of the training inputs.  Variants:

``raw``
    Flattened raw window, the control condition.
``deltas``
    Dynamics without a baseline: 5/10/30-cycle deltas, robust slope, standard
    deviation, persistence and relative variation.
``early_life_residual``
    Per-engine residual against the first ``early_life_cycles`` cycles.
``operating_condition_residual``
    Per-regime residual against training medians/MADs fitted on source windows.
``residual_plus_deltas``
    Residual representation plus the delta features.

Residual z-scores are clipped for numerical stability.  A window whose
early-life baseline is unavailable is returned as neutral zeros, with an
explicit ``early_life_available`` indicator (or raises when the configured
fallback policy is ``error``).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

VARIANTS = (
    'raw',
    'deltas',
    'early_life_residual',
    'operating_condition_residual',
    'residual_plus_deltas',
)

DEFAULT_CONFIG: Dict[str, Any] = {
    'variant': 'raw',
    'early_life_cycles': 20,
    'epsilon': 1.0e-6,
    'fallback_policy': 'neutral',
    'include_raw_features': True,
    'clip': 10.0,
    'n_conditions': None,
    'random_state': 42,
}

DELTA_STAT_NAMES = (
    'delta_5', 'delta_10', 'delta_30', 'slope', 'std',
    'persistence', 'relative_variation',
)


def _as_windows(X: Any) -> np.ndarray:
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            "Physical representation expects windows with shape "
            "(samples, window_size, features)."
        )
    if array.shape[0] == 0:
        raise ValueError("Physical representation received no windows.")
    if not np.isfinite(array).all():
        raise ValueError(
            "Physical representation received NaN or infinite values."
        )
    return array


def _mad(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    return np.median(np.abs(values - median), axis=0)


def _persistence(series: np.ndarray) -> float:
    """Share of trailing differences that keep the latest direction."""
    if len(series) < 2:
        return 0.0
    differences = np.diff(series)
    sign = np.sign(differences[-1])
    if sign == 0:
        return 0.0
    count = 0
    for value in differences[::-1]:
        if np.sign(value) != sign:
            break
        count += 1
    return float(count / len(differences))


class PhysicalRepresentation:
    """Fit and apply one physical representation variant."""

    def __init__(self, feature_names: List[str], variant: str = 'raw',
                 early_life_cycles: int = 20, epsilon: float = 1.0e-6,
                 fallback_policy: str = 'neutral',
                 include_raw_features: bool = True, clip: float = 10.0,
                 n_conditions: Optional[int] = None,
                 random_state: int = 42):
        self.feature_names = [str(name) for name in feature_names]
        if not self.feature_names:
            raise ValueError("feature_names must not be empty.")
        self.variant = str(variant).lower()
        if self.variant not in VARIANTS:
            raise ValueError(
                f"Unsupported physical representation: {variant!r}. "
                f"Allowed: {list(VARIANTS)}."
            )
        self.early_life_cycles = int(early_life_cycles)
        if self.early_life_cycles < 2:
            raise ValueError("early_life_cycles must be an integer >= 2.")
        self.epsilon = float(epsilon)
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be a positive finite value.")
        self.fallback_policy = str(fallback_policy).lower()
        if self.fallback_policy not in {'neutral', 'error'}:
            raise ValueError("fallback_policy must be 'neutral' or 'error'.")
        self.include_raw_features = bool(include_raw_features)
        self.clip = float(clip)
        if self.clip <= 0.0:
            raise ValueError("clip must be positive.")
        self.n_conditions = (
            None if n_conditions is None else int(n_conditions)
        )
        if self.n_conditions is not None and self.n_conditions < 1:
            raise ValueError("n_conditions must be an integer >= 1.")
        self.random_state = int(random_state)

        self.window_size: Optional[int] = None
        self.n_features: Optional[int] = None
        self.feature_names_out: List[str] = []
        self.training_hash: Optional[str] = None
        self.baseline_units: Dict[str, Any] = {}
        self.condition_scaler: Optional[StandardScaler] = None
        self.condition_kmeans: Optional[KMeans] = None
        self.condition_medians: Optional[np.ndarray] = None
        self.condition_mads: Optional[np.ndarray] = None
        self.condition_counts: Optional[List[int]] = None
        self._fitted = False

    @classmethod
    def from_config(cls, feature_names: List[str],
                    config: Dict[str, Any]) -> 'PhysicalRepresentation':
        merged = dict(DEFAULT_CONFIG)
        merged.update(config or {})
        return cls(
            feature_names,
            variant=merged['variant'],
            early_life_cycles=merged['early_life_cycles'],
            epsilon=merged['epsilon'],
            fallback_policy=merged['fallback_policy'],
            include_raw_features=merged['include_raw_features'],
            clip=merged['clip'],
            n_conditions=merged['n_conditions'],
            random_state=merged['random_state'],
        )

    def _hash_inputs(self, X: np.ndarray, units: Optional[np.ndarray],
                     cycles: Optional[np.ndarray],
                     op_settings: Optional[np.ndarray]) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(X, dtype=np.float32).tobytes())
        for values in (units, cycles, op_settings):
            if values is not None:
                digest.update(np.asarray(values).tobytes())
        digest.update(json.dumps({
            'variant': self.variant,
            'early_life_cycles': self.early_life_cycles,
            'epsilon': self.epsilon,
            'fallback_policy': self.fallback_policy,
            'include_raw_features': self.include_raw_features,
            'clip': self.clip,
            'n_conditions': self.n_conditions,
            'features': self.feature_names,
        }, sort_keys=True).encode('utf-8'))
        return digest.hexdigest()

    def _refresh_names(self) -> None:
        if self.variant == 'raw':
            self.feature_names_out = [
                f'raw__{sensor}__{cycle:02d}'
                for sensor in self.feature_names
                for cycle in range(self.window_size or 0)
            ]
            return
        delta_names = self._delta_names()
        if self.variant == 'deltas':
            self.feature_names_out = delta_names
            return
        residual_names = [
            f'residual__{sensor}__{cycle:02d}'
            for sensor in self.feature_names
            for cycle in range(self.window_size or 0)
        ]
        indicator = (
            'early_life_available'
            if self.variant == 'early_life_residual'
            else 'condition_available'
        )
        if self.variant == 'residual_plus_deltas':
            self.feature_names_out = (
                residual_names + delta_names + [indicator]
            )
        else:
            self.feature_names_out = residual_names + [indicator]

    def _delta_names(self) -> List[str]:
        names = [
            f'{stat}__{sensor}'
            for sensor in self.feature_names for stat in DELTA_STAT_NAMES
        ]
        if self.include_raw_features:
            names.extend(f'raw_last__{sensor}' for sensor in self.feature_names)
        return names

    def _unit_baselines(self, X: np.ndarray, units: np.ndarray,
                        cycles: Optional[np.ndarray],
                        record: bool = False
                        ) -> Dict[int, Dict[str, Any]]:
        baselines: Dict[int, Dict[str, Any]] = {}
        for unit in sorted(np.unique(units)):
            mask = np.flatnonzero(np.asarray(units) == unit)
            if not len(mask):
                continue
            unit_cycles = None
            if cycles is not None:
                position = mask[np.argmin(np.asarray(cycles)[mask])]
                unit_cycles = int(np.max(np.asarray(cycles)[mask]))
            else:
                position = int(mask[0])
            window = X[int(position)]
            rows = window[:min(self.early_life_cycles, len(window))]
            # A window always holds window_size timesteps.  Availability must
            # therefore refer to the observed trajectory length when cycles
            # are provided, not to the window length.
            trajectory_length = (
                unit_cycles if unit_cycles is not None else len(window)
            )
            available = trajectory_length >= self.early_life_cycles
            if not available and self.fallback_policy == 'error':
                raise ValueError(
                    f"Unit {int(unit)} has only {trajectory_length} observed "
                    f"cycle(s), fewer than early_life_cycles="
                    f"{self.early_life_cycles}."
                )
            if len(rows) < 2:
                if self.fallback_policy == 'error':
                    raise ValueError(
                        f"Unit {int(unit)} has too few early-life cycles."
                    )
                baselines[int(unit)] = {
                    'median': np.zeros(X.shape[-1]),
                    'scale': np.ones(X.shape[-1]),
                    'available': 0,
                    'cycles': int(len(rows)),
                    'trajectory_cycles': trajectory_length,
                }
                continue
            median = np.median(rows, axis=0)
            scale = _mad(rows)
            scale[~np.isfinite(scale)] = 1.0
            scale[scale < self.epsilon] = 1.0
            baselines[int(unit)] = {
                'median': median,
                'scale': scale,
                'available': int(available),
                'cycles': int(len(rows)),
                'trajectory_cycles': trajectory_length,
            }
            if record:
                self.baseline_units[str(int(unit))] = {
                    'early_life_cycles': int(len(rows)),
                    'trajectory_cycles': trajectory_length,
                    'available': int(available),
                }
        return baselines

    def _fit_conditions(self, X: np.ndarray,
                        op_settings: np.ndarray) -> None:
        settings = np.asarray(op_settings, dtype=np.float64)
        if settings.ndim != 2 or len(settings) != len(X):
            raise ValueError(
                "op_settings must have shape (windows, 3) and match X."
            )
        if not np.isfinite(settings).all():
            raise ValueError("Operating settings contain non-finite values.")
        unique = np.unique(settings, axis=0)
        requested = self.n_conditions or len(unique)
        effective = max(1, min(int(requested), len(unique), len(X)))
        if effective == 1:
            labels = np.zeros(len(X), dtype=int)
            self.condition_scaler = None
            self.condition_kmeans = None
        else:
            self.condition_scaler = StandardScaler()
            scaled = self.condition_scaler.fit_transform(settings)
            self.condition_kmeans = KMeans(
                n_clusters=effective, random_state=self.random_state, n_init=10
            )
            labels = self.condition_kmeans.fit_predict(scaled).astype(int)
        features = X.shape[-1]
        medians = np.zeros((effective, features), dtype=np.float64)
        mads = np.ones((effective, features), dtype=np.float64)
        for condition in range(effective):
            rows = X[labels == condition].reshape(-1, features)
            if not len(rows):
                continue
            medians[condition] = np.median(rows, axis=0)
            deviation = _mad(rows)
            deviation[~np.isfinite(deviation)] = 1.0
            deviation[deviation < self.epsilon] = 1.0
            mads[condition] = deviation
        self.condition_medians = medians
        self.condition_mads = mads
        self.condition_counts = np.bincount(
            labels, minlength=effective
        ).astype(int).tolist()

    def fit(self, X: Any, units: Optional[Any] = None,
            cycles: Optional[Any] = None,
            op_settings: Optional[Any] = None) -> 'PhysicalRepresentation':
        array = _as_windows(X)
        self.window_size = int(array.shape[1])
        self.n_features = int(array.shape[2])
        unit_values = None if units is None else np.asarray(units).ravel()
        cycle_values = None if cycles is None else np.asarray(cycles).ravel()
        op_values = (
            None if op_settings is None
            else np.asarray(op_settings, dtype=np.float64)
        )
        if unit_values is not None and len(unit_values) != len(array):
            raise ValueError("units must contain one value per window.")
        if cycle_values is not None and len(cycle_values) != len(array):
            raise ValueError("cycles must contain one value per window.")
        if self.variant in {'early_life_residual'} and unit_values is None:
            raise ValueError(
                "early_life_residual requires unit_ids for every window."
            )
        if self.variant == 'operating_condition_residual':
            if op_values is None:
                raise ValueError(
                    "operating_condition_residual requires op_settings."
                )
            self._fit_conditions(array, op_values)
        if self.variant == 'residual_plus_deltas':
            if op_values is not None:
                self._fit_conditions(array, op_values)
            elif unit_values is None:
                raise ValueError(
                    "residual_plus_deltas requires op_settings or unit_ids."
                )
        if (self.variant in {'early_life_residual', 'residual_plus_deltas'}
                and unit_values is not None):
            self._unit_baselines(
                array, unit_values, cycle_values, record=True
            )
        self.training_hash = self._hash_inputs(
            array, unit_values, cycle_values, op_values
        )
        self._refresh_names()
        self._fitted = True
        return self

    def _delta_row(self, window: np.ndarray) -> List[float]:
        values: List[float] = []
        for index in range(window.shape[1]):
            series = np.asarray(window[:, index], dtype=np.float64)
            last = float(series[-1])
            values.extend([
                float(last - series[-1 - 5]) if len(series) > 5 else 0.0,
                float(last - series[-1 - 10]) if len(series) > 10 else 0.0,
                float(last - series[-31]) if len(series) > 30 else 0.0,
                float(np.median(np.diff(series))) if len(series) > 1 else 0.0,
                float(series.std()),
                _persistence(series),
                float(series.std() / (abs(series.mean()) + self.epsilon)),
            ])
        if self.include_raw_features:
            values.extend(float(window[-1, index])
                          for index in range(window.shape[1]))
        return values

    def _delta_matrix(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self._delta_row(window) for window in X], dtype=np.float32
        )

    def _residual_matrix(self, X: np.ndarray, units: Optional[np.ndarray],
                         cycles: Optional[np.ndarray],
                         op_settings: Optional[np.ndarray],
                         allow_early_life: bool) -> tuple:
        if op_settings is not None and self.condition_medians is not None:
            settings = np.asarray(op_settings, dtype=np.float64)
            if self.condition_kmeans is None:
                labels = np.zeros(len(X), dtype=int)
            else:
                labels = self.condition_kmeans.predict(
                    self.condition_scaler.transform(settings)
                ).astype(int)
            residual = np.empty_like(X, dtype=np.float64)
            for condition in range(len(self.condition_medians)):
                mask = labels == condition
                if not np.any(mask):
                    continue
                residual[mask] = (
                    (X[mask] - self.condition_medians[condition])
                    / self.condition_mads[condition]
                )
            return residual, np.ones(len(X), dtype=np.float32), 'condition'

        if not allow_early_life or units is None:
            raise ValueError(
                "Residual representation requires op_settings or unit_ids."
            )
        unit_values = np.asarray(units).ravel()
        baselines = self._unit_baselines(
            X, unit_values,
            None if cycles is None else np.asarray(cycles).ravel(),
        )
        residual = np.empty_like(X, dtype=np.float64)
        available = np.zeros(len(X), dtype=np.float32)
        for unit, baseline in baselines.items():
            mask = unit_values == unit
            residual[mask] = (
                (X[mask] - baseline['median']) / baseline['scale']
            )
            available[mask] = float(baseline['available'])
        return residual, available, 'early_life'

    def transform(self, X: Any, units: Optional[Any] = None,
                  cycles: Optional[Any] = None,
                  op_settings: Optional[Any] = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PhysicalRepresentation must be fitted first.")
        array = _as_windows(X)
        if (self.window_size is not None
                and array.shape[1] != self.window_size):
            raise ValueError(
                f"Expected window_size={self.window_size}, "
                f"got {array.shape[1]}."
            )
        if array.shape[2] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} sensors, "
                f"got {array.shape[2]}."
            )
        unit_values = None if units is None else np.asarray(units).ravel()
        cycle_values = None if cycles is None else np.asarray(cycles).ravel()
        op_values = (
            None if op_settings is None
            else np.asarray(op_settings, dtype=np.float64)
        )
        if unit_values is not None and len(unit_values) != len(array):
            raise ValueError("units must contain one value per window.")
        if cycle_values is not None and len(cycle_values) != len(array):
            raise ValueError("cycles must contain one value per window.")
        if op_values is not None and len(op_values) != len(array):
            raise ValueError("op_settings must contain one value per window.")

        if self.variant == 'raw':
            matrix = array.reshape(len(array), -1).astype(np.float32)
        elif self.variant == 'deltas':
            matrix = self._delta_matrix(array)
        else:
            allow_early = self.variant in {
                'early_life_residual', 'residual_plus_deltas'
            }
            residual, available, _ = self._residual_matrix(
                array, unit_values, cycle_values, op_values,
                allow_early_life=allow_early,
            )
            residual = np.clip(residual, -self.clip, self.clip)
            flattened = residual.reshape(len(array), -1).astype(np.float32)
            if self.variant == 'residual_plus_deltas':
                matrix = np.hstack([
                    flattened, self._delta_matrix(array),
                    available.reshape(-1, 1),
                ]).astype(np.float32)
            else:
                matrix = np.hstack([
                    flattened, available.reshape(-1, 1),
                ]).astype(np.float32)
        if matrix.shape[1] != len(self.feature_names_out):
            raise RuntimeError(
                "Physical representation metadata is inconsistent with "
                f"features: {matrix.shape[1]} vs "
                f"{len(self.feature_names_out)}."
            )
        return matrix

    def fit_transform(self, X: Any, units: Optional[Any] = None,
                      cycles: Optional[Any] = None,
                      op_settings: Optional[Any] = None) -> np.ndarray:
        return self.fit(
            X, units=units, cycles=cycles, op_settings=op_settings
        ).transform(
            X, units=units, cycles=cycles, op_settings=op_settings
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            'variant': self.variant,
            'feature_names': list(self.feature_names),
            'feature_names_out': list(self.feature_names_out),
            'early_life_cycles': self.early_life_cycles,
            'epsilon': self.epsilon,
            'fallback_policy': self.fallback_policy,
            'include_raw_features': self.include_raw_features,
            'clip': self.clip,
            'n_conditions': (
                None if self.condition_counts is None
                else len(self.condition_counts)
            ),
            'condition_counts': self.condition_counts,
            'baseline_units': self.baseline_units,
            'window_size': self.window_size,
            'training_hash': self.training_hash,
            'fitted': self._fitted,
        }

    def save(self, path: str) -> str:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        return path

    def save_metadata(self, path: str) -> str:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as file:
            json.dump(self.metadata(), file, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> 'PhysicalRepresentation':
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Physical representation artifact not found at {path}"
            )
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(
                f"Object at {path} is not a PhysicalRepresentation."
            )
        return instance
