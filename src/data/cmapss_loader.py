# src/data/cmapss_loader.py

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# FD001 sensors that are constant or quasi-constant are excluded.
USEFUL_SENSORS: List[str] = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11',
    's12', 's13', 's14', 's15', 's17', 's20', 's21',
]
EXCLUDED_SENSORS: List[str] = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']

RAW_COLUMNS: List[str] = (
    ['unit_number', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    + [f's{i}' for i in range(1, 22)]
)


class CMAPSSLoader:
    """Load FD001 trajectories and build windowed binary failure-risk samples.

    The preferred input is the TwinnableAgent processed parquet file, which
    already contains an ``rul`` column. Raw NASA ``train_FD001.txt`` and
    ``test_FD001.txt`` files are also supported.
    """

    def __init__(self, data_file: str, window_size: int = 30,
                 feature_names: Optional[List[str]] = None,
                 rul_threshold: int = 30, stride: int = 1,
                 test_file: Optional[str] = None, rul_file: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Configure the loader; ``rul_threshold`` defines the positive class."""
        self.data_file = data_file
        self.window_size = int(window_size)
        self.feature_names = list(feature_names or USEFUL_SENSORS)
        self.rul_threshold = int(rul_threshold)
        self.stride = max(1, int(stride))
        self.test_file = test_file
        self.rul_file = rul_file
        self.logger = logger or logging.getLogger(__name__)
        self.unit_ids: Optional[np.ndarray] = None
        self.window_end_cycles: Optional[np.ndarray] = None

    def _read_frame(self, path: str) -> pd.DataFrame:
        """Read a processed parquet or a raw NASA whitespace text file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"CMAPSS file not found: {path}")

        if path.lower().endswith('.parquet'):
            frame = pd.read_parquet(path)
            rename_map = {'unit': 'unit_number', 'op1': 'op_setting_1',
                          'op2': 'op_setting_2', 'op3': 'op_setting_3'}
            frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
        else:
            frame = pd.read_csv(path, sep=r'\s+', header=None, names=RAW_COLUMNS,
                                engine='python')

        missing = [c for c in ['unit_number', 'cycle'] + self.feature_names
                   if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required CMAPSS columns: {missing}")

        if 'rul' not in frame.columns:
            max_cycle = frame.groupby('unit_number')['cycle'].transform('max')
            frame['rul'] = (max_cycle - frame['cycle']).astype(int)

        return frame.sort_values(['unit_number', 'cycle']).reset_index(drop=True)

    def _read_test_frame(self, path: str) -> pd.DataFrame:
        """Read a test trajectory file and attach true RUL when available."""
        frame = self._read_frame(path)
        if self.rul_file and os.path.exists(self.rul_file):
            base_rul = pd.read_csv(self.rul_file, sep=r'\s+', header=None,
                                   names=['rul']).reset_index(drop=True)
            units = sorted(frame['unit_number'].unique())
            if len(base_rul) != len(units):
                raise ValueError(
                    f"RUL file has {len(base_rul)} rows but test file has {len(units)} units"
                )
            base_map = {unit: int(base_rul.iloc[i]['rul']) for i, unit in enumerate(units)}
            max_cycle = frame.groupby('unit_number')['cycle'].transform('max')
            frame['rul'] = frame['unit_number'].map(base_map) + (max_cycle - frame['cycle'])
        return frame

    def _window_frame(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
        """Create per-unit sliding windows, labels, unit ids and end cycles."""
        X_list, y_list, unit_list, cycle_list = [], [], [], []
        for unit, group in frame.groupby('unit_number', sort=True):
            values = group[self.feature_names].to_numpy(dtype=np.float32)
            rul = group['rul'].to_numpy(dtype=np.int64)
            cycles = group['cycle'].to_numpy(dtype=np.int64)
            if len(values) < self.window_size:
                self.logger.debug("Skipping unit %s with %d cycles", unit, len(values))
                continue
            for start in range(0, len(values) - self.window_size + 1, self.stride):
                end = start + self.window_size
                X_list.append(values[start:end])
                y_list.append(1 if rul[end - 1] <= self.rul_threshold else 0)
                unit_list.append(unit)
                cycle_list.append(int(cycles[end - 1]))

        if not X_list:
            raise ValueError(
                f"No windows produced: window_size={self.window_size} exceeds trajectory lengths"
            )
        return (np.asarray(X_list, dtype=np.float32),
                np.asarray(y_list, dtype=np.int64),
                np.asarray(unit_list),
                np.asarray(cycle_list, dtype=np.int64))

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training windows as ``X (samples, window_size, features)`` and ``y``."""
        frame = self._read_frame(self.data_file)
        X, y, units, cycles = self._window_frame(frame)
        self.unit_ids = units
        self.window_end_cycles = cycles
        self.logger.info(
            "Loaded %d training windows from %d units (%d positive)",
            len(X), len(np.unique(units)), int(y.sum()),
        )
        return X, y

    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load held-out test windows from ``test_file`` when configured."""
        if not self.test_file:
            raise ValueError("test_file was not provided to CMAPSSLoader")
        frame = self._read_test_frame(self.test_file)
        X, y, units, cycles = self._window_frame(frame)
        self.unit_ids = units
        self.window_end_cycles = cycles
        self.logger.info("Loaded %d test windows from %d units", len(X), len(np.unique(units)))
        return X, y

    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return already-windowed data unchanged (DataLoader compatibility)."""
        if X.ndim == 3:
            return X, y
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, validation_split: float,
                   test_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, np.ndarray]:
        """Chronological split compatible with ``DataLoader.split_data``."""
        total = len(X)
        test_size = int(total * test_split)
        validation_size = int(total * validation_split)
        X_test, y_test = X[:test_size], y[:test_size]
        X_val = X[test_size:test_size + validation_size]
        y_val = y[test_size:test_size + validation_size]
        X_train = X[test_size + validation_size:]
        y_train = y[test_size + validation_size:]
        self.logger.info(
            "Split: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_by_unit(self, X: np.ndarray, y: np.ndarray,
                      validation_split: float = 0.2, test_split: float = 0.1,
                      random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray, np.ndarray]:
        """Split whole engine trajectories to avoid window-level leakage."""
        if self.unit_ids is None or len(self.unit_ids) != len(X):
            raise ValueError("unit_ids unavailable; call load_data() before split_by_unit()")
        units = np.array(sorted(np.unique(self.unit_ids)))
        rng = np.random.default_rng(random_state)
        rng.shuffle(units)

        n_test = int(len(units) * test_split)
        n_val = int(len(units) * validation_split)
        if test_split > 0:
            n_test = max(1, n_test)
        if validation_split > 0:
            n_val = max(1, n_val)
        if n_test + n_val >= len(units):
            raise ValueError("test_split and validation_split leave no training units")

        test_units = set(units[:n_test])
        val_units = set(units[n_test:n_test + n_val])
        train_units = set(units[n_test + n_val:])

        masks = {
            'train': np.isin(self.unit_ids, list(train_units)),
            'val': np.isin(self.unit_ids, list(val_units)),
            'test': np.isin(self.unit_ids, list(test_units)),
        }
        self.logger.info(
            "Unit split: train=%d val=%d test=%d units",
            len(train_units), len(val_units), len(test_units),
        )
        return (
            X[masks['train']], X[masks['val']], X[masks['test']],
            y[masks['train']], y[masks['val']], y[masks['test']],
        )

    def describe_percentiles(self) -> pd.DataFrame:
        """Return training percentiles for rule-threshold justification."""
        frame = self._read_frame(self.data_file)
        return frame[self.feature_names].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
