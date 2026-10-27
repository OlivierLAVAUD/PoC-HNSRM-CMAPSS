# tests/test_cmapss_loader.py

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.cmapss_loader import CMAPSSLoader, RAW_COLUMNS, USEFUL_SENSORS


def _make_frame(n_units=3, length=40):
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, length + 1):
            row = {
                'unit': unit,
                'cycle': cycle,
                'op1': 0.0,
                'op2': 0.0,
                'op3': 100.0,
                'rul': length - cycle,
            }
            for i in range(1, 22):
                row[f's{i}'] = float(i * 10 + cycle)
            rows.append(row)
    return pd.DataFrame(rows)


class TestCMAPSSLoader(unittest.TestCase):
    def test_parquet_loading_labels_and_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'FD001_processed.parquet')
            _make_frame().to_parquet(path, index=False)

            loader = CMAPSSLoader(path, window_size=5, rul_threshold=5)
            X, y = loader.load_data()

            expected_windows = 3 * (40 - 5 + 1)
            self.assertEqual(X.shape, (expected_windows, 5, len(USEFUL_SENSORS)))
            self.assertEqual(y.shape, (expected_windows,))
            self.assertGreater(int(y.sum()), 0)
            self.assertEqual(len(set(loader.unit_ids.tolist())), 3)

    def test_unit_split_is_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'FD001_processed.parquet')
            _make_frame().to_parquet(path, index=False)

            loader = CMAPSSLoader(path, window_size=5, rul_threshold=5)
            X, y = loader.load_data()
            X_tr, X_va, X_te, y_tr, y_va, y_te = loader.split_by_unit(X, y)

            self.assertEqual(len(X_tr) + len(X_va) + len(X_te), len(X))
            self.assertEqual(len(y_tr) + len(y_va) + len(y_te), len(y))
            self.assertGreater(len(X_tr), 0)
            self.assertGreater(len(X_te), 0)

    def test_raw_text_loading_computes_rul(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'train_FD001.txt')
            frame = _make_frame()
            raw = frame[['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]]
            raw.to_csv(path, sep=' ', header=False, index=False)

            loader = CMAPSSLoader(path, window_size=5, rul_threshold=5)
            X, y = loader.load_data()
            self.assertEqual(X.shape, (3 * 36, 5, 14))
            self.assertGreater(int(y.sum()), 0)
            self.assertTrue(set(RAW_COLUMNS).issuperset({'unit_number', 'cycle'}))


if __name__ == '__main__':
    unittest.main()
