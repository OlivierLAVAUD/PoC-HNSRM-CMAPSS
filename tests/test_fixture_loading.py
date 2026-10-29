"""Smoke tests for the tiny FD001 fixture (no external C-MAPSS data required).

These tests keep the provenance / threshold / loader code paths runnable in CI
without the official ``data/raw/CMAPSS/FD001_processed.parquet`` or SWI-Prolog.
"""

import os
import unittest

import numpy as np

from src.data.cmapss_loader import CMAPSSLoader

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "tiny_fd001.parquet"
)


class TestTinyFixture(unittest.TestCase):
    def setUp(self):
        self.assertTrue(os.path.exists(FIXTURE), "fixture parquet missing")
        self.loader = CMAPSSLoader(
            FIXTURE, window_size=10, rul_threshold=15, stride=1,
        )

    def test_load_builds_windows(self):
        X, y = self.loader.load_data()
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[2], 14)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)
        self.assertIn(y.dtype.kind, "iub")

    def test_split_by_unit_keeps_units_disjoint(self):
        X, y = self.loader.load_data()
        self.loader.split_by_unit(X, y, validation_split=0.25, test_split=0.25)
        masks = self.loader.last_split_masks
        self.assertIsNotNone(masks)
        for name in ("train", "val", "test"):
            self.assertIn(name, masks)
        units = self.loader.unit_ids
        train_units = set(np.unique(units[masks["train"]]))
        val_units = set(np.unique(units[masks["val"]]))
        test_units = set(np.unique(units[masks["test"]]))
        self.assertTrue(train_units.isdisjoint(val_units))
        self.assertTrue(train_units.isdisjoint(test_units))
        self.assertTrue(val_units.isdisjoint(test_units))


if __name__ == "__main__":
    unittest.main()
