# tests/test_condition_adaptation.py

import unittest

import numpy as np

from src.experiments.condition_adaptation import (
    ConditionNormalizer,
    achieved_alert_rate,
    select_alert_rate_threshold,
)


class TestConditionNormalizer(unittest.TestCase):
    def test_standardises_each_condition_separately(self):
        rng = np.random.default_rng(42)
        op_settings = np.vstack([
            np.zeros((60, 3)),
            np.full((60, 3), 5.0),
        ])
        X = rng.normal(size=(120, 8, 2))
        X[:60] += 10.0
        X[60:] -= 7.0

        normalizer = ConditionNormalizer.fit(
            op_settings, X, n_conditions=2, random_state=42
        )
        transformed = normalizer.transform(X, op_settings)

        self.assertEqual(normalizer.n_conditions, 2)
        self.assertEqual(sorted(normalizer.condition_counts), [60, 60])
        for condition in range(2):
            rows = transformed[normalizer.labels == condition]
            self.assertLess(float(np.abs(rows.mean(axis=(0, 1))).max()), 1e-5)
            self.assertLess(
                float(np.abs(rows.std(axis=(0, 1)) - 1.0).max()), 1e-5
            )

    def test_single_condition_falls_back_to_global_standardisation(self):
        rng = np.random.default_rng(7)
        op_settings = np.zeros((30, 3))
        X = rng.normal(loc=4.0, scale=2.0, size=(30, 5, 3))

        normalizer = ConditionNormalizer.fit(
            op_settings, X, n_conditions=6, random_state=42
        )
        transformed = normalizer.transform(X, op_settings)

        self.assertEqual(normalizer.n_conditions, 1)
        self.assertEqual(normalizer.condition_counts, [30])
        self.assertLess(float(np.abs(transformed.mean()).max()), 1e-5)
        self.assertLess(
            float(np.abs(transformed.std(axis=(0, 1)) - 1.0).max()), 1e-5
        )

    def test_rejects_mismatched_or_non_finite_inputs(self):
        with self.assertRaises(ValueError):
            ConditionNormalizer.fit(np.zeros((4, 3)), np.zeros((3, 2, 1)), 2)
        with self.assertRaises(ValueError):
            ConditionNormalizer.fit(
                np.zeros((4, 3)), np.zeros((4, 2, 1)), 0
            )
        X = np.zeros((4, 2, 1))
        X[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            ConditionNormalizer.fit(np.zeros((4, 3)), X, 2)

    def test_transform_requires_matching_operating_settings(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(10, 4, 2))
        normalizer = ConditionNormalizer.fit(np.zeros((10, 3)), X, 1)
        with self.assertRaises(ValueError):
            normalizer.transform(X, np.zeros((9, 3)))


class TestAlertRateThreshold(unittest.TestCase):
    def test_threshold_matches_requested_alert_rate(self):
        scores = np.linspace(0.0, 1.0, 101)
        for rate in (0.1, 0.25, 0.5, 0.9):
            threshold = select_alert_rate_threshold(scores, rate)
            self.assertAlmostEqual(
                achieved_alert_rate(scores, threshold), rate, delta=0.011,
            )

    def test_extreme_rates_are_supported(self):
        scores = np.array([0.0, 0.5, 1.0])
        self.assertGreater(
            select_alert_rate_threshold(scores, 0.0), float(scores.max())
        )
        self.assertEqual(select_alert_rate_threshold(scores, 1.0), 0.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            select_alert_rate_threshold([], 0.5)
        with self.assertRaises(ValueError):
            select_alert_rate_threshold([0.1, np.nan], 0.5)
        with self.assertRaises(ValueError):
            select_alert_rate_threshold([0.1, 0.2], 1.5)
        with self.assertRaises(ValueError):
            achieved_alert_rate([], 0.5)


if __name__ == '__main__':
    unittest.main()
