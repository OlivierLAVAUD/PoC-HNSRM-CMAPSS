# tests/test_hysteresis.py

import unittest

import numpy as np

from src.evaluation.alerting import (
    compare_alerting,
    hysteresis_predictions,
    resolve_hysteresis_thresholds,
)


class TestHysteresis(unittest.TestCase):
    def test_alert_stays_on_between_exit_and_enter(self):
        scores = np.array([0.1, 0.6, 0.4, 0.3, 0.1, 0.2])
        units = np.ones(6, dtype=int)
        cycles = np.arange(1, 7)
        predictions = hysteresis_predictions(scores, units, cycles, 0.5, 0.2)
        np.testing.assert_array_equal(
            predictions, np.array([0, 1, 1, 1, 0, 0])
        )

    def test_state_is_reset_per_engine(self):
        scores = np.array([0.9, 0.9, 0.1, 0.1])
        units = np.array([1, 1, 2, 2])
        cycles = np.array([1, 2, 1, 2])
        predictions = hysteresis_predictions(scores, units, cycles, 0.5, 0.2)
        np.testing.assert_array_equal(
            predictions, np.array([1, 1, 0, 0])
        )

    def test_thresholds_are_derived_and_clipped(self):
        self.assertEqual(
            resolve_hysteresis_thresholds(0.5, None, None, 0.05),
            (0.55, 0.45),
        )
        enter, exit_value = resolve_hysteresis_thresholds(
            0.98, None, None, 0.05
        )
        self.assertEqual(enter, 1.0)
        self.assertAlmostEqual(exit_value, 0.93)
        enter, exit_value = resolve_hysteresis_thresholds(
            0.02, None, None, 0.05
        )
        self.assertAlmostEqual(enter, 0.07)
        self.assertEqual(exit_value, 0.0)

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaises(ValueError):
            hysteresis_predictions(
                np.zeros(2), np.ones(2, dtype=int), np.arange(2), 0.2, 0.5
            )

    def test_compare_alerting_adds_hysteresis_mode(self):
        units = np.array([1, 1, 1, 1, 1, 1])
        cycles = np.arange(1, 7)
        y_true = np.array([0, 0, 1, 1, 1, 1])
        rul = np.array([30, 25, 20, 15, 10, 5])
        scores = np.array([0.1, 0.6, 0.45, 0.45, 0.3, 0.1])
        report = compare_alerting(
            y_true, scores, units, cycles, rul, 0.5,
            consecutive_ks=[2], smoothing_cycles=1,
            hysteresis={'enabled': True, 'margin': 0.05},
        )
        self.assertEqual(report['hysteresis_thresholds']['enter_threshold'], 0.55)
        self.assertEqual(report['hysteresis_thresholds']['exit_threshold'], 0.45)
        self.assertIn('hysteresis', report['modes'])
        self.assertIn('delta_vs_immediate', report['modes']['hysteresis'])


if __name__ == '__main__':
    unittest.main()
