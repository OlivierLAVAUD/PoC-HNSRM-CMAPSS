# tests/test_alerting.py

import unittest

import numpy as np

from src.evaluation.alerting import (
    compare_alerting,
    consecutive_confirmation,
    smoothed_predictions,
)


class TestAlerting(unittest.TestCase):
    def test_consecutive_confirmation_per_unit(self):
        unit_ids = np.array([1, 1, 1, 1, 1, 2, 2, 2])
        cycles = np.array([1, 2, 3, 4, 5, 1, 2, 3])
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])

        confirmed = consecutive_confirmation(predictions, unit_ids, cycles, k=2)
        np.testing.assert_array_equal(
            confirmed, np.array([0, 0, 1, 1, 0, 0, 0, 0])
        )

    def test_smoothing_uses_trailing_window_and_resets_per_unit(self):
        unit_ids = np.array([1, 1, 1, 1, 2, 2])
        cycles = np.array([1, 2, 3, 4, 1, 2])
        scores = np.array([0.9, 0.9, 0.2, 0.9, 0.9, 0.1])

        predictions = smoothed_predictions(
            scores, unit_ids, cycles, threshold=0.6, window=2
        )
        np.testing.assert_array_equal(
            predictions, np.array([0, 1, 0, 0, 0, 0])
        )

    def test_compare_reports_immediate_confirmed_and_smoothed(self):
        unit_ids = np.array([1] * 4)
        cycles = np.arange(1, 5)
        rul = np.array([4, 3, 2, 1])
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.9, 0.9])

        report = compare_alerting(
            y_true, scores, unit_ids, cycles, rul, threshold=0.85,
            consecutive_ks=(2, 3), smoothing_cycles=3,
        )

        self.assertEqual(report['threshold'], 0.85)
        self.assertIn('immediate', report['modes'])
        self.assertIn('consecutive_2', report['modes'])
        self.assertIn('consecutive_3', report['modes'])
        self.assertIn('smoothed_3', report['modes'])
        self.assertGreater(
            report['modes']['consecutive_2']['mean_precision'],
            report['modes']['immediate']['mean_precision'],
        )
        self.assertIn(
            'delta_vs_immediate', report['modes']['consecutive_2']
        )

    def test_rejects_invalid_confirmation_length(self):
        with self.assertRaises(ValueError):
            consecutive_confirmation(
                np.array([1, 1]), np.array([1]), np.array([1, 2]), k=2
            )


if __name__ == '__main__':
    unittest.main()
