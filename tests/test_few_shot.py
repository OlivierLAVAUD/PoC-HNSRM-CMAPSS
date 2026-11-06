# tests/test_few_shot.py

import unittest

import numpy as np

from src.experiments.few_shot import (
    aggregate_threshold_selections,
    balance_validation_prevalence,
    fit_score_calibrator,
    select_few_shot_threshold,
    split_engines,
    validate_few_shot_config,
)
from src.evaluation.thresholding import select_threshold


class TestFewShotHelpers(unittest.TestCase):
    def test_engine_split_is_disjoint_and_complete(self):
        units = np.repeat(np.arange(1, 9), 3)
        calibration, validation = split_engines(units, 0.5, 0.5, 42)
        self.assertFalse(calibration & validation)
        self.assertEqual(calibration | validation, set(range(1, 9)))

    def test_rejects_single_engine(self):
        with self.assertRaises(ValueError):
            split_engines(np.array([1, 1, 1]), 0.5, 0.5)

    def test_sigmoid_and_isotonic_calibration_are_bounded(self):
        rng = np.random.default_rng(3)
        scores = rng.uniform(size=200)
        y = (scores + rng.normal(0, 0.15, 200) > 0.5).astype(int)
        for method in ('sigmoid', 'isotonic'):
            calibrator = fit_score_calibrator(scores, y, method=method)
            calibrated = calibrator.predict(scores)
            self.assertEqual(calibrated.shape, scores.shape)
            self.assertTrue(np.isfinite(calibrated).all())
            self.assertGreaterEqual(calibrated.min(), -0.01)
            self.assertLessEqual(calibrated.max(), 1.01)

    def test_auto_calibration_requires_validation_for_selection(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        y = np.array([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            fit_score_calibrator(scores, y, method='auto')

    def test_threshold_is_selected_on_target_validation_only(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
        y = np.array([1, 1, 1, 1, 0, 0])
        selection = select_threshold(
            y, scores, strategy='f1', min_precision=0.5,
            beta=2.0, fallback=0.5,
        )
        self.assertIn('threshold', selection)
        self.assertGreaterEqual(selection['threshold'], 0.0)

    def test_degenerate_precision_threshold_falls_back_to_fbeta(self):
        # A single high-score island satisfies precision but collapses recall.
        scores = np.array([0.99, 0.20, 0.21, 0.22, 0.23, 0.24])
        y = np.array([1, 1, 1, 1, 0, 0])
        selection = select_few_shot_threshold(
            y, scores, strategy='recall_at_precision', min_precision=0.75,
        )
        self.assertIn('guard_fallback', selection)
        self.assertEqual(selection['strategy'], 'fbeta')
        self.assertLess(selection['threshold'], 0.99)
        self.assertEqual(
            selection['guard_fallback']['selection_split'],
            'target_validation',
        )

    def test_repeated_selection_uses_median_threshold(self):
        rows = [
            {'threshold': 0.1, 'selected_strategy': 'fbeta'},
            {'threshold': 0.4, 'selected_strategy': 'fbeta'},
            {'threshold': 0.9, 'selected_strategy': 'f1'},
        ]
        report = aggregate_threshold_selections(rows, aggregation='median')
        self.assertEqual(report['n_repeats'], 3)
        self.assertAlmostEqual(report['selected_threshold'], 0.4)
        self.assertEqual(report['representative_repeat'], 1)
        self.assertEqual(report['strategy_counts'], {'fbeta': 2, 'f1': 1})
        self.assertFalse(report['test_used_for_selection'])
        self.assertEqual(report['selection_split'], 'target_validation')

    def test_single_aggregation_uses_first_selection(self):
        rows = [
            {'threshold': 0.3, 'selected_strategy': 'f1'},
            {'threshold': 0.8, 'selected_strategy': 'fbeta'},
        ]
        report = aggregate_threshold_selections(rows, aggregation='single')
        self.assertEqual(report['selected_threshold'], 0.3)
        self.assertEqual(report['representative_repeat'], 0)

    def test_prevalence_control_downsamples_positives_deterministically(self):
        units = np.repeat(np.arange(1, 5), 50)
        cycles = np.tile(np.arange(50), 4)
        rul = np.tile(np.arange(50, 0, -1), 4)
        y = (rul <= 10).astype(int)
        scores = np.linspace(0.0, 1.0, len(y))
        first = balance_validation_prevalence(
            y, scores, units, cycles, rul,
            positive_rate=0.10, random_state=7,
        )
        second = balance_validation_prevalence(
            y, scores, units, cycles, rul,
            positive_rate=0.10, random_state=7,
        )
        self.assertTrue(first['metadata']['applied'])
        self.assertEqual(first['indices'].tolist(), second['indices'].tolist())
        self.assertAlmostEqual(
            first['metadata']['achieved_positive_rate'], 0.10, places=2
        )
        self.assertEqual(
            first['metadata']['kept_negatives'],
            first['metadata']['original_negatives'],
        )
        self.assertFalse(first['metadata']['test_used_for_selection'])

    def test_prevalence_control_is_optional(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        units = np.array([1, 1, 1, 1])
        cycles = np.arange(4)
        result = balance_validation_prevalence(
            y, scores, units, cycles, np.arange(4, 0, -1),
        )
        self.assertFalse(result['metadata']['applied'])
        self.assertEqual(result['indices'].tolist(), [0, 1, 2, 3])
        self.assertEqual(result['metadata']['sampling'], 'none')

    def test_config_rejects_test_selection_and_bad_fractions(self):
        with self.assertRaises(ValueError):
            validate_few_shot_config({
                'target_dataset': 'fd003',
                'selection_split': 'test',
            })
        with self.assertRaises(ValueError):
            validate_few_shot_config({
                'target_dataset': 'fd003',
                'calibration_fraction': 0.6,
                'validation_fraction': 0.6,
            })
        with self.assertRaises(ValueError):
            validate_few_shot_config({
                'target_dataset': 'fd001',
                'calibration_fraction': 0.5,
                'validation_fraction': 0.5,
            })
        with self.assertRaises(ValueError):
            validate_few_shot_config({
                'target_dataset': 'fd003',
                'selection_repeats': 0,
            })
        with self.assertRaises(ValueError):
            validate_few_shot_config({
                'target_dataset': 'fd003',
                'threshold_aggregation': 'mean',
            })
        with self.assertRaises(ValueError):
            validate_few_shot_config({
                'target_dataset': 'fd003',
                'validation_positive_rate': 0.0,
            })
        with self.assertRaises(ValueError):
            validate_few_shot_config({
                'target_dataset': 'fd003',
                'validation_positive_rate': True,
            })


if __name__ == '__main__':
    unittest.main()
