# tests/test_fusion_diagnostics.py

import json
import os
import tempfile
import unittest

import numpy as np

from src.fusion.diagnostics import (
    ablation_study,
    cross_validate_fusion,
    group_folds,
    run_fusion_diagnostics,
)
from src.fusion.probability_fusion import ProbabilityFusion

FEATURES = [
    'neural_probability', 'symbolic_rule_score',
    'problog_failure_risk', 'predicted_rul',
]
CONFIG = {'n_splits': 4, 'n_repeats': 1, 'random_state': 42}
THRESHOLD_CONFIG = {
    'threshold_strategy': 'fbeta', 'beta': 2.0,
    'min_precision': 0.75, 'fallback': 0.5,
}


def build_dataset(n_units=12, per_unit=20, seed=0):
    """Return an informative dataset with one pure-noise fusion input."""
    rng = np.random.default_rng(seed)
    total = n_units * per_unit
    neural = rng.uniform(0.0, 1.0, total)
    symbolic = rng.uniform(0.0, 1.0, total)
    problog = rng.uniform(0.0, 1.0, total)
    rul = 60.0 - 50.0 * neural
    labels = ((neural + symbolic) > 1.0).astype(int)
    units = np.repeat(np.arange(1, n_units + 1), per_unit)
    matrix = ProbabilityFusion.build_matrix(
        neural, symbolic, problog, rul, feature_order=FEATURES
    )
    return matrix, labels, units


class TestGroupFolds(unittest.TestCase):
    def test_folds_hold_out_whole_engines(self):
        matrix, _, units = build_dataset()
        folds = group_folds(units, n_splits=4, n_repeats=1, random_state=7)
        self.assertEqual(len(folds), 4)
        covered = np.concatenate([fold['test_indices'] for fold in folds])
        np.testing.assert_array_equal(np.sort(covered), np.arange(len(units)))
        assigned = []
        for fold in folds:
            assigned.extend(fold['test_units'])
        self.assertEqual(sorted(assigned), sorted(set(units)))
        for fold in folds:
            test_units = set(units[fold['test_indices']])
            train_units = set(units[fold['train_indices']])
            self.assertFalse(test_units & train_units)

    def test_repeats_are_reproducible(self):
        _, _, units = build_dataset()
        first = group_folds(units, 4, 2, random_state=11)
        second = group_folds(units, 4, 2, random_state=11)
        self.assertEqual(len(first), 8)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(
                left['test_indices'], right['test_indices']
            )

    def test_invalid_fold_counts_are_rejected(self):
        _, _, units = build_dataset(n_units=3)
        with self.assertRaises(ValueError):
            group_folds(units, n_splits=5)
        with self.assertRaises(ValueError):
            group_folds(units, n_splits=1)
        with self.assertRaises(ValueError):
            group_folds(units, n_splits=2, n_repeats=0)


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        self.matrix, self.y, self.units = build_dataset()

    def test_reports_coefficient_and_threshold_stability(self):
        report = cross_validate_fusion(
            self.matrix, self.y, self.units, FEATURES, CONFIG,
            THRESHOLD_CONFIG,
        )
        self.assertEqual(report['folds_evaluated'], 4)
        self.assertEqual(set(report['coefficients']), set(FEATURES))
        for stats in report['coefficients'].values():
            self.assertEqual(stats['count'], 4)
            self.assertGreaterEqual(stats['std'], 0.0)
        self.assertEqual(report['threshold_stability']['count'], 4)
        self.assertGreaterEqual(report['threshold_stability']['ci95_low'], 0.0)
        total_importance = sum(report['relative_importance'].values())
        self.assertAlmostEqual(total_importance, 1.0)
        self.assertGreater(
            report['relative_importance']['neural_probability'],
            report['relative_importance']['problog_failure_risk'],
        )
        self.assertEqual(report['scope'], 'validation_engines_only')

    def test_single_class_is_rejected(self):
        with self.assertRaises(ValueError):
            cross_validate_fusion(
                self.matrix, np.zeros(len(self.y), dtype=int), self.units,
                FEATURES, CONFIG, THRESHOLD_CONFIG,
            )


class TestAblation(unittest.TestCase):
    def test_removing_an_informative_input_hurts(self):
        matrix, y, units = build_dataset()
        report = ablation_study(matrix, y, units, FEATURES, CONFIG,
                                THRESHOLD_CONFIG)
        by_feature = {
            item['removed_feature']: item for item in report['ablations']
        }
        self.assertEqual(by_feature['neural_probability']['verdict'], 'improves')
        self.assertLess(
            by_feature['neural_probability']['delta_fbeta'], 0.0
        )
        self.assertIn(
            'neural_probability', report['summary']['helpful_features']
        )
        noise = by_feature['problog_failure_risk']
        self.assertNotEqual(noise['verdict'], 'improves')
        self.assertNotIn(
            'problog_failure_risk', report['summary']['helpful_features']
        )
        self.assertLess(abs(noise['delta_fbeta']), 0.05)
        self.assertIn('F-beta', report['summary']['answer'])
        self.assertEqual(
            set(report['full']['features']), set(FEATURES)
        )

    def test_ablation_needs_two_inputs(self):
        matrix, y, units = build_dataset()
        with self.assertRaises(ValueError):
            ablation_study(matrix[:, :1], y, units,
                           ['neural_probability'], CONFIG, THRESHOLD_CONFIG)


class TestDiagnosticsExport(unittest.TestCase):
    def test_reports_are_written(self):
        matrix, y, units = build_dataset()
        config = {
            'fusion': {
                'threshold_strategy': 'fbeta', 'beta': 2.0,
                'min_precision': 0.75, 'fallback': 0.5,
                'validation': CONFIG,
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_fusion_diagnostics(
                matrix, y, units, FEATURES, config, tmpdir
            )
            diagnostics_path = os.path.join(tmpdir, 'fusion_diagnostics.json')
            ablation_path = os.path.join(tmpdir, 'fusion_ablation.json')
            self.assertTrue(os.path.exists(diagnostics_path))
            self.assertTrue(os.path.exists(ablation_path))
            self.assertTrue(summary['applied'])
            with open(ablation_path) as file:
                payload = json.load(file)
            self.assertIn('summary', payload)
            self.assertTrue(payload['ablations'])


if __name__ == '__main__':
    unittest.main()
