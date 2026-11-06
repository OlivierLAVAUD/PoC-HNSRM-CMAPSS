# tests/test_nested_validation.py

import json
import os
import tempfile
import unittest

import numpy as np

from src.fusion.nested_validation import run_nested_validation
from src.fusion.probability_fusion import ProbabilityFusion

FEATURES = [
    'neural_probability', 'symbolic_rule_score',
    'problog_failure_risk', 'predicted_rul',
]


def build_dataset(n_units=8, per_unit=12, seed=3):
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


class TestNestedValidation(unittest.TestCase):
    def test_report_is_validation_only_and_exported(self):
        matrix, y, units = build_dataset()
        config = {
            'threshold_strategy': 'fbeta', 'beta': 2.0,
            'min_precision': 0.75, 'fallback': 0.5,
            'nested_validation': {'n_splits': 4, 'random_state': 42},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_nested_validation(
                matrix, y, units, FEATURES, config, tmpdir
            )
            path = os.path.join(tmpdir, 'fusion_nested_validation.json')
            with open(path) as file:
                persisted = json.load(file)
        self.assertEqual(report['scope'], 'validation_engines_only')
        self.assertEqual(report['folds_evaluated'], 4)
        self.assertEqual(persisted['folds_evaluated'], 4)
        for key in ('f1', 'recall', 'precision', 'roc_auc', 'threshold'):
            self.assertIn(key, report['metrics'])

    def test_single_class_is_rejected(self):
        matrix, _, units = build_dataset()
        with self.assertRaises(ValueError):
            run_nested_validation(
                matrix, np.zeros(len(units), dtype=int), units, FEATURES,
                {'nested_validation': {'n_splits': 4}},
            )


if __name__ == '__main__':
    unittest.main()
