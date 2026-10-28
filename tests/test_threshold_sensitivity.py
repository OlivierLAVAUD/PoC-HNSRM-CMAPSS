# tests/test_threshold_sensitivity.py

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from src.experiments.threshold_sensitivity import (
    build_candidates,
    candidate_label,
    evaluate_candidate,
    run_threshold_sensitivity,
)

CONFIG = {
    'thresholding': {
        'strategy': 'recall_at_precision',
        'min_precision': 0.75,
        'beta': 2.0,
        'fallback': 0.5,
    },
    'operational_policy': {
        'false_negative_cost': 10.0,
        'false_alarm_cost': 1.0,
        'inspection_capacity_per_cycle': None,
    },
    'threshold_sensitivity': {
        'min_precision_values': [0.65, 0.75, 0.85],
        'beta_values': [1.0, 2.0],
        'strategies': ['recall_at_precision', 'fbeta', 'f1'],
    },
}


def build_split(engines=6, cycles=12, seed=0):
    """Return a labelled synthetic split with separable scores."""
    rng = np.random.default_rng(seed)
    labels = []
    scores = []
    units = []
    cycle_values = []
    rul_values = []
    for unit in range(1, engines + 1):
        for cycle in range(1, cycles + 1):
            units.append(unit)
            cycle_values.append(cycle)
            remaining = cycles - cycle
            rul_values.append(remaining)
            if remaining <= 4:
                labels.append(1)
                scores.append(float(rng.uniform(0.65, 0.99)))
            else:
                labels.append(0)
                scores.append(float(rng.uniform(0.0, 0.45)))
    return (
        np.asarray(labels, dtype=int),
        np.asarray(scores, dtype=float),
        np.asarray(units),
        np.asarray(cycle_values, dtype=int),
        np.asarray(rul_values, dtype=int),
    )


class TestCandidateGrid(unittest.TestCase):
    def test_grid_expands_every_combination(self):
        candidates = build_candidates(CONFIG)
        self.assertEqual(len(candidates), 3 + 2 + 1)
        labels = {candidate_label(candidate) for candidate in candidates}
        self.assertIn('recall@prec>=0.75', labels)
        self.assertIn('fbeta(beta=2.00)', labels)
        self.assertIn('f1', labels)

    def test_duplicates_are_removed(self):
        config = {
            'threshold_sensitivity': {
                'min_precision_values': [0.75, 0.75],
                'beta_values': [2.0, 2.0],
                'strategies': ['recall_at_precision', 'f1', 'f1'],
            }
        }
        candidates = build_candidates(config)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [candidate['strategy'] for candidate in candidates],
            ['recall_at_precision', 'f1'],
        )

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            build_candidates({
                'threshold_sensitivity': {'strategies': ['accuracy']}
            })


class TestCandidateEvaluation(unittest.TestCase):
    def setUp(self):
        self.y_val, self.val_scores, self.units, self.cycles, self.rul = (
            build_split()
        )
        self.y_test, self.test_scores, _, _, _ = build_split(seed=1)

    def test_threshold_is_selected_on_validation_only(self):
        candidate = {
            'strategy': 'recall_at_precision', 'min_precision': 0.75,
            'beta': None,
        }
        baseline = evaluate_candidate(
            candidate, self.y_val, self.val_scores, self.units, self.cycles,
            self.rul, self.y_test, self.test_scores, self.units, self.cycles,
            self.rul, CONFIG['thresholding'], CONFIG['operational_policy'],
        )
        degraded = evaluate_candidate(
            candidate, self.y_val, self.val_scores, self.units, self.cycles,
            self.rul, self.y_test, np.zeros_like(self.test_scores), self.units,
            self.cycles, self.rul, CONFIG['thresholding'],
            CONFIG['operational_policy'],
        )
        self.assertAlmostEqual(baseline['threshold'], degraded['threshold'])
        self.assertNotAlmostEqual(
            baseline['test']['recall'], degraded['test']['recall']
        )

    def test_reports_alert_modes_and_engine_metrics(self):
        result = evaluate_candidate(
            {'strategy': 'fbeta', 'min_precision': None, 'beta': 2.0},
            self.y_val, self.val_scores, self.units, self.cycles, self.rul,
            self.y_test, self.test_scores, self.units, self.cycles, self.rul,
            CONFIG['thresholding'], CONFIG['operational_policy'],
        )
        self.assertEqual(result['selection']['strategy'], 'fbeta')
        self.assertGreater(result['threshold'], 0.0)
        self.assertIn('immediate', result['alert_modes']['test'])
        self.assertIn('recall', result['test'])
        self.assertGreaterEqual(result['test']['total_false_alarms'], 0)
        self.assertAlmostEqual(
            result['test']['fbeta_beta'], 2.0
        )


class TestSensitivityExport(unittest.TestCase):
    def test_json_csv_and_plot_are_written(self):
        y_val, val_scores, units, cycles, rul = build_split()
        y_test, test_scores, _, _, _ = build_split(seed=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_threshold_sensitivity(
                y_val, val_scores, units, cycles, rul,
                y_test, test_scores, units, cycles, rul,
                CONFIG, tmpdir,
            )
            self.assertTrue(os.path.exists(
                os.path.join(tmpdir, 'threshold_sensitivity.json')
            ))
            csv_path = os.path.join(tmpdir, 'threshold_sensitivity.csv')
            self.assertTrue(os.path.exists(csv_path))
            plot_path = os.path.join(
                tmpdir, 'visualization', 'threshold_sensitivity.png'
            )
            self.assertTrue(os.path.exists(plot_path))
            self.assertGreater(os.path.getsize(plot_path), 1000)

            with open(os.path.join(
                    tmpdir, 'threshold_sensitivity.json')) as file:
                payload = json.load(file)
            self.assertEqual(payload['selection_split'], 'validation')
            self.assertEqual(
                len(payload['candidates']), len(report['candidates'])
            )

            with open(csv_path) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), len(report['candidates']))
            self.assertIn('test_consecutive_3_false_alarms', rows[0])
            self.assertIn('threshold', rows[0])

            production = report['production_candidate']
            self.assertIsNotNone(production)
            self.assertEqual(
                production['candidate']['strategy'], 'recall_at_precision'
            )
            self.assertAlmostEqual(
                production['candidate']['min_precision'], 0.75
            )

    def test_candidates_are_sorted_by_recall(self):
        y_val, val_scores, units, cycles, rul = build_split()
        y_test, test_scores, _, _, _ = build_split(seed=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_threshold_sensitivity(
                y_val, val_scores, units, cycles, rul,
                y_test, test_scores, units, cycles, rul,
                CONFIG, tmpdir,
            )
        recalls = [
            candidate['test']['recall']
            for candidate in report['candidates']
        ]
        self.assertEqual(recalls, sorted(recalls, reverse=True))


if __name__ == '__main__':
    unittest.main()
