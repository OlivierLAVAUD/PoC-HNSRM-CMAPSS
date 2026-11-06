# tests/test_alert_policy_comparison.py

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from src.evaluation.alert_policy_comparison import (
    POLICY_MODES,
    build_alert_policy_comparison,
    export_alert_policy_comparison_csv,
    save_alert_policy_comparison,
)


def build_split(engines=5, cycles=10, seed=0):
    rng = np.random.default_rng(seed)
    labels, scores, units, cycle_values, rul_values = [], [], [], [], []
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
                scores.append(0.90 if cycle == 2 else float(rng.uniform(0.0, 0.35)))
    return (
        np.asarray(labels, dtype=int), np.asarray(scores, dtype=float),
        np.asarray(units), np.asarray(cycle_values, dtype=int),
        np.asarray(rul_values, dtype=int),
    )


CONFIG = {
    'false_negative_cost': 10.0,
    'false_alarm_cost': 1.0,
    'selection_metric': 'expected_cost',
    'cost_unit': 'episode',
    'hysteresis': {'enter_threshold': 0.45, 'exit_threshold': 0.25},
}


class TestAlertPolicyComparison(unittest.TestCase):
    def setUp(self):
        self.y_val, self.val_scores, self.units, self.cycles, self.rul = (
            build_split(seed=0)
        )
        self.y_test, self.test_scores, _, _, _ = build_split(seed=1)

    def _report(self, **overrides):
        config = dict(CONFIG)
        config.update(overrides)
        return build_alert_policy_comparison(
            self.y_val, self.val_scores, self.units, self.cycles, self.rul,
            self.y_test, self.test_scores, self.units, self.cycles, self.rul,
            0.5, config,
        )

    def test_all_modes_and_units_are_reported(self):
        report = self._report()
        self.assertEqual(set(report['validation']), set(POLICY_MODES))
        self.assertEqual(set(report['test']), set(POLICY_MODES))
        for values in report['validation'].values():
            self.assertIn('episode_recall', values)
            self.assertIn('expected_cost_window', values)
            self.assertIn('expected_cost_engine', values)
            self.assertIn('expected_cost_episode', values)
        self.assertEqual(report['selection_split'], 'validation')
        self.assertEqual(report['test_usage'], 'reporting_only')

    def test_every_mode_reports_hysteresis_thresholds(self):
        report = self._report()
        self.assertIn('hysteresis', report['validation'])
        self.assertEqual(
            report['hysteresis_thresholds'],
            {'enter_threshold': 0.45, 'exit_threshold': 0.25},
        )

    def test_selection_ignores_test_perturbations(self):
        baseline = self._report()
        perturbed = build_alert_policy_comparison(
            self.y_val, self.val_scores, self.units, self.cycles, self.rul,
            np.zeros_like(self.y_test), np.zeros_like(self.test_scores),
            self.units, self.cycles, self.rul, 0.5, CONFIG,
        )
        self.assertEqual(perturbed['selected_mode'], baseline['selected_mode'])
        self.assertNotEqual(
            perturbed['test']['immediate']['expected_cost_episode'],
            baseline['test']['immediate']['expected_cost_episode'],
        )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            self._report(cost_unit='cycle')
        with self.assertRaises(ValueError):
            self._report(selection_metric='accuracy')
        with self.assertRaises(ValueError):
            build_alert_policy_comparison(
                self.y_val, self.val_scores, self.units, self.cycles,
                self.rul, self.y_test, self.test_scores, self.units,
                self.cycles, self.rul, 0.5, CONFIG, modes=('immediate', 'bad'),
            )

    def test_hysteresis_uses_valid_thresholds(self):
        report = self._report()
        thresholds = report['hysteresis_thresholds']
        self.assertGreater(thresholds['enter_threshold'],
                           thresholds['exit_threshold'])
        self.assertGreaterEqual(thresholds['exit_threshold'], 0.0)
        self.assertLessEqual(thresholds['enter_threshold'], 1.0)
        derived = self._report(hysteresis={'margin': 0.1})
        derived_thresholds = derived['hysteresis_thresholds']
        self.assertAlmostEqual(derived_thresholds['enter_threshold'], 0.6)
        self.assertAlmostEqual(derived_thresholds['exit_threshold'], 0.4)
        self.assertGreater(derived_thresholds['enter_threshold'],
                           derived_thresholds['exit_threshold'])

    def test_json_and_csv_metrics_are_coherent(self):
        report = self._report()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(save_alert_policy_comparison(
                    report, os.path.join(tmpdir, 'report.json'))) as file:
                payload = json.load(file)
            csv_path = export_alert_policy_comparison_csv(
                payload, os.path.join(tmpdir, 'report.csv')
            )
            with open(csv_path, newline='') as file:
                rows = list(csv.DictReader(file))
        for row in rows:
            values = payload[row['split']][row['mode']]
            with self.subTest(split=row['split'], mode=row['mode']):
                self.assertAlmostEqual(
                    float(row['episode_recall']),
                    float(values['episode_recall']),
                )
                self.assertAlmostEqual(
                    float(row['expected_cost']),
                    float(values['expected_cost']),
                )
                self.assertEqual(
                    int(row['detected_episodes']),
                    int(values['detected_episodes']),
                )

    def test_json_and_csv_are_written(self):
        report = self._report()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = save_alert_policy_comparison(
                report, os.path.join(tmpdir, 'alert_policy.json')
            )
            csv_path = export_alert_policy_comparison_csv(
                report, os.path.join(tmpdir, 'alert_policy.csv')
            )
            with open(json_path) as file:
                payload = json.load(file)
            self.assertEqual(payload['selected_mode'], report['selected_mode'])
            with open(csv_path, newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2 * len(POLICY_MODES))
            self.assertIn('episode_recall', rows[0])
            self.assertIn('cost_unit', rows[0])


if __name__ == '__main__':
    unittest.main()
