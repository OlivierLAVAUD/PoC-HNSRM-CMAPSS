# tests/test_symbolic_contribution.py

import json
import os
import tempfile
import unittest

import numpy as np

from src.fusion.symbolic_contribution import (
    save_symbolic_contribution,
    symbolic_contribution,
)


class FakeFusion:
    feature_order = [
        'neural_probability', 'symbolic_rule_score',
        'problog_failure_risk', 'predicted_rul',
    ]

    def positive_proba(self, matrix):
        logits = 10.0 * matrix[:, 1] + 10.0 * matrix[:, 2] - 5.0
        return 1.0 / (1.0 + np.exp(-logits))


def build_split(seed=0):
    rng = np.random.default_rng(seed)
    labels, units, cycles, rul = [], [], [], []
    rows = []
    for unit in range(1, 5):
        for cycle in range(1, 9):
            label = int((unit + cycle) % 2 == 0)
            labels.append(label)
            units.append(unit)
            cycles.append(cycle)
            rul.append(8 - cycle)
            rows.append([
                rng.uniform(0.0, 0.1), float(label), float(label), 8 - cycle,
            ])
    return (
        np.asarray(rows, dtype=float), np.asarray(labels, dtype=int),
        np.asarray(units), np.asarray(cycles, dtype=int),
        np.asarray(rul, dtype=int),
    )


class TestSymbolicContribution(unittest.TestCase):
    def setUp(self):
        self.val_matrix, self.y_val, self.units, self.cycles, self.rul = (
            build_split(seed=0)
        )
        self.test_matrix, self.y_test, _, _, _ = build_split(seed=1)

    def _report(self):
        return symbolic_contribution(
            FakeFusion(), self.val_matrix, self.y_val, self.units,
            self.cycles, self.rul, self.test_matrix, self.y_test,
            self.units, self.cycles, self.rul, 0.5,
            {'false_negative_cost': 10.0, 'false_alarm_cost': 1.0},
        )

    def test_reports_signal_variants_and_verdicts(self):
        report = self._report()
        self.assertTrue(report['applicable'])
        self.assertEqual(
            set(report['signals']),
            {'symbolic_rule_score', 'problog_failure_risk'},
        )
        self.assertIn('without_symbolic_score', report['variants'])
        self.assertIn('without_problog_risk', report['variants'])
        self.assertIn('without_symbolic_or_problog', report['variants'])
        self.assertIn('neural_and_rul_only', report['variants'])
        self.assertEqual(
            set(report['contributions']['symbolic_rule_score']
                ['delta_full_minus_ablation']),
            {'validation', 'test'},
        )
        self.assertIn(
            report['contributions']['symbolic_rule_score']['verdict'],
            {'helps', 'hurts', 'neutral'},
        )
        self.assertIn('answer', report['summary'])

    def test_ablation_keeps_the_threshold_and_other_inputs(self):
        report = self._report()
        full = report['variants']['full']
        ablated = report['variants']['without_symbolic_or_problog']
        self.assertEqual(report['threshold'], 0.5)
        self.assertNotEqual(
            full['validation']['window']['recall'],
            ablated['validation']['window']['recall'],
        )
        self.assertEqual(
            report['variants']['neural_and_rul_only']['removed_inputs'],
            ['symbolic_rule_score', 'problog_failure_risk'],
        )

    def test_per_rule_ablation_identifies_useful_and_noisy_rules(self):
        labels = np.array([i % 2 for i in range(8)], dtype=int)
        rows = []
        units, cycles, rul = [], [], []
        for index, label in enumerate(labels):
            rows.append([
                0.1, float(label), 0.0, float(10 - index),
            ])
            units.append(index // 2 + 1)
            cycles.append(index % 2 + 1)
            rul.append(10 - index)
        matrix = np.asarray(rows, dtype=float)
        activations = np.zeros((8, 2), dtype=bool)
        activations[:, 0] = labels == 1
        activations[:, 1] = labels == 0
        report = symbolic_contribution(
            FakeFusion(), matrix, labels, np.asarray(units),
            np.asarray(cycles), np.asarray(rul), matrix, labels,
            np.asarray(units), np.asarray(cycles), np.asarray(rul), 0.5,
            {'false_negative_cost': 10.0, 'false_alarm_cost': 1.0},
            rule_names=['useful_rule', 'noise_rule'],
            rule_activations_val=activations,
            rule_activations_test=activations,
            neural_val=matrix[:, 1], neural_test=matrix[:, 1],
            neural_threshold=0.5,
        )
        rule_report = report['rule_contributions']
        self.assertTrue(rule_report['applicable'])
        self.assertTrue(rule_report['diagnostic'])
        useful = rule_report['contributions']['useful_rule']
        noise = rule_report['contributions']['noise_rule']
        self.assertGreater(useful['marginal_f1_validation'], 0.0)
        self.assertEqual(useful['verdict'], 'helps')
        self.assertEqual(noise['verdict'], 'neutral')
        self.assertAlmostEqual(useful['precision_validation'], 1.0)
        self.assertAlmostEqual(useful['recall_validation'], 1.0)
        self.assertIn(
            'useful_rule',
            rule_report['summary']['most_useful_rules'],
        )
        self.assertIn('answer', rule_report['summary'])

    def test_report_is_json_serialisable(self):
        report = self._report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_symbolic_contribution(
                report, os.path.join(tmpdir, 'symbolic_contribution.json')
            )
            with open(path) as file:
                payload = json.load(file)
        self.assertEqual(payload['scope'], 'validation_and_test_reporting')
        self.assertTrue(payload['applicable'])


class TestSymbolicContributionWithoutInputs(unittest.TestCase):
    def test_absent_inputs_are_reported_as_not_applicable(self):
        class NeuralOnly:
            feature_order = ['neural_probability', 'predicted_rul']

            def positive_proba(self, matrix):
                return matrix[:, 0]

        matrix = np.zeros((2, 2), dtype=float)
        labels = np.array([0, 1])
        units = np.array([1, 1])
        cycles = np.array([1, 2])
        report = symbolic_contribution(
            NeuralOnly(), matrix, labels, units, cycles, labels,
            matrix, labels, units, cycles, labels, 0.5,
        )
        self.assertFalse(report['applicable'])
        self.assertEqual(report['signals'], [])


if __name__ == '__main__':
    unittest.main()
