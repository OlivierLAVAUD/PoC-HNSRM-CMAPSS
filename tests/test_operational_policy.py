# tests/test_operational_policy.py

import unittest

import numpy as np

from src.evaluation.operational_policy import (
    ALERT_MODES,
    alert_predictions,
    build_operational_policy_report,
    evaluate_alert_mode,
    expected_cost,
    validate_costs,
    window_metrics,
)


def build_split(engines=5, cycles=10):
    """Return labels, scores, unit ids and cycles with one isolated false alarm."""
    labels = []
    scores = []
    units = []
    cycle_values = []
    positive_scores = [0.80, 0.85, 0.90, 0.95]
    for unit in range(1, engines + 1):
        for cycle in range(1, cycles + 1):
            units.append(unit)
            cycle_values.append(cycle)
            if cycle <= cycles - 4:
                labels.append(0)
                scores.append(0.95 if cycle == 2 else 0.1)
            else:
                labels.append(1)
                scores.append(positive_scores[cycle - (cycles - 4) - 1])
    return (
        np.asarray(labels, dtype=int),
        np.asarray(scores, dtype=float),
        np.asarray(units),
        np.asarray(cycle_values, dtype=int),
        np.zeros(len(labels), dtype=int),
    )


class TestExpectedCost(unittest.TestCase):
    def test_cost_formula(self):
        self.assertEqual(expected_cost(3, 5, 10.0, 1.0), 35.0)
        self.assertEqual(expected_cost(0, 0, 10.0, 1.0), 0.0)

    def test_invalid_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_costs(-1.0, 1.0)
        with self.assertRaises(ValueError):
            validate_costs(10.0, -0.5)
        with self.assertRaises(ValueError):
            validate_costs(0.0, 0.0)
        with self.assertRaises(ValueError):
            validate_costs('high', 1.0)
        self.assertEqual(validate_costs(10, 1), (10.0, 1.0))


class TestAlertModes(unittest.TestCase):
    def setUp(self):
        (self.y, self.scores, self.units, self.cycles,
         self.rul) = build_split()

    def test_immediate_matches_threshold(self):
        predictions = alert_predictions(
            self.scores, self.units, self.cycles, 0.5, 'immediate'
        )
        np.testing.assert_array_equal(
            predictions, (self.scores >= 0.5).astype(int)
        )

    def test_consecutive_confirmation_removes_isolated_alerts(self):
        predictions = alert_predictions(
            self.scores, self.units, self.cycles, 0.5, 'consecutive_2'
        )
        self.assertEqual(int(predictions.sum()), 4 * 5)
        self.assertEqual(int(predictions[1]), 0)

    def test_smoothed_scores_move_the_first_alert(self):
        predictions = alert_predictions(
            self.scores, self.units, self.cycles, 0.5, 'smoothed_3'
        )
        self.assertEqual(int(predictions[:7].sum()), 0)
        self.assertEqual(int(predictions[7]), 1)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            alert_predictions(self.scores, self.units, self.cycles, 0.5, 'nope')

    def test_window_metrics_are_consistent(self):
        metrics = window_metrics(
            self.y, alert_predictions(
                self.scores, self.units, self.cycles, 0.5, 'immediate'
            )
        )
        self.assertEqual(metrics['true_positives'], 20)
        self.assertEqual(metrics['false_positives'], 5)
        self.assertEqual(metrics['false_negatives'], 0)
        self.assertAlmostEqual(metrics['precision'], 20 / 25)
        self.assertAlmostEqual(metrics['recall'], 1.0)

    def test_evaluate_alert_mode_reports_cost_and_engine_metrics(self):
        report = evaluate_alert_mode(
            self.y, self.scores, self.units, self.cycles, self.rul,
            0.5, 'consecutive_2', 10.0, 1.0, None,
        )
        self.assertEqual(report['mode'], 'consecutive_2')
        self.assertEqual(report['missed_critical_windows'], 0)
        self.assertEqual(report['total_false_alarms'], 0)
        self.assertEqual(report['expected_cost'], 0.0)
        self.assertAlmostEqual(report['mean_recall'], 1.0)
        self.assertTrue(report['capacity_feasible'])
        self.assertIn('mean_anticipation_cycles', report)


class TestOperationalPolicy(unittest.TestCase):
    def setUp(self):
        (self.y, self.scores, self.units, self.cycles,
         self.rul) = build_split()

    def _report(self, **overrides):
        config = {
            'false_negative_cost': 10.0,
            'false_alarm_cost': 1.0,
            'inspection_capacity_per_cycle': None,
            'selection_metric': 'expected_cost',
            'preferred_alert_mode': None,
        }
        config.update(overrides)
        return build_operational_policy_report(
            self.y, self.scores, self.units, self.cycles, self.rul,
            self.y, self.scores, self.units, self.cycles, self.rul,
            0.5, config,
        )

    def test_all_modes_are_compared(self):
        report = self._report()
        self.assertEqual(set(report['validation']), set(ALERT_MODES))
        self.assertEqual(set(report['test']), set(ALERT_MODES))
        self.assertEqual(report['selection_split'], 'validation')
        self.assertEqual(report['test_usage'], 'reporting_only')

    def test_selection_minimises_expected_cost(self):
        report = self._report()
        costs = {
            mode: report['validation'][mode]['expected_cost']
            for mode in ALERT_MODES
        }
        self.assertEqual(report['selected_mode'], 'consecutive_2')
        self.assertEqual(costs['consecutive_2'], min(costs.values()))
        self.assertAlmostEqual(
            report['validation']['immediate']['expected_cost'], 5.0
        )

    def test_selection_never_uses_the_test_split(self):
        baseline = self._report()
        y_test = np.zeros_like(self.y)
        scores_test = np.zeros_like(self.scores)
        config = {
            'false_negative_cost': 10.0,
            'false_alarm_cost': 1.0,
            'inspection_capacity_per_cycle': None,
            'selection_metric': 'expected_cost',
            'preferred_alert_mode': None,
        }
        perturbed = build_operational_policy_report(
            self.y, self.scores, self.units, self.cycles, self.rul,
            y_test, scores_test, self.units, self.cycles, self.rul,
            0.5, config,
        )
        self.assertEqual(
            perturbed['selected_mode'], baseline['selected_mode']
        )
        self.assertNotEqual(
            perturbed['test']['immediate']['expected_cost'],
            baseline['test']['immediate']['expected_cost'],
        )

    def test_preferred_mode_overrides_selection(self):
        report = self._report(preferred_alert_mode='smoothed_3')
        self.assertEqual(report['selected_mode'], 'smoothed_3')
        self.assertEqual(report['selection_reason'], 'configured_preference')

    def test_capacity_marks_modes_infeasible(self):
        report = self._report(inspection_capacity_per_cycle=0.0)
        self.assertEqual(
            report['capacity_feasible_modes'],
            ['consecutive_2', 'consecutive_3', 'smoothed_3', 'smoothed_5'],
        )
        self.assertTrue(report['validation']['immediate']['excluded_by_capacity'])
        self.assertNotIn('immediate', report['capacity_feasible_modes'])
        self.assertEqual(report['selected_mode'], 'consecutive_2')

    def test_impossible_capacity_falls_back_to_all_modes(self):
        report = self._report(inspection_capacity_per_cycle=0.0)
        self.assertTrue(report['validation']['consecutive_2']['capacity_feasible'])
        everything_flagged = build_operational_policy_report(
            self.y, np.full_like(self.scores, 0.99), self.units, self.cycles,
            self.rul, self.y, np.full_like(self.scores, 0.99), self.units,
            self.cycles, self.rul, 0.5,
            {
                'false_negative_cost': 10.0, 'false_alarm_cost': 1.0,
                'inspection_capacity_per_cycle': 0.0,
                'selection_metric': 'expected_cost',
            },
        )
        self.assertEqual(everything_flagged['capacity_feasible_modes'], [])
        self.assertIn(
            everything_flagged['selected_mode'], ALERT_MODES
        )

    def test_invalid_selection_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            self._report(selection_metric='accuracy')

    def test_rationale_documents_the_choice(self):
        report = self._report()
        rationale = report['rationale']
        self.assertEqual(rationale['selected_mode'], 'consecutive_2')
        self.assertIsNotNone(rationale['expected_cost_validation'])
        self.assertTrue(rationale['reasons'])
        self.assertIn('precision', rationale['precision_recall_tradeoff'])


if __name__ == '__main__':
    unittest.main()
