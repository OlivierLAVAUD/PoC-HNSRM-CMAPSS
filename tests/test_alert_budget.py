# tests/test_alert_budget.py

import unittest

import numpy as np

from src.evaluation.alert_budget import (
    evaluate_budget_usage,
    select_alert_budget_threshold,
    validate_budget_config,
)


class TestAlertBudget(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        self.units = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        self.cycles = np.array([10, 20, 30, 40, 10, 20, 30, 40])

    def test_per_engine_budget_selects_highest_feasible_threshold(self):
        report = select_alert_budget_threshold(
            self.scores, self.units, self.cycles,
            {'max_alerts_per_engine': 1.0},
        )
        self.assertEqual(report['selection_split'], 'validation')
        self.assertLessEqual(report['achieved']['alerts_per_engine'], 1.0)
        # 0.7 yields 2 alerts (one per engine); 0.8 yields 2 as well, and
        # 0.9 yields 1, so 0.9 is the highest feasible observed threshold.
        self.assertEqual(report['threshold'], 0.9)
        self.assertEqual(report['achieved']['alerts'], 1)

    def test_per_100_cycles_budget(self):
        report = select_alert_budget_threshold(
            self.scores, self.units, self.cycles,
            {'max_alerts_per_100_cycles': 24.0},
        )
        self.assertLessEqual(
            report['achieved']['alerts_per_100_cycles'], 24.0
        )
        self.assertEqual(report['threshold'], 0.9)

    def test_stricter_constraint_wins(self):
        report = select_alert_budget_threshold(
            self.scores, self.units, self.cycles,
            {
                'max_alerts_per_engine': 2.0,
                'max_alerts_per_100_cycles': 12.5,
            },
        )
        self.assertLessEqual(
            report['achieved']['alerts_per_100_cycles'], 12.5
        )
        self.assertEqual(report['threshold'], 0.9)

    def test_impossible_budget_raises_or_returns_warning(self):
        tied = np.array([0.9, 0.9, 0.9, 0.9])
        units = np.array([1, 1, 2, 2])
        cycles = np.array([1, 2, 1, 2])
        with self.assertRaises(ValueError):
            select_alert_budget_threshold(
                tied, units, cycles, {'max_alerts_per_engine': 0.0},
            )
        report = select_alert_budget_threshold(
            tied, units, cycles,
            {
                'max_alerts_per_engine': 0.0,
                'unreachable_policy': 'most_constraining',
            },
        )
        self.assertFalse(report['feasible'])
        self.assertIn('warning', report)

    def test_config_rejects_test_selection_and_negative_budget(self):
        with self.assertRaises(ValueError):
            validate_budget_config({
                'max_alerts_per_engine': 1.0,
                'selection_split': 'test',
            })
        with self.assertRaises(ValueError):
            validate_budget_config({'max_alerts_per_engine': -1.0})
        with self.assertRaises(ValueError):
            validate_budget_config({})

    def test_frozen_threshold_usage_is_recomputed_on_test(self):
        report = select_alert_budget_threshold(
            self.scores, self.units, self.cycles,
            {'max_alerts_per_engine': 1.0},
        )
        usage = evaluate_budget_usage(
            self.scores, self.units, self.cycles, report['threshold']
        )
        self.assertEqual(usage['alerts'], 1)


if __name__ == '__main__':
    unittest.main()
