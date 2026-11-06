# tests/test_cost_sensitivity.py

import os
import tempfile
import unittest

import numpy as np

from src.evaluation.cost_sensitivity import (
    build_cost_sensitivity,
    save_cost_sensitivity,
)


class TestCostSensitivity(unittest.TestCase):
    def _data(self):
        units = np.repeat([1, 2], 10)
        cycles = np.tile(np.arange(1, 11), 2)
        rul = np.tile(np.arange(10, 0, -1), 2)
        y = (rul <= 3).astype(int)
        scores = np.linspace(0.9, 0.1, 20)
        return y, scores, units, cycles, rul

    def test_rows_cover_ratios_and_select_on_validation(self):
        y, scores, units, cycles, rul = self._data()
        report = build_cost_sensitivity(
            y, scores, units, cycles, rul,
            y, scores, units, cycles, rul, 0.5,
            {
                'false_negative_cost_ratios': [1, 10, 100],
                'false_alarm_cost': 1.0,
                'selection_split': 'validation',
            },
        )
        self.assertEqual(report['selection_split'], 'validation')
        self.assertEqual([row['ratio'] for row in report['rows']], [1, 10, 100])
        self.assertTrue(all(row['selected_mode'] for row in report['rows']))
        self.assertTrue(all(
            row['validation_cost_episode'] >= 0 for row in report['rows']
        ))

    def test_rejects_test_selection(self):
        y, scores, units, cycles, rul = self._data()
        with self.assertRaises(ValueError):
            build_cost_sensitivity(
                y, scores, units, cycles, rul,
                y, scores, units, cycles, rul, 0.5,
                {'selection_split': 'test'},
            )

    def test_artifacts_are_written(self):
        y, scores, units, cycles, rul = self._data()
        report = build_cost_sensitivity(
            y, scores, units, cycles, rul,
            y, scores, units, cycles, rul, 0.5,
            {'false_negative_cost_ratios': [1, 5]},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_cost_sensitivity(report, tmpdir)
            for path in paths.values():
                self.assertTrue(os.path.exists(path), path)


if __name__ == '__main__':
    unittest.main()
