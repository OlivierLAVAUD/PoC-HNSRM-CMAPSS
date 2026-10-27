# tests/test_engine_metrics.py

import csv
import os
import shutil
import tempfile
import unittest

import numpy as np

from src.evaluation.engine_metrics import (
    CSV_FIELDS,
    compute_engine_metrics,
    export_engine_metrics_csv,
)


class TestEngineMetrics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='ansrdt_engine_')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_per_engine_detection_false_alarms_and_anticipation(self):
        unit_ids = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        cycles = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        rul = np.array([50, 40, 20, 10, 50, 40, 20, 10])
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 1])

        metrics = compute_engine_metrics(unit_ids, cycles, rul, y_true, y_pred)
        first, second = metrics['per_engine']

        self.assertEqual(first['false_alarms'], 1)
        self.assertEqual(first['recall'], 0.5)
        self.assertEqual(first['precision'], 0.5)
        self.assertEqual(first['lead_time_cycles'], 0)
        self.assertEqual(first['anticipation_cycles'], 10)
        self.assertFalse(first['alerted_before_eol'])

        self.assertEqual(second['false_alarms'], 1)
        self.assertEqual(second['recall'], 1.0)
        self.assertAlmostEqual(second['precision'], 2 / 3)
        self.assertEqual(second['first_correct_alert_cycle'], 3)
        self.assertEqual(second['lead_time_cycles'], 1)
        self.assertEqual(second['anticipation_cycles'], 20)
        self.assertTrue(second['alerted_before_eol'])

        summary = metrics['summary']
        self.assertEqual(summary['engines'], 2)
        self.assertEqual(summary['total_false_alarms'], 2)
        self.assertAlmostEqual(summary['mean_false_alarms_per_engine'], 1.0)
        self.assertAlmostEqual(summary['mean_recall'], 0.75)
        self.assertAlmostEqual(summary['mean_lead_time_cycles'], 0.5)
        self.assertAlmostEqual(summary['mean_anticipation_cycles'], 15.0)
        self.assertAlmostEqual(summary['proportion_alerted_before_eol'], 0.5)
        self.assertEqual(summary['engines_with_positive_windows'], 2)
        self.assertAlmostEqual(
            summary['proportion_alerted_before_eol_among_relevant'], 0.5
        )

    def test_engine_without_positive_window_is_excluded_from_mean_recall(self):
        unit_ids = np.array([1, 1, 2, 2])
        cycles = np.array([1, 2, 1, 2])
        rul = np.array([50, 40, 50, 40])
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 0])
        metrics = compute_engine_metrics(unit_ids, cycles, rul, y_true, y_pred)
        self.assertEqual(metrics['summary']['mean_recall'], 0.0)
        self.assertEqual(metrics['summary']['total_false_alarms'], 1)
        self.assertEqual(metrics['summary']['engines_with_positive_windows'], 0)
        self.assertEqual(
            metrics['summary']['proportion_alerted_before_eol_among_relevant'], 0.0
        )

    def test_csv_export(self):
        metrics = compute_engine_metrics(
            np.array([1, 1]), np.array([1, 2]), np.array([2, 1]),
            np.array([0, 1]), np.array([1, 1]),
        )
        path = export_engine_metrics_csv(
            metrics, os.path.join(self.temp_dir, 'engine_metrics.csv')
        )
        with open(path, newline='') as file:
            reader = csv.DictReader(file)
            self.assertEqual(reader.fieldnames, CSV_FIELDS)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['unit'], '1')
        self.assertEqual(rows[0]['false_alarms'], '1')


if __name__ == '__main__':
    unittest.main()
