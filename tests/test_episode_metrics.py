# tests/test_episode_metrics.py

import csv
import os
import tempfile
import unittest

import numpy as np

from src.evaluation.episode_metrics import (
    compute_episode_metrics,
    export_episode_metrics_csv,
)


class TestEpisodeMetrics(unittest.TestCase):
    def test_fragmented_alert_detects_one_episode(self):
        units = np.array([1, 1, 1, 1, 1, 1])
        cycles = np.array([1, 2, 3, 4, 5, 6])
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        rul = np.array([30, 20, 15, 14, 13, 12])
        metrics = compute_episode_metrics(y_true, y_pred, units, cycles, rul)
        self.assertEqual(metrics['real_episodes'], 1)
        self.assertEqual(metrics['detected_episodes'], 1)
        self.assertEqual(metrics['episode_recall'], 1.0)
        self.assertEqual(metrics['alert_episodes'], 2)
        self.assertEqual(metrics['false_alarm_episodes'], 1)
        self.assertEqual(metrics['episode_precision'], 0.5)
        self.assertEqual(metrics['mean_anticipation_cycles'], 14.0)
        self.assertEqual(metrics['mean_alert_to_episode_end_cycles'], 1.0)

    def test_false_alarm_only_episode_is_counted(self):
        units = np.array([1, 1, 1, 1])
        cycles = np.array([1, 2, 3, 4])
        y_true = np.zeros(4, dtype=int)
        y_pred = np.array([0, 1, 1, 0])
        metrics = compute_episode_metrics(y_true, y_pred, units, cycles)
        self.assertEqual(metrics['real_episodes'], 0)
        self.assertEqual(metrics['alert_episodes'], 1)
        self.assertIsNone(metrics['episode_recall'])
        self.assertEqual(metrics['episode_precision'], 0.0)

    def test_validation_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            compute_episode_metrics(
                np.zeros(3), np.zeros(2), np.ones(3), np.arange(3)
            )

    def test_csv_contains_one_row_per_episode(self):
        units = np.array([1, 1, 1, 2, 2])
        cycles = np.array([1, 2, 3, 1, 2])
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])
        metrics = compute_episode_metrics(y_true, y_pred, units, cycles)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_episode_metrics_csv(
                metrics, os.path.join(tmpdir, 'episodes.csv')
            )
            with open(path, newline='') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                fieldnames = reader.fieldnames
        self.assertEqual(len(rows), len(metrics['episodes']))
        self.assertEqual(set(rows[0]), set(fieldnames))


if __name__ == '__main__':
    unittest.main()
