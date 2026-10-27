# tests/test_thresholding.py

import unittest

import numpy as np

from src.evaluation.thresholding import (
    precision_recall_analysis,
    select_threshold,
)


class TestThresholding(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.scores = np.array([0.05, 0.20, 0.35, 0.55, 0.60, 0.72, 0.85, 0.95])

    def test_f1_selection(self):
        result = select_threshold(self.y, self.scores, strategy='f1')
        self.assertFalse(result['fallback_used'])
        self.assertGreaterEqual(result['selected_metrics']['f1'], 0.8)

    def test_recall_at_precision(self):
        result = select_threshold(
            self.y, self.scores, strategy='recall_at_precision',
            min_precision=0.9,
        )
        self.assertFalse(result['fallback_used'])
        self.assertGreaterEqual(result['selected_metrics']['precision'], 0.9)

    def test_fbeta_favours_recall(self):
        result = select_threshold(self.y, self.scores, strategy='fbeta', beta=2.0)
        self.assertEqual(result['strategy'], 'fbeta')
        self.assertGreaterEqual(result['selected_metrics']['recall'], 0.75)
        self.assertFalse(result['fallback_used'])

    def test_fallback_when_single_class(self):
        result = select_threshold(
            np.zeros(8, dtype=int), self.scores, strategy='f1', fallback=0.42
        )
        self.assertTrue(result['fallback_used'])
        self.assertAlmostEqual(result['threshold'], 0.42)

    def test_precision_recall_analysis(self):
        analysis = precision_recall_analysis(self.y, self.scores)
        self.assertEqual(len(analysis['precision']), len(analysis['recall']))
        self.assertEqual(len(analysis['thresholds']) + 1, len(analysis['precision']))


if __name__ == '__main__':
    unittest.main()
