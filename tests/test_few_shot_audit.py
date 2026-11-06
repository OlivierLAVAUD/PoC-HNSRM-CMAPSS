# tests/test_few_shot_audit.py

import os
import tempfile
import unittest

import numpy as np

from src.experiments.few_shot_audit import (
    build_few_shot_audit,
    save_few_shot_audit,
)


def make_split(seed, shift=0.0, invert=False):
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(1, 13), 24)
    cycles = np.tile(np.arange(1, 25), 12)
    rul = np.tile(np.arange(24, 0, -1), 12)
    y = (rul <= 5).astype(int)
    latent = -rul / 24.0 + shift
    scores = 1.0 / (1.0 + np.exp(-8.0 * latent))
    if invert:
        scores = 1.0 - scores
    scores += rng.normal(0, 0.01, len(scores))
    return y, np.clip(scores, 0.0, 1.0), units, cycles, rul


class TestFewShotAudit(unittest.TestCase):
    def test_audit_reports_validation_and_test_without_test_selection(self):
        y, scores, units, cycles, rul = make_split(1)
        report = build_few_shot_audit(
            y, scores, units, cycles, rul,
            y, scores, units, cycles, rul,
            0.5, 'recall_at_precision', n_bootstrap=20,
        )
        self.assertEqual(report['selection_split'], 'target_validation')
        self.assertEqual(report['test_usage'], 'reporting_only')
        self.assertFalse(
            report['threshold_stability']['test_used_for_selection']
        )
        self.assertEqual(report['threshold_stability']['bootstrap']['count'], 20)
        self.assertIn('validation', report)
        self.assertIn('test', report)
        self.assertIsInstance(report['flags'], list)

    def test_validation_optimism_is_flagged(self):
        y_val, val_scores, units, cycles, rul = make_split(2)
        y_test, test_scores, _, _, _ = make_split(3, invert=True)
        report = build_few_shot_audit(
            y_val, val_scores, units, cycles, rul,
            y_test, test_scores, units, cycles, rul,
            0.15, 'f1', n_bootstrap=10,
        )
        self.assertGreater(report['deltas']['f1_validation_minus_test'], 0.15)
        self.assertTrue(any('optimistic' in flag for flag in report['flags']))

    def test_audit_artifact_is_written(self):
        y, scores, units, cycles, rul = make_split(4)
        report = build_few_shot_audit(
            y, scores, units, cycles, rul,
            y, scores, units, cycles, rul,
            0.5, 'f1', n_bootstrap=5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'audit.json')
            self.assertEqual(save_few_shot_audit(report, path), path)
            self.assertTrue(os.path.exists(path))

    def test_bootstrap_is_deterministic(self):
        y, scores, units, cycles, rul = make_split(5)
        first = build_few_shot_audit(
            y, scores, units, cycles, rul,
            y, scores, units, cycles, rul,
            0.5, 'f1', n_bootstrap=10, random_state=7,
        )
        second = build_few_shot_audit(
            y, scores, units, cycles, rul,
            y, scores, units, cycles, rul,
            0.5, 'f1', n_bootstrap=10, random_state=7,
        )
        self.assertEqual(
            first['threshold_stability']['bootstrap']['thresholds'],
            second['threshold_stability']['bootstrap']['thresholds'],
        )


if __name__ == '__main__':
    unittest.main()
