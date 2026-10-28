# tests/test_drift.py

import os
import tempfile
import unittest

import numpy as np

from src.monitoring.drift import (
    analyse_drift,
    compare_features,
    drift_level,
    load_reference,
    reference_statistics,
    save_drift_report,
    save_reference,
    validate_config,
)

FEATURES = ['s2', 's4']
CONFIG = {
    'enabled': True,
    'warning_threshold': 0.05,
    'critical_threshold': 0.15,
    'min_windows': 30,
}


def build_windows(windows=60, cycles=10, seed=0):
    """Return normally distributed reference windows."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(windows, cycles, len(FEATURES))).astype(
        np.float32
    )


def inject_out_of_bounds(windows, ratio, value=50.0):
    """Replace a share of the values with an out-of-range value."""
    data = windows.copy()
    flat = data.reshape(-1, data.shape[-1])
    total = flat.size
    count = int(round(ratio * total))
    flat.reshape(-1)[:count] = value
    return data


class TestReferenceStatistics(unittest.TestCase):
    def test_reference_contains_expected_keys(self):
        reference = reference_statistics(build_windows(), FEATURES)
        self.assertEqual(reference['windows'], 60)
        self.assertEqual(reference['window_size'], 10)
        self.assertEqual(set(reference['features']), set(FEATURES))
        stats = reference['features']['s2']
        for key in ('mean', 'std', 'min', 'max', 'p5', 'p25', 'p50', 'p75',
                    'p95'):
            self.assertIn(key, stats)
        self.assertLess(stats['p5'], stats['p95'])

    def test_reference_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            reference_statistics(np.zeros((10, 10)), FEATURES)
        with self.assertRaises(ValueError):
            reference_statistics(np.zeros((0, 10, 2)), FEATURES)
        with self.assertRaises(ValueError):
            reference_statistics(np.zeros((10, 10, 3)), FEATURES)


class TestDriftLevels(unittest.TestCase):
    def setUp(self):
        self.train = build_windows()
        self.reference = reference_statistics(self.train, FEATURES)

    def _analyse(self, incoming, config=None, **kwargs):
        return analyse_drift(
            None, incoming, FEATURES, config or CONFIG,
            reference=self.reference, **kwargs,
        )

    def test_conforming_data_is_ok(self):
        report = self._analyse(self.train)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['level'], 'ok')
        self.assertFalse(report['hitl_recommended'])
        self.assertAlmostEqual(report['score'], 0.0)
        self.assertIn('no action', report['recommendation'].lower())

    def test_light_drift_is_a_warning(self):
        incoming = inject_out_of_bounds(self.train, 0.08)
        report = self._analyse(incoming)
        self.assertEqual(report['level'], 'warning')
        self.assertTrue(report['hitl_recommended'])
        self.assertTrue(report['warning_features'])
        self.assertEqual(report['critical_features'], [])

    def test_strong_drift_is_critical(self):
        incoming = inject_out_of_bounds(self.train, 0.25)
        report = self._analyse(incoming)
        self.assertEqual(report['level'], 'critical')
        self.assertIn('s2', report['critical_features'])
        self.assertIn('human review', report['recommendation'].lower())

    def test_nan_and_inf_are_counted(self):
        incoming = self.train.copy()
        incoming[:, :, 0] = np.nan
        incoming[:, :5, 1] = np.inf
        report = self._analyse(incoming)
        self.assertGreater(report['incoming']['non_finite_proportion'], 0.0)
        self.assertEqual(
            report['incoming']['features']['s2']['non_finite_proportion'], 1.0
        )
        self.assertEqual(report['level'], 'critical')

    def test_min_windows_blocks_the_assessment(self):
        report = self._analyse(self.train[:5])
        self.assertEqual(report['status'], 'insufficient_data')
        self.assertEqual(report['level'], 'ok')
        self.assertIn('at least 30', report['recommendation'])

    def test_signals_are_summarised(self):
        report = self._analyse(
            self.train,
            signals={
                'predicted_rul': np.linspace(40, 0, 100),
                'neural_probability': np.linspace(0, 1, 100),
            },
        )
        self.assertAlmostEqual(
            report['signals']['neural_probability']['mean'], 0.5, places=6
        )
        self.assertEqual(report['signals']['predicted_rul']['count'], 100)

    def test_feature_scores_expose_the_worst_sensor(self):
        incoming = self.train.copy()
        incoming[:, :, 1] = 40.0
        comparison = compare_features(self.reference, incoming, FEATURES)
        self.assertAlmostEqual(
            comparison['s4']['out_of_bounds_proportion'], 1.0
        )
        self.assertEqual(comparison['s2']['out_of_bounds_proportion'], 0.0)
        expected_shift = 40.0 - float(
            self.reference['features']['s4']['mean']
        )
        self.assertAlmostEqual(
            comparison['s4']['mean_shift'], expected_shift, places=6
        )
        report = self._analyse(incoming)
        self.assertEqual(report['most_drifted_features'][0]['feature'], 's4')


class TestConfiguration(unittest.TestCase):
    def test_level_mapping(self):
        self.assertEqual(drift_level(0.01, 0.05, 0.15), 'ok')
        self.assertEqual(drift_level(0.05, 0.05, 0.15), 'warning')
        self.assertEqual(drift_level(0.15, 0.05, 0.15), 'critical')

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({'warning_threshold': 0.2,
                             'critical_threshold': 0.1})
        with self.assertRaises(ValueError):
            validate_config({'warning_threshold': -0.1,
                             'critical_threshold': 0.1})
        with self.assertRaises(ValueError):
            validate_config({'min_windows': 0})

    def test_analysis_without_training_or_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            analyse_drift(None, build_windows(5), FEATURES, CONFIG)


class TestPersistence(unittest.TestCase):
    def test_reference_and_report_round_trip(self):
        train = build_windows()
        reference = reference_statistics(train, FEATURES)
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = os.path.join(tmpdir, 'drift_reference.json')
            save_reference(reference, reference_path)
            loaded = load_reference(reference_path)
            self.assertEqual(loaded['windows'], reference['windows'])
            report = analyse_drift(
                None, train, FEATURES, CONFIG, reference=loaded
            )
            report_path = save_drift_report(
                report, os.path.join(tmpdir, 'drift_report.json')
            )
            self.assertTrue(os.path.exists(report_path))
            self.assertEqual(report['level'], 'ok')


if __name__ == '__main__':
    unittest.main()
