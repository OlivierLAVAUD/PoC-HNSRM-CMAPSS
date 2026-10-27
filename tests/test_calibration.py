# tests/test_calibration.py

import os
import tempfile
import unittest

import numpy as np

from src.evaluation.calibration import (
    calibration_metrics,
    expected_calibration_error,
    reliability_curve,
)
from src.models.light_classifier import LightClassifier


def _imbalanced_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5, 3)).astype(np.float32)
    signal = X[:, -1, 0] + 0.5 * X[:, -1, 1]
    y = (signal > np.quantile(signal, 0.75)).astype(int)
    return X, y


class TestCalibration(unittest.TestCase):
    def test_calibrated_classifier_and_reload(self):
        X, y = _imbalanced_data()
        model = LightClassifier(n_estimators=20, max_depth=4, random_state=0)
        model.fit(X, y).calibrate(X, y, method="sigmoid", cv_folds=3)

        self.assertTrue(model.is_calibrated)
        self.assertEqual(model.calibration_method, "sigmoid")
        proba = model.positive_proba(X)
        self.assertEqual(proba.shape, (len(X),))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

        model.set_threshold(0.7)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.joblib')
            model.save(path)
            loaded = LightClassifier.load(path)
            self.assertAlmostEqual(loaded.decision_threshold, 0.7)
            self.assertTrue(loaded.is_calibrated)
            np.testing.assert_allclose(loaded.positive_proba(X), proba)

    def test_threshold_controls_predict(self):
        X, y = _imbalanced_data(seed=1)
        model = LightClassifier(n_estimators=10, max_depth=3, random_state=0)
        model.fit(X, y)

        model.set_threshold(0.0)
        self.assertTrue(np.all(model.predict(X) == 1))
        model.set_threshold(1.0)
        self.assertTrue(np.all(model.predict(X) == 0))

    def test_calibration_metrics(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = calibration_metrics(y, prob, n_bins=5)
        self.assertIn('brier_score', metrics)
        self.assertIn('ece', metrics)
        self.assertGreaterEqual(metrics['ece'], 0.0)
        curve = reliability_curve(y, prob, n_bins=5)
        self.assertEqual(sum(curve['counts']), len(y))
        self.assertEqual(expected_calibration_error(y, prob, n_bins=5),
                         metrics['ece'])


if __name__ == '__main__':
    unittest.main()
