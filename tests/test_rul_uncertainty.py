# tests/test_rul_uncertainty.py

import unittest

import numpy as np

from src.fusion.probability_fusion import ProbabilityFusion
from src.models.rul_regressor import RULEnsembleRegressor


class TestRULUncertainty(unittest.TestCase):
    def test_ensemble_returns_ordered_interval(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(80, 3)).astype(np.float32)
        y = 20.0 + 5.0 * X[:, 0] - 3.0 * X[:, 1]
        model = RULEnsembleRegressor(n_models=3, n_estimators=25)
        model.fit(X, y)
        distribution = model.predict_distribution(X[:10])
        self.assertEqual(distribution['median'].shape, (10,))
        self.assertTrue(np.all(distribution['q10'] <= distribution['median']))
        self.assertTrue(np.all(distribution['median'] <= distribution['q90']))
        self.assertTrue(np.all(distribution['std'] >= 0.0))
        np.testing.assert_allclose(
            model.predict(X[:10]), distribution['median']
        )

    def test_nan_features_are_accepted(self):
        X = np.array([[0.0, 1.0], [np.nan, 2.0], [1.0, np.nan]], dtype=np.float32)
        y = np.array([10.0, 20.0, 30.0])
        model = RULEnsembleRegressor(n_models=2, n_estimators=10)
        model.fit(X, y)
        self.assertEqual(len(model.predict(X)), 3)

    def test_fusion_matrix_can_include_uncertainty(self):
        n = 5
        matrix = ProbabilityFusion.build_matrix(
            np.linspace(0, 1, n), np.zeros(n), np.zeros(n), np.ones(n),
            feature_order=[
                'neural_probability', 'symbolic_rule_score',
                'problog_failure_risk', 'predicted_rul', 'rul_uncertainty',
            ],
            rul_uncertainty=np.full(n, 2.5),
        )
        self.assertEqual(matrix.shape, (n, 5))
        np.testing.assert_allclose(matrix[:, -1], 2.5)

    def test_default_fusion_matrix_is_unchanged(self):
        n = 4
        matrix = ProbabilityFusion.build_matrix(
            np.arange(n), np.arange(n), np.arange(n), np.arange(n),
        )
        self.assertEqual(matrix.shape, (n, 4))


if __name__ == '__main__':
    unittest.main()
