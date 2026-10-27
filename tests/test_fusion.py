# tests/test_fusion.py

import os
import tempfile
import unittest

import numpy as np

from src.fusion.probability_fusion import (
    ProbabilityFusion,
    evaluate_probability_fusion,
    symbolic_rule_score,
)


class TestProbabilityFusion(unittest.TestCase):
    def test_symbolic_score_is_bounded_and_severity_driven(self):
        self.assertEqual(symbolic_rule_score([]), 0.0)
        self.assertEqual(symbolic_rule_score(['unrelated insight']), 0.0)
        self.assertGreater(
            symbolic_rule_score(['critical overload']),
            symbolic_rule_score(['rising trend']),
        )
        self.assertLessEqual(
            symbolic_rule_score(['critical'] * 50), 1.0
        )

    def test_build_matrix_respects_configured_order(self):
        matrix = ProbabilityFusion.build_matrix(
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([0.5, 0.6]),
            np.array([70.0, 30.0]),
            feature_order=[
                'predicted_rul', 'neural_probability',
                'problog_failure_risk', 'symbolic_rule_score',
            ],
        )
        np.testing.assert_allclose(
            matrix,
            np.array([[70.0, 0.1, 0.5, 0.3], [30.0, 0.2, 0.6, 0.4]]),
        )

    def test_fit_predict_coefficients_and_roundtrip(self):
        rng = np.random.default_rng(0)
        n = 200
        neural = rng.uniform(0, 1, n)
        symbolic = rng.uniform(0, 1, n)
        problog = rng.uniform(0, 1, n)
        rul = rng.uniform(0, 100, n)
        y = (
            (neural + symbolic + problog).reshape(n, 1)
            + (1.0 - rul / 100.0).reshape(n, 1)
            > 2.0
        ).astype(int).ravel()
        X = ProbabilityFusion.build_matrix(neural, symbolic, problog, rul)

        fusion = ProbabilityFusion(random_state=0).fit(X, y)
        probabilities = fusion.positive_proba(X)
        self.assertEqual(probabilities.shape, (n,))
        self.assertEqual(
            set(fusion.coefficients()),
            {'neural_probability', 'symbolic_rule_score',
             'problog_failure_risk', 'predicted_rul'},
        )
        metrics = evaluate_probability_fusion(y, probabilities)
        self.assertGreater(metrics['roc_auc'], 0.5)
        self.assertTrue(0.0 <= metrics['brier_score'] <= 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'fusion.joblib')
            fusion.save(path)
            loaded = ProbabilityFusion.load(path)
            np.testing.assert_allclose(
                loaded.positive_proba(X), probabilities
            )
            self.assertEqual(
                loaded.coefficients(), fusion.coefficients()
            )

    def test_requires_both_classes(self):
        fusion = ProbabilityFusion()
        with self.assertRaises(ValueError):
            fusion.fit(np.zeros((4, 4)), np.zeros(4, dtype=int))


if __name__ == '__main__':
    unittest.main()
