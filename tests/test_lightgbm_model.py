# tests/test_lightgbm_model.py

import os
import tempfile
import unittest

import numpy as np

from src.features.temporal_features import TemporalFeatureExtractor
from src.models.light_classifier import LightClassifier
from src.models.rul_regressor import RULLightGBMRegressor, out_of_fold_predictions


def _data(seed=0, n=120, window=6, sensors=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, window, sensors)).astype(np.float32)
    X[:, -2:, 0] += 1.2
    y = (X[:, -1, 0] > 0.5).astype(int)
    rul = np.clip(80 - 0.5 * np.arange(n) + rng.normal(0, 2, n), 0, None)
    groups = np.repeat(np.arange(12), n // 12)
    return X, y, rul, groups


class TestLightGBMModel(unittest.TestCase):
    def test_lightgbm_classifier_bundle_roundtrip(self):
        X, y, rul, _ = _data()
        extractor = TemporalFeatureExtractor(
            ['a', 'b', 'c'], include_raw_window=True,
            moving_average_cycles=3,
        ).fit(X)
        engineered = extractor.transform(X)
        regressor = RULLightGBMRegressor(
            n_estimators=20, learning_rate=0.1, random_state=0,
            num_leaves=7,
        ).fit(engineered, rul)
        augmented = np.hstack([
            engineered, regressor.predict(engineered).reshape(-1, 1),
        ])

        model = LightClassifier(
            architecture='lightgbm', random_state=0, n_estimators=20,
            learning_rate=0.1, num_leaves=7,
        )
        model.temporal_feature_extractor = extractor
        model.rul_regressor = regressor
        model.feature_names_out = list(extractor.feature_names_out) + [
            'predicted_rul'
        ]
        model.uses_temporal_features = True
        model.uses_rul_feature = True
        model.fit(augmented, y)

        prepared = model.prepare_features(X)
        self.assertEqual(prepared.shape, augmented.shape)
        probabilities = model.positive_proba(prepared)
        self.assertEqual(probabilities.shape, (len(X),))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'lightgbm.joblib')
            model.save(path)
            restored = LightClassifier.load(path)
            restored_prepared = restored.prepare_features(X)
            np.testing.assert_allclose(restored_prepared, prepared)
            np.testing.assert_allclose(
                restored.positive_proba(restored_prepared), probabilities
            )
            self.assertEqual(restored.feature_names_out[-1], 'predicted_rul')
            self.assertTrue(restored.uses_temporal_features)

    def test_rul_regressor_metrics_and_oof_predictions(self):
        X, _, rul, groups = _data(n=120)
        model = RULLightGBMRegressor(
            n_estimators=20, learning_rate=0.1, random_state=0,
            num_leaves=7,
        ).fit(X.reshape(len(X), -1), rul)
        predictions = model.predict(X.reshape(len(X), -1))
        metrics = model.metrics(rul, predictions)
        self.assertEqual(set(metrics), {'rmse', 'mae', 'spearman'})
        self.assertTrue(np.isfinite(list(metrics.values())).all())

        oof = out_of_fold_predictions(
            lambda: RULLightGBMRegressor(
                n_estimators=10, learning_rate=0.1, random_state=0,
                num_leaves=7,
            ),
            X.reshape(len(X), -1), rul, groups, n_splits=4,
        )
        self.assertEqual(oof.shape, rul.shape)
        self.assertTrue(np.isfinite(oof).all())

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            LightClassifier(architecture='transformer')


if __name__ == '__main__':
    unittest.main()
