# tests/test_light_classifier_cmapss.py

import os
import tempfile
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, recall_score

from src.data.cmapss_loader import CMAPSSLoader
from src.models.light_classifier import LightClassifier

FD001_PARQUET = os.path.join('data', 'raw', 'CMAPSS', 'FD001_processed.parquet')


class TestLightClassifierCMAPSS(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(FD001_PARQUET), "FD001 parquet not downloaded")
    def test_twenty_percent_units_reach_validation_accuracy(self):
        loader = CMAPSSLoader(FD001_PARQUET, window_size=30, rul_threshold=30)
        X, y = loader.load_data()
        X_train, X_val, _, y_train, y_val, _ = loader.split_by_unit(
            X, y, validation_split=0.2, test_split=0.6, random_state=42
        )

        model = LightClassifier(architecture="random_forest", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred, zero_division=0)
        self.assertGreater(accuracy, 0.90)
        self.assertGreater(recall, 0.50)

    def test_flatten_save_and_load(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 5, 4)).astype(np.float32)
        y = (X[:, :, 0].mean(axis=1) > 0).astype(int)

        model = LightClassifier(random_state=0, n_estimators=10, max_depth=3)
        model.fit(X, y)
        proba = model.positive_proba(X)
        self.assertEqual(proba.shape, (len(X),))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.joblib')
            model.save(path)
            loaded = LightClassifier.load(path)
            np.testing.assert_allclose(loaded.positive_proba(X), proba)


if __name__ == '__main__':
    unittest.main()
