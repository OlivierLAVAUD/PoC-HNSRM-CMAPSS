# tests/test_temporal_features.py

import os
import tempfile
import unittest

import numpy as np

from src.features.temporal_features import TemporalFeatureExtractor


def _windows(seed=0, n=12, cycles=6, sensors=3):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, cycles, sensors)).astype(np.float32)
    if n:
        values[:, -1, 0] += 2.0
    return values


class TestTemporalFeatures(unittest.TestCase):
    def test_fit_transform_shape_metadata_and_persistence(self):
        train = _windows()
        extractor = TemporalFeatureExtractor(
            ['a', 'b', 'c'], include_raw_window=True,
            moving_average_cycles=3, percentile_low=20, percentile_high=80,
        )
        train_features = extractor.fit_transform(train)

        expected = 13 * 3 + 6 * 3
        self.assertEqual(train_features.shape, (len(train), expected))
        self.assertEqual(len(extractor.feature_names_out), expected)
        self.assertTrue(np.isfinite(train_features).all())
        self.assertEqual(extractor.feature_names_out[0], 'a__first')
        self.assertEqual(extractor.feature_names_out[-1], 'raw__c__05')

        metadata = extractor.metadata()
        self.assertEqual(metadata['window_size'], 6)
        self.assertIn('a', metadata['high_thresholds'])
        self.assertEqual(metadata['raw_order'], 'sensor-major, then cycle')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'extractor.joblib')
            metadata_path = os.path.join(tmpdir, 'extractor.json')
            extractor.save(path)
            extractor.save_metadata(metadata_path)
            loaded = TemporalFeatureExtractor.load(path)
            self.assertTrue(os.path.exists(metadata_path))
            np.testing.assert_allclose(loaded.transform(train), train_features)

    def test_persistence_uses_training_thresholds_only(self):
        train = np.zeros((3, 4, 1), dtype=np.float32)
        train[:, :, 0] = np.arange(12).reshape(3, 4)
        extractor = TemporalFeatureExtractor(
            ['a'], include_raw_window=False,
            percentile_low=25, percentile_high=75,
        ).fit(train)

        validation = np.zeros((1, 4, 1), dtype=np.float32)
        validation[0, :, 0] = [-100, -100, -100, -100]
        features = extractor.transform(validation)
        names = extractor.feature_names_out
        above = features[0, names.index('a__persistence_above')]
        below = features[0, names.index('a__persistence_below')]
        self.assertEqual(above, 0.0)
        self.assertEqual(below, 4.0)
        self.assertLess(extractor.high_thresholds['a'], 11.0)

    def test_raw_window_is_optional(self):
        train = _windows()
        extractor = TemporalFeatureExtractor(
            ['a', 'b', 'c'], include_raw_window=False,
        ).fit(train)
        features = extractor.transform(train)
        self.assertEqual(features.shape[1], 13 * 3)
        self.assertFalse(
            any(name.startswith('raw__') for name in extractor.feature_names_out)
        )

    def test_rejects_non_finite_windows(self):
        train = _windows()
        train[0, 0, 0] = np.nan
        extractor = TemporalFeatureExtractor(['a', 'b', 'c'])
        with self.assertRaises(ValueError):
            extractor.fit(train)


if __name__ == '__main__':
    unittest.main()
