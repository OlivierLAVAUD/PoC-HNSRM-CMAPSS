# tests/test_physical_representation.py

import os
import tempfile
import unittest

import numpy as np

from src.features.physical_representation import (
    VARIANTS,
    PhysicalRepresentation,
)

FEATURES = ['s2', 's4']
WINDOW = 8


def make_windows(offset=0.0, units=(1, 2), cycles_per_unit=12):
    rng = np.random.default_rng(7)
    windows, unit_values, cycle_values = [], [], []
    for unit in units:
        base = np.arange(cycles_per_unit, dtype=float)[:, None] * 0.1
        trajectory = np.hstack([
            base + 640.0 + unit + offset,
            base * 2.0 + 1400.0 + unit + offset,
        ])
        trajectory += rng.normal(0, 0.01, trajectory.shape)
        for end in range(WINDOW, cycles_per_unit + 1):
            windows.append(trajectory[end - WINDOW:end])
            unit_values.append(unit)
            cycle_values.append(end)
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(unit_values), np.asarray(cycle_values),
    )


class TestPhysicalRepresentation(unittest.TestCase):
    def test_raw_matches_flattened_windows(self):
        X, units, cycles = make_windows()
        representation = PhysicalRepresentation(
            FEATURES, variant='raw', early_life_cycles=4
        ).fit(X, units, cycles)
        transformed = representation.transform(X, units, cycles)
        np.testing.assert_allclose(
            transformed, X.reshape(len(X), -1), rtol=1e-6, atol=1e-5
        )
        self.assertEqual(len(representation.feature_names_out), 16)
        self.assertTrue(representation.training_hash)

    def test_deltas_are_finite_and_include_dynamics(self):
        X, units, cycles = make_windows()
        representation = PhysicalRepresentation(
            FEATURES, variant='deltas', early_life_cycles=4
        ).fit(X, units, cycles)
        transformed = representation.transform(X, units, cycles)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape[0], len(X))
        self.assertIn('delta_5__s2', representation.feature_names_out)
        self.assertEqual(
            len(representation.feature_names_out), 2 * 7 + 2
        )

    def test_early_life_residual_uses_training_baseline_consistently(self):
        X, units, cycles = make_windows()
        representation = PhysicalRepresentation(
            FEATURES, variant='early_life_residual', early_life_cycles=4
        ).fit(X, units, cycles)
        first = representation.transform(X, units, cycles)
        second = representation.transform(X, units, cycles)
        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape[1], 2 * WINDOW + 1)
        self.assertEqual(first[:, -1].min(), 1.0)
        self.assertLessEqual(np.abs(first[:, :-1]).max(), representation.clip)

    def test_short_trajectory_is_neutral_and_flagged(self):
        X, units, cycles = make_windows(cycles_per_unit=8)
        representation = PhysicalRepresentation(
            FEATURES, variant='early_life_residual',
            early_life_cycles=10, fallback_policy='neutral',
        ).fit(X, units, cycles)
        transformed = representation.transform(X, units, cycles)
        self.assertEqual(transformed[:, -1].min(), 0.0)
        self.assertTrue(np.isfinite(transformed).all())

    def test_short_trajectory_can_raise(self):
        X, units, cycles = make_windows(cycles_per_unit=8)
        with self.assertRaises(ValueError):
            PhysicalRepresentation(
                FEATURES, variant='early_life_residual',
                early_life_cycles=10, fallback_policy='error',
            ).fit(X, units, cycles)

    def test_operating_condition_residual_requires_and_uses_settings(self):
        X, units, cycles = make_windows()
        op = np.tile([100.0, 0.5, 60.0], (len(X), 1))
        representation = PhysicalRepresentation(
            FEATURES, variant='operating_condition_residual',
            early_life_cycles=4, n_conditions=1,
        ).fit(X, units, cycles, op_settings=op)
        transformed = representation.transform(X, units, cycles, op_settings=op)
        self.assertEqual(transformed.shape[1], 2 * WINDOW + 1)
        self.assertEqual(transformed[:, -1].min(), 1.0)
        self.assertEqual(representation.metadata()['condition_counts'], [
            len(X)
        ])
        with self.assertRaises(ValueError):
            PhysicalRepresentation(
                FEATURES, variant='operating_condition_residual'
            ).fit(X, units, cycles)

    def test_residual_plus_deltas_combines_features(self):
        X, units, cycles = make_windows()
        representation = PhysicalRepresentation(
            FEATURES, variant='residual_plus_deltas', early_life_cycles=4
        ).fit(X, units, cycles)
        transformed = representation.transform(X, units, cycles)
        expected = 2 * WINDOW + 16 + 1
        self.assertEqual(transformed.shape[1], expected)
        self.assertTrue(np.isfinite(transformed).all())

    def test_save_load_round_trip_preserves_transform_and_metadata(self):
        X, units, cycles = make_windows()
        representation = PhysicalRepresentation(
            FEATURES, variant='deltas', early_life_cycles=4
        ).fit(X, units, cycles)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'representation.joblib')
            representation.save(path)
            metadata_path = os.path.join(tmpdir, 'representation.json')
            representation.save_metadata(metadata_path)
            loaded = PhysicalRepresentation.load(path)
            np.testing.assert_allclose(
                loaded.transform(X, units, cycles),
                representation.transform(X, units, cycles),
            )
            self.assertEqual(
                loaded.metadata()['training_hash'],
                representation.training_hash,
            )
            self.assertTrue(os.path.exists(metadata_path))

    def test_rejects_unknown_variant_and_shape_mismatch(self):
        X, units, cycles = make_windows()
        with self.assertRaises(ValueError):
            PhysicalRepresentation(FEATURES, variant='unknown')
        representation = PhysicalRepresentation(
            FEATURES, variant='raw'
        ).fit(X, units, cycles)
        with self.assertRaises(ValueError):
            representation.transform(X[:, :, :1])

    def test_variants_are_the_documented_set(self):
        self.assertEqual(
            set(VARIANTS),
            {
                'raw', 'deltas', 'early_life_residual',
                'operating_condition_residual', 'residual_plus_deltas',
            },
        )


if __name__ == '__main__':
    unittest.main()
