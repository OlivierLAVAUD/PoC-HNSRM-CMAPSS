# tests/test_temporal_reasoning.py

import logging
import os
import shutil
import tempfile
import unittest

import numpy as np

from src.reasoning.prob_query import EXAMPLES
from src.reasoning.reasoning import PYSWIP_AVAILABLE, SymbolicReasoner
from src.reasoning.temporal import compute_temporal_thresholds, window_temporal_features

FEATURES = list(EXAMPLES['normal'].keys())


def _temporal_window():
    window = np.tile(
        np.array([EXAMPLES['normal'][name] for name in FEATURES], dtype=float),
        (30, 1),
    )
    window[:, FEATURES.index('s4')] = np.linspace(1400.0, 1430.0, 30)
    window[:, FEATURES.index('s7')] = np.linspace(560.0, 540.0, 30)
    window[:, FEATURES.index('s11')] = 47.0
    window[-10:, FEATURES.index('s11')] = 48.1
    return window


@unittest.skipUnless(PYSWIP_AVAILABLE, "SWI-Prolog/PySWIP not available")
class TestTemporalReasoning(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='ansrdt_temporal_')
        rules_path = os.path.join(self.temp_dir, 'rules.pl')
        shutil.copy('src/reasoning/rules.pl', rules_path)
        logger = logging.getLogger('TestTemporalReasoning')
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)
        self.reasoner = SymbolicReasoner(
            rules_path, input_shape=(30, len(FEATURES)), model=None, logger=logger
        )
        self.reasoner.get_sensor_thresholds()
        self.reasoner.set_temporal_thresholds({
            's4': {'slope_abs': 0.1, 'relative_change_abs': 0.001,
                   'std': 0.1, 'persistence': 3},
            's7': {'slope_abs': 0.1, 'relative_change_abs': 0.001,
                   'std': 0.1, 'persistence': 3},
            's11': {'slope_abs': 0.1, 'relative_change_abs': 0.001,
                    'std': 0.1, 'persistence': 3},
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_temporal_rules_fire_on_full_window(self):
        window = _temporal_window()
        sensor_dict = {name: float(window[-1, i]) for i, name in enumerate(FEATURES)}
        insights = self.reasoner.reason(sensor_dict, window=window)

        for expected in [
            'HPC temperature rising trend',
            'HPC outlet pressure falling trend',
            'HPC static pressure persistent high',
            'HPC thermal-pressure degradation',
            'HPC temperature variability high',
            'HPC moving-average temperature high',
        ]:
            self.assertIn(expected, insights)

    def test_no_window_clears_temporal_insights(self):
        window = _temporal_window()
        sensor_dict = {name: float(window[-1, i]) for i, name in enumerate(FEATURES)}
        self.reasoner.reason(sensor_dict, window=window)
        insights = self.reasoner.reason(sensor_dict)
        self.assertNotIn('HPC temperature rising trend', insights)
        self.assertNotIn('HPC thermal-pressure degradation', insights)

    def test_temporal_features_and_training_thresholds(self):
        window = _temporal_window()
        features = window_temporal_features(
            window, FEATURES, {'s11': 47.7}, {'s7': 552.81}
        )
        self.assertGreater(features['slope']['s4'], 0)
        self.assertLess(features['slope']['s7'], 0)
        self.assertEqual(features['persistence_above']['s11'], 10)
        self.assertGreaterEqual(features['persistence_below']['s7'], 3)

        windows = np.stack([window, window + 1.0])
        thresholds = compute_temporal_thresholds(
            windows, FEATURES, {'s11': 47.7}, {'s7': 552.81}
        )
        self.assertIn('s4', thresholds)
        self.assertGreaterEqual(thresholds['s4']['persistence'], 3)
        self.assertGreaterEqual(thresholds['s4']['slope_abs'], 0.0)


if __name__ == '__main__':
    unittest.main()
