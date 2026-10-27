# tests/test_core.py

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.ansrdt.core import ANSRDTCore


class TestANSRDTCore(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()
        self.config = {
            'model': {
                'window_size': 10,
                'feature_names': [
                    'temperature', 'vibration', 'pressure', 'operational_hours',
                    'efficiency_index', 'system_state', 'performance_score',
                ],
            },
            'paths': {
                'results_dir': 'results',
                'reasoning_rules_path': 'src/reasoning/rules.pl',
            },
            'symbolic_reasoning': {'enabled': True},
            'logging': {'state_history_limit': 5},
        }

    def _build_core(self, classifier=None, reasoner=None):
        classifier = classifier or MagicMock()
        with patch('src.ansrdt.core.load_config', return_value=self.config), \
             patch.object(ANSRDTCore, '_initialize_reasoner', return_value=reasoner):
            return ANSRDTCore('configs/config.yaml', logger=self.logger, classifier=classifier)

    def test_initialization_with_injected_components(self):
        reasoner = MagicMock()
        classifier = MagicMock()

        core = self._build_core(classifier=classifier, reasoner=reasoner)

        self.assertIs(core.classifier, classifier)
        self.assertIs(core.reasoner, reasoner)
        self.assertEqual(core.window_size, 10)
        self.assertEqual(len(core.feature_names), 7)
        self.assertTrue(core.results_dir.endswith('results'))

    def test_preprocess_data_invalid_shape(self):
        core = self._build_core()
        with self.assertRaises(ValueError):
            core.preprocess_data(np.array([1, 2, 3]))

    def test_update_state(self):
        classifier = MagicMock()
        classifier.positive_proba.return_value = np.array([0.6])

        reasoner = MagicMock()
        reasoner.reason.return_value = ['High temperature', 'Low pressure']
        reasoner.get_rule_activations.return_value = [{'activated_rules_detailed': ['rule_a']}]

        core = self._build_core(classifier=classifier, reasoner=reasoner)
        sensor_data = np.random.rand(10, 7).astype(np.float32)
        state = core.update_state(sensor_data)

        self.assertAlmostEqual(state['failure_probability'], 0.6, places=5)
        self.assertTrue(state['anomaly_detected'])
        self.assertEqual(state['insights'], ['High temperature', 'Low pressure'])
        self.assertEqual(state['rule_activations'], ['rule_a'])
        self.assertEqual(len(core.state_history), 1)


if __name__ == '__main__':
    unittest.main()
