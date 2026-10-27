# tests/test_config_validation.py

import logging
import os
import tempfile
import unittest

from src.pipeline.pipeline import validate_config


class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('TestLogger')
        self.logger.addHandler(logging.NullHandler())

    def test_validate_config_correct_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = tmpdir
            config_dir = os.path.join(project_root, 'configs')
            data_dir = os.path.join(project_root, 'data', 'raw', 'CMAPSS')
            reasoning_dir = os.path.join(project_root, 'src', 'reasoning')
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(reasoning_dir, exist_ok=True)

            data_file = os.path.join(data_dir, 'FD001_processed.parquet')
            rules_file = os.path.join(reasoning_dir, 'rules.pl')
            open(data_file, 'a').close()
            open(rules_file, 'a').close()

            config = {
                'model': {},
                'training': {},
                'paths': {},
            }

            validate_config(config, self.logger, project_root, config_dir)

            self.assertEqual(config['paths']['data_file'], data_file)
            self.assertEqual(config['paths']['results_dir'], os.path.join(project_root, 'results'))
            self.assertEqual(config['paths']['plot_config_path'], os.path.join(config_dir, 'plot_config.yaml'))
            self.assertEqual(config['paths']['reasoning_rules_path'], rules_file)
            self.assertEqual(config['paths']['test_file'], '')
            self.assertEqual(config['paths']['rul_file'], '')
            self.assertTrue(os.path.isdir(config['paths']['results_dir']))

    def _minimal_config(self):
        return {
            'model': {
                'window_size': 3,
                'feature_names': ['a', 'b'],
                'input_shape': [3, 2],
                'architecture': 'random_forest',
            },
            'training': {'validation_split': 0.2, 'test_split': 0.1,
                         'train_fraction': 0.2},
            'paths': {},
        }

    def test_rejects_inconsistent_input_shape(self):
        config = self._minimal_config()
        config['model']['input_shape'] = [3, 1]
        with self.assertRaises(ValueError):
            validate_config(config, self.logger, '/tmp/project', '/tmp/project/configs')

    def test_rejects_duplicate_features_and_bad_architecture(self):
        config = self._minimal_config()
        config['model']['feature_names'] = ['a', 'a']
        with self.assertRaises(ValueError):
            validate_config(config, self.logger, '/tmp/project', '/tmp/project/configs')

        config = self._minimal_config()
        config['model']['architecture'] = 'transformer'
        with self.assertRaises(ValueError):
            validate_config(config, self.logger, '/tmp/project', '/tmp/project/configs')

    def test_rejects_invalid_calibration_and_threshold(self):
        config = self._minimal_config()
        config['calibration'] = {'enabled': True, 'method': 'platt'}
        with self.assertRaises(ValueError):
            validate_config(config, self.logger, '/tmp/project', '/tmp/project/configs')

        config = self._minimal_config()
        config['thresholding'] = {'strategy': 'fbeta', 'beta': 0.0}
        with self.assertRaises(ValueError):
            validate_config(config, self.logger, '/tmp/project', '/tmp/project/configs')

    def test_rejects_invalid_non_finite_policy(self):
        config = self._minimal_config()
        config['preprocessing'] = {'non_finite_policy': 'ignore'}
        with self.assertRaises(ValueError):
            validate_config(config, self.logger, '/tmp/project', '/tmp/project/configs')

    def test_validate_config_missing_key(self):
        config = {
            'model': {},
            'training': {},
        }
        with self.assertRaises(KeyError):
            validate_config(config, self.logger, '/path/to/project', '/path/to/project/configs')


if __name__ == '__main__':
    unittest.main()
