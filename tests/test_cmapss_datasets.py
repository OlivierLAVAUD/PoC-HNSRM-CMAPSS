# tests/test_cmapss_datasets.py

import copy
import os
import unittest

from main import (
    CMAPSS_DATASETS,
    _config_path_for_dataset,
    _parse_datasets,
    build_parser,
)
from src.config.config_manager import load_config
from src.data.cmapss_datasets import (
    missing_data_message,
    validate_dataset_config,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCmapssDatasetConfiguration(unittest.TestCase):
    def test_every_dataset_resolves_to_a_config(self):
        for name in CMAPSS_DATASETS:
            path = _config_path_for_dataset(PROJECT_ROOT, name)
            self.assertTrue(os.path.exists(path), path)

    def test_configs_target_their_own_raw_and_result_paths(self):
        base = load_config(_config_path_for_dataset(PROJECT_ROOT, 'fd001'))
        base_features = list(base['model']['feature_names'])
        for name in ('fd002', 'fd003', 'fd004'):
            config = load_config(_config_path_for_dataset(PROJECT_ROOT, name))
            token = name.upper()
            self.assertEqual(config['dataset']['id'], token)
            self.assertIn(token, config['paths']['data_file'])
            self.assertIn(token, config['paths']['test_file'])
            self.assertIn(token, config['paths']['rul_file'])
            self.assertIn(name, config['paths']['results_dir'])
            self.assertEqual(
                list(config['model']['feature_names']), base_features
            )

    def test_dataset_configs_inherit_the_base_and_override_minimal_keys(self):
        base = load_config(_config_path_for_dataset(PROJECT_ROOT, 'fd001'))
        for name in ('fd002', 'fd003', 'fd004'):
            config = load_config(_config_path_for_dataset(PROJECT_ROOT, name))
            self.assertNotIn('extends', config)
            self.assertEqual(config['fusion'], base['fusion'])
            self.assertEqual(config['alerting'], base['alerting'])
            self.assertEqual(
                config['hitl']['prioritization'],
                base['hitl']['prioritization'],
            )
            self.assertNotEqual(config['paths'], base['paths'])
            report = validate_dataset_config(config)
            self.assertEqual(report['sensor_count'], 14)
            self.assertEqual(report['input_shape'], [30, 14])
            self.assertIsInstance(report['missing_files'], list)

            phantom = copy.deepcopy(config)
            token = name.upper()
            phantom['paths']['data_file'] = (
                f'data/raw/CMAPSS/{token}_missing.parquet'
            )
            phantom['paths']['test_file'] = f'data/raw/CMAPSS/test_{token}_missing.txt'
            phantom['paths']['rul_file'] = f'data/raw/CMAPSS/RUL_{token}_missing.txt'
            phantom_report = validate_dataset_config(phantom)
            self.assertIn('data_file', phantom_report['missing_files'])
            self.assertIn('test_file', phantom_report['missing_files'])
            self.assertIn('rul_file', phantom_report['missing_files'])

    def test_dataset_validation_rejects_sensor_and_path_mismatches(self):
        config = load_config(_config_path_for_dataset(PROJECT_ROOT, 'fd002'))
        broken = dict(config)
        broken['model'] = dict(config['model'])
        broken['model']['input_shape'] = [30, 13]
        with self.assertRaises(ValueError):
            validate_dataset_config(broken)

        broken = dict(config)
        broken['model'] = dict(config['model'])
        broken['model']['feature_names'] = list(
            config['model']['feature_names']
        ) + ['s99']
        broken['model']['input_shape'] = [30, 15]
        with self.assertRaises(ValueError):
            validate_dataset_config(broken)

        broken = dict(config)
        broken['paths'] = dict(config['paths'])
        broken['paths']['test_file'] = 'data/raw/CMAPSS/test_FD001.txt'
        with self.assertRaises(ValueError):
            validate_dataset_config(broken)

    def test_missing_data_message_names_the_convention(self):
        config = load_config(_config_path_for_dataset(PROJECT_ROOT, 'fd003'))
        message = missing_data_message(config, ['data_file', 'test_file'])
        self.assertIn('FD003', message)
        self.assertIn('FD003_processed.parquet', message)
        self.assertIn('test_FD003.txt', message)
        self.assertIn('--cmapss-dataset', message)

    def test_dataset_argument_validation(self):
        self.assertEqual(
            _parse_datasets('FD002, fd004'), ['fd002', 'fd004']
        )
        with self.assertRaises(ValueError):
            _parse_datasets('')
        with self.assertRaises(ValueError):
            _parse_datasets('fd001,fd999')
        with self.assertRaises(ValueError):
            _config_path_for_dataset(PROJECT_ROOT, 'fd999')

    def test_cli_exposes_dataset_and_cross_condition_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            '--cmapss-dataset', 'fd003',
            '--cross-condition',
            '--cross-train-dataset', 'fd001',
            '--cross-test-datasets', 'fd002,fd004',
            '--robustness-report',
        ])
        self.assertEqual(args.cmapss_dataset, 'fd003')
        self.assertTrue(args.cross_condition)
        self.assertEqual(args.cross_train_dataset, 'fd001')
        self.assertEqual(args.cross_test_datasets, 'fd002,fd004')
        self.assertTrue(args.robustness_report)


if __name__ == '__main__':
    unittest.main()
