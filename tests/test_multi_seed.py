# tests/test_multi_seed.py

import csv
import json
import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.experiments.multi_seed import (
    aggregate,
    build_report,
    parse_seeds,
    run_multi_seed,
)

SENSORS = ['s2', 's4']


def write_fixture(root):
    """Create a tiny FD001-like project with parquet data and an official test."""
    config_dir = os.path.join(root, 'configs')
    data_dir = os.path.join(root, 'data', 'raw', 'CMAPSS')
    reasoning_dir = os.path.join(root, 'src', 'reasoning')
    for directory in (config_dir, data_dir, reasoning_dir):
        os.makedirs(directory, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = []
    for unit in range(1, 13):
        for cycle in range(1, 25):
            rows.append({
                'unit_number': unit,
                'cycle': cycle,
                's2': 640.0 + unit + 0.05 * cycle + rng.normal(0, 0.3),
                's4': 1400.0 + 2 * unit + 0.2 * cycle + rng.normal(0, 0.5),
            })
    data_file = os.path.join(data_dir, 'FD001_processed.parquet')
    pd.DataFrame(rows).to_parquet(data_file)

    test_rows = []
    for unit in range(1, 5):
        for cycle in range(1, 21):
            test_rows.append({
                'unit_number': unit, 'cycle': cycle,
                'op_setting_1': 0.0, 'op_setting_2': 0.0, 'op_setting_3': 0.0,
                's2': 640.0 + unit + 0.05 * cycle,
                's4': 1400.0 + 2 * unit + 0.2 * cycle,
            })
    test_file = os.path.join(data_dir, 'test_FD001.txt')
    with open(test_file, 'w') as file:
        for row in test_rows:
            file.write(' '.join(str(value) for value in row.values()) + '\n')
    rul_file = os.path.join(data_dir, 'RUL_FD001.txt')
    with open(rul_file, 'w') as file:
        for value in (4, 3, 2, 1):
            file.write(f'{value}\n')

    rules_path = os.path.join(reasoning_dir, 'rules.pl')
    with open(rules_path, 'w') as file:
        file.write('% symbolic layer disabled in this fixture\n')

    config = {
        'model': {
            'window_size': 5,
            'feature_names': SENSORS,
            'input_shape': [5, 2],
            'architecture': 'random_forest',
            'random_state': 42,
            'light_classifier': {'n_estimators': 8, 'max_depth': 3},
        },
        'training': {
            'train_fraction': 0.4,
            'validation_split': 0.2,
            'test_split': 0.1,
            'rul_threshold': 5,
            'random_state': 42,
        },
        'paths': {
            'data_file': data_file,
            'test_file': test_file,
            'rul_file': rul_file,
            'results_dir': os.path.join(root, 'results'),
            'plot_config_path': '',
            'reasoning_rules_path': rules_path,
        },
        'calibration': {'enabled': False},
        'thresholding': {
            'strategy': 'fbeta', 'beta': 2.0,
            'min_precision': 0.5, 'fallback': 0.5,
        },
        'symbolic_reasoning': {'enabled': False},
        'temporal': {'enabled': False},
        'problog': {'learned_rule_weight': 0.15, 'max_active_learned_rules': 3},
        'fusion': {'enabled': False, 'apply_to_architectures': ['lightgbm']},
        'hitl': {'enabled': False},
        'engine_metrics': {'enabled': True},
        'alerting': {'enabled': True, 'consecutive_ks': [2, 3],
                     'smoothing_cycles': 3},
        'preprocessing': {'non_finite_policy': 'zero'},
        'experiments': {
            'multi_seed': {
                'enabled': True, 'seeds': [42, 43],
                'selection_metric': 'fbeta', 'beta': 2.0,
            }
        },
        'operational_policy': {
            'enabled': True,
            'false_negative_cost': 10.0,
            'false_alarm_cost': 1.0,
            'inspection_capacity_per_cycle': None,
            'selection_metric': 'expected_cost',
            'preferred_alert_mode': None,
        },
        'monitoring': {'drift': {'enabled': False}},
        'audit': {'enabled': False},
    }
    config_path = os.path.join(config_dir, 'config.yaml')
    with open(config_path, 'w') as file:
        file.write('# fixture\n')
    return config, config_path


class TestSeedParsing(unittest.TestCase):
    def test_comma_separated_seeds(self):
        self.assertEqual(parse_seeds('42, 43,44'), [42, 43, 44])

    def test_duplicates_and_order_are_preserved_once(self):
        self.assertEqual(parse_seeds('7,7,3'), [7, 3])

    def test_defaults_come_from_config(self):
        config = {'experiments': {'multi_seed': {'seeds': [1, 2]}}}
        self.assertEqual(parse_seeds(None, config), [1, 2])

    def test_invalid_seeds_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_seeds('a,b')
        with self.assertRaises(ValueError):
            parse_seeds('')
        with self.assertRaises(ValueError):
            parse_seeds('-1')


class TestAggregation(unittest.TestCase):
    def test_aggregate_reports_dispersion_and_interval(self):
        stats = aggregate([0.2, 0.4, 0.6])
        self.assertEqual(stats['count'], 3)
        self.assertAlmostEqual(stats['mean'], 0.4)
        self.assertAlmostEqual(stats['min'], 0.2)
        self.assertAlmostEqual(stats['max'], 0.6)
        self.assertLess(stats['ci95_low'], stats['mean'])
        self.assertGreater(stats['ci95_high'], stats['mean'])

    def test_aggregate_ignores_missing_values(self):
        stats = aggregate([1.0, None, 3.0])
        self.assertEqual(stats['count'], 2)
        self.assertAlmostEqual(stats['mean'], 2.0)

    def test_build_report_selects_on_validation_metric(self):
        per_seed = [
            {
                'seed': 42, 'train_units': [1], 'validation_units': [2],
                'train_engines': 1, 'validation_engines': 1,
                'thresholds': {'neural': 0.5, 'fusion': None},
                'validation': {'accuracy': 0.9, 'precision': 0.8,
                               'recall': 0.7, 'f1': 0.75, 'fbeta': 0.72,
                               'roc_auc': 0.9, 'average_precision': 0.8},
                'test': {'accuracy': 0.9, 'precision': 0.8, 'recall': 0.7,
                         'f1': 0.75, 'fbeta': 0.72, 'roc_auc': 0.9,
                         'average_precision': 0.8},
                'calibration': {
                    'validation': {'brier_score': 0.1, 'ece': 0.05},
                    'test': {'brier_score': 0.1, 'ece': 0.05},
                },
                'engine_level': {
                    'mean_recall': 0.7, 'mean_precision': 0.8,
                    'total_false_alarms': 3, 'mean_false_alarms_per_engine': 1.0,
                    'mean_anticipation_cycles': 10.0,
                    'mean_lead_time_cycles': 5.0,
                    'proportion_alerted_before_eol': 0.5,
                    'proportion_alerted_before_eol_among_relevant': 1.0,
                },
                'alert_modes': {}, 'operational_policy': {
                    'selected_mode': 'consecutive_2',
                    'validation_expected_cost': 3.0,
                    'test_expected_cost': 4.0,
                },
            },
            {
                'seed': 43, 'train_units': [3], 'validation_units': [4],
                'train_engines': 1, 'validation_engines': 1,
                'thresholds': {'neural': 0.5, 'fusion': None},
                'validation': {'accuracy': 0.9, 'precision': 0.9,
                               'recall': 0.9, 'f1': 0.9, 'fbeta': 0.9,
                               'roc_auc': 0.9, 'average_precision': 0.9},
                'test': {'accuracy': 0.9, 'precision': 0.9, 'recall': 0.9,
                         'f1': 0.9, 'fbeta': 0.9, 'roc_auc': 0.9,
                         'average_precision': 0.9},
                'calibration': {
                    'validation': {'brier_score': 0.1, 'ece': 0.05},
                    'test': {'brier_score': 0.1, 'ece': 0.05},
                },
                'engine_level': {
                    'mean_recall': 0.9, 'mean_precision': 0.9,
                    'total_false_alarms': 1, 'mean_false_alarms_per_engine': 0.5,
                    'mean_anticipation_cycles': 20.0,
                    'mean_lead_time_cycles': 9.0,
                    'proportion_alerted_before_eol': 0.9,
                    'proportion_alerted_before_eol_among_relevant': 1.0,
                },
                'alert_modes': {}, 'operational_policy': {
                    'selected_mode': 'immediate',
                    'validation_expected_cost': 1.0,
                    'test_expected_cost': 2.0,
                },
            },
        ]
        config = {'experiments': {'multi_seed': {
            'selection_metric': 'fbeta', 'beta': 2.0,
        }}}
        report = build_report(
            per_seed, config,
            {'ok': True, 'test_units_identical': True, 'test_engines': 4},
        )
        self.assertEqual(report['selection']['selected_seed'], 43)
        self.assertAlmostEqual(
            report['aggregate']['test_f1']['mean'], 0.825
        )
        self.assertTrue(report['protocol']['official_test_unchanged'])


class TestMultiSeedRun(unittest.TestCase):
    """End-to-end protocol check on a tiny fixture."""

    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger('TestMultiSeedRun')
        cls.logger.addHandler(logging.NullHandler())
        cls.logger.setLevel(logging.WARNING)

    def test_report_csv_plot_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, config_path = write_fixture(tmpdir)
            report = run_multi_seed(
                config, config_path, self.logger, seeds=[42, 43],
                results_dir=config['paths']['results_dir'],
            )
            results_dir = config['paths']['results_dir']
            self.assertTrue(os.path.exists(
                os.path.join(results_dir, 'multi_seed_report.json')
            ))
            self.assertTrue(os.path.exists(
                os.path.join(results_dir, 'multi_seed_summary.csv')
            ))
            plot = os.path.join(
                results_dir, 'visualization', 'multi_seed_distributions.png'
            )
            self.assertGreater(os.path.getsize(plot), 1000)

            with open(os.path.join(
                    results_dir, 'multi_seed_report.json')) as file:
                payload = json.load(file)
            self.assertEqual(
                [record['seed'] for record in payload['per_seed']], [42, 43]
            )
            self.assertTrue(payload['leakage_check']['ok'])
            self.assertTrue(
                payload['leakage_check']['test_windows_identical']
            )
            for entry in payload['leakage_check']['details']:
                self.assertEqual(entry['train_validation_overlap'], [])
            fingerprints = {
                entry['test_fingerprint']
                for entry in payload['leakage_check']['details']
            }
            self.assertEqual(len(fingerprints), 1)

            with open(os.path.join(
                    results_dir, 'multi_seed_summary.csv')) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2 + 6)
            self.assertEqual(rows[0]['row'], 'seed')
            self.assertEqual(rows[-1]['row'], 'ci95_high')
            self.assertIsNotNone(rows[0]['test_f1'])

            for seed in (42, 43):
                self.assertTrue(os.path.exists(os.path.join(
                    results_dir, 'multi_seed', f'seed_{seed}',
                    'cmapss_results.json',
                )))
            self.assertEqual(len(report['per_seed']), 2)
            self.assertEqual(
                report['protocol']['test_usage'], 'reporting_only'
            )

    def test_same_seed_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, config_path = write_fixture(tmpdir)
            first = run_multi_seed(
                config, config_path, self.logger, seeds=[42],
                results_dir=config['paths']['results_dir'],
            )
            second = run_multi_seed(
                config, config_path, self.logger, seeds=[42],
                results_dir=config['paths']['results_dir'],
            )
            first_seed = first['per_seed'][0]
            second_seed = second['per_seed'][0]
            self.assertEqual(
                first_seed['train_units'], second_seed['train_units']
            )
            self.assertEqual(
                first_seed['validation_units'],
                second_seed['validation_units']
            )
            self.assertAlmostEqual(
                first_seed['test']['f1'], second_seed['test']['f1']
            )
            self.assertAlmostEqual(
                first_seed['thresholds']['neural'],
                second_seed['thresholds']['neural'],
            )


if __name__ == '__main__':
    unittest.main()
