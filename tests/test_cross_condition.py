# tests/test_cross_condition.py

import csv
import json
import logging
import os
import tempfile
import unittest

import pandas as pd

from src.experiments.cross_condition import (
    _dataset_id,
    _metric_row,
    _target_specs,
    run_cross_condition,
)


def _raw_file(path, units, length, offset=0.0, op_settings=None,
              feature_shifts=None):
    rows = []
    for unit in range(1, units + 1):
        index = (unit - 1) % len(op_settings) if op_settings else 0
        settings = op_settings[index] if op_settings else (0.0, 0.0, 100.0)
        shift = feature_shifts[index] if feature_shifts else 0.0
        for cycle in range(1, length + 1):
            rows.append(
                [unit + offset, cycle, *settings]
                + [10.0 * index + cycle + shift for index in range(1, 22)]
            )
    pd.DataFrame(rows).to_csv(path, sep=' ', header=False, index=False)
    return path


def _test_files(directory, name, units=8, length=35, op_settings=None,
                feature_shifts=None):
    test_path = _raw_file(
        os.path.join(directory, f'test_{name}.txt'), units, length,
        op_settings=op_settings, feature_shifts=feature_shifts,
    )
    rul_path = os.path.join(directory, f'RUL_{name}.txt')
    pd.DataFrame([length] * units).to_csv(
        rul_path, sep=' ', header=False, index=False
    )
    return test_path, rul_path


def _config(directory, name, data_file, test_file, rul_file):
    return {
        'dataset': {'id': name},
        'model': {
            'window_size': 10, 'feature_names': ['s2', 's4'],
            'input_shape': [10, 2], 'architecture': 'random_forest',
            'random_state': 42,
        },
        'training': {
            'validation_split': 0.25, 'test_split': 0.25,
            'train_fraction': 0.5, 'rul_threshold': 30, 'random_state': 42,
        },
        'paths': {
            'data_file': data_file, 'test_file': test_file,
            'rul_file': rul_file,
            'results_dir': os.path.join(directory, f'results_{name}'),
            'reasoning_rules_path': os.path.abspath('src/reasoning/rules.pl'),
        },
        'symbolic_reasoning': {'enabled': False},
        'hitl': {'enabled': False},
        'audit': {'enabled': False},
        'monitoring': {'drift': {'enabled': False}},
        'features': {'temporal': {'enabled': False}},
    }


class TestCrossConditionHelpers(unittest.TestCase):
    def test_dataset_identity_is_detected(self):
        self.assertEqual(_dataset_id({'dataset': {'id': 'FD003'}}), 'FD003')
        self.assertEqual(_dataset_id({
            'paths': {'data_file': 'train_FD004.txt'}
        }), 'FD004')

    def test_target_specs_are_normalised(self):
        specs = _target_specs([{
            'name': 'FD002', 'config': {'dataset': {'id': 'FD002'}},
            'config_path': 'configs/config_fd002.yaml',
        }])
        self.assertEqual(specs[0]['name'], 'FD002')
        with self.assertRaises(ValueError):
            _target_specs([])
        with self.assertRaises(ValueError):
            _target_specs(['not-a-dict'])

    def test_metric_row_has_the_csv_contract(self):
        row = _metric_row(
            'FD002', 'random_forest', 0.5, 10, 2,
            {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
             'roc_auc': 1.0, 'average_precision': 1.0},
            {'mean_recall': 1.0, 'mean_precision': 1.0,
             'total_false_alarms': 0},
            {'episode_recall': 1.0, 'episode_precision': 1.0,
             'false_alarm_episodes': 0, 'mean_anticipation_cycles': 3},
            {'brier_score': 0.12, 'ece': 0.05},
            {'f1': 0.5, 'recall': 0.5, 'precision': 0.5,
             'average_precision': 0.5},
            'test_FD002.txt',
        )
        self.assertEqual(row['target_dataset'], 'FD002')
        self.assertEqual(row['episode_recall'], 1.0)
        self.assertEqual(row['brier_score'], 0.12)
        self.assertIn('f1', row)
        self.assertAlmostEqual(row['relative_degradation_f1'], -1.0)
        self.assertAlmostEqual(row['relative_degradation_recall'], -1.0)


class TestCrossConditionRun(unittest.TestCase):
    def test_source_transforms_are_reused_and_targets_are_reporting_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_train = _raw_file(
                os.path.join(tmpdir, 'train_FD001.txt'), 14, 42
            )
            source_test, source_rul = _test_files(tmpdir, 'FD001')
            target_train = _raw_file(
                os.path.join(tmpdir, 'train_FD002.txt'), 14, 42, offset=100,
            )
            target_test, target_rul = _test_files(tmpdir, 'FD002')
            source_config = _config(
                tmpdir, 'FD001', source_train, source_test, source_rul
            )
            target_config = _config(
                tmpdir, 'FD002', target_train, target_test, target_rul
            )
            source_path = os.path.join(tmpdir, 'configs', 'source.yaml')
            target_path = os.path.join(tmpdir, 'configs', 'target.yaml')
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            open(source_path, 'w').close()
            open(target_path, 'w').close()
            logger = logging.getLogger('TestCrossConditionRun')
            logger.addHandler(logging.NullHandler())
            report = run_cross_condition(
                source_config, source_path,
                [{
                    'name': 'FD002', 'config': target_config,
                    'config_path': target_path,
                }],
                logger, results_dir=os.path.join(tmpdir, 'out'),
            )

            self.assertEqual(
                report['protocol']['source_dataset'], 'FD001'
            )
            self.assertEqual(
                report['protocol']['target_datasets'], ['FD002']
            )
            self.assertEqual(
                report['protocol']['selection_split'], 'source_validation'
            )
            self.assertEqual(
                report['protocol']['target_usage'], 'reporting_only'
            )
            self.assertTrue(report['protocol']['source_transforms_reused'])
            self.assertEqual(len(report['rows']), 1)
            self.assertEqual(report['rows'][0]['target_dataset'], 'FD002')
            self.assertGreater(report['rows'][0]['windows'], 0)
            self.assertIn('brier_score', report['rows'][0])
            self.assertIn('ece', report['rows'][0])
            self.assertIn(
                'relative_degradation_f1', report['rows'][0]
            )
            self.assertEqual(
                report['protocol']['train_conditions'], ['FD001']
            )
            self.assertEqual(
                report['protocol']['test_conditions'], ['FD002']
            )
            self.assertEqual(
                report['protocol']['degradation_reference'],
                'source_validation',
            )
            self.assertIn(
                'calibration', report['targets']['FD002']
            )
            self.assertIn(
                'relative_degradation_vs_source_validation',
                report['targets']['FD002'],
            )
            self.assertTrue(os.path.exists(
                os.path.join(tmpdir, 'out', 'cross_condition_report.json')
            ))
            with open(os.path.join(
                    tmpdir, 'out', 'cross_condition_summary.csv'),
                    newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            with open(os.path.join(
                    tmpdir, 'out', 'cross_condition_report.json')) as file:
                payload = json.load(file)
            self.assertEqual(payload['rows'][0]['target_dataset'], 'FD002')

    def test_per_condition_adaptation_reports_a_full_label_free_ablation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_train = _raw_file(
                os.path.join(tmpdir, 'train_FD001.txt'), 14, 42
            )
            source_test, source_rul = _test_files(tmpdir, 'FD001')
            target_train = _raw_file(
                os.path.join(tmpdir, 'train_FD002.txt'), 14, 42, offset=100,
                op_settings=[(0.0, 0.0, 100.0), (5.0, 5.0, 200.0)],
                feature_shifts=[0.0, 1000.0],
            )
            target_test, target_rul = _test_files(
                tmpdir, 'FD002',
                op_settings=[(0.0, 0.0, 100.0), (5.0, 5.0, 200.0)],
                feature_shifts=[0.0, 1000.0],
            )
            source_config = _config(
                tmpdir, 'FD001', source_train, source_test, source_rul
            )
            target_config = _config(
                tmpdir, 'FD002', target_train, target_test, target_rul
            )
            target_config['dataset']['operating_conditions'] = 2
            source_config['experiments'] = {
                'cross_condition': {
                    'adaptation': {
                        'normalization': 'per_condition',
                        'threshold_strategy': 'target_alert_rate',
                    },
                },
            }
            source_path = os.path.join(tmpdir, 'configs', 'source.yaml')
            target_path = os.path.join(tmpdir, 'configs', 'target.yaml')
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            open(source_path, 'w').close()
            open(target_path, 'w').close()
            logger = logging.getLogger('TestCrossConditionAdaptation')
            logger.addHandler(logging.NullHandler())
            report = run_cross_condition(
                source_config, source_path,
                [{
                    'name': 'FD002', 'config': target_config,
                    'config_path': target_path,
                }],
                logger, results_dir=os.path.join(tmpdir, 'out'),
            )

            protocol = report['protocol']
            self.assertEqual(protocol['normalization'], 'per_condition')
            self.assertEqual(
                protocol['threshold_strategy'], 'target_alert_rate'
            )
            self.assertEqual(protocol['variant'], 'per_condition_alert_rate')
            self.assertEqual(
                protocol['adaptation_usage'], 'unlabeled_target_features'
            )
            self.assertFalse(protocol['source_transforms_reused'])
            self.assertEqual(
                set(protocol['variants_evaluated']),
                {
                    'zero_shot', 'per_condition', 'alert_rate',
                    'per_condition_alert_rate',
                },
            )

            row = report['rows'][0]
            self.assertEqual(row['variant'], 'per_condition_alert_rate')
            self.assertEqual(row['normalization'], 'per_condition')
            self.assertIn('baseline_f1', row)
            self.assertIn('baseline_false_alarm_episodes', row)
            self.assertIsNotNone(row['source_validation_alert_rate'])
            self.assertIsNotNone(row['target_alert_rate'])

            target = report['targets']['FD002']
            self.assertEqual(
                set(target['variants']),
                set(protocol['variants_evaluated']),
            )
            self.assertEqual(
                target['selected_variant'], 'per_condition_alert_rate'
            )
            self.assertEqual(
                target['adaptation']['n_conditions'], 2
            )
            self.assertEqual(
                sum(target['adaptation']['condition_counts']),
                target['windows'],
            )
            self.assertIn('baseline', target)


if __name__ == '__main__':
    unittest.main()
