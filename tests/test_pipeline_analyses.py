# tests/test_pipeline_analyses.py

import json
import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.pipeline.pipeline import ANSRDTPipeline
from src.utils.audit import verify_audit

SENSORS = ['s2', 's4']
ALERT_MODES = {
    'immediate', 'consecutive_2', 'consecutive_3', 'smoothed_3', 'smoothed_5',
}


def write_project(root):
    """Create a small C-MAPSS project exercising every new analysis."""
    config_dir = os.path.join(root, 'configs')
    data_dir = os.path.join(root, 'data', 'raw', 'CMAPSS')
    reasoning_dir = os.path.join(root, 'src', 'reasoning')
    for directory in (config_dir, data_dir, reasoning_dir):
        os.makedirs(directory, exist_ok=True)

    rng = np.random.default_rng(3)
    rows = []
    for unit in range(1, 9):
        for cycle in range(1, 25):
            rows.append({
                'unit_number': unit,
                'cycle': cycle,
                's2': 640.0 + unit + 0.05 * cycle + rng.normal(0, 0.3),
                's4': 1400.0 + 2 * unit + 0.2 * cycle + rng.normal(0, 0.5),
            })
    data_file = os.path.join(data_dir, 'FD001_processed.parquet')
    pd.DataFrame(rows).to_parquet(data_file)

    with open(os.path.join(data_dir, 'test_FD001.txt'), 'w') as file:
        for unit in range(1, 4):
            for cycle in range(1, 16):
                values = [
                    unit, cycle, 0.0, 0.0, 0.0,
                    640.0 + unit + 0.05 * cycle,
                    1400.0 + 2 * unit + 0.2 * cycle,
                ]
                file.write(' '.join(str(value) for value in values) + '\n')
    test_file = os.path.join(data_dir, 'test_FD001.txt')
    rul_file = os.path.join(data_dir, 'RUL_FD001.txt')
    with open(rul_file, 'w') as file:
        for value in (5, 4, 3):
            file.write(f'{value}\n')

    rules_path = os.path.join(reasoning_dir, 'rules.pl')
    with open(rules_path, 'w') as file:
        file.write('% symbolic layer disabled in this fixture\n')
    config_path = os.path.join(config_dir, 'config.yaml')
    with open(config_path, 'w') as file:
        file.write('# fixture\n')

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
            'train_fraction': 0.5,
            'validation_split': 0.25,
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
            'multi_seed': {'enabled': False, 'seeds': [42]},
        },
        'threshold_sensitivity': {
            'enabled': True,
            'min_precision_values': [0.5, 0.75],
            'beta_values': [1.0, 2.0],
            'strategies': ['recall_at_precision', 'fbeta', 'f1'],
        },
        'operational_policy': {
            'enabled': True,
            'false_negative_cost': 10.0,
            'false_alarm_cost': 1.0,
            'inspection_capacity_per_cycle': None,
            'selection_metric': 'expected_cost',
            'preferred_alert_mode': None,
        },
        'monitoring': {
            'drift': {
                'enabled': True,
                'warning_threshold': 0.05,
                'critical_threshold': 0.15,
                'min_windows': 30,
            }
        },
        'audit': {'enabled': True, 'max_records': 0, 'include_insights': 2},
    }
    return config, config_path


class TestPipelineAnalyses(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('TestPipelineAnalyses')
        self.logger.addHandler(logging.NullHandler())
        self.logger.setLevel(logging.WARNING)

    def test_analyses_are_written_and_consistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, config_path = write_project(tmpdir)
            results_dir = config['paths']['results_dir']
            pipeline = ANSRDTPipeline(config, config_path, self.logger)
            results = pipeline.run()

            policy = results['operational_policy']
            self.assertTrue(policy['enabled'])
            self.assertIn(policy['selected_mode'], ALERT_MODES)
            self.assertEqual(policy['selection_split'], 'validation')
            self.assertTrue(os.path.exists(
                os.path.join(results_dir, 'operational_policy.json')
            ))

            sensitivity = results['threshold_sensitivity']
            self.assertTrue(sensitivity['enabled'])
            self.assertEqual(sensitivity['candidate_count'], 5)
            self.assertTrue(os.path.exists(sensitivity['csv_path']))
            self.assertTrue(os.path.exists(sensitivity['json_path']))
            self.assertTrue(os.path.exists(sensitivity['plot_path']))

            drift = results['drift']
            self.assertTrue(drift['enabled'])
            self.assertIn(drift['level'], {'ok', 'warning', 'critical'})
            self.assertTrue(os.path.exists(
                os.path.join(results_dir, 'drift_report.json')
            ))
            self.assertTrue(os.path.exists(
                os.path.join(results_dir, 'drift_reference.json')
            ))

            audit = results['audit']
            self.assertTrue(audit['enabled'])
            self.assertEqual(
                audit['records'], results['splits']['test_windows']
            )
            self.assertEqual(
                audit['alert_mode'], policy['selected_mode']
            )
            journal = os.path.join(results_dir, 'decision_audit.jsonl')
            self.assertTrue(os.path.exists(journal))
            report = verify_audit(results_dir)
            self.assertTrue(report['ok'], report['issues'])
            self.assertEqual(
                report['decisions'], results['splits']['test_windows']
            )

            with open(os.path.join(
                    results_dir, 'cmapss_results.json')) as file:
                payload = json.load(file)
            self.assertIn('drift', payload)
            self.assertIn('decision_audit_path', payload['artifacts'])

    def test_analyses_are_off_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, config_path = write_project(tmpdir)
            config['threshold_sensitivity']['enabled'] = False
            config['operational_policy']['enabled'] = False
            config['monitoring']['drift']['enabled'] = False
            config['audit']['enabled'] = False
            results_dir = config['paths']['results_dir']
            pipeline = ANSRDTPipeline(config, config_path, self.logger)
            results = pipeline.run()
            self.assertEqual(results['operational_policy'], {})
            self.assertEqual(results['threshold_sensitivity'], {})
            self.assertEqual(results['drift'], {})
            self.assertEqual(results['audit'], {})
            self.assertFalse(os.path.exists(
                os.path.join(results_dir, 'decision_audit.jsonl')
            ))


if __name__ == '__main__':
    unittest.main()
