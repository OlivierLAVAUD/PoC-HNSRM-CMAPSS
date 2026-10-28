# tests/test_pipeline.py

import json
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data.cmapss_loader import CMAPSSLoader
from src.pipeline.pipeline import ANSRDTPipeline, validate_config
from src.reasoning.reasoning import PYSWIP_AVAILABLE
from src.reasoning.thresholds import derive_thresholds, update_rules_file


def _derive_training_symbolic_thresholds(data_file, rules_path, config):
    """Rewrite rules.pl so its facts match a synthetic training split."""
    loader = CMAPSSLoader(
        data_file,
        window_size=int(config['model']['window_size']),
        rul_threshold=int(config['training'].get('rul_threshold', 30)),
    )
    X, y = loader.load_data()
    validation_split = float(config['training'].get('validation_split', 0.2))
    held_out = max(
        0.0,
        1.0 - float(config['training'].get('train_fraction', 0.2))
        - validation_split,
    )
    loader.split_by_unit(
        X, y, validation_split=validation_split, test_split=held_out,
        random_state=int(config['training'].get('random_state', 42)),
    )
    train_mask = loader.last_split_masks['train']
    train_units = sorted({
        int(unit) for unit in np.asarray(loader.unit_ids)[train_mask].tolist()
    })
    update_rules_file(
        rules_path,
        derive_thresholds(loader._read_frame(data_file), train_units),
    )


class TestValidateConfig(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('TestValidateConfig')
        self.logger.addHandler(logging.NullHandler())

    @staticmethod
    def _prepare_project(tmpdir, with_rules=True):
        config_dir = os.path.join(tmpdir, 'configs')
        data_dir = os.path.join(tmpdir, 'data', 'raw', 'CMAPSS')
        reasoning_dir = os.path.join(tmpdir, 'src', 'reasoning')
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(reasoning_dir, exist_ok=True)
        open(os.path.join(data_dir, 'FD001_processed.parquet'), 'a').close()
        if with_rules:
            open(os.path.join(reasoning_dir, 'rules.pl'), 'a').close()
        return config_dir

    def test_valid_config_resolves_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = self._prepare_project(tmpdir)
            config = {'model': {}, 'training': {}, 'paths': {}}
            validate_config(config, self.logger, tmpdir, config_dir)

            self.assertTrue(config['paths']['data_file'].endswith('FD001_processed.parquet'))
            self.assertTrue(
                config['paths']['reasoning_rules_path'].endswith(
                    os.path.join('src', 'reasoning', 'rules.pl')
                )
            )
            self.assertTrue(os.path.isdir(config['paths']['results_dir']))

    def test_missing_key(self):
        with self.assertRaises(KeyError):
            validate_config({'model': {}, 'training': {}}, self.logger, '/tmp/project', '/tmp/project/configs')

    def test_missing_plot_config_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = self._prepare_project(tmpdir)
            config = {'model': {}, 'training': {}, 'paths': {'reasoning_rules_path': 'src/reasoning/rules.pl'}}
            validate_config(config, self.logger, tmpdir, config_dir)
            self.assertTrue(config['paths']['plot_config_path'].endswith('plot_config.yaml'))

    def test_optional_test_paths_resolve_to_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = self._prepare_project(tmpdir)
            config = {'model': {}, 'training': {}, 'paths': {}}
            validate_config(config, self.logger, tmpdir, config_dir)
            self.assertEqual(config['paths']['test_file'], '')
            self.assertEqual(config['paths']['rul_file'], '')

    def test_missing_required_rule_file_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = self._prepare_project(tmpdir, with_rules=False)
            config = {'model': {}, 'training': {}, 'paths': {'reasoning_rules_path': 'src/reasoning/rules.pl'}}
            with self.assertRaises(FileNotFoundError):
                validate_config(config, self.logger, tmpdir, config_dir)


class TestPipelineDataHandling(unittest.TestCase):
    """Regression tests for FD001 loading, splits and configuration wiring."""

    def setUp(self):
        self.logger = logging.getLogger('TestPipelineDataHandling')
        self.logger.addHandler(logging.NullHandler())
        self.logger.setLevel(logging.WARNING)

    @staticmethod
    def _write_project(root, *, non_finite_policy='zero', rul_threshold=30,
                       symbolic_enabled=True):
        config_dir = os.path.join(root, 'configs')
        data_dir = os.path.join(root, 'data', 'raw', 'CMAPSS')
        reasoning_dir = os.path.join(root, 'src', 'reasoning')
        for directory in (config_dir, data_dir, reasoning_dir):
            os.makedirs(directory, exist_ok=True)

        data_file = os.path.join(data_dir, 'FD001_processed.parquet')
        frame = pd.DataFrame({
            'unit_number': np.repeat(np.arange(1, 11), 6),
            'cycle': np.tile(np.arange(1, 7), 10),
            's2': 640.0,
            's4': 1400.0,
        })
        frame.loc[frame['cycle'] == 2, 's2'] = np.nan
        frame.to_parquet(data_file)

        rules_file = os.path.join(reasoning_dir, 'rules.pl')
        open(rules_file, 'w').close()
        config_path = os.path.join(config_dir, 'config.yaml')
        open(config_path, 'w').close()

        config = {
            'model': {
                'window_size': 3,
                'feature_names': ['s2', 's4'],
                'input_shape': [3, 2],
                'architecture': 'random_forest',
            },
            'training': {
                'validation_split': 0.2,
                'test_split': 0.1,
                'train_fraction': 0.2,
                'rul_threshold': rul_threshold,
                'random_state': 42,
            },
            'paths': {
                'data_file': data_file,
                'test_file': '',
                'rul_file': '',
                'results_dir': os.path.join(root, 'results'),
                'plot_config_path': '',
                'reasoning_rules_path': rules_file,
            },
            'symbolic_reasoning': {'enabled': symbolic_enabled},
            'preprocessing': {'non_finite_policy': non_finite_policy},
        }
        return config, config_path

    def _build_pipeline(self, root, **kwargs):
        config, config_path = self._write_project(root, **kwargs)
        return ANSRDTPipeline(config, config_path, self.logger)

    def test_non_finite_values_are_replaced_when_policy_is_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            splits = pipeline._load_cmapss_splits()
            for name in ('X_train', 'X_val', 'X_test'):
                self.assertTrue(np.isfinite(splits[name]).all(), name)

    def test_non_finite_values_are_rejected_when_policy_is_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir, non_finite_policy='error')
            with self.assertRaises(ValueError):
                pipeline._load_cmapss_splits()

    def test_rul_threshold_is_forwarded_to_the_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir, rul_threshold=12)
            self.assertEqual(pipeline.data_loader.rul_threshold, 12)

    def test_fallback_split_uses_train_and_validation_fractions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir)
            splits = pipeline._load_cmapss_splits()
            self.assertEqual(len(np.unique(splits['train_units'])), 2)
            self.assertEqual(len(np.unique(splits['val_units'])), 2)
            self.assertEqual(len(np.unique(splits['test_units'])), 6)

    def test_disabled_symbolic_reasoning_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = self._build_pipeline(tmpdir, symbolic_enabled=False)
            self.assertFalse(pipeline.symbolic_enabled)
            with self.assertRaises(RuntimeError):
                pipeline._initialise_reasoner()
            self.assertEqual(
                pipeline._explain_representative_windows(
                    np.zeros((2, 3, 2)), np.array([0.1, 0.9])
                ),
                {},
            )
            matrix = pipeline._collect_fusion_signals(
                np.zeros((2, 3, 2)), np.array([0.1, 0.9]), None, 'test'
            )
            np.testing.assert_allclose(matrix[:, 0], [0.1, 0.9])
            np.testing.assert_allclose(matrix[:, 1], 0.0)
            np.testing.assert_allclose(matrix[:, 2], 0.0)
            np.testing.assert_allclose(matrix[:, 3], 0.0)


class TestLearnedRuleValidationOrder(unittest.TestCase):
    """Lock the contract: learned rules are validated before fusion."""

    def test_learned_rules_are_validated_before_fusion(self):
        order = []
        report = {
            'summary': {
                'rules_evaluated': 0, 'accepted': 0, 'rejected': 0,
            },
            'rules': [],
            'applied': True,
        }
        reasoner = MagicMock()
        reasoner.validate_learned_rules.side_effect = (
            lambda *args, **kwargs: order.append('validation') or report
        )
        metrics = {
            'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
            'confusion_matrix': [[1, 0], [0, 1]], 'roc_auc': 1.0,
            'average_precision': 1.0, 'support': 2, 'positive_rate': 0.5,
        }
        X = np.zeros((2, 3, 2), dtype=np.float32)
        y = np.array([0, 1])
        units = np.array([1, 1])
        cycles = np.array([1, 2])
        rul = np.array([1, 0])
        splits = {
            'X_train': X, 'X_val': X, 'X_test': X,
            'y_train': y, 'y_val': y, 'y_test': y,
            'train_units': units, 'train_cycles': cycles, 'train_rul': rul,
            'val_units': units, 'val_cycles': cycles, 'val_rul': rul,
            'test_units': units, 'test_cycles': cycles, 'test_rul': rul,
            'test_source': 'unit-test',
        }
        prepared = {
            'architecture': 'lightgbm',
            'scaler': None,
            'uses_temporal_features': False,
            'paths': {
                'feature_extractor': 'extractor.joblib',
                'feature_metadata': 'metadata.json',
                'rul_regressor': 'rul.joblib',
            },
            'rul_report': {'applied': False},
        }
        evaluation = {
            'model_path': 'model.joblib',
            'calibration': {'method': 'sigmoid'},
            'calibration_report': {'report_path': 'calibration.json'},
            'decision_threshold': {'threshold': 0.5},
            'decision_threshold_path': 'threshold.json',
            'validation_metrics': metrics,
            'test_metrics': metrics,
            'engine_metrics': {},
            'alerting': {},
            'val_scores': np.array([0.1, 0.9]),
            'test_scores': np.array([0.1, 0.9]),
            'threshold': 0.5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ANSRDTPipeline.__new__(ANSRDTPipeline)
            pipeline.config = {
                'model': {'architecture': 'lightgbm', 'random_state': 42},
                'paths': {
                    'results_dir': tmpdir,
                    'data_file': 'test.parquet',
                },
                'training': {'random_state': 42},
                'temporal': {'enabled': False},
                'hitl': {'rule_validation': {'enabled': True}},
                'symbolic_reasoning': {'enabled': True},
                'engine_metrics': {'enabled': True},
            }
            pipeline.logger = logging.getLogger(
                'TestLearnedRuleValidationOrder'
            )
            pipeline.logger.addHandler(logging.NullHandler())
            pipeline.feature_names = ['s2', 's4']
            pipeline.model = MagicMock(decision_threshold=0.5)
            pipeline.reasoner = reasoner
            pipeline.temporal_thresholds = {}
            pipeline.symbolic_enabled = True

            pipeline._load_cmapss_splits = MagicMock(return_value=splits)
            pipeline._prepare_model_data = MagicMock(return_value=prepared)
            pipeline._evaluate_architecture = MagicMock(
                return_value=evaluation
            )
            pipeline._apply_fusion = MagicMock(
                side_effect=lambda *args, **kwargs: (
                    order.append('fusion') or {
                        'enabled': False, 'applied': False,
                    }
                )
            )
            pipeline._explain_representative_windows = MagicMock(
                return_value={}
            )
            pipeline._build_hitl_queue = MagicMock(return_value={})

            pipeline._run_cmapss()

        self.assertEqual(order, ['validation', 'fusion'])


class TestHITLProbLogConflict(unittest.TestCase):
    """HITL classification must use the computed ProbLog risk."""

    def test_low_probability_with_problog_alarm_is_queued(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ANSRDTPipeline.__new__(ANSRDTPipeline)
            pipeline.config = {
                'paths': {'results_dir': tmpdir},
                'training': {'random_state': 42},
                'hitl': {
                    'enabled': True,
                    'center_on_threshold': True,
                    'uncertainty_margin': 0.1,
                    'max_cases': 10,
                    'sample_per_bucket': 10,
                },
                'problog': {
                    'learned_rule_weight': 0.15,
                    'max_active_learned_rules': 3,
                },
            }
            pipeline.logger = logging.getLogger('TestHITLProbLogConflict')
            pipeline.logger.addHandler(logging.NullHandler())
            pipeline.feature_names = ['s2', 's4']
            pipeline.model = MagicMock(decision_threshold=0.5)
            pipeline.symbolic_enabled = True
            pipeline.reasoner = MagicMock()
            pipeline.reasoner.reason.return_value = []
            pipeline.reasoner.get_active_learned_rule_names.return_value = []
            pipeline.data_loader = MagicMock()
            pipeline.data_loader.unit_ids = None
            pipeline.data_loader.window_end_cycles = None

            X_raw = np.zeros((2, 3, 2), dtype=np.float32)
            scores = np.array([0.1, 0.5])

            with patch(
                'src.pipeline.pipeline.evaluate_state',
                return_value={'failure_risk': 0.9},
            ):
                summary = pipeline._build_hitl_queue(
                    X_raw, scores, decision_threshold=0.5
                )

            self.assertEqual(
                summary['reasons'].get('problog_probability_conflict_low'), 1
            )
            with open(os.path.join(tmpdir, 'hitl_queue.json')) as file:
                payload = json.load(file)
            cases = {case['reason']: case for case in payload['cases']}
            self.assertAlmostEqual(
                cases['problog_probability_conflict_low'][
                    'problog_failure_risk'
                ],
                0.9,
            )


@unittest.skipUnless(PYSWIP_AVAILABLE, "SWI-Prolog/PySWIP not available")
class TestBroadLearnedRuleIntegration(unittest.TestCase):
    """End-to-end: a broad HITL rule is pruned before fusion is fitted."""

    SENSORS = [
        's2', 's3', 's4', 's7', 's8', 's9', 's11',
        's12', 's13', 's14', 's15', 's17', 's20', 's21',
    ]

    def test_broad_rule_is_rejected_before_fusion(self):
        logger = logging.getLogger('TestBroadLearnedRuleIntegration')
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, 'data', 'raw', 'CMAPSS')
            config_dir = os.path.join(tmpdir, 'configs')
            reasoning_dir = os.path.join(tmpdir, 'src', 'reasoning')
            results_dir = os.path.join(tmpdir, 'results')
            for directory in (data_dir, config_dir, reasoning_dir):
                os.makedirs(directory, exist_ok=True)

            data_file = os.path.join(data_dir, 'FD001_processed.parquet')
            rows = []
            for unit in range(1, 7):
                for cycle in range(1, 21):
                    row = {'unit_number': unit, 'cycle': cycle}
                    for index, name in enumerate(self.SENSORS):
                        row[name] = 100.0 + unit + 0.1 * cycle + index
                    rows.append(row)
            pd.DataFrame(rows).to_parquet(data_file)

            rules_path = os.path.join(reasoning_dir, 'rules.pl')
            shutil.copy('src/reasoning/rules.pl', rules_path)
            with open(os.path.join(reasoning_dir, 'rules_learned.pl'), 'w') as f:
                f.write(
                    '% test sidecar\n'
                    ':- discontiguous(neural_rule/0).\n\n'
                    'hitl_rule_bad :- current_sensor_value(s2, _).\n'
                )

            config = {
                'model': {
                    'window_size': 5,
                    'feature_names': self.SENSORS,
                    'input_shape': [5, 14],
                    'architecture': 'lightgbm',
                    'random_state': 42,
                    'lightgbm': {
                        'n_estimators': 20, 'learning_rate': 0.2,
                        'num_leaves': 7, 'max_depth': -1,
                        'min_child_samples': 5, 'subsample': 1.0,
                        'colsample_bytree': 1.0, 'reg_lambda': 0.0,
                        'class_weight': 'balanced',
                    },
                },
                'training': {
                    'train_fraction': 0.2,
                    'validation_split': 0.2,
                    'test_split': 0.1,
                    'rul_threshold': 8,
                    'random_state': 42,
                },
                'paths': {
                    'data_file': data_file,
                    'test_file': '',
                    'rul_file': '',
                    'results_dir': results_dir,
                    'plot_config_path': '',
                    'reasoning_rules_path': rules_path,
                },
                'calibration': {
                    'enabled': True, 'method': 'sigmoid',
                    'cv_folds': 2, 'n_bins': 3,
                },
                'thresholding': {
                    'strategy': 'recall_at_precision',
                    'min_precision': 0.5, 'beta': 2.0, 'fallback': 0.5,
                },
                'symbolic_reasoning': {'enabled': True},
                'temporal': {'enabled': False},
                'features': {
                    'temporal': {
                        'enabled': True,
                        'include_raw_window': False,
                        'moving_average_cycles': 3,
                        'percentile_low': 25.0,
                        'percentile_high': 75.0,
                        'apply_to_architectures': ['lightgbm'],
                    },
                },
                'rul_regression': {'enabled': False},
                'fusion': {
                    'enabled': True,
                    'apply_to_architectures': ['lightgbm'],
                    'threshold_strategy': 'recall_at_precision',
                    'min_precision': 0.5, 'beta': 2.0, 'fallback': 0.5,
                    'features': [
                        'neural_probability', 'symbolic_rule_score',
                        'problog_failure_risk', 'predicted_rul',
                    ],
                },
                'hitl': {
                    'enabled': True,
                    'center_on_threshold': True,
                    'uncertainty_margin': 0.1,
                    'max_cases': 5,
                    'sample_per_bucket': 5,
                    'rule_validation': {
                        'enabled': True,
                        'min_precision': 0.9,
                        'min_support': 2,
                        'max_false_positive_rate': 0.05,
                        'apply': True,
                    },
                },
                'problog': {
                    'learned_rule_weight': 0.15,
                    'max_active_learned_rules': 3,
                },
                'alerting': {'enabled': False},
                'engine_metrics': {'enabled': False},
                'preprocessing': {'non_finite_policy': 'zero'},
            }
            config_path = os.path.join(config_dir, 'config.yaml')
            open(config_path, 'w').close()

            _derive_training_symbolic_thresholds(
                data_file, rules_path, config
            )
            pipeline = ANSRDTPipeline(config, config_path, logger)
            observed = {}
            original_fusion = pipeline._apply_fusion

            def spy_apply_fusion(*args, **kwargs):
                observed['bad_rule_present'] = any(
                    'hitl_rule_bad' in rule
                    for rule in pipeline.reasoner.learned_rules
                )
                observed['report_written'] = os.path.exists(
                    os.path.join(results_dir, 'hitl_rule_validation.json')
                )
                return original_fusion(*args, **kwargs)

            pipeline._apply_fusion = spy_apply_fusion
            fake_problog = {
                'failure_risk': 0.4,
                'hpc_degradation': 0.2,
                'active_conditions': [],
                'learned_rules_active': [],
            }
            with patch(
                'src.pipeline.pipeline.evaluate_state',
                return_value=fake_problog,
            ):
                summary = pipeline.run()

            self.assertFalse(observed.get('bad_rule_present'), observed)
            self.assertTrue(observed.get('report_written'), observed)
            self.assertEqual(
                summary['hitl_rule_validation']['summary']['rejected'], 1
            )
            self.assertTrue(summary['probability_fusion']['applied'])
            with open(
                os.path.join(reasoning_dir, 'rules_learned.pl')
            ) as file:
                self.assertNotIn('hitl_rule_bad', file.read())


class TestCompareModelsFeatureCount(unittest.TestCase):
    """Lock feature_count to the flattened model input dimension."""

    SENSORS = TestBroadLearnedRuleIntegration.SENSORS

    def test_feature_count_matches_model_inputs(self):
        logger = logging.getLogger('TestCompareModelsFeatureCount')
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, 'data', 'raw', 'CMAPSS')
            config_dir = os.path.join(tmpdir, 'configs')
            reasoning_dir = os.path.join(tmpdir, 'src', 'reasoning')
            for directory in (data_dir, config_dir, reasoning_dir):
                os.makedirs(directory, exist_ok=True)

            data_file = os.path.join(data_dir, 'FD001_processed.parquet')
            rows = []
            for unit in range(1, 7):
                for cycle in range(1, 21):
                    row = {'unit_number': unit, 'cycle': cycle}
                    for index, name in enumerate(self.SENSORS):
                        row[name] = 100.0 + unit + 0.1 * cycle + index
                    rows.append(row)
            pd.DataFrame(rows).to_parquet(data_file)

            rules_path = os.path.join(reasoning_dir, 'rules.pl')
            shutil.copy('src/reasoning/rules.pl', rules_path)
            open(
                os.path.join(reasoning_dir, 'rules_learned.pl'), 'w'
            ).close()

            config = {
                'model': {
                    'window_size': 5,
                    'feature_names': self.SENSORS,
                    'input_shape': [5, 14],
                    'architecture': 'random_forest',
                    'random_state': 42,
                    'light_classifier': {
                        'n_estimators': 20, 'max_depth': 3,
                        'class_weight': 'balanced',
                    },
                    'lightgbm': {
                        'n_estimators': 20, 'learning_rate': 0.2,
                        'num_leaves': 7, 'max_depth': -1,
                        'min_child_samples': 5, 'subsample': 1.0,
                        'colsample_bytree': 1.0, 'reg_lambda': 0.0,
                        'class_weight': 'balanced',
                    },
                },
                'training': {
                    'train_fraction': 0.2,
                    'validation_split': 0.2,
                    'test_split': 0.1,
                    'rul_threshold': 8,
                    'random_state': 42,
                },
                'paths': {
                    'data_file': data_file,
                    'test_file': '',
                    'rul_file': '',
                    'results_dir': os.path.join(tmpdir, 'results'),
                    'plot_config_path': '',
                    'reasoning_rules_path': rules_path,
                },
                'calibration': {'enabled': False},
                'thresholding': {
                    'strategy': 'f1', 'min_precision': 0.5,
                    'beta': 2.0, 'fallback': 0.5,
                },
                'symbolic_reasoning': {'enabled': False},
                'temporal': {'enabled': False},
                'features': {
                    'temporal': {
                        'enabled': True,
                        'include_raw_window': False,
                        'moving_average_cycles': 3,
                        'percentile_low': 25.0,
                        'percentile_high': 75.0,
                        'apply_to_architectures': ['lightgbm'],
                    },
                },
                'rul_regression': {'enabled': False},
                'fusion': {'enabled': False},
                'hitl': {'enabled': False},
                'alerting': {'enabled': False},
                'engine_metrics': {'enabled': False},
                'experiments': {
                    'compare_models': {
                        'enabled': True,
                        'models': ['random_forest', 'lightgbm'],
                        'selection_metric': 'fbeta',
                        'beta': 2.0,
                        'selection_split': 'validation',
                    },
                },
                'preprocessing': {'non_finite_policy': 'zero'},
            }
            config_path = os.path.join(config_dir, 'config.yaml')
            open(config_path, 'w').close()

            pipeline = ANSRDTPipeline(config, config_path, logger)
            with patch(
                'src.pipeline.pipeline.save_reliability_plot',
                side_effect=lambda metrics, path, title: path,
            ), patch(
                'src.pipeline.pipeline.save_precision_recall_plot',
                side_effect=lambda *args, **kwargs: 'plot.png',
            ):
                report = pipeline.run(compare_models=True)

            self.assertEqual(
                report['models']['random_forest']['feature_count'],
                5 * len(self.SENSORS),
            )
            self.assertEqual(
                report['models']['lightgbm']['feature_count'],
                13 * len(self.SENSORS),
            )
            self.assertEqual(report['selection_split'], 'validation')
            self.assertTrue(os.path.exists(report['report_path']))


if __name__ == '__main__':
    unittest.main()
