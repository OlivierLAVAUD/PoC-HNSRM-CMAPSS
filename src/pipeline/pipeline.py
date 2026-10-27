# src/pipeline/pipeline.py

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.cmapss_loader import CMAPSSLoader
from src.data.data_loader import DataLoader, create_data_loader
from src.data.data_processing import map_labels
from src.evaluation.calibration import calibration_metrics, save_reliability_plot
from src.evaluation.engine_metrics import (
    compute_engine_metrics,
    export_engine_metrics_csv,
)
from src.evaluation.thresholding import (
    save_precision_recall_plot,
    select_threshold,
)
from src.hitl.queue import HITLQueue, REASON_PRIORITY
from src.models.light_classifier import LightClassifier
from src.preprocessing.preprocessing import preprocess_sequences
from src.reasoning.prob_query import RULES_PATH as PROB_RULES_PATH, evaluate_state
from src.reasoning.reasoning import SymbolicReasoner
from src.reasoning.temporal import compute_temporal_thresholds
from src.utils.model_utils import save_scaler


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy scalars, arrays and datetimes."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def _is_number(value: Any) -> bool:
    """Return True for real scalar configuration values (not booleans)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_range(value: Any, name: str, low: float, high: float,
                   allow_low: bool = True, allow_high: bool = True) -> None:
    """Raise a clear error when a numeric setting is out of range."""
    if not _is_number(value):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}.")
    if (value < low) or (value == low and not allow_low):
        raise ValueError(f"{name} must be >= {low}, got {value}.")
    if (value > high) or (value == high and not allow_high):
        raise ValueError(f"{name} must be <= {high}, got {value}.")


def _validate_model_and_training(config: dict) -> None:
    """Validate model coherence and the main training/calibration settings."""
    model = config['model']
    training = config['training']

    if model:
        window_size = model.get('window_size')
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
            raise ValueError("model.window_size must be a positive integer.")

        feature_names = model.get('feature_names')
        if (not isinstance(feature_names, list) or not feature_names
                or not all(isinstance(name, str) and name for name in feature_names)):
            raise ValueError("model.feature_names must be a non-empty list of strings.")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("model.feature_names must not contain duplicates.")

        input_shape = model.get('input_shape')
        expected_shape = [window_size, len(feature_names)]
        if (not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2
                or list(input_shape) != expected_shape):
            raise ValueError(
                f"model.input_shape must be {expected_shape}, got {input_shape!r}."
            )

        architecture = model.get('architecture', 'random_forest')
        if architecture not in {'random_forest', 'light_classifier', 'cnn_lstm'}:
            raise ValueError(f"Unsupported model.architecture: {architecture!r}.")

    if training:
        if 'validation_split' in training:
            _require_range(training['validation_split'], 'training.validation_split',
                           0.0, 1.0, allow_low=False, allow_high=False)
        if 'test_split' in training:
            _require_range(training['test_split'], 'training.test_split',
                           0.0, 1.0, allow_low=True, allow_high=False)
        if 'train_fraction' in training:
            _require_range(training['train_fraction'], 'training.train_fraction',
                           0.0, 1.0, allow_low=False, allow_high=True)
        if ('train_fraction' in training and 'validation_split' in training
                and training['train_fraction'] + training['validation_split'] > 1.0):
            raise ValueError(
                "training.train_fraction + training.validation_split must be <= 1.0."
            )
        if 'rul_threshold' in training:
            _require_range(training['rul_threshold'], 'training.rul_threshold',
                           0.0, 100000.0, allow_low=False)

    calibration = config.get('calibration', {})
    if calibration.get('enabled', False):
        if calibration.get('method', 'sigmoid') not in {'sigmoid', 'isotonic'}:
            raise ValueError("calibration.method must be 'sigmoid' or 'isotonic'.")
        if 'cv_folds' in calibration and (
                not isinstance(calibration['cv_folds'], int)
                or calibration['cv_folds'] < 2):
            raise ValueError("calibration.cv_folds must be an integer >= 2.")

    thresholding = config.get('thresholding', {})
    if thresholding:
        if thresholding.get('strategy', 'f1') not in {
                'f1', 'recall_at_precision', 'fbeta'}:
            raise ValueError(
                "thresholding.strategy must be 'f1', 'recall_at_precision' or 'fbeta'."
            )
        if 'beta' in thresholding:
            _require_range(thresholding['beta'], 'thresholding.beta', 0.0, 100.0,
                           allow_low=False)
        for key in ('min_precision', 'fallback'):
            if key in thresholding:
                _require_range(thresholding[key], f'thresholding.{key}', 0.0, 1.0)

    problog = config.get('problog', {})
    if 'learned_rule_weight' in problog:
        _require_range(problog['learned_rule_weight'],
                       'problog.learned_rule_weight', 0.0, 1.0)
    if 'max_active_learned_rules' in problog and (
            not isinstance(problog['max_active_learned_rules'], int)
            or problog['max_active_learned_rules'] < 0):
        raise ValueError("problog.max_active_learned_rules must be an integer >= 0.")

    temporal = config.get('temporal', {})
    if 'percentile' in temporal:
        _require_range(temporal['percentile'], 'temporal.percentile', 0.0, 100.0)
    if 'moving_average_cycles' in temporal and (
            not isinstance(temporal['moving_average_cycles'], int)
            or temporal['moving_average_cycles'] < 1):
        raise ValueError("temporal.moving_average_cycles must be an integer >= 1.")

    preprocessing = config.get('preprocessing', {})
    if preprocessing.get('non_finite_policy', 'zero') not in {
            'zero', 'replace_zero', 'error'}:
        raise ValueError(
            "preprocessing.non_finite_policy must be 'zero', 'replace_zero' or 'error'."
        )


def validate_config(config: dict, logger: logging.Logger, project_root: str, config_dir: str):
    """Validate configuration and resolve paths in place."""
    try:
        required_keys = ['model', 'training', 'paths']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                raise KeyError(f"Missing required configuration key: {key}")

        _validate_model_and_training(config)

        required_paths_info = {
            'data_file': {
                'relative_to': project_root,
                'default': 'data/raw/CMAPSS/FD001_processed.parquet',
            },
            'test_file': {'relative_to': project_root, 'default': '', 'optional': True},
            'rul_file': {'relative_to': project_root, 'default': '', 'optional': True},
            'results_dir': {'relative_to': project_root, 'default': 'results'},
            'plot_config_path': {
                'relative_to': config_dir,
                'default': 'plot_config.yaml',
                'optional': True,
            },
            'reasoning_rules_path': {
                'relative_to': project_root,
                'default': 'src/reasoning/rules.pl',
            },
        }

        for key, path_info in required_paths_info.items():
            relative_path = config['paths'].get(key, path_info['default'])
            if not relative_path:
                config['paths'][key] = ''
                continue

            full_path = os.path.normpath(os.path.join(path_info['relative_to'], relative_path))
            config['paths'][key] = full_path
            logger.debug(f"Validated and resolved path for '{key}': {full_path}")

            if key.endswith('_dir'):
                os.makedirs(full_path, exist_ok=True)
            elif not os.path.exists(full_path):
                if path_info.get('optional', False):
                    logger.warning(f"Optional file '{key}' not found at: {full_path}")
                else:
                    raise FileNotFoundError(f"Required file '{key}' not found at: {full_path}")

        logger.info("Configuration paths validated and resolved.")
    except Exception as e:
        logger.exception(f"Configuration validation failed: {e}")
        raise


class ANSRDTPipeline:
    """Train the lightweight neural component and evaluate the symbolic bridge."""

    NumpyEncoder = NumpyEncoder

    def __init__(self, config: dict, config_path: str, logger: logging.Logger):
        """Initialize the pipeline from a validated configuration."""
        self.config = config
        self.logger = logger
        self.config_path = os.path.abspath(config_path)
        self.config_dir = os.path.dirname(self.config_path)
        self.project_root = os.path.dirname(self.config_dir)

        validate_config(self.config, self.logger, self.project_root, self.config_dir)

        self.feature_names = self.config['model'].get('feature_names', [])
        if not self.feature_names:
            raise ValueError("feature_names missing in model configuration.")

        self.data_loader = create_data_loader(
            self.config['paths']['data_file'],
            window_size=self.config['model']['window_size'],
            feature_names=self.feature_names,
            test_file=self.config['paths'].get('test_file'),
            rul_file=self.config['paths'].get('rul_file'),
            logger=self.logger,
        )
        self.is_cmapss = isinstance(self.data_loader, CMAPSSLoader)

        self.model: Optional[LightClassifier] = None
        self.reasoner: Optional[SymbolicReasoner] = None
        self.temporal_thresholds: Dict[str, Any] = {}

    def _build_classifier(self) -> LightClassifier:
        """Create the configured lightweight classifier."""
        model_config = self.config.get('model', {})
        architecture = model_config.get('architecture', 'random_forest')
        if architecture in ('cnn_lstm', 'light_classifier'):
            self.logger.warning(
                "Architecture alias '%s' is deprecated; using 'random_forest'.",
                architecture,
            )
            architecture = 'random_forest'
        options = dict(model_config.get('light_classifier', {}))
        return LightClassifier(
            architecture=architecture,
            random_state=int(model_config.get('random_state', 42)),
            logger=self.logger,
            **options,
        )

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_scores: np.ndarray) -> Dict[str, Any]:
        """Compute classification metrics and a confusion matrix."""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'support': int(len(y_true)),
            'positive_rate': float(np.mean(y_true)),
        }
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
            metrics['average_precision'] = float(average_precision_score(y_true, y_scores))
        except ValueError:
            metrics['roc_auc'] = None
            metrics['average_precision'] = None
        return metrics

    @staticmethod
    def _split_scaled(scaler, X: np.ndarray) -> np.ndarray:
        """Apply a fitted scaler to a 3D window array."""
        return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    def _calibrate_model(self, X_train_scaled: np.ndarray,
                         y_train: np.ndarray) -> Dict[str, Any]:
        """Calibrate the classifier on training data only."""
        calibration_config = self.config.get('calibration', {})
        if not calibration_config.get('enabled', True):
            self.logger.info("Probability calibration disabled by configuration.")
            return {'enabled': False, 'applied': False}

        method = calibration_config.get('method', 'sigmoid')
        cv_folds = int(calibration_config.get('cv_folds', 5))
        self.model.calibrate(X_train_scaled, y_train, method=method, cv_folds=cv_folds)
        return {
            'enabled': True,
            'applied': bool(getattr(self.model, 'is_calibrated', False)),
            'method': getattr(self.model, 'calibration_method', None),
            'cv_folds': getattr(self.model, 'calibration_cv_folds', None),
        }

    def _select_decision_threshold(self, y_val: np.ndarray,
                                   val_scores: np.ndarray) -> Dict[str, Any]:
        """Select a decision threshold on validation data only."""
        threshold_config = self.config.get('thresholding', {})
        selection = select_threshold(
            y_val, val_scores,
            strategy=threshold_config.get('strategy', 'fbeta'),
            min_precision=float(threshold_config.get('min_precision', 0.7)),
            beta=float(threshold_config.get('beta', 2.0)),
            fallback=float(threshold_config.get('fallback', 0.5)),
        )
        self.model.set_threshold(selection['threshold'])
        return selection

    def _calibration_report(self, y_val: np.ndarray, val_scores: np.ndarray,
                            y_test: np.ndarray, test_scores: np.ndarray,
                            results_dir: str) -> Dict[str, Any]:
        """Compute calibration diagnostics and persist curves."""
        n_bins = int(self.config.get('calibration', {}).get('n_bins', 10))
        report = {
            'validation': calibration_metrics(y_val, val_scores, n_bins=n_bins),
            'test': calibration_metrics(y_test, test_scores, n_bins=n_bins),
        }

        visualization_dir = os.path.join(results_dir, 'visualization')
        os.makedirs(visualization_dir, exist_ok=True)
        report['validation']['reliability_plot'] = save_reliability_plot(
            report['validation'],
            os.path.join(visualization_dir, 'reliability_validation.png'),
            title='Validation reliability curve',
        )
        report['test']['reliability_plot'] = save_reliability_plot(
            report['test'],
            os.path.join(visualization_dir, 'reliability_test.png'),
            title='Test reliability curve',
        )
        report_path = os.path.join(results_dir, 'calibration_report.json')
        with open(report_path, 'w') as file:
            json.dump(report, file, indent=2, cls=NumpyEncoder)
        report['report_path'] = report_path
        return report

    def _initialise_reasoner(self) -> SymbolicReasoner:
        """Build the Prolog bridge for representative test windows."""
        if self.reasoner is None:
            self.reasoner = SymbolicReasoner(
                rules_path=self.config['paths']['reasoning_rules_path'],
                input_shape=(self.config['model']['window_size'], len(self.feature_names)),
                model=self.model,
                logger=self.logger,
            )
            if self.temporal_thresholds:
                self.reasoner.set_temporal_thresholds(self.temporal_thresholds)
        return self.reasoner

    def _explain_representative_windows(self, X_raw: np.ndarray,
                                        scores: np.ndarray) -> Dict[str, Any]:
        """Reason over normal, borderline and critical raw test windows."""
        if len(X_raw) == 0:
            return {}

        reasoner = self._initialise_reasoner()
        with open(PROB_RULES_PATH, 'r') as file:
            problog_program = file.read()

        decision_threshold = float(getattr(self.model, 'decision_threshold', 0.5))
        normal_idx = int(np.argmin(scores))
        critical_idx = int(np.argmax(scores))
        degraded_idx = int(np.argmin(np.abs(scores - decision_threshold)))
        indices = {'normal': normal_idx, 'degraded': degraded_idx, 'critical': critical_idx}

        explanations = {}
        for label, idx in indices.items():
            sensor_dict = {
                name: float(X_raw[idx, -1, feature_idx])
                for feature_idx, name in enumerate(self.feature_names)
            }
            insights = reasoner.reason(sensor_dict, window=X_raw[idx])
            problog_config = self.config.get('problog', {})
            problog = evaluate_state(
                problog_program, sensor_dict,
                learned_rules=reasoner.get_active_learned_rule_names(),
                learned_rule_weight=float(problog_config.get('learned_rule_weight', 0.15)),
                max_active_learned_rules=int(
                    problog_config.get('max_active_learned_rules', 3)
                ),
            )
            explanations[label] = {
                'window_index': idx,
                'neural_probability': float(scores[idx]),
                'problog_failure_risk': float(problog['failure_risk']),
                'problog_hpc_degradation': float(problog['hpc_degradation']),
                'active_symbolic_conditions': problog['active_conditions'],
                'active_learned_rules': problog['learned_rules_active'],
                'insights': insights,
                'sensor_values': sensor_dict,
            }
        return explanations

    def _build_hitl_queue(self, X_raw: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Queue uncertain and rule/probability-conflicting test windows."""
        hitl_config = self.config.get('hitl', {})
        if not hitl_config.get('enabled', True) or len(X_raw) == 0:
            return {}

        decision_threshold = float(
            getattr(self.model, 'decision_threshold',
                    self.config.get('thresholding', {}).get('fallback', 0.5))
        )
        if hitl_config.get('center_on_threshold', True):
            margin = float(hitl_config.get('uncertainty_margin', 0.10))
            uncertain_low = max(0.0, decision_threshold - margin)
            uncertain_high = min(1.0, decision_threshold + margin)
        else:
            uncertain_low = float(hitl_config.get('uncertain_low', 0.4))
            uncertain_high = float(hitl_config.get('uncertain_high', 0.6))
        max_cases = int(hitl_config.get('max_cases', 200))
        sample_per_bucket = max(1, int(hitl_config.get('sample_per_bucket', 300)))
        random_state = int(self.config['training'].get('random_state', 42))

        scores = np.asarray(scores, dtype=float)
        low_mask = scores < uncertain_low
        high_mask = scores > uncertain_high
        uncertain_mask = ~(low_mask | high_mask)

        rng = np.random.default_rng(random_state)
        buckets = []
        for mask in (uncertain_mask, high_mask, low_mask):
            indices = np.flatnonzero(mask)
            if len(indices) > sample_per_bucket:
                indices = rng.choice(indices, size=sample_per_bucket, replace=False)
            if len(indices):
                buckets.append(indices)
        candidates = (np.unique(np.concatenate(buckets)) if buckets
                      else np.asarray([], dtype=int))

        reasoner = self._initialise_reasoner()
        records = []
        for raw_index in candidates:
            index = int(raw_index)
            sensor_dict = {
                name: float(X_raw[index, -1, feature_idx])
                for feature_idx, name in enumerate(self.feature_names)
            }
            reasoner.state_history = []
            insights = reasoner.reason(sensor_dict, window=X_raw[index])
            active_learned_rules = reasoner.get_active_learned_rule_names()
            reason = HITLQueue.classify_reason(
                float(scores[index]), insights, 0.0,
                uncertain_low, uncertain_high,
            )
            if reason is None:
                continue
            records.append({
                'priority': REASON_PRIORITY.get(reason, 99),
                'window_index': index,
                'neural_probability': float(scores[index]),
                'reason': reason,
                'insights': insights,
                'sensor_values': sensor_dict,
                'active_learned_rules': active_learned_rules,
            })

        records.sort(key=lambda item: (item['priority'], item['window_index']))
        with open(PROB_RULES_PATH, 'r') as file:
            problog_program = file.read()

        results_dir = self.config['paths']['results_dir']
        queue = HITLQueue(
            uncertain_low=uncertain_low,
            uncertain_high=uncertain_high,
            output_dir=results_dir,
            max_cases=max_cases,
            decision_threshold=decision_threshold,
        )
        unit_ids = getattr(self.data_loader, 'unit_ids', None)
        end_cycles = getattr(self.data_loader, 'window_end_cycles', None)

        def metadata(array, index):
            if array is not None and len(array) == len(scores):
                return int(array[index])
            return None

        problog_logger = logging.getLogger('problog')
        previous_problog_level = problog_logger.level
        problog_logger.setLevel(logging.WARNING)
        try:
            for record in records:
                if len(queue.cases) >= max_cases:
                    break
                index = record['window_index']
                problog_config = self.config.get('problog', {})
                problog = evaluate_state(
                    problog_program, record['sensor_values'],
                    learned_rules=record.get('active_learned_rules', []),
                    learned_rule_weight=float(
                        problog_config.get('learned_rule_weight', 0.15)
                    ),
                    max_active_learned_rules=int(
                        problog_config.get('max_active_learned_rules', 3)
                    ),
                )
                queue.add_case(
                    window_index=index,
                    neural_probability=record['neural_probability'],
                    problog_failure_risk=float(problog['failure_risk']),
                    reason=record['reason'],
                    insights=record['insights'],
                    sensor_values=record['sensor_values'],
                    unit=metadata(unit_ids, index),
                    cycle=metadata(end_cycles, index),
                )
        finally:
            problog_logger.setLevel(previous_problog_level)

        queue.export_csv()
        queue.save_json()
        summary = queue.summary()
        summary['candidates_evaluated'] = int(len(candidates))
        summary['eligible_records'] = len(records)
        self.logger.info(
            "HITL queue: %d cases from %d candidates (%d eligible).",
            summary['total_cases'], summary['candidates_evaluated'],
            summary['eligible_records'],
        )
        return summary

    def _run_cmapss(self) -> Dict[str, Any]:
        """Train and evaluate the FD001 model with an engine-level split."""
        self.logger.info("Starting C-MAPSS pipeline.")
        X, y = self.data_loader.load_data()
        train_units_all = np.asarray(self.data_loader.unit_ids)
        train_cycles_all = np.asarray(self.data_loader.window_end_cycles)
        train_rul_all = np.asarray(self.data_loader.window_rul)

        validation_split = float(self.config['training'].get('validation_split', 0.2))
        test_split = float(self.config['training'].get('test_split', 0.1))
        train_fraction = float(self.config['training'].get('train_fraction', 0.2))
        random_state = int(self.config['training'].get('random_state', 42))

        if self.config['paths'].get('test_file'):
            held_out_fraction = max(0.0, 1.0 - train_fraction - validation_split)
            X_train, X_val, _, y_train, y_val, _ = self.data_loader.split_by_unit(
                X, y, validation_split=validation_split,
                test_split=held_out_fraction, random_state=random_state,
            )
            val_mask = self.data_loader.last_split_masks['val']
            val_units = train_units_all[val_mask]
            val_cycles = train_cycles_all[val_mask]
            val_rul = train_rul_all[val_mask]
            X_test, y_test = self.data_loader.load_test_data()
            test_units = np.asarray(self.data_loader.unit_ids)
            test_cycles = np.asarray(self.data_loader.window_end_cycles)
            test_rul = np.asarray(self.data_loader.window_rul)
            test_source = self.config['paths']['test_file']
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_by_unit(
                X, y, validation_split=validation_split,
                test_split=test_split, random_state=random_state,
            )
            masks = self.data_loader.last_split_masks
            val_units = train_units_all[masks['val']]
            val_cycles = train_cycles_all[masks['val']]
            val_rul = train_rul_all[masks['val']]
            test_units = train_units_all[masks['test']]
            test_cycles = train_cycles_all[masks['test']]
            test_rul = train_rul_all[masks['test']]
            test_source = 'held-out units from training trajectories'

        self.logger.info(
            "Windows: train=%d val=%d test=%d | positives: train=%d val=%d test=%d",
            len(X_train), len(X_val), len(X_test),
            int(y_train.sum()), int(y_val.sum()), int(y_test.sum()),
        )

        scaler, X_train_scaled = preprocess_sequences(X_train)
        X_val_scaled = self._split_scaled(scaler, X_val)
        X_test_scaled = self._split_scaled(scaler, X_test)

        scaler_path = os.path.join(self.config['paths']['results_dir'], 'scaler.pkl')
        save_scaler(scaler, scaler_path, self.logger)

        self.model = self._build_classifier()
        self.model.fit(X_train_scaled, y_train)
        calibration_info = self._calibrate_model(X_train_scaled, y_train)

        val_scores = self.model.positive_proba(X_val_scaled)
        threshold_info = self._select_decision_threshold(y_val, val_scores)
        model_path = os.path.join(
            self.config['paths']['results_dir'], 'models', 'light_classifier.joblib'
        )
        self.model.save(model_path)

        val_pred = (val_scores >= self.model.decision_threshold).astype(int)
        test_scores = self.model.positive_proba(X_test_scaled)
        test_pred = (test_scores >= self.model.decision_threshold).astype(int)

        validation_metrics = self._compute_metrics(y_val, val_pred, val_scores)
        test_metrics = self._compute_metrics(y_test, test_pred, test_scores)
        calibration_report = self._calibration_report(
            y_val, val_scores, y_test, test_scores,
            self.config['paths']['results_dir'],
        )
        threshold_report = {
            key: value for key, value in threshold_info.items()
            if key != 'precision_recall_curve'
        }
        threshold_report['precision_recall_plot'] = save_precision_recall_plot(
            threshold_info.get('precision_recall_curve', {}),
            os.path.join(
                self.config['paths']['results_dir'], 'visualization',
                'precision_recall_curve.png',
            ),
            threshold=self.model.decision_threshold,
        )
        threshold_path = os.path.join(
            self.config['paths']['results_dir'], 'decision_threshold.json'
        )
        with open(threshold_path, 'w') as file:
            json.dump(threshold_info, file, indent=2, cls=NumpyEncoder)
        self.logger.info("Validation metrics: %s", validation_metrics)
        self.logger.info("Test metrics: %s", test_metrics)

        engine_metrics = {}
        if self.config.get('engine_metrics', {}).get('enabled', True):
            val_engine = compute_engine_metrics(
                val_units, val_cycles, val_rul, y_val, val_pred
            )
            test_engine = compute_engine_metrics(
                test_units, test_cycles, test_rul, y_test, test_pred
            )
            val_engine_path = export_engine_metrics_csv(
                val_engine,
                os.path.join(self.config['paths']['results_dir'],
                             'engine_metrics_validation.csv'),
            )
            test_engine_path = export_engine_metrics_csv(
                test_engine,
                os.path.join(self.config['paths']['results_dir'],
                             'engine_metrics_test.csv'),
            )
            engine_metrics = {
                'validation': {**val_engine, 'csv_path': val_engine_path},
                'test': {**test_engine, 'csv_path': test_engine_path},
            }
            self.logger.info(
                "Engine metrics (test): %s", test_engine['summary']
            )

        temporal_config = self.config.get('temporal', {})
        temporal_path = None
        if temporal_config.get('enabled', True):
            reasoner = self._initialise_reasoner()
            sensor_thresholds = reasoner.get_sensor_thresholds()
            self.temporal_thresholds = compute_temporal_thresholds(
                X_train, self.feature_names,
                high_thresholds=sensor_thresholds.get('high', {}),
                low_thresholds=sensor_thresholds.get('low', {}),
                percentile=float(temporal_config.get('percentile', 75)),
                moving_average_cycles=int(
                    temporal_config.get('moving_average_cycles', 5)
                ),
            )
            reasoner.set_temporal_thresholds(self.temporal_thresholds)
            temporal_path = os.path.join(
                self.config['paths']['results_dir'], 'temporal_thresholds.json'
            )
            with open(temporal_path, 'w') as file:
                json.dump(self.temporal_thresholds, file, indent=2, cls=NumpyEncoder)
            self.logger.info(
                "Temporal thresholds saved to %s (training windows only).",
                temporal_path,
            )

        rule_validation = None
        rule_validation_path = None
        rule_validation_config = self.config.get('hitl', {}).get(
            'rule_validation', {}
        )
        if rule_validation_config.get('enabled', True):
            reasoner = self._initialise_reasoner()
            rule_validation = reasoner.validate_learned_rules(
                X_val, self.feature_names, y_val,
                min_precision=float(
                    rule_validation_config.get('min_precision', 0.9)
                ),
                min_support=int(
                    rule_validation_config.get('min_support', 5)
                ),
                max_false_positive_rate=float(
                    rule_validation_config.get('max_false_positive_rate', 0.05)
                ),
                apply=bool(rule_validation_config.get('apply', True)),
            )
            rule_validation_path = os.path.join(
                self.config['paths']['results_dir'],
                'hitl_rule_validation.json',
            )
            with open(rule_validation_path, 'w') as file:
                json.dump(rule_validation, file, indent=2, cls=NumpyEncoder)
            self.logger.info(
                "Learned-rule validation: %s", rule_validation['summary']
            )

        explanations = self._explain_representative_windows(X_test, test_scores)
        hitl_queue = self._build_hitl_queue(X_test, test_scores)
        results = {
            'dataset': 'NASA C-MAPSS FD001',
            'test_source': test_source,
            'data_file': self.config['paths']['data_file'],
            'splits': {
                'train_windows': int(len(X_train)),
                'validation_windows': int(len(X_val)),
                'test_windows': int(len(X_test)),
                'train_positive_rate': float(np.mean(y_train)),
                'validation_positive_rate': float(np.mean(y_val)),
                'test_positive_rate': float(np.mean(y_test)),
            },
            'metrics': {
                'validation': validation_metrics,
                'test': test_metrics,
            },
            'calibration': {
                **calibration_info,
                'metrics': calibration_report,
            },
            'decision_threshold': {
                **threshold_report,
                'path': threshold_path,
            },
            'engine_metrics': engine_metrics,
            'hitl_rule_validation': rule_validation,
            'representative_windows': explanations,
            'hitl_queue': hitl_queue,
            'artifacts': {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'problog_rules_path': PROB_RULES_PATH,
                'calibration_report_path': calibration_report.get('report_path'),
                'decision_threshold_path': threshold_path,
                'temporal_thresholds_path': temporal_path,
                'hitl_rule_validation_path': rule_validation_path,
                'hitl_queue_csv': hitl_queue.get('csv_path'),
                'hitl_queue_json': hitl_queue.get('json_path'),
            },
            'timestamp': datetime.now().isoformat(),
        }

        results_path = os.path.join(self.config['paths']['results_dir'], 'cmapss_results.json')
        with open(results_path, 'w') as file:
            json.dump(results, file, indent=2, cls=NumpyEncoder)
        self.logger.info("C-MAPSS results saved to %s", results_path)
        return results

    def _run_legacy(self) -> Dict[str, Any]:
        """Run the original synthetic-data path for backward compatibility."""
        self.logger.info("Starting legacy synthetic-data pipeline.")
        X, y = self.data_loader.load_data()
        X_seq, y_seq = self.data_loader.create_sequences(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
            X_seq, y_seq,
            validation_split=self.config['training']['validation_split'],
            test_split=self.config['training']['test_split'],
        )

        y_train = map_labels(y_train, self.logger)
        y_val = map_labels(y_val, self.logger)
        y_test = map_labels(y_test, self.logger)

        scaler, X_train_scaled = preprocess_sequences(X_train)
        X_val_scaled = self._split_scaled(scaler, X_val)
        X_test_scaled = self._split_scaled(scaler, X_test)

        scaler_path = os.path.join(self.config['paths']['results_dir'], 'scaler.pkl')
        save_scaler(scaler, scaler_path, self.logger)

        self.model = self._build_classifier()
        self.model.fit(X_train_scaled, y_train)
        model_path = os.path.join(
            self.config['paths']['results_dir'], 'models', 'light_classifier.joblib'
        )
        self.model.save(model_path)

        val_scores = self.model.positive_proba(X_val_scaled)
        test_scores = self.model.positive_proba(X_test_scaled)
        results = {
            'dataset': 'synthetic',
            'data_file': self.config['paths']['data_file'],
            'splits': {
                'train_windows': int(len(X_train)),
                'validation_windows': int(len(X_val)),
                'test_windows': int(len(X_test)),
            },
            'metrics': {
                'validation': self._compute_metrics(
                    y_val, (val_scores >= 0.5).astype(int), val_scores
                ),
                'test': self._compute_metrics(
                    y_test, (test_scores >= 0.5).astype(int), test_scores
                ),
            },
            'artifacts': {'model_path': model_path, 'scaler_path': scaler_path},
            'timestamp': datetime.now().isoformat(),
        }
        return results

    def run(self) -> Dict[str, Any]:
        """Execute the configured pipeline and return a run summary."""
        try:
            if self.is_cmapss:
                return self._run_cmapss()
            return self._run_legacy()
        except Exception as e:
            self.logger.exception(f"ANSR-DT pipeline failed: {e}")
            raise

    def extract_and_update_neural_rules(self, model, input_data: np.ndarray,
                                        y_pred: np.ndarray, threshold: float = 0.8) -> None:
        """Extract symbolic candidates from high-confidence neural predictions."""
        if self.reasoner is None:
            self.logger.error("Symbolic Reasoner not initialized. Cannot extract/update rules.")
            return

        try:
            potential_new_rules = self.reasoner.extract_rules_from_neural_model(
                input_data=input_data,
                feature_names=self.feature_names,
                threshold=threshold,
                model=model,
            )
            if potential_new_rules:
                reasoning_config = self.config.get('symbolic_reasoning', {})
                self.reasoner.update_rules(
                    potential_new_rules,
                    min_confidence=reasoning_config.get('min_confidence', 0.7),
                    max_learned_rules=reasoning_config.get('max_rules', 100),
                    pruning_strategy=reasoning_config.get('pruning_strategy', 'confidence'),
                )
                stats = self.reasoner.get_rule_statistics()
                summary = {
                    'reasoner_statistics': stats,
                    'learned_rules_in_reasoner': self.reasoner.learned_rules,
                    'timestamp': datetime.now().isoformat(),
                }
                summary_path = os.path.join(
                    self.config['paths']['results_dir'],
                    'neurosymbolic_rules_summary_pipeline.json',
                )
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, cls=NumpyEncoder)
                self.logger.info(f"Rule summary saved to {summary_path}")
            else:
                self.logger.info("No new rule candidates extracted in this step.")
        except Exception as e:
            self.logger.error(f"Error during neural rule extraction/update: {e}", exc_info=True)

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Return rule statistics from the reasoner when available."""
        if self.reasoner and hasattr(self.reasoner, 'get_rule_statistics'):
            try:
                return self.reasoner.get_rule_statistics()
            except Exception as e:
                self.logger.error(f"Failed to get rule statistics: {e}")
        return {}
