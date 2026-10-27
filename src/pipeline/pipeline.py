# src/pipeline/pipeline.py

import json
import logging
import os
import time
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
from src.evaluation.alerting import compare_alerting
from src.evaluation.engine_metrics import (
    compute_engine_metrics,
    export_engine_metrics_csv,
)
from src.evaluation.thresholding import (
    save_precision_recall_plot,
    select_threshold,
)
from src.features.temporal_features import TemporalFeatureExtractor
from src.fusion.probability_fusion import (
    ProbabilityFusion,
    evaluate_probability_fusion,
    symbolic_rule_score,
)
from src.hitl.queue import HITLQueue, REASON_PRIORITY
from src.models.light_classifier import LightClassifier
from src.models.rul_regressor import (
    RULLightGBMRegressor,
    out_of_fold_predictions,
)
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
        if architecture not in {
                'random_forest', 'light_classifier', 'lightgbm',
                'lgbm', 'lightgbm_classifier', 'cnn_lstm'}:
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
        if calibration.get('method', 'sigmoid') not in {
                'sigmoid', 'isotonic', 'auto'}:
            raise ValueError(
                "calibration.method must be 'sigmoid', 'isotonic' or 'auto'."
            )
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

    features = config.get('features', {})
    temporal_features = features.get('temporal', {})
    if temporal_features.get('enabled', False):
        for key in ('percentile_low', 'percentile_high'):
            if key in temporal_features:
                _require_range(
                    temporal_features[key], f'features.temporal.{key}',
                    0.0, 100.0,
                )
        low = temporal_features.get('percentile_low', 25.0)
        high = temporal_features.get('percentile_high', 75.0)
        if low >= high:
            raise ValueError(
                "features.temporal.percentile_low must be smaller than "
                "features.temporal.percentile_high."
            )
        if 'moving_average_cycles' in temporal_features and (
                not isinstance(
                    temporal_features['moving_average_cycles'], int
                ) or temporal_features['moving_average_cycles'] < 1):
            raise ValueError(
                "features.temporal.moving_average_cycles must be an integer >= 1."
            )
        apply_to = temporal_features.get('apply_to_architectures')
        if apply_to is not None and not isinstance(apply_to, list):
            raise ValueError(
                "features.temporal.apply_to_architectures must be a list."
            )

    alerting = config.get('alerting', {})
    if alerting.get('enabled', False):
        ks = alerting.get('consecutive_ks', [2, 3])
        if (not isinstance(ks, list)
                or not all(isinstance(k, int) and k >= 1 for k in ks)):
            raise ValueError(
                "alerting.consecutive_ks must be a list of integers >= 1."
            )
        if 'smoothing_cycles' in alerting and (
                not isinstance(alerting['smoothing_cycles'], int)
                or alerting['smoothing_cycles'] < 1):
            raise ValueError(
                "alerting.smoothing_cycles must be an integer >= 1."
            )

    rul = config.get('rul_regression', {})
    if rul.get('enabled', False):
        if 'cv_folds' in rul and (
                not isinstance(rul['cv_folds'], int) or rul['cv_folds'] < 2):
            raise ValueError("rul_regression.cv_folds must be an integer >= 2.")
        apply_to = rul.get('apply_to_architectures')
        if apply_to is not None and not isinstance(apply_to, list):
            raise ValueError(
                "rul_regression.apply_to_architectures must be a list."
            )

    fusion = config.get('fusion', {})
    if fusion.get('enabled', False):
        _require_range(fusion.get('min_precision', 0.75),
                       'fusion.min_precision', 0.0, 1.0)
        if fusion.get('threshold_strategy', 'recall_at_precision') not in {
                'f1', 'recall_at_precision', 'fbeta'}:
            raise ValueError(
                "fusion.threshold_strategy must be 'f1', "
                "'recall_at_precision' or 'fbeta'."
            )
        apply_to = fusion.get('apply_to_architectures')
        if apply_to is not None and not isinstance(apply_to, list):
            raise ValueError(
                "fusion.apply_to_architectures must be a list."
            )

    experiments = config.get('experiments', {}).get('compare_models', {})
    if experiments.get('enabled', False):
        models = experiments.get('models', ['random_forest', 'lightgbm'])
        allowed = {'random_forest', 'lightgbm'}
        if (not isinstance(models, list) or not models
                or not all(isinstance(name, str) for name in models)):
            raise ValueError(
                "experiments.compare_models.models must be a non-empty list."
            )
        invalid = [name for name in models if name not in allowed]
        if invalid:
            raise ValueError(
                f"Unsupported comparison models: {invalid}. "
                f"Allowed: {sorted(allowed)}."
            )
        metric = experiments.get('selection_metric', 'fbeta')
        if metric not in {'f1', 'fbeta', 'recall', 'average_precision', 'roc_auc'}:
            raise ValueError(
                "experiments.compare_models.selection_metric must be one of "
                "f1, fbeta, recall, average_precision, roc_auc."
            )
        if experiments.get('selection_split', 'validation') != 'validation':
            raise ValueError(
                "experiments.compare_models.selection_split must be 'validation'; "
                "the official test split cannot drive model selection."
            )

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

    @staticmethod
    def _normalise_architecture(architecture: str) -> str:
        """Return the canonical architecture name."""
        name = str(architecture or 'random_forest').lower()
        if name in ('cnn_lstm', 'light_classifier'):
            return 'random_forest'
        if name in ('lgbm', 'lightgbm_classifier'):
            return 'lightgbm'
        return name

    @staticmethod
    def _applies_to(section: dict, architecture: str) -> bool:
        """Return True when a feature section targets this architecture."""
        if not section.get('enabled', False):
            return False
        targets = section.get('apply_to_architectures')
        if targets is None:
            return True
        return architecture in {str(target).lower() for target in targets}

    def _build_classifier(
            self, architecture: Optional[str] = None) -> LightClassifier:
        """Create the configured lightweight classifier."""
        model_config = self.config.get('model', {})
        architecture = self._normalise_architecture(
            architecture or model_config.get('architecture', 'random_forest')
        )
        if architecture not in {'random_forest', 'lightgbm', 'mlp'}:
            raise ValueError(f"Unsupported architecture: {architecture!r}")
        option_key = 'lightgbm' if architecture == 'lightgbm' else 'light_classifier'
        options = dict(model_config.get(option_key, {}))
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

    def _calibrate_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray) -> Dict[str, Any]:
        """Calibrate on training data and select the method on validation only."""
        calibration_config = self.config.get('calibration', {})
        if not calibration_config.get('enabled', True):
            self.logger.info("Probability calibration disabled by configuration.")
            return {'enabled': False, 'applied': False}

        method = str(calibration_config.get('method', 'sigmoid')).lower()
        cv_folds = int(calibration_config.get('cv_folds', 5))
        requested_method = method
        candidates = []
        if method == 'auto':
            methods = ['sigmoid', 'isotonic']
        else:
            methods = [method]

        for candidate in methods:
            try:
                self.model.calibrate(
                    X_train, y_train, method=candidate, cv_folds=cv_folds
                )
            except (ValueError, RuntimeError) as error:
                self.logger.warning(
                    "Calibration candidate %s failed: %s", candidate, error
                )
                continue
            if not getattr(self.model, 'is_calibrated', False):
                continue
            candidate_scores = self.model.positive_proba(X_val)
            candidate_metrics = calibration_metrics(
                y_val, candidate_scores,
                n_bins=int(calibration_config.get('n_bins', 10)),
            )
            candidates.append({
                'method': candidate,
                'validation_brier_score': candidate_metrics['brier_score'],
                'validation_ece': candidate_metrics['ece'],
            })

        selected_method = getattr(self.model, 'calibration_method', None)
        if method == 'auto' and candidates:
            selected_method = min(
                candidates,
                key=lambda item: (
                    item['validation_brier_score'], item['validation_ece']
                ),
            )['method']
            if selected_method != getattr(self.model, 'calibration_method', None):
                self.model.calibrate(
                    X_train, y_train, method=selected_method,
                    cv_folds=cv_folds,
                )
            self.logger.info(
                "Calibration method selected on validation: %s", selected_method
            )

        return {
            'enabled': True,
            'applied': bool(getattr(self.model, 'is_calibrated', False)),
            'requested_method': requested_method,
            'method': selected_method,
            'selection_metric': (
                'validation_brier_score' if method == 'auto' else None
            ),
            'candidates': candidates,
            'cv_folds': getattr(self.model, 'calibration_cv_folds', None),
        }

    def _select_decision_threshold(
            self, y_val: np.ndarray, val_scores: np.ndarray,
            strategy: Optional[str] = None,
            min_precision: Optional[float] = None) -> Dict[str, Any]:
        """Select a decision threshold on validation data only."""
        threshold_config = self.config.get('thresholding', {})
        selection = select_threshold(
            y_val, val_scores,
            strategy=strategy or threshold_config.get('strategy', 'fbeta'),
            min_precision=float(
                threshold_config.get('min_precision', 0.7)
                if min_precision is None else min_precision
            ),
            beta=float(threshold_config.get('beta', 2.0)),
            fallback=float(threshold_config.get('fallback', 0.5)),
        )
        self.model.set_threshold(selection['threshold'])
        return selection

    def _calibration_report(self, y_val: np.ndarray, val_scores: np.ndarray,
                            y_test: np.ndarray, test_scores: np.ndarray,
                            results_dir: str,
                            suffix: str = '') -> Dict[str, Any]:
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
            os.path.join(
                visualization_dir,
                f'reliability_validation{suffix}.png',
            ),
            title='Validation reliability curve',
        )
        report['test']['reliability_plot'] = save_reliability_plot(
            report['test'],
            os.path.join(visualization_dir, f'reliability_test{suffix}.png'),
            title='Test reliability curve',
        )
        report_path = os.path.join(
            results_dir, f'calibration_report{suffix}.json'
        )
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

    def _explain_representative_windows(
            self, X_raw: np.ndarray, scores: np.ndarray,
            decision_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Reason over normal, borderline and critical raw test windows."""
        if len(X_raw) == 0:
            return {}

        reasoner = self._initialise_reasoner()
        with open(PROB_RULES_PATH, 'r') as file:
            problog_program = file.read()

        decision_threshold = float(
            getattr(self.model, 'decision_threshold', 0.5)
            if decision_threshold is None else decision_threshold
        )
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

    def _build_hitl_queue(self, X_raw: np.ndarray, scores: np.ndarray,
                          decision_threshold: Optional[float] = None
                          ) -> Dict[str, Any]:
        """Queue uncertain and rule/probability-conflicting test windows."""
        hitl_config = self.config.get('hitl', {})
        if not hitl_config.get('enabled', True) or len(X_raw) == 0:
            return {}

        decision_threshold = float(
            getattr(self.model, 'decision_threshold',
                    self.config.get('thresholding', {}).get('fallback', 0.5))
            if decision_threshold is None else decision_threshold
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

    def _load_cmapss_splits(self) -> Dict[str, Any]:
        """Load FD001 windows and split whole engines into train/val/test."""
        X, y = self.data_loader.load_data()
        train_units_all = np.asarray(self.data_loader.unit_ids)
        train_cycles_all = np.asarray(self.data_loader.window_end_cycles)
        train_rul_all = np.asarray(self.data_loader.window_rul)

        validation_split = float(
            self.config['training'].get('validation_split', 0.2)
        )
        test_split = float(self.config['training'].get('test_split', 0.1))
        train_fraction = float(
            self.config['training'].get('train_fraction', 0.2)
        )
        random_state = int(self.config['training'].get('random_state', 42))

        if self.config['paths'].get('test_file'):
            held_out_fraction = max(
                0.0, 1.0 - train_fraction - validation_split
            )
            X_train, X_val, _, y_train, y_val, _ = (
                self.data_loader.split_by_unit(
                    X, y, validation_split=validation_split,
                    test_split=held_out_fraction, random_state=random_state,
                )
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
            X_train, X_val, X_test, y_train, y_val, y_test = (
                self.data_loader.split_by_unit(
                    X, y, validation_split=validation_split,
                    test_split=test_split, random_state=random_state,
                )
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
            "Windows: train=%d val=%d test=%d | "
            "positives: train=%d val=%d test=%d",
            len(X_train), len(X_val), len(X_test),
            int(y_train.sum()), int(y_val.sum()), int(y_test.sum()),
        )
        train_mask = self.data_loader.last_split_masks['train']
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'train_units': np.asarray(train_units_all)[train_mask],
            'train_cycles': np.asarray(train_cycles_all)[train_mask],
            'train_rul': np.asarray(train_rul_all)[train_mask],
            'val_units': val_units, 'test_units': test_units,
            'val_cycles': val_cycles, 'test_cycles': test_cycles,
            'val_rul': val_rul, 'test_rul': test_rul,
            'test_source': test_source,
        }

    def _model_paths(self, architecture: str,
                     suffix: str = '') -> Dict[str, str]:
        """Return architecture-specific model artifact paths."""
        models_dir = os.path.join(
            self.config['paths']['results_dir'], 'models'
        )
        if architecture == 'lightgbm':
            model_name = 'lightgbm_classifier'
        else:
            model_name = 'light_classifier'
        return {
            'model': os.path.join(models_dir, f'{model_name}{suffix}.joblib'),
            'feature_extractor': os.path.join(
                models_dir, f'temporal_features{suffix}.joblib'
            ),
            'feature_metadata': os.path.join(
                self.config['paths']['results_dir'],
                f'temporal_feature_metadata{suffix}.json'
            ),
            'rul_regressor': os.path.join(
                models_dir, f'lightgbm_rul_regressor{suffix}.joblib'
            ),
        }

    def _prepare_model_data(self, splits: Dict[str, Any],
                            architecture: str,
                            suffix: str = '') -> Dict[str, Any]:
        """Build architecture-specific model inputs without test leakage."""
        architecture = self._normalise_architecture(architecture)
        temporal_config = self.config.get('features', {}).get('temporal', {})
        use_temporal = (
            architecture == 'lightgbm'
            and self._applies_to(temporal_config, architecture)
        )
        extractor = None
        scaler = None
        feature_names_out = None
        paths = self._model_paths(architecture, suffix)

        X_train = splits['X_train']
        X_val = splits['X_val']
        X_test = splits['X_test']
        if use_temporal:
            extractor = TemporalFeatureExtractor(
                self.feature_names,
                include_raw_window=bool(
                    temporal_config.get('include_raw_window', True)
                ),
                moving_average_cycles=int(
                    temporal_config.get('moving_average_cycles', 5)
                ),
                percentile_low=float(
                    temporal_config.get('percentile_low', 25.0)
                ),
                percentile_high=float(
                    temporal_config.get('percentile_high', 75.0)
                ),
            )
            X_train_model = extractor.fit_transform(X_train)
            X_val_model = extractor.transform(X_val)
            X_test_model = extractor.transform(X_test)
            feature_names_out = list(extractor.feature_names_out)
            extractor.save(paths['feature_extractor'])
            extractor.save_metadata(paths['feature_metadata'])
            self.logger.info(
                "Temporal features: %d statistics + raw=%s -> %d features.",
                len(extractor.feature_names) * 13,
                extractor.include_raw_window,
                len(feature_names_out),
            )
        elif architecture == 'lightgbm':
            X_train_model = np.asarray(X_train).reshape(len(X_train), -1)
            X_val_model = np.asarray(X_val).reshape(len(X_val), -1)
            X_test_model = np.asarray(X_test).reshape(len(X_test), -1)
            feature_names_out = [
                f'raw__{index}' for index in range(X_train_model.shape[1])
            ]
        else:
            scaler, X_train_model = preprocess_sequences(X_train)
            X_val_model = self._split_scaled(scaler, X_val)
            X_test_model = self._split_scaled(scaler, X_test)
            scaler_path = os.path.join(
                self.config['paths']['results_dir'], 'scaler.pkl'
            )
            save_scaler(scaler, scaler_path, self.logger)

        rul_config = self.config.get('rul_regression', {})
        regressor = None
        rul_report = {'enabled': False, 'applied': False}
        val_rul_prediction = None
        test_rul_prediction = None
        if (architecture == 'lightgbm'
                and self._applies_to(rul_config, architecture)):
            params = {
                key: value for key, value in rul_config.items()
                if key not in {
                    'enabled', 'apply_to_architectures',
                    'use_as_classifier_feature', 'cv_folds',
                }
            }
            random_state = int(
                self.config['model'].get('random_state', 42)
            )

            def make_regressor():
                return RULLightGBMRegressor(
                    random_state=random_state, logger=self.logger, **params
                )

            regressor = make_regressor()
            train_rul = np.asarray(splits['train_rul'], dtype=float)
            regressor.fit(X_train_model, train_rul)
            oof_rul = out_of_fold_predictions(
                make_regressor, X_train_model, train_rul,
                splits['train_units'],
                n_splits=int(rul_config.get('cv_folds', 5)),
            )
            val_rul_prediction = regressor.predict(X_val_model)
            test_rul_prediction = regressor.predict(X_test_model)
            use_as_feature = bool(
                rul_config.get('use_as_classifier_feature', True)
            )
            if use_as_feature:
                X_train_model = np.hstack([
                    X_train_model, oof_rul.reshape(-1, 1)
                ]).astype(np.float32)
                X_val_model = np.hstack([
                    X_val_model, val_rul_prediction.reshape(-1, 1)
                ]).astype(np.float32)
                X_test_model = np.hstack([
                    X_test_model, test_rul_prediction.reshape(-1, 1)
                ]).astype(np.float32)
                feature_names_out.append('predicted_rul')
            regressor.save(paths['rul_regressor'])
            rul_report = {
                'enabled': True,
                'applied': True,
                'model': 'lightgbm',
                'used_as_classifier_feature': use_as_feature,
                'model_path': paths['rul_regressor'],
                'metrics': {
                    'train_out_of_fold': RULLightGBMRegressor.metrics(
                        train_rul, oof_rul
                    ),
                    'validation': RULLightGBMRegressor.metrics(
                        splits['val_rul'], val_rul_prediction
                    ),
                    'test': RULLightGBMRegressor.metrics(
                        splits['test_rul'], test_rul_prediction
                    ),
                },
                'validation_predictions': val_rul_prediction.tolist(),
                'test_predictions': test_rul_prediction.tolist(),
            }
            self.logger.info(
                "RUL regression (validation): %s", rul_report['metrics'][
                    'validation'
                ]
            )

        return {
            'architecture': architecture,
            'X_train': X_train_model,
            'X_val': X_val_model,
            'X_test': X_test_model,
            'feature_names_out': feature_names_out,
            'extractor': extractor,
            'scaler': scaler,
            'rul_regressor': regressor,
            'rul_report': rul_report,
            'paths': paths,
            'uses_temporal_features': use_temporal,
            'uses_rul_feature': bool(
                regressor is not None
                and rul_report.get('used_as_classifier_feature', False)
            ),
        }

    def _engine_metrics(self, units, cycles, rul, y_true, y_pred,
                        suffix: str = '') -> Dict[str, Any]:
        """Compute and persist per-engine metrics."""
        if not self.config.get('engine_metrics', {}).get('enabled', True):
            return {}
        metrics = compute_engine_metrics(
            units, cycles, rul, y_true, y_pred
        )
        results_dir = self.config['paths']['results_dir']
        csv_path = export_engine_metrics_csv(
            metrics,
            os.path.join(results_dir, f'engine_metrics{suffix}.csv'),
        )
        return {**metrics, 'csv_path': csv_path}

    def _alerting_report(self, y_true, scores, units, cycles, rul,
                         threshold: float, suffix: str = '') -> Dict[str, Any]:
        """Compare immediate and confirmed alerts, then persist the report."""
        alerting_config = self.config.get('alerting', {})
        if not self._applies_to(alerting_config, 'any'):
            return {}
        report = compare_alerting(
            y_true, scores, units, cycles, rul, threshold,
            consecutive_ks=alerting_config.get('consecutive_ks', [2, 3]),
            smoothing_cycles=int(
                alerting_config.get('smoothing_cycles', 3)
            ),
        )
        path = os.path.join(
            self.config['paths']['results_dir'],
            f'alerting_analysis{suffix}.json',
        )
        with open(path, 'w') as file:
            json.dump(report, file, indent=2, cls=NumpyEncoder)
        report['report_path'] = path
        return report

    def _evaluate_architecture(self, prepared: Dict[str, Any],
                               splits: Dict[str, Any],
                               suffix: str = '') -> Dict[str, Any]:
        """Fit, calibrate, threshold and evaluate one architecture."""
        architecture = prepared['architecture']
        self.model = self._build_classifier(architecture)
        self.model.feature_names_out = prepared['feature_names_out']
        self.model.uses_temporal_features = prepared['uses_temporal_features']
        self.model.uses_rul_feature = prepared['uses_rul_feature']
        self.model.temporal_feature_extractor = prepared['extractor']
        self.model.rul_regressor = prepared['rul_regressor']

        started = time.perf_counter()
        self.model.fit(prepared['X_train'], splits['y_train'])
        calibration_info = self._calibrate_model(
            prepared['X_train'], splits['y_train'],
            prepared['X_val'], splits['y_val'],
        )
        training_seconds = time.perf_counter() - started

        inference_started = time.perf_counter()
        val_scores = self.model.positive_proba(prepared['X_val'])
        test_scores = self.model.positive_proba(prepared['X_test'])
        inference_seconds = time.perf_counter() - inference_started

        threshold_info = self._select_decision_threshold(
            splits['y_val'], val_scores
        )
        model_path = prepared['paths']['model']
        self.model.save(model_path)

        threshold = float(self.model.decision_threshold)
        val_pred = (val_scores >= threshold).astype(int)
        test_pred = (test_scores >= threshold).astype(int)
        validation_metrics = self._compute_metrics(
            splits['y_val'], val_pred, val_scores
        )
        test_metrics = self._compute_metrics(
            splits['y_test'], test_pred, test_scores
        )
        calibration_report = self._calibration_report(
            splits['y_val'], val_scores, splits['y_test'], test_scores,
            self.config['paths']['results_dir'], suffix=suffix,
        )
        threshold_report = {
            key: value for key, value in threshold_info.items()
            if key != 'precision_recall_curve'
        }
        threshold_report['precision_recall_plot'] = save_precision_recall_plot(
            threshold_info.get('precision_recall_curve', {}),
            os.path.join(
                self.config['paths']['results_dir'], 'visualization',
                f'precision_recall_curve{suffix}.png',
            ),
            threshold=threshold,
        )
        threshold_path = os.path.join(
            self.config['paths']['results_dir'],
            f'decision_threshold{suffix}.json',
        )
        with open(threshold_path, 'w') as file:
            json.dump(threshold_info, file, indent=2, cls=NumpyEncoder)

        engine_metrics = {
            'validation': self._engine_metrics(
                splits['val_units'], splits['val_cycles'], splits['val_rul'],
                splits['y_val'], val_pred, suffix=f'{suffix}_validation',
            ),
            'test': self._engine_metrics(
                splits['test_units'], splits['test_cycles'], splits['test_rul'],
                splits['y_test'], test_pred, suffix=f'{suffix}_test',
            ),
        }
        alerting = self._alerting_report(
            splits['y_test'], test_scores, splits['test_units'],
            splits['test_cycles'], splits['test_rul'], threshold,
            suffix=suffix,
        )
        self.logger.info(
            "%s validation metrics: %s", architecture, validation_metrics
        )
        self.logger.info("%s test metrics: %s", architecture, test_metrics)
        return {
            'architecture': architecture,
            'model_path': model_path,
            'feature_count': int(prepared['X_train'].shape[1]),
            'calibration': calibration_info,
            'calibration_report': calibration_report,
            'decision_threshold': threshold_report,
            'decision_threshold_path': threshold_path,
            'validation_metrics': validation_metrics,
            'test_metrics': test_metrics,
            'engine_metrics': engine_metrics,
            'alerting': alerting,
            'rul_regression': prepared['rul_report'],
            'feature_metadata_path': (
                prepared['paths']['feature_metadata']
                if prepared['uses_temporal_features'] else None
            ),
            'training_seconds': float(training_seconds),
            'inference_seconds': float(inference_seconds),
            'val_scores': val_scores,
            'test_scores': test_scores,
            'threshold': threshold,
        }

    def _collect_fusion_signals(self, X_raw: np.ndarray,
                                neural_scores: np.ndarray,
                                predicted_rul: Optional[np.ndarray],
                                label: str) -> np.ndarray:
        """Reason over every window and build the logistic-fusion matrix."""
        reasoner = self._initialise_reasoner()
        with open(PROB_RULES_PATH, 'r') as file:
            problog_program = file.read()
        symbolic_scores = np.zeros(len(X_raw), dtype=float)
        problog_risks = np.zeros(len(X_raw), dtype=float)
        problog_config = self.config.get('problog', {})
        problog_logger = logging.getLogger('problog')
        previous_problog_level = problog_logger.level
        problog_logger.setLevel(logging.WARNING)
        try:
            for index, window in enumerate(X_raw):
                sensor_dict = {
                    name: float(window[-1, feature_index])
                    for feature_index, name in enumerate(self.feature_names)
                }
                reasoner.state_history = []
                insights = reasoner.reason(sensor_dict, window=window)
                symbolic_scores[index] = symbolic_rule_score(insights)
                problog = evaluate_state(
                    problog_program, sensor_dict,
                    learned_rules=reasoner.get_active_learned_rule_names(),
                    learned_rule_weight=float(
                        problog_config.get('learned_rule_weight', 0.15)
                    ),
                    max_active_learned_rules=int(
                        problog_config.get('max_active_learned_rules', 3)
                    ),
                )
                problog_risks[index] = float(problog['failure_risk'])
                if (index + 1) % 1000 == 0:
                    self.logger.info(
                        "Fusion signals %s: %d/%d windows", label,
                        index + 1, len(X_raw),
                    )
        finally:
            problog_logger.setLevel(previous_problog_level)
        if predicted_rul is None:
            predicted_rul = np.zeros(len(X_raw), dtype=float)
        fusion_config = self.config.get('fusion', {})
        return ProbabilityFusion.build_matrix(
            neural_scores, symbolic_scores, problog_risks,
            predicted_rul,
            feature_order=fusion_config.get(
                'features',
                [
                    'neural_probability', 'symbolic_rule_score',
                    'problog_failure_risk', 'predicted_rul',
                ],
            ),
        )

    def _apply_fusion(self, prepared: Dict[str, Any],
                      splits: Dict[str, Any],
                      neural_val_scores: np.ndarray,
                      neural_test_scores: np.ndarray,
                      neural_threshold: float) -> Dict[str, Any]:
        """Fit validation-only fusion and return fused scores and metrics."""
        fusion_config = self.config.get('fusion', {})
        if not self._applies_to(
                fusion_config, prepared['architecture']):
            return {'enabled': False, 'applied': False}

        val_matrix = self._collect_fusion_signals(
            splits['X_val'], neural_val_scores,
            prepared['rul_report'].get('validation_predictions'),
            'validation',
        )
        test_matrix = self._collect_fusion_signals(
            splits['X_test'], neural_test_scores,
            prepared['rul_report'].get('test_predictions'),
            'test',
        )
        fusion = ProbabilityFusion(
            random_state=int(self.config['model'].get('random_state', 42)),
            feature_order=fusion_config.get(
                'features',
                [
                    'neural_probability', 'symbolic_rule_score',
                    'problog_failure_risk', 'predicted_rul',
                ],
            ),
            logger=self.logger,
        )
        fusion.fit(val_matrix, splits['y_val'])
        val_fused = fusion.positive_proba(val_matrix)
        test_fused = fusion.positive_proba(test_matrix)
        fusion_selection = select_threshold(
            splits['y_val'], val_fused,
            strategy=fusion_config.get('threshold_strategy', 'recall_at_precision'),
            min_precision=float(fusion_config.get('min_precision', 0.75)),
            beta=float(fusion_config.get('beta', 2.0)),
            fallback=float(fusion_config.get('fallback', 0.5)),
        )
        fusion_threshold = float(fusion_selection['threshold'])
        val_pred = (val_fused >= fusion_threshold).astype(int)
        test_pred = (test_fused >= fusion_threshold).astype(int)
        fusion_model_path = os.path.join(
            self.config['paths']['results_dir'], 'models',
            'probability_fusion.joblib',
        )
        fusion.save(fusion_model_path)
        threshold_path = os.path.join(
            self.config['paths']['results_dir'], 'fusion_threshold.json'
        )
        with open(threshold_path, 'w') as file:
            json.dump(fusion_selection, file, indent=2, cls=NumpyEncoder)
        report = {
            'enabled': True,
            'applied': True,
            'trained_on': 'validation_split_only',
            'features': fusion.feature_order,
            'coefficients': fusion.coefficients(),
            'model_path': fusion_model_path,
            'threshold_path': threshold_path,
            'threshold': fusion_threshold,
            'threshold_strategy': fusion_selection['strategy'],
            'validation_metrics': self._compute_metrics(
                splits['y_val'], val_pred, val_fused
            ),
            'test_metrics': self._compute_metrics(
                splits['y_test'], test_pred, test_fused
            ),
            'validation_calibration': evaluate_probability_fusion(
                splits['y_val'], val_fused
            ),
            'test_calibration': evaluate_probability_fusion(
                splits['y_test'], test_fused
            ),
            'neural_threshold': float(neural_threshold),
            'val_scores': val_fused,
            'test_scores': test_fused,
        }
        self.logger.info(
            "Fusion validation metrics: %s", report['validation_metrics']
        )
        self.logger.info("Fusion test metrics: %s", report['test_metrics'])
        return report

    def _run_cmapss(self) -> Dict[str, Any]:
        """Train and evaluate the FD001 model with an engine-level split."""
        self.logger.info("Starting C-MAPSS pipeline.")
        splits = self._load_cmapss_splits()
        architecture = self._normalise_architecture(
            self.config.get('model', {}).get('architecture', 'random_forest')
        )
        prepared = self._prepare_model_data(splits, architecture)
        evaluation = self._evaluate_architecture(prepared, splits)
        val_scores = evaluation['val_scores']
        test_scores = evaluation['test_scores']
        calibration_info = evaluation['calibration']
        calibration_report = evaluation['calibration_report']
        threshold_report = evaluation['decision_threshold']
        threshold_path = evaluation['decision_threshold_path']
        validation_metrics = evaluation['validation_metrics']
        test_metrics = evaluation['test_metrics']
        engine_metrics = evaluation['engine_metrics']
        model_path = evaluation['model_path']
        scaler_path = (
            os.path.join(self.config['paths']['results_dir'], 'scaler.pkl')
            if prepared['scaler'] is not None else None
        )
        decision_threshold = evaluation['threshold']

        temporal_config = self.config.get('temporal', {})
        temporal_path = None
        if temporal_config.get('enabled', True):
            reasoner = self._initialise_reasoner()
            sensor_thresholds = reasoner.get_sensor_thresholds()
            self.temporal_thresholds = compute_temporal_thresholds(
                splits['X_train'], self.feature_names,
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

        fusion = self._apply_fusion(
            prepared, splits, val_scores, test_scores, decision_threshold
        )
        if fusion.get('applied'):
            val_scores = fusion['val_scores']
            test_scores = fusion['test_scores']
            decision_threshold = fusion['threshold']
            validation_metrics = fusion['validation_metrics']
            test_metrics = fusion['test_metrics']
            val_pred = (val_scores >= decision_threshold).astype(int)
            test_pred = (test_scores >= decision_threshold).astype(int)
            engine_metrics = {
                'validation': self._engine_metrics(
                    splits['val_units'], splits['val_cycles'],
                    splits['val_rul'], splits['y_val'], val_pred,
                    suffix='_fusion_validation',
                ),
                'test': self._engine_metrics(
                    splits['test_units'], splits['test_cycles'],
                    splits['test_rul'], splits['y_test'], test_pred,
                    suffix='_fusion_test',
                ),
            }
            alerting = self._alerting_report(
                splits['y_test'], test_scores, splits['test_units'],
                splits['test_cycles'], splits['test_rul'],
                decision_threshold, suffix='_fusion',
            )
        else:
            alerting = evaluation['alerting']

        rule_validation = None
        rule_validation_path = None
        rule_validation_config = self.config.get('hitl', {}).get(
            'rule_validation', {}
        )
        if rule_validation_config.get('enabled', True):
            reasoner = self._initialise_reasoner()
            rule_validation = reasoner.validate_learned_rules(
                splits['X_val'], self.feature_names, splits['y_val'],
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

        explanations = self._explain_representative_windows(
            splits['X_test'], test_scores, decision_threshold
        )
        hitl_queue = self._build_hitl_queue(
            splits['X_test'], test_scores, decision_threshold
        )
        if fusion.get('applied'):
            decision_section = {
                'source': 'probability_fusion',
                'threshold': fusion['threshold'],
                'strategy': fusion['threshold_strategy'],
                'path': fusion['threshold_path'],
                'neural_threshold': fusion['neural_threshold'],
                'neural_decision': {
                    **threshold_report, 'path': threshold_path,
                },
            }
            primary_threshold_path = fusion['threshold_path']
        else:
            decision_section = {
                'source': prepared['architecture'],
                **threshold_report,
                'path': threshold_path,
            }
            primary_threshold_path = threshold_path

        fusion_report = {
            key: value for key, value in fusion.items()
            if key not in {'val_scores', 'test_scores'}
        }
        rul_report = {
            key: value for key, value in prepared['rul_report'].items()
            if key not in {'validation_predictions', 'test_predictions'}
        }

        results = {
            'dataset': 'NASA C-MAPSS FD001',
            'architecture': prepared['architecture'],
            'test_source': splits['test_source'],
            'data_file': self.config['paths']['data_file'],
            'splits': {
                'train_windows': int(len(splits['X_train'])),
                'validation_windows': int(len(splits['X_val'])),
                'test_windows': int(len(splits['X_test'])),
                'train_positive_rate': float(np.mean(splits['y_train'])),
                'validation_positive_rate': float(np.mean(splits['y_val'])),
                'test_positive_rate': float(np.mean(splits['y_test'])),
            },
            'metrics': {
                'validation': validation_metrics,
                'test': test_metrics,
            },
            'neural_metrics': {
                'validation': evaluation['validation_metrics'],
                'test': evaluation['test_metrics'],
            },
            'decision_threshold': decision_section,
            'probability_fusion': fusion_report,
            'rul_regression': rul_report,
            'alerting': alerting,
            'calibration': {
                **calibration_info,
                'metrics': calibration_report,
            },
            'engine_metrics': engine_metrics,
            'hitl_rule_validation': rule_validation,
            'representative_windows': explanations,
            'hitl_queue': hitl_queue,
            'artifacts': {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'feature_extractor_path': (
                    prepared['paths']['feature_extractor']
                    if prepared['uses_temporal_features'] else None
                ),
                'feature_metadata_path': (
                    prepared['paths']['feature_metadata']
                    if prepared['uses_temporal_features'] else None
                ),
                'rul_regressor_path': (
                    prepared['paths']['rul_regressor']
                    if prepared['rul_report'].get('applied') else None
                ),
                'fusion_model_path': fusion.get('model_path'),
                'problog_rules_path': PROB_RULES_PATH,
                'calibration_report_path': calibration_report.get('report_path'),
                'decision_threshold_path': primary_threshold_path,
                'temporal_thresholds_path': temporal_path,
                'alerting_report_path': alerting.get('report_path'),
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

    def _selection_score(self, metrics: Dict[str, Any]) -> float:
        """Return the configured validation-only model-selection score."""
        compare_config = self.config.get('experiments', {}).get(
            'compare_models', {}
        )
        metric = compare_config.get('selection_metric', 'fbeta')
        precision = float(metrics.get('precision') or 0.0)
        recall = float(metrics.get('recall') or 0.0)
        if metric == 'fbeta':
            beta = float(compare_config.get('beta', 2.0))
            beta_sq = beta * beta
            denominator = beta_sq * precision + recall
            return (
                (1.0 + beta_sq) * precision * recall / denominator
                if denominator else 0.0
            )
        if metric == 'average_precision':
            return float(metrics.get('average_precision') or 0.0)
        if metric == 'roc_auc':
            return float(metrics.get('roc_auc') or 0.0)
        return float(metrics.get(metric) or 0.0)

    def _run_model_comparison(self) -> Dict[str, Any]:
        """Compare configured architectures using validation-only selection."""
        self.logger.info("Starting model comparison.")
        compare_config = self.config.get('experiments', {}).get(
            'compare_models', {}
        )
        architectures = [
            self._normalise_architecture(name)
            for name in compare_config.get(
                'models', ['random_forest', 'lightgbm']
            )
        ]
        splits = self._load_cmapss_splits()
        comparison = {}
        for architecture in architectures:
            suffix = f'_{architecture}'
            prepared = self._prepare_model_data(
                splits, architecture, suffix=suffix
            )
            evaluation = self._evaluate_architecture(
                prepared, splits, suffix=suffix
            )
            comparison[architecture] = {
                'architecture': architecture,
                'model_path': evaluation['model_path'],
                'feature_count': evaluation['feature_count'],
                'validation_metrics': evaluation['validation_metrics'],
                'test_metrics': evaluation['test_metrics'],
                'calibration_method': evaluation['calibration'].get('method'),
                'calibration_candidates': evaluation['calibration'].get(
                    'candidates', []
                ),
                'decision_threshold': evaluation['threshold'],
                'engine_metrics': {
                    split: details.get('summary', {})
                    for split, details in evaluation['engine_metrics'].items()
                },
                'alerting': evaluation['alerting'].get('modes', {}),
                'rul_regression': {
                    key: value for key, value in
                    evaluation['rul_regression'].items()
                    if key not in {
                        'validation_predictions', 'test_predictions'
                    }
                },
                'training_seconds': evaluation['training_seconds'],
                'inference_seconds': evaluation['inference_seconds'],
                'validation_selection_score': self._selection_score(
                    evaluation['validation_metrics']
                ),
            }
            self.logger.info(
                "Comparison %s: validation score=%.4f test F1=%.4f",
                architecture,
                comparison[architecture]['validation_selection_score'],
                float(evaluation['test_metrics'].get('f1') or 0.0),
            )

        selected = max(
            comparison,
            key=lambda name: comparison[name]['validation_selection_score'],
        )
        report = {
            'dataset': 'NASA C-MAPSS FD001',
            'selection_metric': compare_config.get(
                'selection_metric', 'fbeta'
            ),
            'selection_beta': compare_config.get('beta', 2.0),
            'selection_split': 'validation',
            'selected_model': selected,
            'models': comparison,
            'splits': {
                'train_windows': int(len(splits['X_train'])),
                'validation_windows': int(len(splits['X_val'])),
                'test_windows': int(len(splits['X_test'])),
            },
            'timestamp': datetime.now().isoformat(),
        }
        path = os.path.join(
            self.config['paths']['results_dir'], 'model_comparison.json'
        )
        with open(path, 'w') as file:
            json.dump(report, file, indent=2, cls=NumpyEncoder)
        report['report_path'] = path
        self.logger.info(
            "Model comparison saved to %s; selected=%s", path, selected
        )
        return report

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

    def run(self, compare_models: Optional[bool] = None) -> Dict[str, Any]:
        """Execute the configured pipeline and return a run summary."""
        try:
            if not self.is_cmapss:
                return self._run_legacy()
            if compare_models is None:
                compare_models = bool(
                    self.config.get('experiments', {}).get(
                        'compare_models', {}
                    ).get('enabled', False)
                )
            if compare_models:
                return self._run_model_comparison()
            return self._run_cmapss()
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
