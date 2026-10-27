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
from src.hitl.queue import HITLQueue, REASON_PRIORITY
from src.models.light_classifier import LightClassifier
from src.preprocessing.preprocessing import preprocess_sequences
from src.reasoning.prob_query import RULES_PATH as PROB_RULES_PATH, evaluate_state
from src.reasoning.reasoning import SymbolicReasoner
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


def validate_config(config: dict, logger: logging.Logger, project_root: str, config_dir: str):
    """Validate configuration and resolve paths in place."""
    try:
        required_keys = ['model', 'training', 'paths']
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                raise KeyError(f"Missing required configuration key: {key}")

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

    def _build_classifier(self) -> LightClassifier:
        """Create the configured lightweight classifier."""
        model_config = self.config.get('model', {})
        architecture = model_config.get('architecture', 'light_classifier')
        if architecture in ('cnn_lstm', 'light_classifier'):
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

    def _initialise_reasoner(self) -> SymbolicReasoner:
        """Build the Prolog bridge for representative test windows."""
        if self.reasoner is None:
            self.reasoner = SymbolicReasoner(
                rules_path=self.config['paths']['reasoning_rules_path'],
                input_shape=(self.config['model']['window_size'], len(self.feature_names)),
                model=self.model,
                logger=self.logger,
            )
        return self.reasoner

    def _explain_representative_windows(self, X_raw: np.ndarray,
                                        scores: np.ndarray) -> Dict[str, Any]:
        """Reason over normal, borderline and critical raw test windows."""
        if len(X_raw) == 0:
            return {}

        reasoner = self._initialise_reasoner()
        with open(PROB_RULES_PATH, 'r') as file:
            problog_program = file.read()

        normal_idx = int(np.argmin(scores))
        critical_idx = int(np.argmax(scores))
        degraded_idx = int(np.argmin(np.abs(scores - 0.5)))
        indices = {'normal': normal_idx, 'degraded': degraded_idx, 'critical': critical_idx}

        explanations = {}
        for label, idx in indices.items():
            sensor_dict = {
                name: float(X_raw[idx, -1, feature_idx])
                for feature_idx, name in enumerate(self.feature_names)
            }
            insights = reasoner.reason(sensor_dict)
            problog = evaluate_state(problog_program, sensor_dict)
            explanations[label] = {
                'window_index': idx,
                'neural_probability': float(scores[idx]),
                'problog_failure_risk': float(problog['failure_risk']),
                'problog_hpc_degradation': float(problog['hpc_degradation']),
                'active_symbolic_conditions': problog['active_conditions'],
                'insights': insights,
                'sensor_values': sensor_dict,
            }
        return explanations

    def _build_hitl_queue(self, X_raw: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Queue uncertain and rule/probability-conflicting test windows."""
        hitl_config = self.config.get('hitl', {})
        if not hitl_config.get('enabled', True) or len(X_raw) == 0:
            return {}

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
            insights = reasoner.reason(sensor_dict)
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
                problog = evaluate_state(problog_program, record['sensor_values'])
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
            X_test, y_test = self.data_loader.load_test_data()
            test_source = self.config['paths']['test_file']
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_by_unit(
                X, y, validation_split=validation_split,
                test_split=test_split, random_state=random_state,
            )
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
        model_path = os.path.join(
            self.config['paths']['results_dir'], 'models', 'light_classifier.joblib'
        )
        self.model.save(model_path)

        val_scores = self.model.positive_proba(X_val_scaled)
        val_pred = (val_scores >= 0.5).astype(int)
        test_scores = self.model.positive_proba(X_test_scaled)
        test_pred = (test_scores >= 0.5).astype(int)

        validation_metrics = self._compute_metrics(y_val, val_pred, val_scores)
        test_metrics = self._compute_metrics(y_test, test_pred, test_scores)
        self.logger.info("Validation metrics: %s", validation_metrics)
        self.logger.info("Test metrics: %s", test_metrics)

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
            'representative_windows': explanations,
            'hitl_queue': hitl_queue,
            'artifacts': {
                'model_path': model_path,
                'scaler_path': scaler_path,
                'problog_rules_path': PROB_RULES_PATH,
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
