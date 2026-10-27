# src/ansrdt/core.py

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from src.config.config_manager import load_config
from src.models.light_classifier import LightClassifier
from src.reasoning.reasoning import SymbolicReasoner
from src.utils.model_utils import load_scaler


class ANSRDTCore:
    """Perception-to-symbols core: light classifier, Prolog bridge and state tracking."""

    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None,
                 classifier: Optional[LightClassifier] = None):
        """Load configuration, classifier and symbolic reasoner."""
        self.logger = logger or logging.getLogger(__name__)
        self.config = load_config(config_path)

        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self.project_root = os.path.dirname(self.config_dir)
        results_rel_path = self.config.get('paths', {}).get('results_dir', 'results')
        self.results_dir = os.path.join(self.project_root, results_rel_path)
        os.makedirs(self.results_dir, exist_ok=True)

        self.window_size = int(self.config['model']['window_size'])
        self.feature_names = self.config['model']['feature_names']
        if not self.feature_names:
            raise ValueError("feature_names must be a non-empty list in configuration.")

        self.classifier = classifier if classifier is not None else self._load_classifier()
        self.scaler = self._load_scaler()

        self.reasoner = self._initialize_reasoner()
        self.last_data_quality: Dict[str, Any] = {}

        self.current_state: Optional[Dict[str, Any]] = None
        self.state_history_limit = self.config.get('logging', {}).get('state_history_limit', 1000)
        self.state_history = []

    def _load_classifier(self) -> LightClassifier:
        """Load the serialized LightClassifier from results/models."""
        model_path = os.path.join(self.results_dir, 'models', 'light_classifier.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LightClassifier not found at {model_path}")
        return LightClassifier.load(model_path, logger=self.logger)

    def _load_scaler(self) -> Optional[Any]:
        """Load the fitted scaler used during pipeline training."""
        scaler_path = os.path.join(self.results_dir, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            self.logger.warning(
                "Scaler not found at %s; classifier inputs will not be scaled.",
                scaler_path,
            )
            return None
        return load_scaler(scaler_path, self.logger)

    def _apply_scaler(self, data: np.ndarray) -> np.ndarray:
        """Scale a 3D window batch when a scaler is available."""
        if self.scaler is None:
            return data
        expected_features = getattr(self.scaler, 'n_features_in_', data.shape[-1])
        if data.shape[-1] != expected_features:
            self.logger.warning(
                "Scaler expects %d features but received %d; skipping scaling.",
                expected_features, data.shape[-1],
            )
            return data
        flat = data.reshape(-1, data.shape[-1])
        return self.scaler.transform(flat).reshape(data.shape)

    def _initialize_reasoner(self) -> Optional[SymbolicReasoner]:
        """Initialize the symbolic reasoner when enabled in configuration."""
        symbolic_config = self.config.get('symbolic_reasoning', {})
        if not symbolic_config.get('enabled', False):
            self.logger.info("Symbolic reasoning is disabled.")
            return None

        rules_path = self.config.get('paths', {}).get('reasoning_rules_path')
        if not rules_path:
            raise KeyError("'reasoning_rules_path' not found in configuration paths.")
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Rules file not found at: {rules_path}")

        reasoner = SymbolicReasoner(
            rules_path=rules_path,
            input_shape=(self.window_size, len(self.feature_names)),
            model=self.classifier,
            logger=self.logger,
        )
        temporal_path = os.path.join(self.results_dir, 'temporal_thresholds.json')
        if os.path.exists(temporal_path):
            try:
                with open(temporal_path) as file:
                    reasoner.set_temporal_thresholds(json.load(file))
                self.logger.info("Loaded temporal thresholds from %s", temporal_path)
            except (OSError, ValueError, TypeError) as error:
                self.logger.warning(
                    "Could not load temporal thresholds from %s: %s",
                    temporal_path, error,
                )
        return reasoner

    def preprocess_data(self, sensor_data: np.ndarray) -> np.ndarray:
        """Validate and reshape input to ``(batch, window_size, features)``."""
        if not isinstance(sensor_data, np.ndarray):
            sensor_data = np.array(sensor_data, dtype=np.float32)
        else:
            sensor_data = sensor_data.astype(np.float32)

        if sensor_data.ndim == 2:
            if sensor_data.shape != (self.window_size, len(self.feature_names)):
                raise ValueError(
                    f"Expected 2D shape ({self.window_size}, {len(self.feature_names)}), got {sensor_data.shape}"
                )
            sensor_data = np.expand_dims(sensor_data, axis=0)
        elif sensor_data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {sensor_data.shape}")

        expected_inner = (self.window_size, len(self.feature_names))
        if sensor_data.shape[1:] != expected_inner:
            raise ValueError(
                f"Expected inner shape {expected_inner}, got {sensor_data.shape[1:]}"
            )

        non_finite = ~np.isfinite(sensor_data)
        non_finite_count = int(non_finite.sum())
        self.last_data_quality = {
            'non_finite_inputs': non_finite_count,
            'total_values': int(sensor_data.size),
            'policy': self.config.get('preprocessing', {}).get(
                'non_finite_policy', 'zero'
            ),
        }
        if non_finite_count:
            policy = self.last_data_quality['policy']
            if policy == 'error':
                raise ValueError(
                    f"Input contains {non_finite_count} NaN/Inf value(s) and "
                    "preprocessing.non_finite_policy is 'error'."
                )
            if policy not in {'zero', 'replace_zero'}:
                raise ValueError(f"Unsupported non-finite policy: {policy!r}")
            self.logger.warning(
                "Replacing %d NaN/Inf input value(s) with zeros.",
                non_finite_count,
            )
            sensor_data = np.nan_to_num(sensor_data, nan=0.0, posinf=0.0, neginf=0.0)
        return sensor_data

    def update_state(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """Run neural perception, symbolic reasoning and compile the system state."""
        raw_data = self.preprocess_data(sensor_data)
        model_data = self._apply_scaler(raw_data)

        if hasattr(self.classifier, 'positive_proba'):
            failure_probability = float(self.classifier.positive_proba(model_data)[0])
        else:
            failure_probability = float(np.asarray(self.classifier.predict_proba(model_data))[0, -1])
        anomaly_detected = failure_probability >= 0.5
        self.logger.debug(
            f"Failure probability: {failure_probability:.3f}, detected: {anomaly_detected}"
        )

        last_timestep = raw_data[0, -1, :]
        current_readings = self._extract_current_state(last_timestep)

        insights = []
        rule_activations = []
        if self.reasoner:
            try:
                insights = self.reasoner.reason(
                    current_readings, window=raw_data[0]
                )
                history = self.reasoner.get_rule_activations()
                if history:
                    rule_activations = history[-1].get('activated_rules_detailed', [])
            except Exception as e:
                self.logger.warning(f"Symbolic reasoning failed: {e}", exc_info=True)

        self.current_state = {
            'failure_probability': failure_probability,
            'confidence': failure_probability,
            'anomaly_score': failure_probability,
            'anomaly_detected': anomaly_detected,
            'sensor_readings': current_readings,
            'input_quality': dict(self.last_data_quality),
            'insights': insights,
            'rule_activations': rule_activations,
            'timestamp': str(np.datetime64('now')),
            'history_length': len(self.state_history),
        }

        self.state_history.append(self.current_state)
        if len(self.state_history) > self.state_history_limit:
            self.state_history.pop(0)

        if anomaly_detected:
            self.logger.info(
                f"Failure risk {failure_probability:.2f}. Insights: {insights}"
            )
        return self.current_state

    def adapt_and_explain(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """Update state and trigger rule learning when a strong anomaly is detected."""
        try:
            state = self.update_state(sensor_data)
            extraction_threshold = self.config.get('symbolic_reasoning', {}).get(
                'extraction_threshold', 0.6
            )
            if state.get('anomaly_detected') and state['failure_probability'] >= extraction_threshold:
                if self.reasoner:
                    input_data = self._apply_scaler(self.preprocess_data(sensor_data))
                    new_rules = self.reasoner.extract_rules_from_neural_model(
                        model=self.classifier,
                        input_data=input_data,
                        feature_names=self.feature_names,
                        threshold=extraction_threshold,
                    )
                    if new_rules:
                        min_conf = self.config.get('symbolic_reasoning', {}).get('min_confidence', 0.7)
                        self.reasoner.update_rules(new_rules, min_confidence=min_conf)
            return state
        except (ValueError, RuntimeError) as e:
            self.logger.error(f"Error in adapt_and_explain: {e}", exc_info=True)
            return {'error': str(e), 'status': 'error'}
        except Exception as e:
            self.logger.error(f"Unexpected error in adapt_and_explain: {e}", exc_info=True)
            return {'error': str(e), 'status': 'error'}

    def _extract_current_state(self, sensor_values: np.ndarray) -> Dict[str, Any]:
        """Map the last timestep's values to a named sensor dictionary."""
        if len(sensor_values) != len(self.feature_names):
            self.logger.error(
                f"Sensor length {len(sensor_values)} != feature count {len(self.feature_names)}"
            )
            return {name: 0.0 for name in self.feature_names}

        state_dict: Dict[str, Any] = {}
        for i, name in enumerate(self.feature_names):
            value = sensor_values[i]
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            if name in ('system_state', 'operational_hours'):
                state_dict[name] = int(round(float(value)))
            else:
                state_dict[name] = float(value)
        return state_dict

    def save_state(self, output_path: str) -> None:
        """Write the current state and recent history to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        state_data = {
            'current_state': self.current_state,
            'state_history_tail': self.state_history[-100:],
            'timestamp': str(np.datetime64('now')),
        }
        with open(output_path, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)

    def load_state(self, input_path: str) -> None:
        """Restore state and recent history from JSON."""
        if not os.path.exists(input_path):
            self.logger.error(f"State file not found: {input_path}")
            return
        with open(input_path, 'r') as f:
            state_data = json.load(f)
        self.current_state = state_data.get('current_state')
        self.state_history = state_data.get('state_history_tail', [])
