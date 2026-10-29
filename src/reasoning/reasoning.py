# src/reasoning/reasoning.py

import logging
import os
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from .temporal import window_temporal_features
from datetime import datetime
import math  # For isnan checks
import re  # For parsing metadata and rule strings

# PySWIP locates and loads SWI-Prolog at import time. Keeping that import
# optional lets lightweight modules (and their tests) import this module on
# machines without SWI-Prolog; only SymbolicReasoner instantiation needs it.
try:
    from pyswip import Prolog
except Exception as _pyswip_error:  # ImportError or SwiPrologNotFoundError
    Prolog = None
    PYSWIP_AVAILABLE = False
    PYSWIP_IMPORT_ERROR: Optional[Exception] = _pyswip_error
else:
    PYSWIP_AVAILABLE = True
    PYSWIP_IMPORT_ERROR = None

def _extract_rule_body(rule_string: str) -> Optional[str]:
    """Extract the body of a Prolog rule (everything after ':-' before trailing '.').

    Used for deduplication: two rules with the same body but different names
    (e.g. neural_rule_1 vs neural_rule_15) are semantically identical.
    Returns None if the rule has no ':-' separator.
    """
    if ':-' not in rule_string:
        return None
    body = rule_string.split(':-', 1)[1].strip()
    if body.endswith('.'):
        body = body[:-1].strip()
    return body


class SymbolicReasoner:
    def __init__(
            self,
            rules_path: str,
            input_shape: tuple,
            model: Optional[Any] = None,
            logger: Optional[logging.Logger] = None
    ):
        """
        Initializes the Symbolic Reasoner.
        Manages interaction with Prolog engine, loads base rules,
        and dynamically manages learned rules in a separate file.
        """
        self.logger = logger or logging.getLogger(__name__)
        if Prolog is None:
            raise RuntimeError(
                "SymbolicReasoner requires SWI-Prolog and PySWIP, which are "
                f"not available: {PYSWIP_IMPORT_ERROR}"
            )
        try:
            self.prolog = Prolog()
            self.logger.info("PySWIP Prolog instance initialized.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize PySWIP Prolog instance: {e}", exc_info=True)
            raise RuntimeError(f"Prolog initialization failed: {e}") from e

        self.rules_path = os.path.abspath(rules_path)  # Store absolute path to base rules
        self.rules_dir = os.path.dirname(self.rules_path)
        self.learned_rules_path = os.path.splitext(self.rules_path)[0] + "_learned.pl"

        self.learned_rules: Dict[str, Dict[str, Any]] = {}  # Store learned rules in memory {rule_string: metadata}
        # Track rules by body so semantically identical clauses are merged even if
        # they were generated under different rule names across training cycles.
        self._body_to_key: Dict[str, str] = {}
        self.model = model  # Expect model instance to be passed
        self.input_shape = input_shape
        self.rule_activations: List[Dict[str, Any]] = []  # History of activations per step
        self.state_history_limit = 10  # Keep short history for change calculation
        self.state_history: List[Dict[str, Any]] = []
        self._sensor_thresholds: Dict[str, Dict[str, float]] = {}
        self._temporal_thresholds: Dict[str, Dict[str, float]] = {}

        if not os.path.exists(self.rules_path):
            self.logger.error(f"Base Prolog rules file not found at: {self.rules_path}")
            raise FileNotFoundError(f"Base Prolog rules file not found at: {self.rules_path}")

        self._load_prolog_files([
            os.path.basename(self.rules_path),
            os.path.basename(self.learned_rules_path)
        ])

        self._load_dynamic_rules_from_file()

        if self.model is not None:
            self.logger.info("Model provided directly to SymbolicReasoner.")
        else:
            self.logger.warning("No neural model provided to SymbolicReasoner. Rule extraction will be skipped.")

    def _load_prolog_files(self, files: List[str]):
        """
        Load Prolog files, handling potential errors during consult.
        Ensures the learned rules file exists.
        """
        for file in files:
            path = os.path.join(self.rules_dir, file)

            if file == os.path.basename(self.learned_rules_path) and not os.path.exists(path):
                try:
                    with open(path, 'w') as f:
                        f.write(f"% Dynamically learned rules for ANSR-DT - Automatically managed\n")
                        f.write(":- discontiguous(neural_rule/0).\n")
                        f.write(":- discontiguous(gradient_rule/0).\n")
                        f.write(":- discontiguous(pattern_rule/0).\n")
                        f.write(":- discontiguous(abstract_pattern/0).\n\n")
                    self.logger.info(f"Created empty learned rules file: {path}")
                except IOError as e:
                    self.logger.error(f"Failed to create learned rules file {path}: {e}")

            if os.path.exists(path):
                try:
                    prolog_path = path.replace("\\", "/")
                    query = f"consult('{prolog_path}')"
                    result = list(self.prolog.query(query))
                    self.logger.info(f"Consulted Prolog file: {path}")
                except Exception as e:
                    self.logger.error(f"Error consulting {path}: {type(e).__name__} - {e}", exc_info=True)
                    raise RuntimeError(f"Failed to consult Prolog file: {path}") from e
            else:
                if file != os.path.basename(self.learned_rules_path):
                    self.logger.warning(f"Prolog file not found, skipping: {path}")

    def _load_dynamic_rules_from_file(self):
        """Loads existing dynamic rules from the separate learned rules file.

        Deduplicates by rule body: if two rules share the same body but have
        different names, the one with higher confidence (or more activations as
        tiebreaker) is kept and the duplicate is discarded.
        """
        self.learned_rules = {}
        self._body_to_key = {}
        try:
            if not os.path.exists(self.learned_rules_path):
                self.logger.info(f"Learned rules file {self.learned_rules_path} not found, starting fresh.")
                return

            with open(self.learned_rules_path, 'r') as f:
                content = f.readlines()

            duplicates_removed = 0
            for line_num, line in enumerate(content):
                stripped_line = line.strip()
                if not stripped_line or (stripped_line.startswith('%') and 'Confidence:' not in stripped_line) or stripped_line.startswith(':- discontiguous'):
                    continue

                confidence = 0.7
                timestamp_str = datetime.now().isoformat()
                activations = 0
                rule_part = stripped_line
                comment_part = ""

                if '%' in stripped_line:
                    try:
                        rule_part, comment_part = stripped_line.split('%', 1)
                        rule_part = rule_part.strip()
                        comment_part = comment_part.strip()
                        conf_match = re.search(r'Confidence:\s*([0-9.]+)', comment_part, re.IGNORECASE)
                        ts_match = re.search(r'Extracted:\s*([\d\-T:.+Z]+)', comment_part, re.IGNORECASE)
                        act_match = re.search(r'Activations:\s*(\d+)', comment_part, re.IGNORECASE)

                        if conf_match:
                            confidence = float(conf_match.group(1))
                        if ts_match:
                            timestamp_str = ts_match.group(1)
                        if act_match:
                            activations = int(act_match.group(1))
                    except Exception as parse_error:
                        self.logger.warning(f"L{line_num+1} in {self.learned_rules_path}: Could not parse metadata '{comment_part}': {parse_error}")
                        if '%' in stripped_line:
                            rule_part = stripped_line.split('%', 1)[0].strip()
                        else:
                            rule_part = stripped_line

                if rule_part.endswith('.'):
                    rule_key = rule_part
                    body = _extract_rule_body(rule_key)

                    try:
                        timestamp_str_cleaned = timestamp_str.replace('Z', '+00:00')
                        timestamp_dt = datetime.fromisoformat(timestamp_str_cleaned)
                    except ValueError:
                        self.logger.warning(f"L{line_num+1}: Invalid ISO timestamp '{timestamp_str}' for rule '{rule_key}', using current time.")
                        timestamp_dt = datetime.now()

                    # Deduplicate at the clause-body level so the sidecar rule file
                    # remains compact and interpretable rather than name-fragmented.
                    if body and body in self._body_to_key:
                        existing_key = self._body_to_key[body]
                        existing_meta = self.learned_rules[existing_key]
                        # Keep the strongest representative for each symbolic
                        # body so repeated extractions do not fragment the rule base.
                        if (confidence > existing_meta['confidence'] or
                                (confidence == existing_meta['confidence'] and activations > existing_meta['activations'])):
                            # Replace the existing rule with this better one
                            del self.learned_rules[existing_key]
                            self.learned_rules[rule_key] = {
                                'confidence': confidence, 'timestamp': timestamp_dt,
                                'activations': activations + existing_meta['activations']
                            }
                            self._body_to_key[body] = rule_key
                        else:
                            # Merge activations into the existing rule
                            existing_meta['activations'] += activations
                        duplicates_removed += 1
                    elif rule_key not in self.learned_rules:
                        self.learned_rules[rule_key] = {
                            'confidence': confidence, 'timestamp': timestamp_dt, 'activations': activations
                        }
                        if body:
                            self._body_to_key[body] = rule_key
                elif rule_part and not rule_part.startswith(':-'):
                    self.logger.warning(f"L{line_num+1} in {self.learned_rules_path}: Skipping potential rule due to missing period: '{rule_part}'")

            if duplicates_removed > 0:
                self.logger.info(f"Deduplicated {duplicates_removed} rules with identical bodies. Kept {len(self.learned_rules)} unique rules.")
                self._rewrite_rules_file()
            self.logger.info(f"Loaded {len(self.learned_rules)} dynamic rules from {self.learned_rules_path}")

        except FileNotFoundError:
            self.logger.warning(f"Learned rules file {self.learned_rules_path} not found during load attempt.")
        except Exception as e:
            self.logger.error(f"Error loading dynamic rules from file {self.learned_rules_path}: {e}", exc_info=True)

    def _rewrite_rules_file(self):
        """Rewrites ONLY the learned rules file safely."""
        temp_learned_rules_path = self.learned_rules_path + ".tmp"
        try:
            # Rewrite the learned-rule sidecar atomically so partial writes do not
            # leave the Prolog state and on-disk rule set out of sync.
            with open(temp_learned_rules_path, 'w') as f_temp:
                f_temp.write(f"% Dynamically learned rules for ANSR-DT - Automatically managed\n")
                f_temp.write(f"% Last updated: {datetime.now().isoformat()}\n\n")
                f_temp.write(":- discontiguous(neural_rule/0).\n")
                f_temp.write(":- discontiguous(gradient_rule/0).\n")
                f_temp.write(":- discontiguous(pattern_rule/0).\n")
                f_temp.write(":- discontiguous(abstract_pattern/0).\n\n")

                sorted_rule_items = sorted(self.learned_rules.items())
                for rule, metadata in sorted_rule_items:
                    rule_str = rule if rule.endswith('.') else rule + '.'
                    formatted_rule = (
                        f"{rule_str}  "
                        f"% Confidence: {metadata.get('confidence', 0.0):.3f}, "
                        f"Extracted: {metadata.get('timestamp', datetime.now()).isoformat()}, "
                        f"Activations: {metadata.get('activations', 0)}\n"
                    )
                    f_temp.write(formatted_rule)

            os.replace(temp_learned_rules_path, self.learned_rules_path)
            self.logger.info(
                f"Safely rewrote learned rules file {self.learned_rules_path} with {len(self.learned_rules)} rules.")

            try:
                prolog_path = self.learned_rules_path.replace("\\", "/")
                query = f"consult('{prolog_path}')"
                list(self.prolog.query(query))
                self.logger.info(f"Re-consulted Prolog learned rules file: {prolog_path}")
            except Exception as e:
                self.logger.error(f"Failed to re-consult {prolog_path} after rewrite: {type(e).__name__} - {e}",
                                  exc_info=True)

        except Exception as e:
            self.logger.error(f"Error rewriting learned rules file {self.learned_rules_path}: {e}", exc_info=True)
            if os.path.exists(temp_learned_rules_path):
                try:
                    os.remove(temp_learned_rules_path)
                except OSError as rm_error:
                    self.logger.error(
                        f"Could not remove temporary learned rules file {temp_learned_rules_path}: {rm_error}")

    def _query_numeric_fact(self, predicate: str, subject: str) -> Optional[float]:
        """Read the numeric second argument of a binary Prolog fact."""
        try:
            solutions = list(self.prolog.query(f"{predicate}({subject}, Value)"))
            if solutions:
                return float(solutions[0]['Value'])
        except Exception as e:
            self.logger.debug(f"Could not query {predicate}/{subject}: {e}")
        return None

    def get_sensor_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Read the training-derived static thresholds from ``rules.pl``."""
        high, low = {}, {}
        for solution in self.prolog.query("sensor_high_threshold(S, V)."):
            high[str(solution['S'])] = float(solution['V'])
        for solution in self.prolog.query("sensor_low_threshold(S, V)."):
            low[str(solution['S'])] = float(solution['V'])
        self._sensor_thresholds = {'high': high, 'low': low}
        return self._sensor_thresholds

    def set_temporal_thresholds(self, thresholds: Dict[str, Dict[str, float]]) -> None:
        """Assert training-derived temporal thresholds into the Prolog KB."""
        self._temporal_thresholds = thresholds or {}
        list(self.prolog.query("retractall(temporal_threshold(_, _, _))"))
        for sensor, values in self._temporal_thresholds.items():
            if not re.match(r'^[a-z][a-zA-Z0-9_]*$', str(sensor)):
                continue
            for kind, value in values.items():
                if kind == 'moving_average_cycles' or kind == 'training_windows':
                    continue
                list(self.prolog.query(
                    f"assertz(temporal_threshold({sensor}, {kind}, {float(value):.6f}))"
                ))
        self.logger.info("Asserted temporal thresholds for %d sensors.",
                         len(self._temporal_thresholds))

    def _retract_temporal_facts(self) -> None:
        """Remove temporal facts that belong to the previous window."""
        for query in [
            "retractall(sensor_slope(_, _))",
            "retractall(sensor_relative_change(_, _))",
            "retractall(sensor_moving_average(_, _))",
            "retractall(sensor_std(_, _))",
            "retractall(sensor_persistence(_, _, _))",
        ]:
            list(self.prolog.query(query))

    def _assert_temporal_snapshot(self, window: np.ndarray,
                                  feature_names: List[str]) -> None:
        """Assert slope, relative change, moving average, std and persistence."""
        values = np.asarray(window, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(feature_names):
            self.logger.warning(
                "Skipping temporal snapshot: expected (n, %d) window, got %s.",
                len(feature_names), values.shape,
            )
            return
        if not np.isfinite(values).all():
            self.logger.warning("Skipping temporal snapshot with NaN/Inf values.")
            return

        high = self._sensor_thresholds.get('high', {})
        low = self._sensor_thresholds.get('low', {})
        features = window_temporal_features(values, feature_names, high, low)
        for name in feature_names:
            if not re.match(r'^[a-z][a-zA-Z0-9_]*$', str(name)):
                continue
            list(self.prolog.query(
                f"assertz(sensor_slope({name}, {features['slope'][name]:.6f}))"
            ))
            list(self.prolog.query(
                "assertz(sensor_relative_change("
                f"{name}, {features['relative_change'][name]:.6f}))"
            ))
            list(self.prolog.query(
                "assertz(sensor_moving_average("
                f"{name}, {features['moving_average'][name]:.6f}))"
            ))
            list(self.prolog.query(
                f"assertz(sensor_std({name}, {features['std'][name]:.6f}))"
            ))
            list(self.prolog.query(
                "assertz(sensor_persistence("
                f"{name}, above, {features['persistence_above'][name]}))"
            ))
            list(self.prolog.query(
                "assertz(sensor_persistence("
                f"{name}, below, {features['persistence_below'][name]}))"
            ))

    def _assert_sensor_snapshot(self, feature_names: List[str],
                                current_values: np.ndarray,
                                previous_values: np.ndarray) -> None:
        """Assert one window's last two timesteps as Prolog sensor facts."""
        for query in [
            "retractall(current_sensor_value(_, _))",
            "retractall(sensor_change(_, _))",
            "retractall(sensor_delta(_, _))",
        ]:
            list(self.prolog.query(query))

        for idx, name in enumerate(feature_names):
            if not re.match(r'^[a-z][a-zA-Z0-9_]*$', str(name)):
                continue
            current = float(current_values[idx])
            previous = float(previous_values[idx])
            delta = current - previous
            list(self.prolog.query(f"assertz(current_sensor_value({name}, {current:.6f}))"))
            list(self.prolog.query(f"assertz(sensor_change({name}, {abs(delta):.6f}))"))
            list(self.prolog.query(f"assertz(sensor_delta({name}, {delta:.6f}))"))

    def conditions_for_sensor_values(self, sensor_values: Dict[str, Any]) -> List[str]:
        """Return active base-rule predicates for one raw sensor snapshot."""
        numeric_values = {}
        for key, value in sensor_values.items():
            key_name = str(key)
            if not re.match(r'^[a-z][a-zA-Z0-9_]*$', key_name):
                continue
            try:
                numeric_values[key_name] = float(value)
            except (TypeError, ValueError):
                continue
        if not numeric_values:
            return []

        for query in [
            "retractall(current_sensor_value(_, _))",
            "retractall(sensor_change(_, _))",
            "retractall(sensor_delta(_, _))",
        ]:
            list(self.prolog.query(query))
        self._retract_temporal_facts()
        for key, value in numeric_values.items():
            list(self.prolog.query(f"assertz(current_sensor_value({key}, {value:.6f}))"))

        atomic_predicates = [
            'hpc_temp_elevated',
            'hpc_temp_critical',
            'hpc_static_pressure_high',
            'hpc_outlet_pressure_low',
            'fuel_flow_low',
            'coolant_bleed_low',
        ]
        conditions = [
            predicate for predicate in atomic_predicates
            if list(self.prolog.query(f"{predicate}."))
        ]
        if conditions:
            return sorted(set(conditions))

        for key in numeric_values:
            if list(self.prolog.query(f"feature_threshold({key}, _, high).")):
                conditions.append(f"feature_threshold({key}, _, high)")
            elif list(self.prolog.query(f"feature_threshold({key}, _, low).")):
                conditions.append(f"feature_threshold({key}, _, low)")

        if list(self.prolog.query("correlated(s4, s11).")):
            conditions.append("correlated(s4, s11)")
        return sorted(set(conditions))

    def extract_rules_from_neural_model(
            self,
            input_data: np.ndarray,
            feature_names: List[str],
            threshold: float = 0.7,
            model: Optional[Any] = None
    ) -> List[Tuple[str, float]]:
        """
        Extracts potential rule candidates from neural model predictions.
        Focuses on translating high-confidence anomaly predictions into symbolic preconditions.
        """
        try:
            model_to_use = model if model is not None else self.model
            if model_to_use is None:
                self.logger.error("No model available for rule extraction.")
                return []

            if hasattr(model_to_use, 'built') and not model_to_use.built:
                self.logger.warning("Keras model not built; attempting build for rule extraction.")
                if self.input_shape and input_data.shape[1:] == self.input_shape:
                    try:
                        _ = model_to_use.predict(input_data[:1].astype(np.float32), verbose=0)
                    except Exception as e:
                        self.logger.error(f"Failed to build model during rule extraction: {e}")
                        return []
                else:
                    self.logger.error("Cannot build model: input shape mismatch.")
                    return []

            if input_data.ndim == 2:
                if input_data.shape[0] == self.input_shape[0] and input_data.shape[1] == self.input_shape[1]:
                    input_data = np.expand_dims(input_data, axis=0)
                else:
                    self.logger.error(
                        f"Invalid 2D input shape: {input_data.shape}. Expected ({self.input_shape[0]}, {self.input_shape[1]}).")
                    return []
            elif input_data.ndim != 3:
                self.logger.error(f"Invalid input shape: {input_data.shape}. Expected 3D (batch, timesteps, features).")
                return []

            if input_data.shape[2] != len(feature_names):
                self.logger.error(
                    f"Input data feature dim ({input_data.shape[2]}) != feature_names len ({len(feature_names)}).")
                return []

            self.logger.debug(f"Rule extraction input data shape: {input_data.shape}")
            input_data = input_data.astype(np.float32)

            if hasattr(model_to_use, 'positive_proba'):
                predictions = np.asarray(model_to_use.positive_proba(input_data)).flatten()
            elif hasattr(model_to_use, 'predict_proba'):
                proba = np.asarray(model_to_use.predict_proba(input_data))
                predictions = proba[:, -1].flatten()
            else:
                try:
                    predictions = np.asarray(model_to_use.predict(input_data, verbose=0)).flatten()
                except TypeError:
                    predictions = np.asarray(model_to_use.predict(input_data)).flatten()
            anomaly_indices = np.where(predictions > threshold)[0]
            self.logger.info(
                f"Found {len(anomaly_indices)} sequences meeting rule extraction threshold (prediction > {threshold}).")

            if len(anomaly_indices) == 0:
                return []

            potential_new_rules = []
            current_learned_rule_count = len(self.learned_rules)
            unique_rule_bodies_this_batch = set()

            for i, sequence_idx in enumerate(anomaly_indices):
                sequence = input_data[sequence_idx]
                if sequence.shape[0] < 2:
                    self.logger.warning(
                        f"Skipping sequence index {sequence_idx}: Not enough timesteps ({sequence.shape[0]})")
                    continue

                current_values = sequence[-1, :]
                previous_values = sequence[-2, :]
                self._assert_sensor_snapshot(feature_names, current_values, previous_values)
                feature_conditions = []
                condition_confidences = []

                for feat_idx, feat_name in enumerate(feature_names):
                    try:
                        feat_value = float(current_values[feat_idx])
                        prev_value = float(previous_values[feat_idx])
                        if math.isnan(feat_value) or math.isinf(feat_value) or \
                                math.isnan(prev_value) or math.isinf(prev_value):
                            self.logger.debug(f"Skipping feature '{feat_name}' due to NaN/Inf.")
                            continue
                        change = abs(feat_value - prev_value)
                    except (IndexError, ValueError, TypeError) as e:
                        self.logger.warning(
                            f"Error processing feature '{feat_name}' (idx {feat_idx}) for rule gen at sequence {sequence_idx}: {e}")
                        continue

                    try:
                        if list(self.prolog.query(f"feature_threshold({feat_name}, _, high).")):
                            feature_conditions.append(f"feature_threshold({feat_name}, _, high)")
                            high_threshold = self._query_numeric_fact('sensor_high_threshold', feat_name)
                            if high_threshold is not None:
                                scale = max(abs(high_threshold) * 0.005, 1e-6)
                                condition_confidences.append(
                                    min(1.0, max(0.0, (feat_value - high_threshold) / scale)))
                        elif list(self.prolog.query(f"feature_threshold({feat_name}, _, low).")):
                            feature_conditions.append(f"feature_threshold({feat_name}, _, low)")
                            low_threshold = self._query_numeric_fact('sensor_low_threshold', feat_name)
                            if low_threshold is not None:
                                scale = max(abs(low_threshold) * 0.005, 1e-6)
                                condition_confidences.append(
                                    min(1.0, max(0.0, (low_threshold - feat_value) / scale)))

                        if list(self.prolog.query(f"feature_gradient({feat_name}, _, high).")):
                            feature_conditions.append(f"feature_gradient({feat_name}, _, high)")
                            gradient_threshold = self._query_numeric_fact('gradient_threshold', feat_name)
                            if gradient_threshold is not None and gradient_threshold > 0:
                                condition_confidences.append(
                                    min(1.0, max(0.0, (change - gradient_threshold) / gradient_threshold)))

                        if feat_value > prev_value and list(self.prolog.query(f"trend({feat_name}, increasing).")):
                            feature_conditions.append(f"trend({feat_name}, increasing)")
                            condition_confidences.append(0.8)
                        elif feat_value < prev_value and list(self.prolog.query(f"trend({feat_name}, decreasing).")):
                            feature_conditions.append(f"trend({feat_name}, decreasing)")
                            condition_confidences.append(0.8)

                    except Exception as e:
                        self.logger.warning(
                            f"Unexpected error querying Prolog for feature '{feat_name}': {type(e).__name__} - {e}")
                        continue

                try:
                    if list(self.prolog.query("correlated(s4, s11).")):
                        feature_conditions.append("correlated(s4, s11)")
                        condition_confidences.append(0.85)
                    if list(self.prolog.query("sequence_pattern(hpc_degradation_trend).")):
                        feature_conditions.append("sequence_pattern(hpc_degradation_trend)")
                        condition_confidences.append(0.9)
                except Exception as e:
                    self.logger.warning(f"Error querying correlation/sequence patterns: {type(e).__name__} - {e}")
                    continue

                if feature_conditions:
                    unique_sorted_conditions = sorted(list(set(feature_conditions)))
                    rule_body = ", ".join(unique_sorted_conditions)
                    if not rule_body:
                        continue

                    if rule_body in unique_rule_bodies_this_batch:
                        continue
                    unique_rule_bodies_this_batch.add(rule_body)

                    rule_name = f"neural_rule_{current_learned_rule_count + len(potential_new_rules) + 1}"
                    rule_string = f"{rule_name} :- {rule_body}."

                    neural_confidence = float(predictions[sequence_idx])
                    avg_condition_confidence = np.mean(condition_confidences) if condition_confidences else 0.0
                    final_confidence = (neural_confidence * 0.6) + (avg_condition_confidence * 0.4)
                    final_confidence = max(0.0, min(1.0, final_confidence))

                    potential_new_rules.append((rule_string, final_confidence))

            self.logger.info(
                f"Generated {len(potential_new_rules)} potential new rule candidates from {len(anomaly_indices)} sequences.")
            return potential_new_rules

        except Exception as e:
            self.logger.error(f"Error extracting rules from neural model: {e}", exc_info=True)
            return []

    def analyze_neural_patterns(
            self,
            anomalous_sequences: np.ndarray,
            normal_sequences: np.ndarray,
            feature_names: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Analyzes patterns in neural model activations to derive abstract rules.
        (Experimental - Requires careful validation and refinement).
        """
        try:
            pattern_rules = []
            if self.model is None:
                self.logger.error("Cannot analyze patterns: Model not available.")
                return []
            if anomalous_sequences.size == 0 or normal_sequences.size == 0:
                self.logger.warning("Insufficient sequences provided for pattern analysis.")
                return pattern_rules

            if not hasattr(self.model, 'layers'):
                self.logger.info("Model has no Keras layers; skipping activation-pattern analysis.")
                return pattern_rules

            self.logger.info("Activation-pattern analysis is unavailable for non-Keras models.")
            return pattern_rules

        except Exception as e:
            self.logger.error(f"Error in analyze_neural_patterns: {e}", exc_info=True)
            return []

    def update_rules(self,
                     potential_new_rules: List[Tuple[str, float]],
                     min_confidence: float = 0.7,
                     max_learned_rules: int = 100,
                     pruning_strategy: str = 'confidence'
                     ):
        """
        Adds new rules meeting confidence to the in-memory store,
        potentially prunes old rules, and triggers rewrite of the learned rules file.
        """
        try:
            added_count = 0
            updated_count = 0
            now = datetime.now()
            needs_rewrite = False

            for rule_string, instance_confidence in potential_new_rules:
                if not isinstance(rule_string, str) or not rule_string.endswith('.'):
                    self.logger.warning(f"Skipping invalid rule string format: {rule_string}")
                    continue

                try:
                    instance_confidence = float(instance_confidence)
                    if not (0.0 <= instance_confidence <= 1.0):
                        raise ValueError("Confidence out of range")
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"Invalid confidence value '{instance_confidence}' for rule '{rule_string}'. Skipping.")
                    continue

                if instance_confidence >= min_confidence:
                    body = _extract_rule_body(rule_string)

                    # Check for exact key match OR body-level duplicate
                    if rule_string in self.learned_rules:
                        self.learned_rules[rule_string]['timestamp'] = now
                        if instance_confidence > self.learned_rules[rule_string]['confidence']:
                            self.learned_rules[rule_string]['confidence'] = instance_confidence
                        updated_count += 1
                        needs_rewrite = True
                    elif body and body in self._body_to_key:
                        # Same body under a different name — update existing, skip adding duplicate
                        existing_key = self._body_to_key[body]
                        existing_meta = self.learned_rules[existing_key]
                        existing_meta['timestamp'] = now
                        if instance_confidence > existing_meta['confidence']:
                            existing_meta['confidence'] = instance_confidence
                        updated_count += 1
                        needs_rewrite = True
                        self.logger.debug(f"Deduplicated rule by body: '{rule_string}' matches existing '{existing_key}'")
                    else:
                        self.learned_rules[rule_string] = {
                            'confidence': instance_confidence,
                            'timestamp': now,
                            'activations': 0
                        }
                        if body:
                            self._body_to_key[body] = rule_string
                        added_count += 1
                        needs_rewrite = True

            if added_count > 0 or updated_count > 0:
                self.logger.info(
                    f"Processed {len(potential_new_rules)} potential rules. Added: {added_count}, Updated existing: {updated_count}.")

            if len(self.learned_rules) > max_learned_rules:
                num_to_remove = len(self.learned_rules) - max_learned_rules
                rules_to_remove_keys = []
                needs_rewrite = True

                self.logger.info(
                    f"Learned rules ({len(self.learned_rules)}) exceed limit ({max_learned_rules}). Pruning {num_to_remove} rules using '{pruning_strategy}' strategy.")

                if pruning_strategy == 'confidence':
                    sorted_rules = sorted(self.learned_rules.items(),
                                          key=lambda item: (item[1]['confidence'], item[1]['timestamp']))
                elif pruning_strategy == 'lru':
                    sorted_rules = sorted(self.learned_rules.items(), key=lambda item: item[1]['timestamp'])
                elif pruning_strategy == 'lra':
                    sorted_rules = sorted(self.learned_rules.items(),
                                          key=lambda item: (item[1]['activations'], item[1]['timestamp']))
                else:
                    self.logger.warning(f"Unknown pruning strategy '{pruning_strategy}'. Defaulting to 'confidence'.")
                    sorted_rules = sorted(self.learned_rules.items(),
                                          key=lambda item: (item[1]['confidence'], item[1]['timestamp']))

                rules_to_remove_keys = [item[0] for item in sorted_rules[:num_to_remove]]

                for rule_key in rules_to_remove_keys:
                    if rule_key in self.learned_rules:
                        del self.learned_rules[rule_key]
                self.logger.info(f"Pruned {len(rules_to_remove_keys)} rules. New count: {len(self.learned_rules)}.")

            if needs_rewrite:
                self._rewrite_rules_file()

        except Exception as e:
            self.logger.error(f"Error updating Prolog rules: {e}", exc_info=True)

    def get_rule_activations(self) -> List[Dict]:
        """Get the complete history of detailed rule activations recorded by the reasoner."""
        return list(self.rule_activations)

    def get_active_learned_rule_names(self) -> List[str]:
        """Return learned rule heads activated during the latest ``reason`` call."""
        if not self.rule_activations:
            return []
        return [
            detail.get('rule')
            for detail in self.rule_activations[-1].get('activated_rules_detailed', [])
            if detail.get('type') == 'learned' and detail.get('rule')
        ]

    def validate_learned_rules(
            self,
            input_data: np.ndarray,
            feature_names: List[str],
            labels: np.ndarray,
            min_precision: float = 0.9,
            min_support: int = 5,
            max_false_positive_rate: float = 0.05,
            apply: bool = False,
    ) -> Dict[str, Any]:
        """Validate learned rules on held-out validation windows.

        A rule is accepted only if it fires at least ``min_support`` times,
        reaches ``min_precision`` among its alerts and its false-positive rate
        over all negative windows stays below ``max_false_positive_rate``.
        When ``apply`` is true, rejected rules are removed from the learned
        rule store and the sidecar file is atomically rewritten.
        """
        X = np.asarray(input_data)
        y = np.asarray(labels, dtype=int).ravel()
        if X.ndim != 3:
            raise ValueError("input_data must have shape (windows, timesteps, features).")
        if len(X) != len(y):
            raise ValueError("input_data and labels must contain the same number of windows.")

        rule_keys = list(self.learned_rules.keys())
        if not rule_keys:
            return {
                'summary': {
                    'rules_evaluated': 0,
                    'accepted': 0,
                    'rejected': 0,
                    'min_precision': float(min_precision),
                    'min_support': int(min_support),
                    'max_false_positive_rate': float(max_false_positive_rate),
                },
                'rules': [],
                'applied': bool(apply),
            }

        heads = {}
        for rule_key in rule_keys:
            match = re.match(
                r"^\s*([a-zA-Z0-9_]+)\s*(?::-\s*.*)?\.",
                rule_key.strip(),
            )
            if match:
                heads[rule_key] = match.group(1)

        stats = {
            rule_key: {'support': 0, 'true_positives': 0, 'false_positives': 0}
            for rule_key in rule_keys
        }
        total_negatives = int(np.sum(y == 0))
        self._retract_temporal_facts()

        for index, sequence in enumerate(X):
            sequence = np.asarray(sequence)
            if sequence.shape[0] < 2:
                continue
            self._assert_sensor_snapshot(
                feature_names, sequence[-1], sequence[-2]
            )
            label = int(y[index])
            for rule_key, head in heads.items():
                try:
                    fired = bool(list(self.prolog.query(f"{head}.")))
                except Exception as error:
                    self.logger.warning(
                        "Could not validate learned rule %s: %s - %s",
                        head, type(error).__name__, error,
                    )
                    fired = False
                if not fired:
                    continue
                stats[rule_key]['support'] += 1
                if label == 1:
                    stats[rule_key]['true_positives'] += 1
                else:
                    stats[rule_key]['false_positives'] += 1

        rows = []
        for rule_key in rule_keys:
            rule_stats = stats[rule_key]
            support = rule_stats['support']
            true_positives = rule_stats['true_positives']
            false_positives = rule_stats['false_positives']
            precision = true_positives / support if support else 0.0
            false_positive_rate = (
                false_positives / total_negatives if total_negatives else 0.0
            )
            reasons = []
            if rule_key not in heads:
                reasons.append('unparseable_rule')
            if support < min_support:
                reasons.append('insufficient_support')
            if precision < min_precision:
                reasons.append('low_precision')
            if false_positive_rate > max_false_positive_rate:
                reasons.append('high_false_positive_rate')
            rows.append({
                'rule': rule_key,
                'support': support,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'precision': precision,
                'false_positive_rate': false_positive_rate,
                'accepted': not reasons,
                'rejection_reasons': reasons,
            })

        rejected = [row['rule'] for row in rows if not row['accepted']]
        if apply and rejected:
            for rule_key in rejected:
                self.learned_rules.pop(rule_key, None)
            self._body_to_key = {}
            for rule_key in self.learned_rules:
                body = _extract_rule_body(rule_key)
                if body:
                    self._body_to_key[body] = rule_key
            self._rewrite_rules_file()
            self.logger.info(
                "Removed %d learned rule(s) rejected on validation.",
                len(rejected),
            )

        return {
            'summary': {
                'rules_evaluated': len(rows),
                'accepted': len(rows) - len(rejected),
                'rejected': len(rejected),
                'min_precision': float(min_precision),
                'min_support': int(min_support),
                'max_false_positive_rate': float(max_false_positive_rate),
            },
            'rules': rows,
            'applied': bool(apply),
        }

    def reason(self, sensor_dict: Dict[str, Any],
               window: Optional[np.ndarray] = None) -> List[str]:
        """
        Apply symbolic reasoning rules based on the current sensor state.
        Asserts current facts, queries base and learned rules, and records activations.

        Args:
            sensor_dict (Dict[str, Any]): Dictionary containing current sensor readings
                                          and potentially other metrics (e.g., efficiency).
                                          Keys must match Prolog expectations (e.g., 'temperature').
            window (Optional[np.ndarray]): Full ``(window_size, features)`` window used
                                          for temporal rules. If omitted, temporal facts are cleared.

        Returns:
            List[str]: A list of human-readable insights derived from activated rules.
        """
        insights = []
        activated_rules_details = []

        if not isinstance(sensor_dict, dict) or not sensor_dict:
            self.logger.error("reason() requires a non-empty sensor dictionary.")
            return []

        try:
            current_values = {}
            for key, value in sensor_dict.items():
                if value is None:
                    continue
                key_name = str(key)
                if not re.match(r'^[a-z][a-zA-Z0-9_]*$', key_name):
                    self.logger.warning(f"Ignoring invalid sensor name: {key!r}")
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    self.logger.warning(f"Ignoring non-numeric sensor value for {key_name}: {value!r}")
                    continue
                if math.isnan(numeric_value) or math.isinf(numeric_value):
                    self.logger.warning(f"NaN/Inf found for key '{key_name}'. Replacing with 0.")
                    numeric_value = 0.0
                current_values[key_name] = numeric_value
            if not current_values:
                self.logger.error("No numeric sensor values available for reasoning.")
                return []
            self.logger.debug(f"Current values for reasoning: {current_values}")
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid sensor value format provided to reason(): {e}")
            return []

        previous_values = self.state_history[-1] if self.state_history else None
        self.state_history.append(current_values)
        if len(self.state_history) > self.state_history_limit:
            self.state_history.pop(0)

        changes = {}
        deltas = {}
        if previous_values:
            for key, current_val in current_values.items():
                previous_val = previous_values.get(key)
                if isinstance(previous_val, (int, float)):
                    delta = float(current_val) - float(previous_val)
                    deltas[key] = delta
                    changes[key] = abs(delta)
        self.logger.debug(f"Calculated changes: {changes}")

        try:
            retract_queries = [
                "retractall(current_sensor_value(_, _))",
                "retractall(sensor_change(_, _))",
                "retractall(sensor_delta(_, _))",
            ]
            for q in retract_queries:
                list(self.prolog.query(q))

            for key, value in current_values.items():
                list(self.prolog.query(f"assertz(current_sensor_value({key}, {value:.6f}))"))
            for key, change in changes.items():
                list(self.prolog.query(f"assertz(sensor_change({key}, {change:.6f}))"))
                list(self.prolog.query(f"assertz(sensor_delta({key}, {deltas[key]:.6f}))"))

            self._retract_temporal_facts()
            if window is not None:
                self._assert_temporal_snapshot(window, list(current_values.keys()))

            self.logger.debug("Asserted facts into Prolog KB")

        except Exception as e:
            self.logger.error(f"Error asserting facts: {type(e).__name__} - {e}", exc_info=True)
            return []

        base_queries = {
            "HPC outlet temperature elevated": "hpc_temp_elevated",
            "HPC outlet temperature critical": "hpc_temp_critical",
            "HPC static pressure high": "hpc_static_pressure_high",
            "HPC outlet pressure low": "hpc_outlet_pressure_low",
            "Fuel flow low": "fuel_flow_low",
            "Coolant bleed low": "coolant_bleed_low",
            "HPC degradation pattern": "hpc_degradation_pattern",
            "HPC critical pattern": "hpc_critical_pattern",
            "HPC thermal overload": "hpc_thermal_overload",
            "HPC fuel starvation": "hpc_fuel_starvation",
            "HPC temperature rising trend": "temporal_trend(s4, increasing)",
            "HPC outlet pressure falling trend": "temporal_trend(s7, decreasing)",
            "HPC static pressure persistent high": "temporal_persistence(s11, above)",
            "HPC thermal-pressure degradation": "temporal_thermal_pressure_degradation",
            "HPC temperature variability high": "temporal_high_variability(s4)",
            "HPC moving-average temperature high": "temporal_moving_average_high(s4)",
        }
        for insight_desc, query_string in base_queries.items():
            try:
                solutions = list(self.prolog.query(query_string))
                if solutions:
                    insights.append(insight_desc)

                    rule_name = query_string.split('(')[0]
                    activated_rules_details.append({
                        'rule': rule_name + "_base",
                        'confidence': 1.0,
                        'type': 'base',
                        'details': solutions[0] if solutions else {}
                    })
                    self.logger.debug(f"Base Rule Activated: {insight_desc}")
            except Exception as e:
                self.logger.warning(
                    f"Error querying base rule '{insight_desc}' ({query_string}): {type(e).__name__} - {e}")

        updated_learned_rules = False
        for rule_string, metadata in list(self.learned_rules.items()):
            try:
                rule_head_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*(?::-\s*.*)?\.", rule_string.strip())
                if not rule_head_match:
                    self.logger.warning(f"Could not parse head from learned rule: {rule_string}")
                    continue
                rule_head = rule_head_match.group(1)

                query_string = f"{rule_head}."
                solutions = list(self.prolog.query(query_string))

                if solutions:
                    confidence = metadata.get('confidence', 0.0)
                    insight = f"Learned Rule Activated: {rule_head} (Conf: {confidence:.2f})"
                    insights.append(insight)
                    activated_rules_details.append({
                        'rule': rule_head,
                        'confidence': confidence,
                        'type': 'learned',
                        'details': solutions[0] if solutions else {}
                    })
                    current_activations = metadata.get('activations', 0)
                    self.learned_rules[rule_string]['activations'] = current_activations + 1
                    self.learned_rules[rule_string]['last_activated'] = datetime.now()
                    updated_learned_rules = True
                    self.logger.debug(
                        f"Learned Rule Activated: {insight} (Total Activations: {current_activations + 1})")

            except Exception as e:
                self.logger.warning(f"Failed applying learned rule {rule_string}: {type(e).__name__} - {e}")

        activation_record = {
            'timestep': len(self.rule_activations),
            'activated_rules_detailed': activated_rules_details,
            'activated_rule_names': [r['rule'] for r in activated_rules_details],
            'insights_generated': insights,
            'sensor_values': current_values,
            'timestamp': datetime.now().isoformat()
        }
        self.rule_activations.append(activation_record)
        history_limit = 1000
        if len(self.rule_activations) > history_limit:
            self.rule_activations.pop(0)

        self.logger.debug(f"Reasoning complete. Generated {len(insights)} insights.")
        return insights

    def extract_rules_from_neural_model(
        self,
        model: Any,
        input_data: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.6,
    ) -> List[Tuple[str, float]]:
        """Extract symbolic threshold rules from a fitted neural/tabular model.

        For tree-based estimators (RandomForest, etc.) this walks the decision
        nodes and turns each ``(feature, threshold)`` split into a Prolog rule
        of the form::

            hpc_neural_<sensor>_high :- sensor_<sensor>(V), V > <threshold>.

        The rules are asserted into the Prolog knowledge base so they are
        immediately available to :meth:`reason`, and returned as
        ``(rule_string, confidence)`` pairs for :meth:`update_rules`.

        Returns an empty list when the model exposes no inspectable structure
        (e.g. a black-box Keras model) or when extraction fails.
        """
        rules: List[Tuple[str, float]] = []
        try:
            estimator = getattr(model, "model", model)
            base_estimator = getattr(estimator, "base_estimator_", None)
            if base_estimator is not None:
                estimator = base_estimator
            estimators = getattr(estimator, "estimators_", None)
            if estimators is None:
                single = getattr(estimator, "tree_", None)
                estimators = [single] if single is not None else []
            if not estimators:
                return rules
            seen_bodies: set = set()
            for tree in estimators:
                if tree is None:
                    continue
                tree_ = tree.tree_ if hasattr(tree, "tree_") else tree
                n_nodes = int(tree_.node_count)
                children_left = tree_.children_left
                children_right = tree_.children_right
                feature = tree_.feature
                threshold_arr = tree_.threshold
                for node_id in range(n_nodes):
                    if children_left[node_id] == children_right[node_id]:
                        continue  # leaf
                    fid = int(feature[node_id])
                    if fid < 0 or fid >= len(feature_names):
                        continue
                    thr = float(threshold_arr[node_id])
                    if math.isnan(thr) or math.isinf(thr):
                        continue
                    name = str(feature_names[fid])
                    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
                    head = f"hpc_neural_{safe}_high"
                    body = f"sensor_{safe}(V), V > {thr:.6f}"
                    rule_string = f"{head} :- {body}."
                    if body in seen_bodies:
                        continue
                    seen_bodies.add(body)
                    # Assert into the live Prolog KB so reason() can fire it.
                    try:
                        self.prolog.assertz(rule_string)
                    except Exception:
                        pass
                    confidence = float(min(0.95, float(threshold) + 0.1))
                    rules.append((rule_string, confidence))
            self.logger.info(
                "Extracted %d neural threshold rules from %d estimator(s).",
                len(rules), len(estimators),
            )
        except Exception as error:
            self.logger.warning(
                "extract_rules_from_neural_model failed: %s", error
            )
        return rules

    def analyze_neural_patterns(
        self,
        model: Any,
        input_data: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.6,
    ) -> List[Tuple[str, float]]:
        """Convenience wrapper around :meth:`extract_rules_from_neural_model`.

        Kept for API compatibility with the original ANSR-DT rule-extraction
        path. Returns ``(rule_string, confidence)`` pairs.
        """
        return self.extract_rules_from_neural_model(
            model=model,
            input_data=input_data,
            feature_names=feature_names,
            threshold=threshold,
        )

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Retrieve statistics about the current rule base (base + learned in memory)."""
        num_base_rules_queried = 10
        num_learned_rules = len(self.learned_rules)

        learned_stats = {
            'count': num_learned_rules,
            'high_confidence_count': 0,
            'average_confidence': 0.0,
            'total_activations': 0,
            'confidence_distribution': {}
        }

        if num_learned_rules > 0:
            confidences = [meta.get('confidence', 0.0) for meta in self.learned_rules.values()]
            activations = [meta.get('activations', 0) for meta in self.learned_rules.values()]

            learned_stats['high_confidence_count'] = sum(1 for conf in confidences if conf >= 0.7)
            learned_stats['average_confidence'] = float(np.mean(confidences)) if confidences else 0.0
            learned_stats['total_activations'] = sum(activations)

            bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]
            try:
                valid_confidences = [c for c in confidences if isinstance(c, (int, float)) and not math.isnan(c)]
                if valid_confidences:
                    hist, _ = np.histogram(valid_confidences, bins=bins)
                    learned_stats['confidence_distribution'] = {
                        f"{bins[i]:.1f}-{bins[i + 1]:.1f}": int(count) for i, count in enumerate(hist)
                    }
                else:
                    learned_stats['confidence_distribution'] = {f"{bins[i]:.1f}-{bins[i + 1]:.1f}": 0 for i in
                                                                range(len(bins) - 1)}
            except Exception as hist_err:
                self.logger.warning(f"Could not generate confidence distribution histogram: {hist_err}")
                learned_stats['confidence_distribution'] = {"error": str(hist_err)}

        stats = {
            'total_rules_managed': num_learned_rules,
            'base_rules_queried_approx': num_base_rules_queried,
            'learned_rules': learned_stats,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.debug(f"Rule Statistics: {stats}")
        return stats