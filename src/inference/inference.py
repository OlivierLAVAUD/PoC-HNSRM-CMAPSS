# src/inference/inference.py

import os
import sys
import logging
import numpy as np
import json
import argparse
from typing import Dict, Tuple, List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.model_utils import load_scaler
from src.models.light_classifier import LightClassifier
from src.models.rul_regressor import RULLightGBMRegressor
from src.fusion.probability_fusion import (
    ProbabilityFusion,
    symbolic_rule_score,
)
from src.reasoning.prob_query import RULES_PATH as PROB_RULES_PATH, evaluate_state
from src.reasoning.reasoning import SymbolicReasoner
from src.config.config_manager import load_config
from src.data.cmapss_loader import CMAPSSLoader
from src.data.data_loader import DataLoader


class InferencePipeline:
    """Pipeline for running inference on data using trained ANSR-DT artifacts."""

    def __init__(self, config_path: str, logger: logging.Logger,
                 model_architecture: str = 'auto',
                 use_fusion: str = 'auto'):
        self.config_path = os.path.abspath(config_path)
        self.config = load_config(self.config_path)
        self.logger = logger
        self.window_size = self.config['model']['window_size']
        self.feature_names = self.config['model']['feature_names']
        self.input_shape = tuple(self.config['model']['input_shape'])

        config_dir = os.path.dirname(self.config_path)
        self.project_root = os.path.dirname(config_dir)
        self.results_dir = self._resolve_path(self.config['paths'].get('results_dir', 'results'))
        self.rules_path = self._resolve_path(self.config['paths']['reasoning_rules_path'])

        configured = str(
            self.config.get('model', {}).get('architecture', 'random_forest')
        ).lower()
        if configured in ('lgbm', 'lightgbm_classifier'):
            configured = 'lightgbm'
        if configured in ('light_classifier', 'cnn_lstm'):
            configured = 'random_forest'
        if model_architecture in (None, 'auto'):
            model_architecture = configured
        self.model_architecture = model_architecture
        if self.model_architecture not in {'random_forest', 'lightgbm'}:
            raise ValueError(
                f"Unsupported inference model: {self.model_architecture!r}"
            )

        model_name = (
            'lightgbm_classifier'
            if self.model_architecture == 'lightgbm' else 'light_classifier'
        )
        model_path = os.path.join(
            self.results_dir, 'models', f'{model_name}.joblib'
        )
        scaler_path = os.path.join(self.results_dir, 'scaler.pkl')

        self.logger.info(f"Resolved model path: {model_path}")
        self.logger.info(f"Resolved scaler path: {scaler_path}")

        self.model = LightClassifier.load(model_path, logger=logger)
        self.needs_feature_bundle = bool(
            getattr(self.model, 'uses_temporal_features', False) is True
            or isinstance(
                getattr(self.model, 'rul_regressor', None),
                RULLightGBMRegressor,
            )
        )
        self.scaler = None
        if not self.needs_feature_bundle and os.path.exists(scaler_path):
            self.scaler = load_scaler(scaler_path, logger)
        elif not self.needs_feature_bundle:
            self.logger.warning(
                "Scaler not found at %s; inference will use raw features.",
                scaler_path,
            )

        fusion_config = self.config.get('fusion', {})
        fusion_targets = {
            str(name).lower()
            for name in fusion_config.get(
                'apply_to_architectures', ['lightgbm']
            )
        }
        fusion_requested = (
            str(use_fusion).lower() != 'off'
            and bool(fusion_config.get('enabled', False))
            and self.model_architecture in fusion_targets
        )
        self.fusion: Optional[ProbabilityFusion] = None
        self.fusion_threshold: Optional[float] = None
        self.fusion_model_path: Optional[str] = None
        if fusion_requested:
            fusion_model_path = os.path.join(
                self.results_dir, 'models', 'probability_fusion.joblib'
            )
            fusion_threshold_path = os.path.join(
                self.results_dir, 'fusion_threshold.json'
            )
            if (os.path.exists(fusion_model_path)
                    and os.path.exists(fusion_threshold_path)):
                self.fusion = ProbabilityFusion.load(
                    fusion_model_path, logger=logger
                )
                with open(fusion_threshold_path) as file:
                    self.fusion_threshold = float(json.load(file)['threshold'])
                self.fusion_model_path = fusion_model_path
                logger.info(
                    "Loaded probability fusion (threshold=%.4f) from %s",
                    self.fusion_threshold, fusion_model_path,
                )
            elif str(use_fusion).lower() == 'on':
                raise FileNotFoundError(
                    "Fusion was requested but artifacts are missing at "
                    f"{fusion_model_path} or {fusion_threshold_path}."
                )
            else:
                logger.warning(
                    "Fusion artifacts not found in %s; using the neural "
                    "decision threshold.", self.results_dir,
                )
        self.reasoner: Optional[SymbolicReasoner] = None

        if self.config.get('symbolic_reasoning', {}).get('enabled', False):
            self.reasoner = SymbolicReasoner(
                rules_path=self.rules_path,
                input_shape=self.input_shape,
                model=self.model,
                logger=self.logger,
            )
            temporal_path = os.path.join(self.results_dir, 'temporal_thresholds.json')
            if os.path.exists(temporal_path):
                try:
                    with open(temporal_path) as file:
                        self.reasoner.set_temporal_thresholds(json.load(file))
                except (OSError, ValueError, TypeError) as error:
                    self.logger.warning(
                        "Could not load temporal thresholds from %s: %s",
                        temporal_path, error,
                    )

    def _resolve_path(self, path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(self.project_root, path_value))

    def _sanitize_inputs(self, data: np.ndarray) -> np.ndarray:
        """Apply the configured NaN/Inf policy before inference."""
        data = np.asarray(data, dtype=np.float32)
        count = int((~np.isfinite(data)).sum())
        if not count:
            return data
        policy = self.config.get('preprocessing', {}).get(
            'non_finite_policy', 'zero'
        )
        if policy == 'error':
            raise ValueError(
                f"Input contains {count} NaN/Inf value(s) and "
                "preprocessing.non_finite_policy is 'error'."
            )
        if policy not in {'zero', 'replace_zero'}:
            raise ValueError(f"Unsupported non-finite policy: {policy!r}")
        self.logger.warning(
            "Replacing %d NaN/Inf inference value(s) with zeros.", count
        )
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    def load_and_preprocess_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load raw data, create sequences, and scale them."""
        try:
            resolved_data_file = data_file if os.path.isabs(data_file) else os.path.join(self.project_root, data_file)
            data_loader = DataLoader(resolved_data_file, self.window_size)
            X, y = data_loader.load_data()
            X_seq, y_seq = data_loader.create_sequences(X, y)
            X_seq = self._sanitize_inputs(X_seq)
            X_model = self._prepare_model_input(X_seq)
            return X_model, X_seq, y_seq
        except Exception as e:
            self.logger.error(f"Failed to load and preprocess data: {e}")
            raise

    def _prepare_model_input(self, X_raw: np.ndarray) -> np.ndarray:
        """Apply the same feature contract used during training."""
        if self.needs_feature_bundle:
            return self.model.prepare_features(X_raw)
        if self.scaler is not None:
            flat = X_raw.reshape(-1, X_raw.shape[-1])
            expected = getattr(self.scaler, 'n_features_in_', X_raw.shape[-1])
            if X_raw.shape[-1] != expected:
                raise ValueError(
                    f"Scaler expects {expected} features but data has "
                    f"{X_raw.shape[-1]}."
                )
            return self.scaler.transform(flat).reshape(X_raw.shape)
        return X_raw

    def _fusion_scores(self, X_raw: np.ndarray,
                       neural_scores: np.ndarray,
                       insights: Optional[List[list]] = None) -> np.ndarray:
        """Fuse neural, symbolic, ProbLog and RUL signals for raw windows."""
        if self.fusion is None:
            return neural_scores
        if insights is None:
            insights = self.get_symbolic_insights(X_raw)

        base_features = self.model.prepare_base_features(X_raw)
        if self.model.rul_regressor is not None:
            predicted_rul = self.model.rul_regressor.predict(base_features)
        else:
            predicted_rul = np.zeros(len(X_raw), dtype=float)
        symbolic = np.asarray(
            [symbolic_rule_score(item) for item in insights], dtype=float
        )
        problog_risks = np.zeros(len(X_raw), dtype=float)
        problog_config = self.config.get('problog', {})
        if not os.path.exists(PROB_RULES_PATH):
            self.logger.warning(
                "ProbLog rules not found at %s; fusion uses zero risk.",
                PROB_RULES_PATH,
            )
        else:
            with open(PROB_RULES_PATH) as file:
                problog_program = file.read()
            learned_rules = (
                self.reasoner.get_active_learned_rule_names()
                if self.reasoner is not None else []
            )
            problog_logger = logging.getLogger('problog')
            previous_problog_level = problog_logger.level
            problog_logger.setLevel(logging.WARNING)
            try:
                for index, window in enumerate(X_raw):
                    sensor_dict = {
                        name: float(window[-1, feature_index])
                        for feature_index, name in enumerate(self.feature_names)
                    }
                    problog = evaluate_state(
                        problog_program, sensor_dict,
                        learned_rules=learned_rules,
                        learned_rule_weight=float(
                            problog_config.get('learned_rule_weight', 0.15)
                        ),
                        max_active_learned_rules=int(
                            problog_config.get('max_active_learned_rules', 3)
                        ),
                    )
                    problog_risks[index] = float(problog['failure_risk'])
            finally:
                problog_logger.setLevel(previous_problog_level)
        matrix = ProbabilityFusion.build_matrix(
            neural_scores, symbolic, problog_risks, predicted_rul,
            feature_order=self.fusion.feature_order,
        )
        return self.fusion.positive_proba(matrix)

    def _decision_threshold(self) -> float:
        """Return the fusion threshold when active, else the neural one."""
        if self.fusion_threshold is not None:
            return float(self.fusion_threshold)
        return float(getattr(self.model, 'decision_threshold', 0.5))

    def get_predictions(self, X_scaled: np.ndarray,
                        X_raw: Optional[np.ndarray] = None,
                        insights: Optional[List[list]] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Get failure probabilities and binary predictions for model features."""
        y_scores = self.model.positive_proba(X_scaled).ravel()
        if self.fusion is not None and X_raw is not None:
            y_scores = self._fusion_scores(X_raw, y_scores, insights=insights)
        threshold = self._decision_threshold()
        y_pred = (y_scores >= threshold).astype(int)
        return y_scores, y_pred

    def get_symbolic_insights(self, X_seq: np.ndarray) -> List[list]:
        """Get symbolic reasoning insights over the last timestep of each sequence."""
        insights: List[list] = []
        if not self.reasoner:
            return insights

        for i in range(len(X_seq)):
            sensor_dict = {
                feature_name: float(X_seq[i][-1, feature_idx])
                for feature_idx, feature_name in enumerate(self.feature_names)
            }
            self.reasoner.state_history = []
            insight = self.reasoner.reason(sensor_dict, window=X_seq[i])
            insights.append(insight)
        return insights

    def load_cmapss_windows(self, data_file: str):
        """Load unlabeled C-MAPSS text/parquet data and build inference windows."""
        resolved = self._resolve_path(data_file)
        loader = CMAPSSLoader(
            resolved, window_size=self.window_size,
            feature_names=self.feature_names,
            logger=self.logger,
        )
        X_raw = self._sanitize_inputs(loader.load_unlabeled_data())
        return X_raw, loader

    def run_cmapss_inference(self, data_file: str,
                             output_path: str) -> Dict[str, object]:
        """Run inference on a raw C-MAPSS FD001 file without true RUL labels."""
        try:
            X_raw, loader = self.load_cmapss_windows(data_file)
            X_model = self._prepare_model_input(X_raw)
            insights = self.get_symbolic_insights(X_raw)
            y_scores, y_pred = self.get_predictions(
                X_model, X_raw=X_raw, insights=insights
            )

            results = {
                'mode': 'cmapss',
                'unit_ids': loader.unit_ids.tolist(),
                'cycles': loader.window_end_cycles.tolist(),
                'failure_probabilities': y_scores.tolist(),
                'predictions': y_pred.tolist(),
                'insights': insights,
                'decision_threshold': self._decision_threshold(),
                'metadata': {
                    'data_file': data_file,
                    'resolved_results_dir': self.results_dir,
                    'resolved_rules_path': self.rules_path,
                    'feature_names': self.feature_names,
                    'window_size': self.window_size,
                    'architecture': self.model_architecture,
                    'uses_temporal_features': bool(
                        getattr(self.model, 'uses_temporal_features', False)
                    ),
                    'uses_rul_feature': bool(
                        getattr(self.model, 'uses_rul_feature', False)
                    ),
                    'fusion_applied': self.fusion is not None,
                    'fusion_model_path': self.fusion_model_path,
                    'fusion_threshold': self.fusion_threshold,
                    'timestamp': str(np.datetime64('now')),
                },
            }

            resolved_output_path = self._resolve_path(output_path)
            os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)
            with open(resolved_output_path, 'w') as file:
                json.dump(results, file, indent=2)
            self.logger.info("C-MAPSS inference results saved to %s", resolved_output_path)
            return results
        except Exception as error:
            self.logger.error("Error during C-MAPSS inference: %s", error)
            raise

    def run_inference(self, data_file: str, output_path: str) -> Dict[str, object]:
        """Run complete inference pipeline and persist results."""
        try:
            X_scaled, X_seq, y_seq = self.load_and_preprocess_data(data_file)
            self.logger.info(f"Data loaded and preprocessed with shape {X_scaled.shape}")

            insights = self.get_symbolic_insights(X_seq)
            y_scores, y_pred = self.get_predictions(
                X_scaled, X_raw=X_seq, insights=insights
            )
            self.logger.info(f"Predictions generated for {len(y_pred)} samples")

            results = {
                'y_true': y_seq.tolist(),
                'y_scores': y_scores.tolist(),
                'y_pred': y_pred.tolist(),
                'insights': insights,
                'metadata': {
                    'data_file': data_file,
                    'resolved_results_dir': self.results_dir,
                    'resolved_rules_path': self.rules_path,
                    'decision_threshold': self._decision_threshold(),
                    'fusion_applied': self.fusion is not None,
                    'fusion_model_path': self.fusion_model_path,
                    'fusion_threshold': self.fusion_threshold,
                    'timestamp': str(np.datetime64('now')),
                },
            }

            resolved_output_path = output_path if os.path.isabs(output_path) else os.path.join(self.project_root, output_path)
            os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)
            with open(resolved_output_path, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Inference results saved to {resolved_output_path}")
            return results
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='ANSR-DT Inference Module')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the .npz data file')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml')
    parser.add_argument('--output', type=str, default='results/inference_results.json', help='Path to save inference results')
    parser.add_argument('--mode', choices=['legacy', 'cmapss'], default='legacy',
                        help='Input format: legacy .npz or raw C-MAPSS text/parquet')
    parser.add_argument('--model', choices=['auto', 'random_forest', 'lightgbm'],
                        default='auto',
                        help='Architecture artifact to load (default: config).')
    parser.add_argument('--fusion', choices=['auto', 'on', 'off'],
                        default='auto',
                        help='Use the validation-fitted fusion model if available.')
    args = parser.parse_args()

    logger = logging.getLogger('InferenceLogger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    pipeline = InferencePipeline(
        args.config, logger, model_architecture=args.model,
        use_fusion=args.fusion,
    )
    if args.mode == 'cmapss':
        pipeline.run_cmapss_inference(args.data_file, args.output)
    else:
        pipeline.run_inference(args.data_file, args.output)


if __name__ == "__main__":
    main()
