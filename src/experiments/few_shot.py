# src/experiments/few_shot.py
"""Few-shot target adaptation for one C-MAPSS dataset.

This is deliberately not a zero-shot protocol.  It uses target *training*
engines only, split by engine into calibration and validation subsets.  The
official target test engines are loaded once, after the threshold is frozen,
and are used exclusively for reporting.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.data.cmapss_loader import CMAPSSLoader
from src.evaluation.alert_budget import (
    evaluate_budget_usage,
    select_alert_budget_threshold,
)
from src.evaluation.calibration import calibration_metrics
from src.evaluation.engine_metrics import compute_engine_metrics
from src.evaluation.episode_metrics import compute_episode_metrics
from src.evaluation.thresholding import select_threshold
from src.experiments.few_shot_audit import (
    build_few_shot_audit,
    save_few_shot_audit,
)
from src.models.light_classifier import LightClassifier

FS_CSV_FIELDS = [
    'split', 'threshold', 'roc_auc', 'average_precision', 'f1', 'precision',
    'recall', 'brier_score', 'ece', 'real_episodes', 'detected_episodes',
    'episode_recall', 'episode_precision', 'false_alarm_episodes',
    'false_alarms_per_100_cycles', 'mean_anticipation_cycles',
]


def validate_few_shot_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a few-shot block without allowing test selection."""
    settings = dict(config or {})
    target = str(settings.get('target_dataset', 'fd003')).lower()
    if target not in {'fd003', 'fd002', 'fd004'}:
        raise ValueError("few_shot_target.target_dataset must be fd002-fd004.")
    calibration = float(settings.get('calibration_fraction', 0.5))
    validation = float(settings.get('validation_fraction', 0.5))
    if not 0.0 < calibration < 1.0 or not 0.0 < validation < 1.0:
        raise ValueError("few-shot fractions must be strictly between 0 and 1.")
    if abs(calibration + validation - 1.0) > 1e-9:
        raise ValueError("few-shot calibration and validation fractions must sum to 1.")
    if settings.get('selection_split', 'target_validation') not in {
            'target_validation', 'validation'}:
        raise ValueError(
            "few_shot_target.selection_split must be target_validation; the "
            "official target test split cannot be used."
        )
    method = str(settings.get('recalibration_method', 'auto')).lower()
    if method not in {'auto', 'sigmoid', 'isotonic'}:
        raise ValueError("few-shot recalibration_method must be auto, sigmoid or isotonic.")
    strategy = str(settings.get('threshold_strategy', 'recall_at_precision'))
    if strategy not in {'f1', 'recall_at_precision', 'fbeta'}:
        raise ValueError("few-shot threshold_strategy is unsupported.")
    repeats = settings.get('selection_repeats', 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise ValueError("few-shot selection_repeats must be an integer.")
    if not 1 <= repeats <= 20:
        raise ValueError("few-shot selection_repeats must be between 1 and 20.")
    aggregation = str(settings.get('threshold_aggregation', 'median')).lower()
    if aggregation not in {'median', 'single'}:
        raise ValueError(
            "few-shot threshold_aggregation must be 'median' or 'single'."
        )
    positive_rate = settings.get('validation_positive_rate')
    if positive_rate is not None:
        if (isinstance(positive_rate, bool)
                or not isinstance(positive_rate, (int, float))):
            raise ValueError(
                "few-shot validation_positive_rate must be numeric or null."
            )
        positive_rate = float(positive_rate)
        if not 0.0 < positive_rate < 1.0:
            raise ValueError(
                "few-shot validation_positive_rate must be strictly "
                "between 0 and 1."
            )
    return {
        'target_dataset': target,
        'calibration_fraction': calibration,
        'validation_fraction': validation,
        'random_state': int(settings.get('random_state', 42)),
        'recalibrate': bool(settings.get('recalibrate', True)),
        'recalibration_method': method,
        'threshold_strategy': strategy,
        'min_precision': float(settings.get('min_precision', 0.75)),
        'selection_repeats': repeats,
        'threshold_aggregation': aggregation,
        'validation_positive_rate': positive_rate,
        'selection_split': 'target_validation',
    }


def split_engines(units: np.ndarray, calibration_fraction: float,
                  validation_fraction: float,
                  random_state: int = 42) -> tuple:
    """Split whole target training engines into disjoint calibration/validation."""
    unique = np.array(sorted({int(unit) for unit in np.asarray(units).tolist()}))
    if len(unique) < 2:
        raise ValueError("Few-shot adaptation needs at least two target engines.")
    rng = np.random.default_rng(int(random_state))
    rng.shuffle(unique)
    n_calibration = max(1, int(round(len(unique) * calibration_fraction)))
    n_calibration = min(n_calibration, len(unique) - 1)
    calibration = set(unique[:n_calibration].tolist())
    validation = set(unique[n_calibration:].tolist())
    if calibration & validation:
        raise RuntimeError("Few-shot engine split is not disjoint.")
    if not calibration or not validation:
        raise ValueError("Few-shot split must leave both subsets non-empty.")
    return calibration, validation


def balance_validation_prevalence(
        y: np.ndarray, scores: np.ndarray, units: np.ndarray,
        cycles: np.ndarray, rul: np.ndarray,
        positive_rate: Optional[float] = None,
        random_state: int = 42) -> Dict[str, Any]:
    """Resample validation windows to a predeclared positive rate.

    This changes only target-validation composition.  It never reads target
    test labels; the requested rate is an operating assumption, not a value
    estimated from the official test split.  The positive or negative class is
    downsampled without replacement, preserving all windows of the other class.
    """
    labels = np.asarray(y, dtype=int).ravel()
    values = np.asarray(scores, dtype=float).ravel()
    unit_values = np.asarray(units).ravel()
    cycle_values = np.asarray(cycles).ravel()
    rul_values = np.asarray(rul).ravel()
    lengths = {
        len(labels), len(values), len(unit_values),
        len(cycle_values), len(rul_values),
    }
    if len(lengths) != 1:
        raise ValueError(
            "Validation arrays must have the same length for prevalence "
            "control."
        )
    if len(labels) == 0:
        raise ValueError("Cannot balance an empty validation set.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Validation labels must be binary.")

    indices = np.arange(len(labels), dtype=int)
    metadata: Dict[str, Any] = {
        'applied': False,
        'requested_positive_rate': (
            None if positive_rate is None else float(positive_rate)
        ),
        'original_positive_rate': float(labels.mean()),
        'achieved_positive_rate': float(labels.mean()),
        'original_windows': int(len(labels)),
        'kept_windows': int(len(labels)),
        'original_positives': int(labels.sum()),
        'kept_positives': int(labels.sum()),
        'original_negatives': int((labels == 0).sum()),
        'kept_negatives': int((labels == 0).sum()),
        'sampling': 'none',
        'selection_split': 'target_validation',
        'test_used_for_selection': False,
    }
    if positive_rate is None:
        return {
            'indices': indices, 'y': labels, 'scores': values,
            'units': unit_values, 'cycles': cycle_values,
            'rul': rul_values, 'metadata': metadata,
        }

    target = float(positive_rate)
    if not 0.0 < target < 1.0:
        raise ValueError(
            "validation positive_rate must be strictly between 0 and 1."
        )
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if not len(positives) or not len(negatives):
        raise ValueError(
            "Prevalence control needs both positive and negative validation "
            "windows."
        )
    current = float(labels.mean())
    sampling = 'none'
    keep = indices
    if current > target + 1e-12:
        desired_positives = int(round(len(negatives) * target / (1.0 - target)))
        desired_positives = max(1, min(len(positives), desired_positives))
        rng = np.random.default_rng(int(random_state))
        sampled = rng.choice(positives, size=desired_positives, replace=False)
        keep = np.sort(np.concatenate([negatives, sampled]))
        sampling = 'downsample_positives'
    elif current < target - 1e-12:
        desired_negatives = int(round(len(positives) * (1.0 - target) / target))
        desired_negatives = max(1, min(len(negatives), desired_negatives))
        rng = np.random.default_rng(int(random_state))
        sampled = rng.choice(negatives, size=desired_negatives, replace=False)
        keep = np.sort(np.concatenate([positives, sampled]))
        sampling = 'downsample_negatives'

    kept_labels = labels[keep]
    metadata.update({
        'applied': bool(sampling != 'none'),
        'achieved_positive_rate': float(kept_labels.mean()),
        'kept_windows': int(len(keep)),
        'kept_positives': int(kept_labels.sum()),
        'kept_negatives': int((kept_labels == 0).sum()),
        'sampling': sampling,
    })
    return {
        'indices': keep,
        'y': kept_labels,
        'scores': values[keep],
        'units': unit_values[keep],
        'cycles': cycle_values[keep],
        'rul': rul_values[keep],
        'metadata': metadata,
    }


class ScoreCalibrator:
    """Map source scores to target probabilities without touching test data."""

    def __init__(self, method: str, estimator: Any):
        self.method = method
        self.estimator = estimator

    def predict(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self.method == 'sigmoid':
            return self.estimator.predict_proba(values)[:, 1]
        return np.asarray(self.estimator.predict(values.ravel()), dtype=float)

    def save(self, path: str) -> str:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        return path


def fit_score_calibrator(scores: np.ndarray, y: np.ndarray,
                         method: str = 'auto',
                         validation_scores: Optional[np.ndarray] = None,
                         validation_y: Optional[np.ndarray] = None
                         ) -> ScoreCalibrator:
    """Fit sigmoid/isotonic score calibration on target calibration engines."""
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(y, dtype=int)
    if len(values) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Few-shot calibration needs both classes.")
    method = str(method).lower()
    if method not in {'auto', 'sigmoid', 'isotonic'}:
        raise ValueError(f"Unsupported recalibration method: {method!r}")

    candidates: List[ScoreCalibrator] = []
    if method in {'auto', 'sigmoid'}:
        sigmoid = LogisticRegression(max_iter=1000, random_state=42)
        sigmoid.fit(values.reshape(-1, 1), labels)
        candidates.append(ScoreCalibrator('sigmoid', sigmoid))
    if method in {'auto', 'isotonic'}:
        isotonic = IsotonicRegression(out_of_bounds='clip')
        isotonic.fit(values, labels)
        candidates.append(ScoreCalibrator('isotonic', isotonic))
    if method != 'auto':
        return candidates[0]
    if validation_scores is None or validation_y is None:
        raise ValueError("auto recalibration requires validation scores and labels.")
    return min(
        candidates,
        key=lambda candidate: calibration_metrics(
            validation_y, candidate.predict(validation_scores), n_bins=10
        )['brier_score'],
    )


class SourceScorer:
    """Load the published source model and apply its persisted transforms."""

    def __init__(self, results_dir: str, architecture: str, logger: Any):
        self.results_dir = results_dir
        self.architecture = architecture
        model_name = (
            'lightgbm_classifier' if architecture == 'lightgbm'
            else 'light_classifier'
        )
        candidates = [
            os.path.join(results_dir, 'models', f'{model_name}.joblib'),
            os.path.join(
                results_dir, 'models', f'{model_name}_{architecture}.joblib'
            ),
        ]
        model_path = next(
            (path for path in candidates if os.path.exists(path)), candidates[0]
        )
        self.model = LightClassifier.load(model_path, logger=logger)
        self.scaler = None
        self.needs_bundle = bool(
            getattr(self.model, 'uses_temporal_features', False)
            or getattr(self.model, 'uses_rul_feature', False)
        )
        if not self.needs_bundle:
            scaler_path = os.path.join(results_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                import joblib
                self.scaler = joblib.load(scaler_path)

    def prepare(self, X_raw: np.ndarray) -> np.ndarray:
        if self.needs_bundle:
            return self.model.prepare_features(X_raw)
        if self.scaler is not None:
            flat = X_raw.reshape(-1, X_raw.shape[-1])
            return self.scaler.transform(flat).reshape(X_raw.shape)
        return X_raw

    def scores(self, X_raw: np.ndarray) -> np.ndarray:
        return self.model.positive_proba(self.prepare(X_raw)).ravel()


def _evaluate(y: np.ndarray, scores: np.ndarray, units: np.ndarray,
              cycles: np.ndarray, rul: np.ndarray,
              threshold: float) -> Dict[str, Any]:
    predictions = (np.asarray(scores) >= float(threshold)).astype(int)
    from src.pipeline.pipeline import ANSRDTPipeline
    metrics = ANSRDTPipeline._compute_metrics(y, predictions, scores)
    cal = calibration_metrics(y, scores, n_bins=10)
    engine = compute_engine_metrics(units, cycles, rul, y, predictions)
    episode = compute_episode_metrics(y, predictions, units, cycles, rul)
    observed = int(len(np.unique(np.column_stack([units, cycles]), axis=0)))
    return {
        'metrics': metrics,
        'calibration': {
            'brier_score': float(cal['brier_score']),
            'ece': float(cal['ece']),
        },
        'engine_metrics': engine['summary'],
        'episode_metrics': {
            key: value for key, value in episode.items() if key != 'episodes'
        },
        'false_alarms_per_100_cycles': (
            100.0 * float(engine['summary']['total_false_alarms']) / observed
            if observed else 0.0
        ),
        'predictions': predictions,
        'scores': scores,
    }


def select_few_shot_threshold(
        y: np.ndarray, scores: np.ndarray, strategy: str,
        min_precision: float, beta: float = 2.0) -> Dict[str, Any]:
    """Select on validation, rejecting a threshold that collapses recall.

    ``recall_at_precision`` can return a threshold on a tiny high-score island
    when the precision constraint is otherwise unreachable.  Such a threshold
    is valid on paper but operationally useless.  This guard retries with
    ``fbeta`` on the same validation data and records the decision.  Test data
    is never consulted.
    """
    primary = select_threshold(
        y, scores, strategy=strategy, min_precision=min_precision,
        beta=beta, fallback=0.5,
    )
    primary_threshold = float(primary['threshold'])
    primary_recall = float(
        primary.get('selected_metrics', {}).get('recall', 0.0)
    )
    alerts = int((np.asarray(scores) >= primary_threshold).sum())
    if primary_recall >= 0.01 and alerts >= 2:
        return primary
    if str(strategy).lower() == 'fbeta':
        return primary
    fallback = select_threshold(
        y, scores, strategy='fbeta', min_precision=min_precision,
        beta=beta, fallback=0.5,
    )
    fallback_recall = float(
        fallback.get('selected_metrics', {}).get('recall', 0.0)
    )
    if fallback_recall <= primary_recall:
        return primary
    fallback['guard_fallback'] = {
        'reason': 'selected_threshold_collapses_recall',
        'requested_strategy': str(strategy),
        'requested_threshold': primary_threshold,
        'requested_recall': primary_recall,
        'fallback_strategy': 'fbeta',
        'fallback_threshold': float(fallback['threshold']),
        'fallback_recall': fallback_recall,
        'selection_split': 'target_validation',
    }
    return fallback


def aggregate_threshold_selections(
        selections: List[Dict[str, Any]],
        aggregation: str = 'median') -> Dict[str, Any]:
    """Aggregate thresholds selected on repeated disjoint engine splits.

    The median is chosen *a priori* to reduce split sensitivity.  It is still
    a target-validation statistic and never consults the official test split.
    """
    if not selections:
        raise ValueError("At least one threshold selection is required.")
    aggregation = str(aggregation).lower()
    if aggregation not in {'median', 'single'}:
        raise ValueError(
            "threshold aggregation must be 'median' or 'single'."
        )
    thresholds = np.asarray(
        [float(row['threshold']) for row in selections], dtype=float
    )
    median = float(np.median(thresholds))
    representative = (
        0 if aggregation == 'single'
        else int(np.argmin(np.abs(thresholds - median)))
    )
    selected = (
        float(thresholds[0]) if aggregation == 'single' else median
    )
    strategy_counts: Dict[str, int] = {}
    for row in selections:
        key = str(row.get('selected_strategy', 'unknown'))
        strategy_counts[key] = strategy_counts.get(key, 0) + 1
    return {
        'aggregation': aggregation,
        'n_repeats': int(len(thresholds)),
        'thresholds': thresholds.tolist(),
        'selected_threshold': selected,
        'median': median,
        'mean': float(thresholds.mean()),
        'std': float(thresholds.std()),
        'p05': float(np.quantile(thresholds, 0.05)),
        'p95': float(np.quantile(thresholds, 0.95)),
        'representative_repeat': representative,
        'representative_threshold': float(thresholds[representative]),
        'strategy_counts': strategy_counts,
        'selection_split': 'target_validation',
        'test_used_for_selection': False,
    }


def _row(split: str, threshold: float, result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result['metrics']
    episode = result['episode_metrics']
    return {
        'split': split,
        'threshold': float(threshold),
        'roc_auc': metrics.get('roc_auc'),
        'average_precision': metrics.get('average_precision'),
        'f1': metrics.get('f1'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'brier_score': result['calibration']['brier_score'],
        'ece': result['calibration']['ece'],
        'real_episodes': episode.get('real_episodes'),
        'detected_episodes': episode.get('detected_episodes'),
        'episode_recall': episode.get('episode_recall'),
        'episode_precision': episode.get('episode_precision'),
        'false_alarm_episodes': episode.get('false_alarm_episodes'),
        'false_alarms_per_100_cycles': result['false_alarms_per_100_cycles'],
        'mean_anticipation_cycles': episode.get('mean_anticipation_cycles'),
    }


def run_few_shot_target(
        source_config: Dict[str, Any], target_config: Dict[str, Any],
        target_config_path: str, logger: Any,
        results_dir: Optional[str] = None,
        architecture: Optional[str] = None) -> Dict[str, Any]:
    """Run one target adaptation and return its report."""
    settings = validate_few_shot_config(
        target_config.get('few_shot_target', {})
    )
    target_name = settings['target_dataset']
    output_dir = results_dir or os.path.join(
        target_config.get('paths', {}).get('results_dir', 'results'),
        'few_shot_target', target_name,
    )
    os.makedirs(output_dir, exist_ok=True)
    architecture = str(
        architecture
        or source_config.get('model', {}).get('architecture', 'random_forest')
    ).lower()
    if architecture in {'lgbm', 'lightgbm_classifier'}:
        architecture = 'lightgbm'
    if architecture == 'light_classifier':
        architecture = 'random_forest'
    source_results = source_config['paths']['results_dir']
    scorer = SourceScorer(source_results, architecture, logger)

    paths = target_config['paths']
    loader = CMAPSSLoader(
        paths['data_file'], window_size=int(
            target_config['model']['window_size']
        ),
        feature_names=list(target_config['model']['feature_names']),
        test_file=paths.get('test_file'), rul_file=paths.get('rul_file'),
        rul_threshold=int(target_config['training'].get('rul_threshold', 30)),
        logger=logger,
    )
    X_train, y_train = loader.load_data()
    train_units = np.asarray(loader.unit_ids)
    train_cycles = np.asarray(loader.window_end_cycles)
    train_rul = np.asarray(loader.window_rul)
    # Source scoring is label-free, so it can be shared by every target split.
    # Calibration and threshold selection below remain target-training-only.
    train_scores = scorer.scores(X_train)
    selection_rows: List[Dict[str, Any]] = []
    selection_contexts: List[Dict[str, Any]] = []
    for repeat in range(settings['selection_repeats']):
        split_seed = settings['random_state'] + repeat
        calibration_units, validation_units = split_engines(
            train_units, settings['calibration_fraction'],
            settings['validation_fraction'], split_seed,
        )
        cal_mask = np.isin(train_units, list(calibration_units))
        val_mask = np.isin(train_units, list(validation_units))
        if cal_mask.sum() == 0 or val_mask.sum() == 0:
            raise ValueError(
                "Few-shot split produced an empty calibration or validation "
                "set."
            )
        cal_scores = train_scores[cal_mask]
        val_scores = train_scores[val_mask]
        calibrator = (
            fit_score_calibrator(
                cal_scores, y_train[cal_mask],
                method=settings['recalibration_method'],
                validation_scores=val_scores,
                validation_y=y_train[val_mask],
            )
            if settings['recalibrate'] else None
        )
        calibrated_val = (
            calibrator.predict(val_scores)
            if calibrator is not None else val_scores
        )
        validation_data = balance_validation_prevalence(
            y_train[val_mask], calibrated_val,
            train_units[val_mask], train_cycles[val_mask],
            train_rul[val_mask],
            settings['validation_positive_rate'], split_seed,
        )
        threshold_info = select_few_shot_threshold(
            validation_data['y'], validation_data['scores'],
            strategy=settings['threshold_strategy'],
            min_precision=settings['min_precision'],
            beta=2.0,
        )
        selection_rows.append({
            'repeat': repeat,
            'split_seed': split_seed,
            'calibration_engines': sorted(calibration_units),
            'validation_engines': sorted(validation_units),
            'validation_windows': int(val_mask.sum()),
            'validation_positive_rate': float(y_train[val_mask].mean()),
            'selection_windows': int(
                validation_data['metadata']['kept_windows']
            ),
            'selection_positive_rate': float(
                validation_data['metadata']['achieved_positive_rate']
            ),
            'prevalence_control': validation_data['metadata'],
            'recalibration_method': (
                None if calibrator is None else calibrator.method
            ),
            'requested_strategy': settings['threshold_strategy'],
            'selected_strategy': threshold_info['strategy'],
            'threshold': float(threshold_info['threshold']),
            'guard_fallback': bool(threshold_info.get('guard_fallback')),
            'selected_metrics': threshold_info.get('selected_metrics', {}),
        })
        selection_contexts.append({
            'calibrator': calibrator,
            'calibration_units': calibration_units,
            'validation_units': validation_units,
            'cal_mask': cal_mask,
            'val_mask': val_mask,
            'calibrated_val': calibrated_val,
            'selection_y': validation_data['y'],
            'selection_scores': validation_data['scores'],
            'selection_units': validation_data['units'],
            'selection_cycles': validation_data['cycles'],
            'selection_rul': validation_data['rul'],
            'prevalence_control': validation_data['metadata'],
            'threshold_info': threshold_info,
        })

    threshold_selection = aggregate_threshold_selections(
        selection_rows, settings['threshold_aggregation']
    )
    threshold = float(threshold_selection['selected_threshold'])
    representative = int(threshold_selection['representative_repeat'])
    selected_context = selection_contexts[representative]
    threshold_info = selected_context['threshold_info']
    calibrator = selected_context['calibrator']
    calibration_units = selected_context['calibration_units']
    validation_units = selected_context['validation_units']
    val_mask = selected_context['val_mask']
    calibrated_val = selected_context['calibrated_val']
    selection_y = selected_context['selection_y']
    selection_scores = selected_context['selection_scores']
    selection_units = selected_context['selection_units']
    selection_cycles = selected_context['selection_cycles']
    selection_rul = selected_context['selection_rul']
    prevalence_control = selected_context['prevalence_control']

    # The official target test engines are loaded exactly once, after the
    # repeated target-validation selection has been frozen.
    X_test, y_test = loader.load_test_data()
    test_units = np.asarray(loader.unit_ids)
    test_cycles = np.asarray(loader.window_end_cycles)
    test_rul = np.asarray(loader.window_rul)
    test_raw_scores = scorer.scores(X_test)
    test_scores = (
        calibrator.predict(test_raw_scores)
        if calibrator is not None else test_raw_scores
    )
    test_result = _evaluate(
        y_test, test_scores, test_units, test_cycles, test_rul, threshold
    )

    val_result = _evaluate(
        selection_y, selection_scores, selection_units,
        selection_cycles, selection_rul, threshold,
    )
    raw_val_result = _evaluate(
        y_train[val_mask], calibrated_val, train_units[val_mask],
        train_cycles[val_mask], train_rul[val_mask], threshold,
    )
    audit = build_few_shot_audit(
        selection_y, selection_scores, selection_units,
        selection_cycles, selection_rul,
        y_test, test_scores, test_units, test_cycles, test_rul,
        threshold, settings['threshold_strategy'],
        min_precision=settings['min_precision'],
        beta=2.0, n_bootstrap=200,
        random_state=settings['random_state'],
    )
    budget_config = {
        **dict(target_config.get('alert_budget', {}) or {}),
        'enabled': True,
    }
    if (budget_config.get('max_alerts_per_engine') is None
            and budget_config.get('max_alerts_per_100_cycles') is None):
        budget_config['max_alerts_per_engine'] = max(
            1.0,
            float(val_result['engine_metrics']['total_false_alarms'])
            / max(1, len(validation_units)),
        )
        # A derived budget may be below the granularity of observed
        # thresholds; keep the report useful and flag feasibility explicitly.
        budget_config['unreachable_policy'] = 'most_constraining'
    budget_config['selection_split'] = 'validation'
    budget_selection = select_alert_budget_threshold(
        selection_scores, selection_units, selection_cycles,
        budget_config,
    )
    budget_test_usage = evaluate_budget_usage(
        test_scores, test_units, test_cycles, budget_selection['threshold']
    )

    report = {
        'protocol': 'few_shot_target_adaptation',
        'warning': (
            'This protocol is not zero-shot: target training engines were '
            'used for calibration and threshold selection. Official target '
            'test engines were reporting-only.'
        ),
        'source_dataset': 'FD001',
        'target_dataset': target_name,
        'source_architecture': architecture,
        'calibration_engines': sorted(calibration_units),
        'validation_engines': sorted(validation_units),
        'n_calibration_engines': len(calibration_units),
        'n_validation_engines': len(validation_units),
        'n_test_engines': int(len(np.unique(test_units))),
        'recalibration_method': (
            None if calibrator is None else calibrator.method
        ),
        'recalibrated': calibrator is not None,
        'threshold': threshold,
        'threshold_strategy': threshold_info['strategy'],
        'requested_threshold_strategy': settings['threshold_strategy'],
        'threshold_aggregation': settings['threshold_aggregation'],
        'threshold_selection': {
            **threshold_selection,
            'rows': selection_rows,
            'representative_selection': {
                key: value for key, value in selection_rows[representative].items()
                if key not in {'calibration_engines', 'validation_engines'}
            },
        },
        'threshold_guard_fallback': threshold_info.get('guard_fallback'),
        'validation_prevalence_control': prevalence_control,
        'metrics': test_result['metrics'],
        'calibration': test_result['calibration'],
        'episode_metrics': test_result['episode_metrics'],
        'engine_metrics': test_result['engine_metrics'],
        'false_alarms_per_100_cycles': test_result[
            'false_alarms_per_100_cycles'
        ],
        'alert_budget': {
            'selection': budget_selection,
            'frozen_threshold': budget_selection['threshold'],
            'test_usage': budget_test_usage,
        },
        'robustness_audit': audit,
        'validation': _row('target_validation', threshold, val_result),
        'validation_unbalanced': _row(
            'target_validation_unbalanced', threshold, raw_val_result
        ),
        'test': _row('official_target_test', threshold, test_result),
        'selection_split': 'target_validation',
        'test_usage': 'reporting_only_once',
        'timestamp': datetime.now().isoformat(),
    }
    audit_path = save_few_shot_audit(
        audit, os.path.join(output_dir, 'robustness_audit.json')
    )
    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w') as file:
        json.dump(report, file, indent=2, default=str)
    summary_path = os.path.join(output_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FS_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(_row('target_validation', threshold, val_result))
        writer.writerow(_row('official_target_test', threshold, test_result))
    calibration_path = os.path.join(output_dir, 'calibration_report.json')
    with open(calibration_path, 'w') as file:
        json.dump({
            'method': report['recalibration_method'],
            'calibration': report['calibration'],
            'n_calibration_engines': report['n_calibration_engines'],
            'threshold_selection': report['threshold_selection'],
            'validation_prevalence_control': prevalence_control,
        }, file, indent=2)
    threshold_path = os.path.join(output_dir, 'threshold_report.json')
    with open(threshold_path, 'w') as file:
        json.dump({
            **threshold_info,
            'threshold': threshold,
            'representative_threshold': float(
                selection_rows[representative]['threshold']
            ),
            'aggregation': settings['threshold_aggregation'],
            'n_selection_repeats': settings['selection_repeats'],
            'threshold_selection': report['threshold_selection'],
            'validation_prevalence_control': prevalence_control,
            'selection_split': 'target_validation',
            'test_usage': 'reporting_only_once',
        }, file, indent=2, default=str)
    report['artifacts'] = {
        'report_path': report_path, 'summary_csv': summary_path,
        'calibration_report': calibration_path,
        'threshold_report': threshold_path,
        'robustness_audit': audit_path,
    }
    with open(report_path, 'w') as file:
        json.dump(report, file, indent=2, default=str)
    return report
