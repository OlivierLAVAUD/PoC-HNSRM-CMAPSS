# src/experiments/few_shot_audit.py
"""Validation/test robustness audit for few-shot target adaptation.

The audit answers a narrow question: is the target-validation decision point a
reliable estimate of target-test behaviour?  It never selects anything on the
test split.  Threshold stability is measured by resampling *validation engines*
and by leave-one-engine-out selection; the official target test split is used
only to describe the frozen threshold's behaviour.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.evaluation.calibration import calibration_metrics
from src.evaluation.engine_metrics import compute_engine_metrics
from src.evaluation.thresholding import select_threshold


def _score_summary(scores: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(y, dtype=int).ravel()
    quantiles = np.quantile(values, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75,
                                     0.95, 0.99, 1.0])
    calibration = calibration_metrics(labels, values, n_bins=10)
    return {
        'windows': int(len(values)),
        'positive_rate': float(labels.mean()) if len(labels) else 0.0,
        'mean': float(values.mean()) if len(values) else 0.0,
        'std': float(values.std()) if len(values) else 0.0,
        'quantiles': {
            str(q): float(v) for q, v in zip(
                [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
                quantiles,
            )
        },
        'brier_score': float(calibration['brier_score']),
        'ece': float(calibration['ece']),
    }


def _metrics_at(y: np.ndarray, scores: np.ndarray,
                threshold: float) -> Dict[str, Any]:
    predictions = (np.asarray(scores) >= float(threshold)).astype(int)
    return {
        'f1': float(f1_score(y, predictions, zero_division=0)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'alert_rate': float(predictions.mean()) if len(predictions) else 0.0,
    }


def _select(y: np.ndarray, scores: np.ndarray, units: np.ndarray,
            cycles: np.ndarray, strategy: str, min_precision: float,
            beta: float) -> Dict[str, Any]:
    selection = select_threshold(
        y, scores, strategy=strategy, min_precision=min_precision,
        beta=beta, fallback=0.5,
    )
    return {
        'threshold': float(selection['threshold']),
        'fallback_used': bool(selection.get('fallback_used', False)),
        'selected_metrics': selection.get('selected_metrics', {}),
    }


def _bootstrap_thresholds(y: np.ndarray, scores: np.ndarray,
                          units: np.ndarray, cycles: np.ndarray,
                          strategy: str, min_precision: float, beta: float,
                          n_bootstrap: int, random_state: int
                          ) -> Dict[str, Any]:
    unique = np.array(sorted({int(unit) for unit in np.asarray(units).tolist()}))
    if len(unique) < 2:
        return {
            'count': 0, 'thresholds': [], 'mean': None, 'std': None,
            'p05': None, 'p50': None, 'p95': None,
            'fallback_fraction': 0.0,
        }
    rng = np.random.default_rng(int(random_state))
    values: List[float] = []
    fallbacks = 0
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            np.flatnonzero(units == unit) for unit in sampled
        ])
        result = _select(
            y[indices], scores[indices], units[indices], cycles[indices],
            strategy, min_precision, beta,
        )
        values.append(result['threshold'])
        fallbacks += int(result['fallback_used'])
    values_array = np.asarray(values, dtype=float)
    return {
        'count': int(len(values)),
        'thresholds': values,
        'mean': float(values_array.mean()),
        'std': float(values_array.std()),
        'p05': float(np.quantile(values_array, 0.05)),
        'p50': float(np.quantile(values_array, 0.50)),
        'p95': float(np.quantile(values_array, 0.95)),
        'fallback_fraction': float(fallbacks / len(values)),
    }


def _leave_one_engine_out(y: np.ndarray, scores: np.ndarray,
                          units: np.ndarray, cycles: np.ndarray,
                          strategy: str, min_precision: float,
                          beta: float) -> Dict[str, Any]:
    unique = sorted({int(unit) for unit in np.asarray(units).tolist()})
    values = []
    for unit in unique:
        mask = units != unit
        if len(np.unique(y[mask])) < 2:
            continue
        result = _select(
            y[mask], scores[mask], units[mask], cycles[mask],
            strategy, min_precision, beta,
        )
        values.append({
            'held_out_engine': int(unit),
            'threshold': result['threshold'],
            'fallback_used': result['fallback_used'],
        })
    thresholds = np.asarray([row['threshold'] for row in values], dtype=float)
    return {
        'count': len(values),
        'rows': values,
        'mean': float(thresholds.mean()) if len(thresholds) else None,
        'std': float(thresholds.std()) if len(thresholds) else None,
        'min': float(thresholds.min()) if len(thresholds) else None,
        'max': float(thresholds.max()) if len(thresholds) else None,
    }


def build_few_shot_audit(
        y_val: np.ndarray, val_scores: np.ndarray, val_units: np.ndarray,
        val_cycles: np.ndarray, val_rul: np.ndarray,
        y_test: np.ndarray, test_scores: np.ndarray, test_units: np.ndarray,
        test_cycles: np.ndarray, test_rul: np.ndarray,
        selected_threshold: float, threshold_strategy: str,
        min_precision: float = 0.75, beta: float = 2.0,
        n_bootstrap: int = 200, random_state: int = 42) -> Dict[str, Any]:
    """Compare target validation and test without selecting on test."""
    y_val = np.asarray(y_val, dtype=int).ravel()
    y_test = np.asarray(y_test, dtype=int).ravel()
    val_scores = np.asarray(val_scores, dtype=float).ravel()
    test_scores = np.asarray(test_scores, dtype=float).ravel()
    val_units = np.asarray(val_units).ravel()
    test_units = np.asarray(test_units).ravel()
    val_cycles = np.asarray(val_cycles).ravel()
    test_cycles = np.asarray(test_cycles).ravel()
    val_rul = np.asarray(val_rul).ravel()
    test_rul = np.asarray(test_rul).ravel()
    if not (len(y_val) == len(val_scores) == len(val_units)
            == len(val_cycles) == len(val_rul)):
        raise ValueError("Validation arrays must have the same length.")
    if not (len(y_test) == len(test_scores) == len(test_units)
            == len(test_cycles) == len(test_rul)):
        raise ValueError("Test arrays must have the same length.")

    val_metrics = _metrics_at(y_val, val_scores, selected_threshold)
    test_metrics = _metrics_at(y_test, test_scores, selected_threshold)
    val_summary = _score_summary(val_scores, y_val)
    test_summary = _score_summary(test_scores, y_test)
    val_summary['engines'] = int(len(np.unique(val_units)))
    test_summary['engines'] = int(len(np.unique(test_units)))

    val_predictions = (val_scores >= selected_threshold).astype(int)
    test_predictions = (test_scores >= selected_threshold).astype(int)
    val_engine = compute_engine_metrics(
        val_units, val_cycles, val_rul, y_val, val_predictions
    )['summary']
    test_engine = compute_engine_metrics(
        test_units, test_cycles, test_rul, y_test, test_predictions
    )['summary']

    bootstrap = _bootstrap_thresholds(
        y_val, val_scores, val_units, val_cycles, threshold_strategy,
        min_precision, beta, n_bootstrap, random_state,
    )
    leave_one_out = _leave_one_engine_out(
        y_val, val_scores, val_units, val_cycles, threshold_strategy,
        min_precision, beta,
    )

    f1_gap = val_metrics['f1'] - test_metrics['f1']
    prevalence_delta = test_summary['positive_rate'] - val_summary['positive_rate']
    mean_score_delta = test_summary['mean'] - val_summary['mean']
    threshold_std = (
        float(bootstrap['std']) if bootstrap['std'] is not None else None
    )
    flags = []
    if f1_gap > 0.15:
        flags.append(
            f'validation F1 is {f1_gap:.3f} above test F1: threshold '
            'selection may be optimistic or the splits may differ.'
        )
    if abs(prevalence_delta) > 0.03:
        flags.append(
            f'positive-rate shift is {prevalence_delta:+.3f}: validation and '
            'test do not share the same class balance.'
        )
    if abs(mean_score_delta) > 0.05:
        flags.append(
            f'mean score shift is {mean_score_delta:+.3f}: score distributions '
            'differ between validation and test.'
        )
    if threshold_std is not None and threshold_std > 0.05:
        flags.append(
            f'bootstrap threshold std is {threshold_std:.3f}: the decision '
            'point is unstable across validation engines.'
        )
    if test_metrics['recall'] < 0.5:
        flags.append(
            'test recall is below 0.50 at the frozen threshold; inspect the '
            'threshold/calibration trade-off before claiming operational gain.'
        )

    return {
        'protocol': 'few_shot_robustness_audit',
        'selection_split': 'target_validation',
        'test_usage': 'reporting_only',
        'selected_threshold': float(selected_threshold),
        'threshold_strategy': str(threshold_strategy),
        'validation': {
            **val_summary, **val_metrics, 'engine_summary': val_engine,
        },
        'test': {
            **test_summary, **test_metrics, 'engine_summary': test_engine,
        },
        'deltas': {
            'f1_validation_minus_test': float(f1_gap),
            'positive_rate_test_minus_validation': float(prevalence_delta),
            'mean_score_test_minus_validation': float(mean_score_delta),
        },
        'threshold_stability': {
            'selected': float(selected_threshold),
            'bootstrap': bootstrap,
            'leave_one_engine_out': leave_one_out,
            'test_used_for_selection': False,
        },
        'flags': flags,
        'conclusion': (
            'No strong validation/test mismatch detected.'
            if not flags else
            'Validation/test mismatch detected; treat the validation-selected '
            'threshold as provisional and audit engine-level stability before '
            'scaling the protocol.'
        ),
    }


def save_few_shot_audit(report: Dict[str, Any], path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(report, file, indent=2, default=str)
    return path
