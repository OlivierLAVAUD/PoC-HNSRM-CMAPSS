# src/fusion/diagnostics.py
"""Robustness diagnostics for the validation-fitted probability fusion.

The production fusion is a single logistic regression fitted on the whole
validation split. This module adds two checks on top of it, both computed on
validation engines only:

- a grouped cross-validation (``group_kfold``), where every fold holds out
  whole engines, so the coefficient and threshold stability can be measured
  without letting windows of one engine appear on both sides;
- an ablation study that removes one fusion input at a time and reports the
  validation performance delta, answering whether each input actually helps.

The official test engines are never used here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.thresholding import select_threshold
from src.fusion.probability_fusion import ProbabilityFusion


def _summary(values: Iterable[Any]) -> Dict[str, Any]:
    """Return count, mean, standard deviation, extremes and the 95% interval."""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {'count': 0, 'mean': None, 'std': None, 'min': None,
                'max': None, 'ci95_low': None, 'ci95_high': None}
    array = np.asarray(clean, dtype=float)
    count = int(len(array))
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    margin = 1.96 * std / np.sqrt(count) if count > 1 else 0.0
    return {
        'count': count,
        'mean': mean,
        'std': std,
        'min': float(array.min()),
        'max': float(array.max()),
        'ci95_low': float(mean - margin),
        'ci95_high': float(mean + margin),
    }


def group_folds(units: np.ndarray, n_splits: int = 5, n_repeats: int = 1,
                random_state: int = 42) -> List[Dict[str, Any]]:
    """Build engine-grouped folds and return their held-out indices.

    Whole engines are assigned to folds, so no engine is split between the
    fitting and the evaluation part of a fold. Repeats reshuffle the engine
    order, which gives a repeated grouped cross-validation.
    """
    unit_values = np.asarray(units).ravel()
    unique_units = np.unique(unit_values)
    n_splits = int(n_splits)
    n_repeats = int(n_repeats)
    if n_splits < 2:
        raise ValueError("fusion.validation.n_splits must be >= 2.")
    if n_splits > len(unique_units):
        raise ValueError(
            f"Cannot build {n_splits} folds from {len(unique_units)} engines."
        )
    if n_repeats < 1:
        raise ValueError("fusion.validation.n_repeats must be >= 1.")

    folds: List[Dict[str, Any]] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(int(random_state) + repeat)
        shuffled = unique_units.copy()
        rng.shuffle(shuffled)
        for fold_index, chunk in enumerate(np.array_split(shuffled, n_splits)):
            test_mask = np.isin(unit_values, chunk)
            folds.append({
                'repeat': repeat,
                'fold': fold_index,
                'test_units': [int(unit) for unit in chunk],
                'test_indices': np.flatnonzero(test_mask),
                'train_indices': np.flatnonzero(~test_mask),
            })
    return folds


def _fold_metrics(y_true: np.ndarray, scores: np.ndarray,
                  threshold: float, beta: float) -> Dict[str, Any]:
    """Return held-out metrics for one fold at its validation threshold."""
    from src.evaluation.operational_policy import window_metrics

    predictions = (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)
    metrics = window_metrics(y_true, predictions)
    beta_sq = float(beta) * beta
    precision = float(metrics['precision'])
    recall = float(metrics['recall'])
    denominator = beta_sq * precision + recall
    metrics['fbeta'] = (
        (1.0 + beta_sq) * precision * recall / denominator
        if denominator else 0.0
    )
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, scores))
        metrics['average_precision'] = float(
            average_precision_score(y_true, scores)
        )
    except ValueError:
        metrics['roc_auc'] = None
        metrics['average_precision'] = None
    return metrics


def _fit_fold(X: np.ndarray, y: np.ndarray, feature_order: Sequence[str],
              random_state: int) -> ProbabilityFusion:
    """Fit one fusion model on a fold."""
    fusion = ProbabilityFusion(
        random_state=int(random_state), feature_order=list(feature_order)
    )
    return fusion.fit(X, y)


def cross_validate_fusion(
        X: np.ndarray, y: np.ndarray, units: np.ndarray,
        feature_order: Sequence[str],
        config: Optional[Dict[str, Any]] = None,
        threshold_config: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
    """Run the grouped cross-validation of the fusion on validation engines."""
    config = config or {}
    threshold_config = threshold_config or {}
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError("Fusion diagnostics need a 2D matrix and one label per row.")
    if len(np.unique(y)) < 2:
        raise ValueError("Fusion diagnostics require both classes.")

    folds = group_folds(
        units,
        n_splits=int(config.get('n_splits', 5)),
        n_repeats=int(config.get('n_repeats', 1)),
        random_state=int(config.get('random_state', 42)),
    )
    beta = float(threshold_config.get('beta', 2.0))
    fold_reports: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for fold in folds:
        train_idx = fold['train_indices']
        test_idx = fold['test_indices']
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            skipped.append({
                'repeat': fold['repeat'],
                'fold': fold['fold'],
                'reason': 'single_class_in_fold',
                'train_units': len(np.unique(np.asarray(units)[train_idx])),
                'test_units': len(np.unique(np.asarray(units)[test_idx])),
            })
            continue
        fusion = _fit_fold(
            X[train_idx], y[train_idx], feature_order,
            int(config.get('random_state', 42)),
        )
        train_scores = fusion.positive_proba(X[train_idx])
        test_scores = fusion.positive_proba(X[test_idx])
        selection = select_threshold(
            y[train_idx], train_scores,
            strategy=threshold_config.get(
                'threshold_strategy', 'recall_at_precision'
            ),
            min_precision=float(threshold_config.get('min_precision', 0.75)),
            beta=beta,
            fallback=float(threshold_config.get('fallback', 0.5)),
        )
        threshold = float(selection['threshold'])
        metrics = _fold_metrics(y[test_idx], test_scores, threshold, beta)
        fold_reports.append({
            'repeat': fold['repeat'],
            'fold': fold['fold'],
            'test_units': fold['test_units'],
            'train_windows': int(len(train_idx)),
            'test_windows': int(len(test_idx)),
            'threshold': threshold,
            'coefficients': fusion.coefficients(),
            'metrics': metrics,
        })

    if not fold_reports:
        raise ValueError("No fusion fold could be evaluated.")

    coefficients = {
        name: _summary([
            fold['coefficients'].get(name) for fold in fold_reports
        ])
        for name in feature_order
    }
    absolute = {
        name: abs(stats['mean'] or 0.0) for name, stats in coefficients.items()
    }
    total = sum(absolute.values()) or 1.0
    relative_importance = {
        name: float(value / total) for name, value in absolute.items()
    }
    thresholds = _summary(fold['threshold'] for fold in fold_reports)
    metrics = {
        key: _summary(fold['metrics'].get(key) for fold in fold_reports)
        for key in ('f1', 'fbeta', 'precision', 'recall',
                    'roc_auc', 'average_precision')
    }
    return {
        'method': 'group_kfold',
        'scope': 'validation_engines_only',
        'n_splits': int(config.get('n_splits', 5)),
        'n_repeats': int(config.get('n_repeats', 1)),
        'random_state': int(config.get('random_state', 42)),
        'folds_evaluated': len(fold_reports),
        'skipped_folds': skipped,
        'folds': fold_reports,
        'coefficients': coefficients,
        'relative_importance': relative_importance,
        'threshold_stability': thresholds,
        'metrics': metrics,
    }


def ablation_study(
        X: np.ndarray, y: np.ndarray, units: np.ndarray,
        feature_order: Sequence[str],
        config: Optional[Dict[str, Any]] = None,
        threshold_config: Optional[Dict[str, Any]] = None,
        tolerance: float = 1e-4) -> Dict[str, Any]:
    """Remove one fusion input at a time and report the validation delta."""
    config = config or {}
    feature_order = list(feature_order)
    if len(feature_order) < 2:
        raise ValueError("Ablation needs at least two fusion inputs.")
    X = np.asarray(X, dtype=float)

    def evaluate(indices: List[int]) -> Dict[str, Any]:
        subset = [feature_order[index] for index in indices]
        report = cross_validate_fusion(
            X[:, indices], y, units, subset, config, threshold_config
        )
        return {
            'features': subset,
            'fbeta': report['metrics']['fbeta'],
            'average_precision': report['metrics']['average_precision'],
            'roc_auc': report['metrics']['roc_auc'],
            'f1': report['metrics']['f1'],
        }

    full = evaluate(list(range(len(feature_order))))
    variants = []
    for index, name in enumerate(feature_order):
        remaining = [position for position in range(len(feature_order))
                     if position != index]
        result = evaluate(remaining)
        delta_fbeta = (
            None if result['fbeta']['mean'] is None or full['fbeta']['mean'] is None
            else float(result['fbeta']['mean'] - full['fbeta']['mean'])
        )
        delta_ap = (
            None
            if result['average_precision']['mean'] is None
            or full['average_precision']['mean'] is None
            else float(
                result['average_precision']['mean']
                - full['average_precision']['mean']
            )
        )
        if delta_fbeta is None:
            verdict = 'inconclusive'
        elif delta_fbeta < -float(tolerance):
            verdict = 'improves'
        elif delta_fbeta > float(tolerance):
            verdict = 'degrades'
        else:
            verdict = 'neutral'
        variants.append({
            'removed_feature': name,
            'remaining_features': result['features'],
            'fbeta': result['fbeta'],
            'average_precision': result['average_precision'],
            'roc_auc': result['roc_auc'],
            'f1': result['f1'],
            'delta_fbeta': delta_fbeta,
            'delta_average_precision': delta_ap,
            'verdict': verdict,
        })

    helpful = [item['removed_feature'] for item in variants
               if item['verdict'] == 'improves']
    return {
        'scope': 'validation_engines_only',
        'full': full,
        'ablations': variants,
        'summary': {
            'helpful_features': helpful,
            'neutral_features': [
                item['removed_feature'] for item in variants
                if item['verdict'] == 'neutral'
            ],
            'harmful_features': [
                item['removed_feature'] for item in variants
                if item['verdict'] == 'degrades'
            ],
            'answer': (
                'Removing ' + ', '.join(helpful) + ' lowers the grouped-CV '
                'F-beta, so those inputs improve validation performance.'
                if helpful else
                'No input improves the grouped-CV F-beta on validation when '
                'removed alone; every input is either neutral or harmful.'
            ),
        },
    }


def run_fusion_diagnostics(
        X_val: np.ndarray, y_val: np.ndarray, val_units: np.ndarray,
        feature_order: Sequence[str], config: Dict[str, Any],
        results_dir: str) -> Dict[str, Any]:
    """Compute cross-validation diagnostics and the ablation study."""
    fusion_config = config.get('fusion', {})
    validation_config = fusion_config.get('validation', {})
    diagnostics = cross_validate_fusion(
        X_val, y_val, val_units, feature_order, validation_config,
        threshold_config=fusion_config,
    )
    ablation = ablation_study(
        X_val, y_val, val_units, feature_order, validation_config,
        threshold_config=fusion_config,
    )
    os.makedirs(results_dir, exist_ok=True)
    diagnostics_path = os.path.join(results_dir, 'fusion_diagnostics.json')
    with open(diagnostics_path, 'w') as file:
        json.dump(diagnostics, file, indent=2)
    ablation_path = os.path.join(results_dir, 'fusion_ablation.json')
    with open(ablation_path, 'w') as file:
        json.dump(ablation, file, indent=2)
    return {
        'enabled': True,
        'applied': True,
        'diagnostics_path': diagnostics_path,
        'ablation_path': ablation_path,
        'threshold_stability': diagnostics['threshold_stability'],
        'relative_importance': diagnostics['relative_importance'],
        'ablation_summary': ablation['summary'],
        'folds_evaluated': diagnostics['folds_evaluated'],
        'timestamp': datetime.now().isoformat(),
    }
