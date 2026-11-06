# src/fusion/nested_validation.py
"""Engine-grouped nested validation of the probability fusion.

The production fusion stays fitted on the whole validation split. This module
only measures how stable that recipe is: whole validation engines are assigned
to outer folds, the fusion and its threshold are fitted on the inner training
fold, and the held-out engines are scored once. The official test split is
never used.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.evaluation.operational_policy import window_metrics
from src.evaluation.thresholding import select_threshold
from src.fusion.probability_fusion import ProbabilityFusion
from src.fusion.diagnostics import group_folds

METRIC_KEYS: Sequence[str] = (
    'f1', 'recall', 'precision', 'roc_auc', 'brier_score', 'threshold',
)


def _summary(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    """Return count, mean, standard deviation, minimum and maximum."""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            'count': 0, 'mean': None, 'std': None,
            'min': None, 'max': None,
        }
    array = np.asarray(clean, dtype=float)
    return {
        'count': int(len(array)),
        'mean': float(array.mean()),
        'std': float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        'min': float(array.min()),
        'max': float(array.max()),
    }


def _fold_metrics(y_true: np.ndarray, scores: np.ndarray,
                  threshold: float, beta: float) -> Dict[str, Any]:
    predictions = (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)
    metrics = window_metrics(y_true, predictions)
    precision = float(metrics['precision'])
    recall = float(metrics['recall'])
    beta_sq = float(beta) * beta
    denominator = beta_sq * precision + recall
    try:
        roc_auc: Optional[float] = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc_auc = None
    return {
        'f1': float(metrics['f1']),
        'fbeta': (
            (1.0 + beta_sq) * precision * recall / denominator
            if denominator else 0.0
        ),
        'recall': recall,
        'precision': precision,
        'roc_auc': roc_auc,
        'brier_score': float(brier_score_loss(y_true, scores)),
        'threshold': float(threshold),
    }


def run_nested_validation(
        X: np.ndarray, y: np.ndarray, units: np.ndarray,
        feature_order: Sequence[str],
        fusion_config: Optional[Dict[str, Any]] = None,
        results_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run an engine-grouped nested validation of the fusion recipe."""
    fusion_config = fusion_config or {}
    nested_config = fusion_config.get('nested_validation', {}) or {}
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    units = np.asarray(units).ravel()
    if X.ndim != 2 or X.shape[0] != len(y) or len(y) != len(units):
        raise ValueError(
            "Nested fusion validation needs a 2D matrix, one label and one "
            "engine id per row."
        )
    if len(np.unique(y)) < 2:
        raise ValueError("Nested fusion validation requires both classes.")

    folds = group_folds(
        units,
        n_splits=int(nested_config.get('n_splits', 5)),
        n_repeats=1,
        random_state=int(nested_config.get('random_state', 42)),
    )
    beta = float(fusion_config.get('beta', 2.0))
    fold_reports: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for fold in folds:
        train_idx = fold['train_indices']
        test_idx = fold['test_indices']
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            skipped.append({
                'fold': fold['fold'],
                'reason': 'single_class_in_fold',
                'test_units': fold['test_units'],
            })
            continue
        model = ProbabilityFusion(
            random_state=int(nested_config.get('random_state', 42)),
            feature_order=list(feature_order),
        )
        model.fit(X[train_idx], y[train_idx])
        train_scores = model.positive_proba(X[train_idx])
        test_scores = model.positive_proba(X[test_idx])
        selection = select_threshold(
            y[train_idx], train_scores,
            strategy=fusion_config.get(
                'threshold_strategy', 'recall_at_precision'
            ),
            min_precision=float(fusion_config.get('min_precision', 0.75)),
            beta=beta,
            fallback=float(fusion_config.get('fallback', 0.5)),
        )
        threshold = float(selection['threshold'])
        fold_reports.append({
            'fold': fold['fold'],
            'test_units': fold['test_units'],
            'train_windows': int(len(train_idx)),
            'test_windows': int(len(test_idx)),
            'threshold': threshold,
            'metrics': _fold_metrics(y[test_idx], test_scores, threshold, beta),
        })

    if not fold_reports:
        raise ValueError("No nested fusion fold could be evaluated.")

    metrics = {
        key: _summary(fold['metrics'].get(key) for fold in fold_reports)
        for key in METRIC_KEYS
    }
    report = {
        'enabled': True,
        'applied': True,
        'method': 'engine_grouped_nested_validation',
        'scope': 'validation_engines_only',
        'n_splits': int(nested_config.get('n_splits', 5)),
        'random_state': int(nested_config.get('random_state', 42)),
        'folds_evaluated': len(fold_reports),
        'skipped_folds': skipped,
        'metrics': metrics,
        'folds': fold_reports,
    }
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, 'fusion_nested_validation.json')
        with open(path, 'w') as file:
            json.dump(report, file, indent=2, cls=_NumpyEncoder)
        report['path'] = path
    return report


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
