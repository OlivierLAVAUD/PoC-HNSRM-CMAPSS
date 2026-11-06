# src/evaluation/alert_budget.py
"""Validation-only alert-budget threshold selection.

An alert budget is an operational constraint, not a copy of the source alert
rate: at most ``max_alerts_per_engine`` alert windows on average, and/or at
most ``max_alerts_per_100_cycles`` alert windows per 100 observed cycles.  The
threshold is selected on validation and then frozen for test reporting.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def validate_budget_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise an ``alert_budget`` configuration block."""
    if not isinstance(config, dict):
        raise ValueError("alert_budget must be a mapping.")
    per_engine = config.get('max_alerts_per_engine')
    per_100 = config.get('max_alerts_per_100_cycles')
    if per_engine is None and per_100 is None:
        raise ValueError(
            "alert_budget requires max_alerts_per_engine and/or "
            "max_alerts_per_100_cycles."
        )
    for name, value in (
            ('max_alerts_per_engine', per_engine),
            ('max_alerts_per_100_cycles', per_100)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"alert_budget.{name} must be numeric or null.")
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"alert_budget.{name} must be finite and >= 0.")
    policy = str(config.get('unreachable_policy', 'error')).lower()
    if policy not in {'error', 'most_constraining'}:
        raise ValueError(
            "alert_budget.unreachable_policy must be 'error' or "
            "'most_constraining'."
        )
    split = str(config.get('selection_split', 'validation')).lower()
    if split != 'validation':
        raise ValueError(
            "alert_budget.selection_split must be 'validation'; the official "
            "test split cannot select a threshold."
        )
    return {
        'max_alerts_per_engine': (
            None if per_engine is None else float(per_engine)
        ),
        'max_alerts_per_100_cycles': (
            None if per_100 is None else float(per_100)
        ),
        'unreachable_policy': policy,
        'selection_split': split,
    }


def _budget_usage(scores: np.ndarray, threshold: float,
                  units: np.ndarray, cycles: np.ndarray) -> Dict[str, float]:
    alerts = np.asarray(scores, dtype=float).ravel() >= float(threshold)
    total = int(alerts.sum())
    engines = int(len(np.unique(units)))
    observed_cycles = int(len(np.unique(np.column_stack([
        np.asarray(units).ravel(), np.asarray(cycles).ravel()
    ]), axis=0)))
    return {
        'alerts': total,
        'engines': engines,
        'cycles': observed_cycles,
        'alerts_per_engine': (
            float(total / engines) if engines else 0.0
        ),
        'alerts_per_100_cycles': (
            float(100.0 * total / observed_cycles)
            if observed_cycles else 0.0
        ),
    }


def _feasible(usage: Dict[str, float], config: Dict[str, Any]) -> bool:
    per_engine = config['max_alerts_per_engine']
    per_100 = config['max_alerts_per_100_cycles']
    if per_engine is not None and usage['alerts_per_engine'] > per_engine + 1e-12:
        return False
    if per_100 is not None and usage['alerts_per_100_cycles'] > per_100 + 1e-12:
        return False
    return True


def select_alert_budget_threshold(
        scores: Any, units: Any, cycles: Any,
        config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the highest observed-score threshold satisfying the budget.

    The official test split is never accepted by this function.  If no
    observed threshold satisfies the budget and ``unreachable_policy`` is
    ``error``, a clear error is raised; otherwise the highest threshold is
    returned with ``feasible=False``.
    """
    settings = validate_budget_config(config)
    values = np.asarray(scores, dtype=float).ravel()
    unit_values = np.asarray(units).ravel()
    cycle_values = np.asarray(cycles).ravel()
    if not len(values):
        raise ValueError("Cannot select an alert budget from empty scores.")
    if not np.isfinite(values).all():
        raise ValueError("Scores contain NaN or infinite values.")
    if len(unit_values) != len(values) or len(cycle_values) != len(values):
        raise ValueError("scores, units and cycles must have the same length.")

    candidates = np.unique(values)[::-1]
    evaluated = []
    for threshold in candidates:
        usage = _budget_usage(values, threshold, unit_values, cycle_values)
        feasible = _feasible(usage, settings)
        evaluated.append({
            'threshold': float(threshold), 'feasible': bool(feasible),
            **usage,
        })
        if feasible:
            return {
                'threshold': float(threshold),
                'feasible': True,
                'selection_split': 'validation',
                'constraints': settings,
                'achieved': usage,
                'candidates_evaluated': len(evaluated),
            }

    if settings['unreachable_policy'] == 'error':
        raise ValueError(
            "No observed threshold satisfies the alert budget "
            f"(max_alerts_per_engine={settings['max_alerts_per_engine']}, "
            f"max_alerts_per_100_cycles="
            f"{settings['max_alerts_per_100_cycles']})."
        )
    threshold = float(candidates[0])
    usage = _budget_usage(values, threshold, unit_values, cycle_values)
    return {
        'threshold': threshold,
        'feasible': False,
        'selection_split': 'validation',
        'constraints': settings,
        'achieved': usage,
        'candidates_evaluated': len(evaluated),
        'warning': 'no observed threshold satisfies the configured budget',
    }


def evaluate_budget_usage(scores: Any, units: Any, cycles: Any,
                          threshold: float) -> Dict[str, float]:
    """Return the budget usage at a frozen threshold."""
    return _budget_usage(
        np.asarray(scores, dtype=float).ravel(),
        float(threshold),
        np.asarray(units).ravel(),
        np.asarray(cycles).ravel(),
    )
