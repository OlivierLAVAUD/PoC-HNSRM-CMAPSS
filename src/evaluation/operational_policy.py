# src/evaluation/operational_policy.py
"""Choose an operational alert mode from asymmetric maintenance costs.

Each alert mode turns risk scores into binary alerts:

- ``immediate``: alert as soon as the score crosses the threshold;
- ``consecutive_2`` / ``consecutive_3``: keep only runs of 2 or 3 positive
  windows per engine;
- ``smoothed_3`` / ``smoothed_5``: threshold a trailing moving average.

For every mode the module reports window-level confusion counts, engine-level
recall/precision, false alarms, anticipation and the expected cost

    expected_cost = false_negatives * false_negative_cost
                  + false_alarms * false_alarm_cost

The mode is selected on validation data only; the official test split is
reported for information and never drives the choice.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.evaluation.alerting import consecutive_confirmation, smoothed_predictions
from src.evaluation.engine_metrics import compute_engine_metrics

ALERT_MODES: tuple = (
    'immediate',
    'consecutive_2',
    'consecutive_3',
    'smoothed_3',
    'smoothed_5',
)

SELECTION_METRICS: tuple = ('expected_cost', 'fbeta', 'recall', 'f1', 'precision')

CSV_FIELDS: List[str] = [
    'mode', 'split', 'threshold', 'true_positives', 'false_positives',
    'false_negatives', 'true_negatives', 'recall', 'precision', 'f1', 'f2',
    'total_false_alarms', 'missed_critical_windows', 'mean_anticipation_cycles',
    'mean_lead_time_cycles', 'proportion_alerted_before_eol',
    'false_alarms_per_cycle', 'expected_cost', 'capacity_feasible',
    'selected',
]


def _safe_div(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` or 0.0 when the denominator is zero."""
    return float(numerator / denominator) if denominator else 0.0


def validate_costs(false_negative_cost: Any,
                   false_alarm_cost: Any) -> tuple:
    """Validate the cost weights and return them as floats.

    Costs must be finite and non-negative, and at least one of them must be
    positive: two zero costs make every mode equivalent and the selection
    meaningless, so this is rejected instead of silently picking the first mode.
    """
    try:
        fn_cost = float(false_negative_cost)
        fa_cost = float(false_alarm_cost)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "false_negative_cost and false_alarm_cost must be numeric."
        ) from error
    if not np.isfinite(fn_cost) or not np.isfinite(fa_cost):
        raise ValueError(
            "false_negative_cost and false_alarm_cost must be finite."
        )
    if fn_cost < 0.0 or fa_cost < 0.0:
        raise ValueError(
            "false_negative_cost and false_alarm_cost must be >= 0."
        )
    if fn_cost == 0.0 and fa_cost == 0.0:
        raise ValueError(
            "false_negative_cost and false_alarm_cost cannot both be zero; "
            "the operational policy would be undefined."
        )
    return fn_cost, fa_cost


def expected_cost(false_negatives: int, false_alarms: int,
                  false_negative_cost: float,
                  false_alarm_cost: float) -> float:
    """Return the expected maintenance cost of one alert mode."""
    return float(
        int(false_negatives) * float(false_negative_cost)
        + int(false_alarms) * float(false_alarm_cost)
    )


def window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Return window-level confusion counts, recall, precision, F1 and F2."""
    true = np.asarray(y_true, dtype=int).ravel()
    pred = np.asarray(y_pred, dtype=int).ravel()
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length.")
    tp = int(np.sum((pred == 1) & (true == 1)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))
    tn = int(np.sum((pred == 0) & (true == 0)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    beta_sq = 4.0
    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'recall': recall,
        'precision': precision,
        'f1': _safe_div(2 * precision * recall, precision + recall),
        'f2': _safe_div(
            (1 + beta_sq) * precision * recall,
            beta_sq * precision + recall,
        ),
        'alert_windows': int(pred.sum()),
        'windows': int(len(pred)),
    }


def alert_predictions(scores: np.ndarray, unit_ids: np.ndarray,
                      cycles: np.ndarray, threshold: float,
                      mode: str) -> np.ndarray:
    """Return binary alerts produced by one operational mode."""
    values = np.asarray(scores, dtype=float).ravel()
    threshold = float(threshold)
    name = str(mode).lower()
    if name == 'immediate' or name == 'threshold':
        return (values >= threshold).astype(int)
    base = (values >= threshold).astype(int)
    if name.startswith('consecutive_'):
        try:
            k = int(name.split('_', 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Unsupported alert mode: {mode!r}") from error
        return consecutive_confirmation(base, unit_ids, cycles, k=k)
    if name.startswith('smoothed_'):
        try:
            window = int(name.split('_', 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Unsupported alert mode: {mode!r}") from error
        return smoothed_predictions(
            values, unit_ids, cycles, threshold, window=window
        )
    raise ValueError(f"Unsupported alert mode: {mode!r}")


def evaluate_alert_mode(y_true: np.ndarray, scores: np.ndarray,
                        unit_ids: np.ndarray, cycles: np.ndarray,
                        rul: np.ndarray, threshold: float, mode: str,
                        false_negative_cost: float = 1.0,
                        false_alarm_cost: float = 0.0,
                        inspection_capacity_per_cycle: Optional[float] = None
                        ) -> Dict[str, Any]:
    """Describe one alert mode on one split."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    predictions = alert_predictions(scores, unit_ids, cycles, threshold, mode)
    window = window_metrics(y_true, predictions)
    engine = compute_engine_metrics(
        unit_ids, cycles, rul, y_true, predictions
    )['summary']
    cycles_observed = len(np.unique(np.column_stack([
        np.asarray(unit_ids).ravel(), np.asarray(cycles).ravel()
    ]), axis=0)) if len(y_true) else 0
    false_alarms_per_cycle = _safe_div(window['false_positives'], cycles_observed)
    capacity = (
        None if inspection_capacity_per_cycle is None
        else float(inspection_capacity_per_cycle)
    )
    return {
        'mode': str(mode),
        'threshold': float(threshold),
        **window,
        'total_false_alarms': int(engine['total_false_alarms']),
        'mean_false_alarms_per_engine': float(
            engine['mean_false_alarms_per_engine']
        ),
        'missed_critical_windows': int(window['false_negatives']),
        'mean_recall': float(engine['mean_recall']),
        'mean_precision': float(engine['mean_precision']),
        'mean_anticipation_cycles': float(engine['mean_anticipation_cycles']),
        'mean_lead_time_cycles': float(engine['mean_lead_time_cycles']),
        'proportion_alerted_before_eol': float(
            engine['proportion_alerted_before_eol']
        ),
        'proportion_alerted_before_eol_among_relevant': float(
            engine['proportion_alerted_before_eol_among_relevant']
        ),
        'false_alarms_per_cycle': false_alarms_per_cycle,
        'inspection_capacity_per_cycle': capacity,
        'capacity_feasible': (
            True if capacity is None else bool(false_alarms_per_cycle <= capacity)
        ),
        'cycles_observed': int(cycles_observed),
        'expected_cost': expected_cost(
            window['false_negatives'], window['false_positives'],
            false_negative_cost, false_alarm_cost,
        ),
    }


def _selection_key(mode_report: Dict[str, Any], metric: str,
                   beta: float) -> float:
    """Return the validation score used to rank alert modes."""
    if metric == 'expected_cost':
        return -float(mode_report['expected_cost'])
    if metric == 'recall':
        return float(mode_report['mean_recall'])
    if metric == 'f1':
        return float(mode_report['f1'])
    if metric == 'precision':
        return float(mode_report['precision'])
    if metric == 'fbeta':
        precision = float(mode_report['precision'])
        recall = float(mode_report['recall'])
        beta_sq = beta * beta
        denominator = beta_sq * precision + recall
        return (
            (1.0 + beta_sq) * precision * recall / denominator
            if denominator else 0.0
        )
    raise ValueError(f"Unsupported operational selection metric: {metric!r}")


def _rationale(selected: str, validation: Dict[str, Dict[str, Any]],
               test: Optional[Dict[str, Dict[str, Any]]],
               false_negative_cost: float, false_alarm_cost: float,
               metric: str, capacity: Optional[float]) -> Dict[str, Any]:
    """Explain the selection in plain language."""
    chosen = validation[selected]
    immediate = validation.get('immediate', chosen)
    recall_delta = float(chosen['mean_recall'] - immediate['mean_recall'])
    precision_delta = float(chosen['precision'] - immediate['precision'])
    lead_delta = float(
        chosen['mean_lead_time_cycles'] - immediate['mean_lead_time_cycles']
    )
    if false_alarm_cost > 0:
        ratio_str = f"{false_negative_cost / false_alarm_cost:.1f}"
    else:
        ratio_str = "inf (false_alarm_cost=0)"
    reasons = [
        (
            f"false negatives cost {false_negative_cost:g}x a false alarm "
            f"(ratio {ratio_str})"
        ),
        (
            f"{selected} expected validation cost "
            f"{chosen['expected_cost']:.2f} vs {immediate['expected_cost']:.2f} "
            f"for immediate alerts"
        ),
        (
            f"validation recall {chosen['mean_recall']:.4f} "
            f"({recall_delta:+.4f} vs immediate) and precision "
            f"{chosen['precision']:.4f} ({precision_delta:+.4f})"
        ),
        (
            f"false alarms {chosen['total_false_alarms']} "
            f"({chosen['total_false_alarms'] - immediate['total_false_alarms']:+d} "
            f"vs immediate)"
        ),
    ]
    if capacity is not None:
        reasons.append(
            f"inspection capacity is {capacity:g} per engine-cycle; "
            f"{selected} uses {chosen['false_alarms_per_cycle']:.4f}"
        )
    return {
        'selected_mode': selected,
        'selection_metric': metric,
        'expected_cost_validation': float(chosen['expected_cost']),
        'expected_cost_test': (
            None if test is None else float(test[selected]['expected_cost'])
        ),
        'precision_recall_tradeoff': (
            f"{selected}: recall {chosen['mean_recall']:.4f} vs "
            f"{immediate['mean_recall']:.4f} for immediate alerts; precision "
            f"{chosen['precision']:.4f} vs {immediate['precision']:.4f}"
        ),
        'anticipation_impact': (
            f"mean engine lead time {chosen['mean_lead_time_cycles']:.2f} cycles "
            f"vs {immediate['mean_lead_time_cycles']:.2f} "
            f"({lead_delta:+.2f}); mean anticipation "
            f"{chosen['mean_anticipation_cycles']:.2f} cycles"
        ),
        'reasons': reasons,
    }


def build_operational_policy_report(
        y_val: np.ndarray, val_scores: np.ndarray,
        val_units: np.ndarray, val_cycles: np.ndarray, val_rul: np.ndarray,
        y_test: np.ndarray, test_scores: np.ndarray,
        test_units: np.ndarray, test_cycles: np.ndarray,
        test_rul: np.ndarray, threshold: float,
        config: Optional[Dict[str, Any]] = None,
        modes: Iterable[str] = ALERT_MODES) -> Dict[str, Any]:
    """Select an alert mode on validation and report it on test."""
    config = config or {}
    fn_cost, fa_cost = validate_costs(
        config.get('false_negative_cost', 10.0),
        config.get('false_alarm_cost', 1.0),
    )
    metric = str(config.get('selection_metric', 'expected_cost')).lower()
    if metric not in SELECTION_METRICS:
        raise ValueError(
            f"operational_policy.selection_metric must be one of "
            f"{SELECTION_METRICS}, got {metric!r}."
        )
    capacity = config.get('inspection_capacity_per_cycle')
    if capacity is not None:
        capacity = float(capacity)
        if capacity < 0.0:
            raise ValueError(
                "inspection_capacity_per_cycle must be >= 0 or null."
            )
    beta = float(
        config.get('beta', 2.0)
    )
    preferred = config.get('preferred_alert_mode')
    if preferred is not None and str(preferred) not in modes:
        raise ValueError(
            f"preferred_alert_mode {preferred!r} is not one of {list(modes)}."
        )

    validation = {
        mode: evaluate_alert_mode(
            y_val, val_scores, val_units, val_cycles, val_rul,
            threshold, mode, fn_cost, fa_cost, capacity,
        )
        for mode in modes
    }
    test = {
        mode: evaluate_alert_mode(
            y_test, test_scores, test_units, test_cycles, test_rul,
            threshold, mode, fn_cost, fa_cost, capacity,
        )
        for mode in modes
    }

    feasible = [
        mode for mode, report in validation.items()
        if report['capacity_feasible']
    ]
    candidates = feasible or list(validation)
    if len(candidates) < len(validation):
        for report in validation.values():
            report['excluded_by_capacity'] = not report['capacity_feasible']

    if preferred is not None:
        selected = str(preferred)
        selection_reason = 'configured_preference'
    else:
        selected = max(
            candidates,
            key=lambda mode: _selection_key(validation[mode], metric, beta),
        )
        selection_reason = f'best_validation_{metric}'

    for mode, report in validation.items():
        report['selected'] = mode == selected
        report['selection_reason'] = selection_reason if mode == selected else None
    for mode, report in test.items():
        report['selected'] = mode == selected

    rationale = _rationale(
        selected, validation, test, fn_cost, fa_cost, metric, capacity
    )
    return {
        'selection_split': 'validation',
        'test_usage': 'reporting_only',
        'threshold': float(threshold),
        'false_negative_cost': fn_cost,
        'false_alarm_cost': fa_cost,
        'inspection_capacity_per_cycle': capacity,
        'selection_metric': metric,
        'capacity_feasible_modes': feasible,
        'selected_mode': selected,
        'selection_reason': selection_reason,
        'validation': validation,
        'test': test,
        'rationale': rationale,
    }


def save_operational_policy(report: Dict[str, Any], path: str) -> str:
    """Persist the operational policy decision as JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(report, file, indent=2)
    return path


def export_operational_policy_csv(report: Dict[str, Any], path: str) -> str:
    """Write the per-mode validation and test comparison to CSV."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS,
                                extrasaction='ignore')
        writer.writeheader()
        for split in ('validation', 'test'):
            for mode, values in report.get(split, {}).items():
                writer.writerow({**values, 'split': split})
    return path
