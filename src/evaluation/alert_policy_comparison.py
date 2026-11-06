# src/evaluation/alert_policy_comparison.py
"""Consolidated per-episode comparison of operational alert policies.

The existing ``operational_policy`` module prices window-level alerts. This
module adds the episode view requested for operational reviews: every alert
mode is described by its window confusion counts, engine metrics, degradation
episode detection and the cost of the selected decision unit. The mode is
selected on validation only; the official test split is reported for
information and never drives the choice.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, Optional

import numpy as np

from src.evaluation.alerting import (
    hysteresis_predictions,
    resolve_hysteresis_thresholds,
)
from src.evaluation.operational_policy import (
    SELECTION_METRICS,
    SELECTION_UNITS,
    _metric_values,
    _selection_key,
    evaluate_alert_mode,
    validate_costs,
)

POLICY_MODES: tuple = (
    'immediate',
    'consecutive_2',
    'consecutive_3',
    'smoothed_3',
    'hysteresis',
)

CSV_FIELDS = [
    'mode', 'split', 'matched', 'threshold', 'recall', 'precision', 'f1',
    'mean_recall', 'mean_precision', 'episode_recall', 'episode_precision',
    'real_episodes', 'detected_episodes', 'false_alarm_episodes',
    'total_false_alarms', 'mean_lead_time_cycles',
    'mean_anticipation_cycles', 'episode_mean_anticipation_cycles',
    'episode_mean_alert_to_end_cycles', 'expected_cost', 'expected_cost_window',
    'expected_cost_engine', 'expected_cost_episode', 'cost_unit',
    'capacity_feasible', 'selected',
]


def _hysteresis_predictions(scores: np.ndarray, units: np.ndarray,
                            cycles: np.ndarray, threshold: float,
                            config: Optional[Dict[str, Any]]) -> tuple:
    config = config or {}
    enter, exit_value = resolve_hysteresis_thresholds(
        threshold,
        config.get('enter_threshold'),
        config.get('exit_threshold'),
        margin=float(config.get('margin', 0.05)),
    )
    return (
        hysteresis_predictions(scores, units, cycles, enter, exit_value),
        {'enter_threshold': enter, 'exit_threshold': exit_value},
    )


def _evaluate_modes(y_true: np.ndarray, scores: np.ndarray,
                    units: np.ndarray, cycles: np.ndarray, rul: np.ndarray,
                    threshold: float, modes: Iterable[str],
                    fn_cost: float, fa_cost: float, cost_unit: str,
                    capacity: Optional[float],
                    hysteresis_config: Optional[Dict[str, Any]],
                    ) -> tuple:
    reports: Dict[str, Dict[str, Any]] = {}
    hysteresis_thresholds = None
    for mode in modes:
        if mode == 'hysteresis':
            predictions, hysteresis_thresholds = _hysteresis_predictions(
                scores, units, cycles, threshold, hysteresis_config
            )
            report = evaluate_alert_mode(
                y_true, scores, units, cycles, rul, threshold, mode,
                fn_cost, fa_cost, capacity, cost_unit,
                predictions_override=predictions,
            )
        else:
            report = evaluate_alert_mode(
                y_true, scores, units, cycles, rul, threshold, mode,
                fn_cost, fa_cost, capacity, cost_unit,
            )
        reports[mode] = report
    return reports, hysteresis_thresholds


def _with_deltas(reports: Dict[str, Dict[str, Any]],
                 cost_unit: str) -> None:
    baseline = reports.get('immediate')
    if baseline is None:
        return
    baseline_values = _metric_values(baseline, cost_unit)
    for mode, report in reports.items():
        if mode == 'immediate':
            report['delta_vs_immediate'] = {}
            continue
        values = _metric_values(report, cost_unit)
        report['delta_vs_immediate'] = {
            'recall': float(values['recall'] - baseline_values['recall']),
            'precision': float(
                values['precision'] - baseline_values['precision']
            ),
            'f1': float(values['f1'] - baseline_values['f1']),
            'episode_recall': float(
                report['episode_recall'] - baseline['episode_recall']
            ),
            'false_alarm_episodes': int(
                report['false_alarm_episodes']
                - baseline['false_alarm_episodes']
            ),
            'expected_cost': float(
                report['expected_cost'] - baseline['expected_cost']
            ),
        }


def build_alert_policy_comparison(
        y_val: np.ndarray, val_scores: np.ndarray,
        val_units: np.ndarray, val_cycles: np.ndarray, val_rul: np.ndarray,
        y_test: np.ndarray, test_scores: np.ndarray,
        test_units: np.ndarray, test_cycles: np.ndarray,
        test_rul: np.ndarray, threshold: float,
        config: Optional[Dict[str, Any]] = None,
        modes: Iterable[str] = POLICY_MODES) -> Dict[str, Any]:
    """Compare alert policies by window, engine and episode metrics."""
    config = config or {}
    modes = tuple(dict.fromkeys(str(mode) for mode in modes))
    invalid = [mode for mode in modes if mode not in POLICY_MODES]
    if invalid:
        raise ValueError(
            f"Unsupported alert comparison mode(s): {invalid}. "
            f"Allowed: {list(POLICY_MODES)}."
        )
    fn_cost, fa_cost = validate_costs(
        config.get('false_negative_cost', 10.0),
        config.get('false_alarm_cost', 1.0),
    )
    metric = str(config.get('selection_metric', 'expected_cost')).lower()
    if metric not in SELECTION_METRICS:
        raise ValueError(
            f"alert comparison selection_metric must be one of "
            f"{SELECTION_METRICS}, got {metric!r}."
        )
    cost_unit = str(
        config.get('cost_unit', config.get('selection_unit', 'episode'))
    ).lower()
    if cost_unit not in SELECTION_UNITS:
        raise ValueError(
            f"alert comparison cost_unit must be one of {SELECTION_UNITS}, "
            f"got {cost_unit!r}."
        )
    capacity = config.get('inspection_capacity_per_cycle')
    if capacity is not None:
        capacity = float(capacity)
        if capacity < 0.0:
            raise ValueError(
                "inspection_capacity_per_cycle must be >= 0 or null."
            )
    beta = float(config.get('beta', 2.0))
    hysteresis_config = config.get('hysteresis') or {}

    validation, hysteresis_thresholds = _evaluate_modes(
        y_val, val_scores, val_units, val_cycles, val_rul,
        threshold, modes, fn_cost, fa_cost, cost_unit, capacity,
        hysteresis_config,
    )
    test, test_hysteresis = _evaluate_modes(
        y_test, test_scores, test_units, test_cycles, test_rul,
        threshold, modes, fn_cost, fa_cost, cost_unit, capacity,
        hysteresis_config,
    )

    feasible = [
        mode for mode, report in validation.items()
        if report['capacity_feasible']
    ]
    candidates = feasible or list(validation)
    if len(candidates) < len(validation):
        for report in validation.values():
            report['excluded_by_capacity'] = not report['capacity_feasible']

    selected = max(
        candidates,
        key=lambda mode: _selection_key(
            validation[mode], metric, beta, cost_unit
        ),
    )
    selection_reason = f'best_validation_{metric}_{cost_unit}'
    for mode, report in validation.items():
        report['selected'] = mode == selected
        report['selection_reason'] = (
            selection_reason if mode == selected else None
        )
    for mode, report in test.items():
        report['selected'] = mode == selected

    _with_deltas(validation, cost_unit)
    _with_deltas(test, cost_unit)

    chosen = validation[selected]
    immediate = validation['immediate']
    return {
        'selection_split': 'validation',
        'test_usage': 'reporting_only',
        'threshold': float(threshold),
        'hysteresis_thresholds': hysteresis_thresholds,
        'test_hysteresis_thresholds': test_hysteresis,
        'false_negative_cost': fn_cost,
        'false_alarm_cost': fa_cost,
        'cost_unit': cost_unit,
        'selection_metric': metric,
        'inspection_capacity_per_cycle': capacity,
        'capacity_feasible_modes': feasible,
        'selected_mode': selected,
        'selection_reason': selection_reason,
        'validation': validation,
        'test': test,
        'rationale': {
            'selected_mode': selected,
            'cost_unit': cost_unit,
            'validation_cost': float(chosen['expected_cost']),
            'immediate_validation_cost': float(immediate['expected_cost']),
            'validation_episode_recall': float(chosen['episode_recall']),
            'validation_episode_precision': float(chosen['episode_precision']),
            'false_alarm_episodes': int(chosen['false_alarm_episodes']),
            'reasons': [
                (
                    f"{selected} minimises validation {cost_unit} cost "
                    f"({chosen['expected_cost']:.2f}) under cost weights "
                    f"FN={fn_cost:g}, FA={fa_cost:g}."
                ),
                (
                    f"It detects {chosen['detected_episodes']}/"
                    f"{chosen['real_episodes']} degradation episodes with "
                    f"{chosen['false_alarm_episodes']} false-alarm episode(s)."
                ),
                (
                    "Selection uses validation only; test metrics are "
                    "reported for information."
                ),
            ],
        },
    }


def save_alert_policy_comparison(report: Dict[str, Any], path: str) -> str:
    """Persist the consolidated alert-policy comparison as JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(report, file, indent=2)
    return path


def export_alert_policy_comparison_csv(report: Dict[str, Any],
                                       path: str) -> str:
    """Write one row per mode and split, including episode metrics."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS,
                                extrasaction='ignore')
        writer.writeheader()
        for split in ('validation', 'test'):
            for mode, values in report.get(split, {}).items():
                row = {
                    **values,
                    'mode': mode,
                    'split': split,
                    'matched': int(bool(values.get('selected'))),
                    'cost_unit': report.get('cost_unit'),
                }
                writer.writerow(row)
    return path
