# src/evaluation/cost_sensitivity.py
"""Validation-only sensitivity of alert policies to the cost ratio.

The ratio ``false_negative_cost / false_alarm_cost`` is an assumption, not a
measurement.  This module replays each ratio on validation, selects the best
alert mode there, and reports its test behaviour.  The official test split is
never used for selection.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.evaluation.operational_policy import (
    ALERT_MODES,
    _selection_key,
    evaluate_alert_mode,
)

CSV_FIELDS = [
    'ratio', 'false_negative_cost', 'false_alarm_cost', 'selected_mode',
    'validation_cost_window', 'validation_cost_engine',
    'validation_cost_episode', 'test_cost_window', 'test_cost_engine',
    'test_cost_episode', 'missed_episodes', 'false_alarm_episodes',
    'false_alarms_per_100_cycles', 'mean_anticipation_cycles',
]


def _false_alarms_per_100(report: Dict[str, Any]) -> float:
    cycles = int(report.get('cycles_observed') or 0)
    if not cycles:
        return 0.0
    return float(100.0 * report['false_positives'] / cycles)


def build_cost_sensitivity(
        y_val: np.ndarray, val_scores: np.ndarray, val_units: np.ndarray,
        val_cycles: np.ndarray, val_rul: np.ndarray,
        y_test: np.ndarray, test_scores: np.ndarray, test_units: np.ndarray,
        test_cycles: np.ndarray, test_rul: np.ndarray, threshold: float,
        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Replay cost ratios and select one alert mode per ratio on validation."""
    config = config or {}
    ratios = config.get(
        'false_negative_cost_ratios', [1, 5, 10, 25, 50, 100]
    )
    if not isinstance(ratios, list) or not ratios:
        raise ValueError(
            "cost_sensitivity.false_negative_cost_ratios must be a non-empty "
            "list."
        )
    fa_cost = float(config.get('false_alarm_cost', 1.0))
    if not np.isfinite(fa_cost) or fa_cost <= 0.0:
        raise ValueError("cost_sensitivity.false_alarm_cost must be > 0.")
    selection_unit = str(config.get('selection_unit', 'window')).lower()
    if selection_unit not in {'window', 'engine', 'episode'}:
        raise ValueError(
            "cost_sensitivity.selection_unit must be window, engine or episode."
        )
    split = str(config.get('selection_split', 'validation')).lower()
    if split != 'validation':
        raise ValueError(
            "cost_sensitivity.selection_split must be 'validation'."
        )
    beta = float(config.get('beta', 2.0))

    rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        ratio = float(ratio)
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(
                "cost_sensitivity ratios must be positive finite numbers."
            )
        fn_cost = ratio * fa_cost
        validation = {
            mode: evaluate_alert_mode(
                y_val, val_scores, val_units, val_cycles, val_rul,
                threshold, mode, fn_cost, fa_cost, None, selection_unit,
            )
            for mode in ALERT_MODES
        }
        selected = max(
            validation,
            key=lambda mode: _selection_key(
                validation[mode], 'expected_cost', beta, selection_unit
            ),
        )
        test = evaluate_alert_mode(
            y_test, test_scores, test_units, test_cycles, test_rul,
            threshold, selected, fn_cost, fa_cost, None, selection_unit,
        )
        chosen = validation[selected]
        rows.append({
            'ratio': ratio,
            'false_negative_cost': fn_cost,
            'false_alarm_cost': fa_cost,
            'selected_mode': selected,
            'validation_cost_window': chosen['expected_cost_window'],
            'validation_cost_engine': chosen['expected_cost_engine'],
            'validation_cost_episode': chosen['expected_cost_episode'],
            'test_cost_window': test['expected_cost_window'],
            'test_cost_engine': test['expected_cost_engine'],
            'test_cost_episode': test['expected_cost_episode'],
            'missed_episodes': max(
                0, int(test['real_episodes']) - int(test['detected_episodes'])
            ),
            'false_alarm_episodes': int(test['false_alarm_episodes']),
            'false_alarms_per_100_cycles': _false_alarms_per_100(test),
            'mean_anticipation_cycles': test['mean_anticipation_cycles'],
        })
    return {
        'selection_split': 'validation',
        'test_usage': 'reporting_only',
        'selection_unit': selection_unit,
        'threshold': float(threshold),
        'rows': rows,
    }


def save_cost_sensitivity(report: Dict[str, Any], results_dir: str) -> Dict[str, str]:
    """Write JSON, CSV and a cost-ratio plot."""
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, 'cost_sensitivity.json')
    csv_path = os.path.join(results_dir, 'cost_sensitivity.csv')
    plot_path = os.path.join(
        results_dir, 'visualization', 'cost_sensitivity.png'
    )
    with open(json_path, 'w') as file:
        json.dump(report, file, indent=2)
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in report['rows']:
            writer.writerow(row)

    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    ratios = [row['ratio'] for row in report['rows']]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ratios, [row['test_cost_episode'] for row in report['rows']],
            marker='o', label='Test episode cost')
    ax.plot(ratios, [row['validation_cost_episode'] for row in report['rows']],
            marker='s', label='Validation episode cost')
    ax.set_xscale('log')
    ax.set_xlabel('False-negative / false-alarm cost ratio')
    ax.set_ylabel('Expected cost')
    ax.set_title('Cost sensitivity (mode selected on validation)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return {
        'json_path': json_path, 'csv_path': csv_path, 'plot_path': plot_path,
    }
