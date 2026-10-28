# src/experiments/threshold_sensitivity.py
"""Sensitivity of the FD001 decision threshold on validation-selected rules.

Every candidate follows the production contract: the threshold is chosen on
validation scores only and then reported on validation and on the official
test engines. Alert modes are evaluated for each candidate so the interaction
between the threshold and the confirmation logic stays visible.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.evaluation.engine_metrics import compute_engine_metrics
from src.evaluation.operational_policy import (
    ALERT_MODES,
    evaluate_alert_mode,
    validate_costs,
    window_metrics,
)
from src.evaluation.thresholding import select_threshold

CSV_FIELDS: List[str] = [
    'strategy', 'min_precision', 'beta', 'threshold', 'fallback_used',
    'validation_precision', 'validation_recall', 'validation_f1',
    'validation_fbeta', 'validation_roc_auc', 'validation_average_precision',
    'validation_false_alarms', 'validation_missed_critical_windows',
    'test_precision', 'test_recall', 'test_f1', 'test_fbeta',
    'test_roc_auc', 'test_average_precision', 'test_false_alarms',
    'test_missed_critical_windows', 'test_mean_anticipation_cycles',
    'test_proportion_alerted_before_eol',
    'test_immediate_false_alarms', 'test_immediate_recall',
    'test_consecutive_2_false_alarms', 'test_consecutive_2_recall',
    'test_consecutive_3_false_alarms', 'test_consecutive_3_recall',
    'test_smoothed_3_false_alarms', 'test_smoothed_3_recall',
    'test_expected_cost',
]

SENSITIVITY_MODES: tuple = (
    'immediate', 'consecutive_2', 'consecutive_3', 'smoothed_3', 'smoothed_5',
)


def build_candidates(config: Optional[Dict[str, Any]] = None
                     ) -> List[Dict[str, Any]]:
    """Expand the configured grid into explicit threshold-rule candidates."""
    config = config or {}
    section = config.get('threshold_sensitivity', config)
    strategies = section.get(
        'strategies', ['recall_at_precision', 'fbeta', 'f1']
    )
    min_precision_values = section.get(
        'min_precision_values', [0.65, 0.70, 0.75, 0.80, 0.85]
    )
    beta_values = section.get('beta_values', [1.0, 1.5, 2.0, 3.0])

    candidates: List[Dict[str, Any]] = []
    seen = set()
    for strategy in strategies:
        name = str(strategy).lower()
        if name == 'recall_at_precision':
            for value in min_precision_values or [None]:
                key = (name, None if value is None else float(value), None)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    'strategy': name,
                    'min_precision': None if value is None else float(value),
                    'beta': None,
                })
        elif name == 'fbeta':
            for value in beta_values or [None]:
                key = (name, None, None if value is None else float(value))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    'strategy': name,
                    'min_precision': None,
                    'beta': None if value is None else float(value),
                })
        elif name == 'f1':
            key = (name, None, None)
            if key not in seen:
                seen.add(key)
                candidates.append({
                    'strategy': name, 'min_precision': None, 'beta': None,
                })
        else:
            raise ValueError(
                f"Unsupported threshold_sensitivity strategy: {strategy!r}"
            )
    if not candidates:
        raise ValueError("threshold_sensitivity produced no candidate.")
    return candidates


def candidate_label(candidate: Dict[str, Any]) -> str:
    """Return a short human-readable label for one candidate."""
    strategy = candidate['strategy']
    if strategy == 'recall_at_precision':
        return f"recall@prec>={candidate['min_precision']:.2f}"
    if strategy == 'fbeta':
        return f"fbeta(beta={candidate['beta']:.2f})"
    return 'f1'


def _split_metrics(y_true: np.ndarray, scores: np.ndarray,
                   units: np.ndarray, cycles: np.ndarray,
                   rul: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Return window and engine metrics for one split at a fixed threshold."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    predictions = (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)
    window = window_metrics(y_true, predictions)
    engine = compute_engine_metrics(
        units, cycles, rul, y_true, predictions
    )['summary']
    metrics = {
        **window,
        'engine_mean_recall': float(engine['mean_recall']),
        'engine_mean_precision': float(engine['mean_precision']),
        'total_false_alarms': int(engine['total_false_alarms']),
        'missed_critical_windows': int(window['false_negatives']),
        'mean_anticipation_cycles': float(engine['mean_anticipation_cycles']),
        'mean_lead_time_cycles': float(engine['mean_lead_time_cycles']),
        'proportion_alerted_before_eol': float(
            engine['proportion_alerted_before_eol']
        ),
    }
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, scores))
        metrics['average_precision'] = float(
            average_precision_score(y_true, scores)
        )
    except ValueError:
        metrics['roc_auc'] = None
        metrics['average_precision'] = None
    return metrics


def evaluate_candidate(
        candidate: Dict[str, Any],
        y_val: np.ndarray, val_scores: np.ndarray,
        val_units: np.ndarray, val_cycles: np.ndarray, val_rul: np.ndarray,
        y_test: np.ndarray, test_scores: np.ndarray,
        test_units: np.ndarray, test_cycles: np.ndarray,
        test_rul: np.ndarray,
        threshold_config: Optional[Dict[str, Any]] = None,
        policy_config: Optional[Dict[str, Any]] = None,
        modes: Iterable[str] = SENSITIVITY_MODES) -> Dict[str, Any]:
    """Select one threshold on validation and report it on both splits."""
    threshold_config = threshold_config or {}
    policy_config = policy_config or {}
    fallback = float(threshold_config.get('fallback', 0.5))
    selection = select_threshold(
        y_val, val_scores,
        strategy=candidate['strategy'],
        min_precision=float(
            candidate['min_precision']
            if candidate['min_precision'] is not None
            else threshold_config.get('min_precision', 0.75)
        ),
        beta=float(
            candidate['beta']
            if candidate['beta'] is not None
            else threshold_config.get('beta', 2.0)
        ),
        fallback=fallback,
    )
    threshold = float(selection['threshold'])
    validation = _split_metrics(
        y_val, val_scores, val_units, val_cycles, val_rul, threshold
    )
    test = _split_metrics(
        y_test, test_scores, test_units, test_cycles, test_rul, threshold
    )
    beta = float(
        candidate['beta']
        if candidate['beta'] is not None
        else threshold_config.get('beta', 2.0)
    )
    for metrics in (validation, test):
        precision = float(metrics['precision'])
        recall = float(metrics['recall'])
        beta_sq = beta * beta
        denominator = beta_sq * precision + recall
        metrics['fbeta'] = (
            (1.0 + beta_sq) * precision * recall / denominator
            if denominator else 0.0
        )
        metrics['fbeta_beta'] = beta

    fn_cost, fa_cost = validate_costs(
        policy_config.get('false_negative_cost', 10.0),
        policy_config.get('false_alarm_cost', 1.0),
    )
    capacity = policy_config.get('inspection_capacity_per_cycle')
    mode_reports = {
        mode: evaluate_alert_mode(
            y_test, test_scores, test_units, test_cycles, test_rul,
            threshold, mode, fn_cost, fa_cost,
            None if capacity is None else float(capacity),
        )
        for mode in modes
    }
    validation_modes = {
        mode: evaluate_alert_mode(
            y_val, val_scores, val_units, val_cycles, val_rul,
            threshold, mode, fn_cost, fa_cost,
            None if capacity is None else float(capacity),
        )
        for mode in modes
    }
    return {
        'candidate': {
            **candidate,
            'label': candidate_label(candidate),
        },
        'threshold': threshold,
        'selection': {
            'strategy': selection['strategy'],
            'beta': selection['beta'],
            'min_precision': selection['min_precision'],
            'fallback_used': selection['fallback_used'],
            'reason': selection['reason'],
            'selected_metrics': selection['selected_metrics'],
            'candidates_evaluated': selection['candidates_evaluated'],
        },
        'validation': validation,
        'test': test,
        'alert_modes': {
            'validation': validation_modes,
            'test': mode_reports,
        },
        'expected_cost_test': float(mode_reports['immediate']['expected_cost']),
    }


def _candidate_sort_key(result: Dict[str, Any]) -> tuple:
    """Order candidates from most to least recall for readable exports."""
    return (
        -float(result['test']['recall']),
        -float(result['threshold']),
        result['candidate']['strategy'],
    )


def run_threshold_sensitivity(
        y_val: np.ndarray, val_scores: np.ndarray,
        val_units: np.ndarray, val_cycles: np.ndarray, val_rul: np.ndarray,
        y_test: np.ndarray, test_scores: np.ndarray,
        test_units: np.ndarray, test_cycles: np.ndarray,
        test_rul: np.ndarray, config: Dict[str, Any],
        results_dir: str) -> Dict[str, Any]:
    """Evaluate the configured threshold grid and persist JSON, CSV and plot."""
    candidates = build_candidates(config)
    results = [
        evaluate_candidate(
            candidate, y_val, val_scores, val_units, val_cycles, val_rul,
            y_test, test_scores, test_units, test_cycles, test_rul,
            threshold_config=config.get('thresholding', {}),
            policy_config=config.get('operational_policy', {}),
        )
        for candidate in candidates
    ]
    results.sort(key=_candidate_sort_key)

    production = {
        'strategy': config.get('thresholding', {}).get(
            'strategy', 'recall_at_precision'
        ),
        'min_precision': float(
            config.get('thresholding', {}).get('min_precision', 0.75)
        ),
        'beta': float(config.get('thresholding', {}).get('beta', 2.0)),
    }
    matching = [
        item for item in results
        if item['candidate']['strategy'] == production['strategy']
        and (
            production['min_precision'] is None
            or item['candidate']['min_precision'] == production['min_precision']
        )
    ]
    report = {
        'selection_split': 'validation',
        'test_usage': 'reporting_only',
        'production_setting': production,
        'production_candidate': matching[0] if matching else None,
        'candidates': results,
        'grid': {
            'strategies': sorted({item['candidate']['strategy']
                                  for item in results}),
            'min_precision_values': [
                item['candidate']['min_precision'] for item in results
                if item['candidate']['min_precision'] is not None
            ],
            'beta_values': [
                item['candidate']['beta'] for item in results
                if item['candidate']['beta'] is not None
            ],
        },
    }

    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, 'threshold_sensitivity.json')
    with open(json_path, 'w') as file:
        json.dump(report, file, indent=2)
    csv_path = export_sensitivity_csv(
        report, os.path.join(results_dir, 'threshold_sensitivity.csv')
    )
    plot_path = save_sensitivity_plot(
        report,
        os.path.join(results_dir, 'visualization',
                     'threshold_sensitivity.png'),
    )
    report['json_path'] = json_path
    report['csv_path'] = csv_path
    report['plot_path'] = plot_path
    return report


def export_sensitivity_csv(report: Dict[str, Any], path: str) -> str:
    """Write one row per threshold candidate to CSV."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in report.get('candidates', []):
            validation = item['validation']
            test = item['test']
            modes = item['alert_modes']['test']
            row = {
                'strategy': item['candidate']['strategy'],
                'min_precision': item['candidate']['min_precision'],
                'beta': item['candidate']['beta'],
                'threshold': item['threshold'],
                'fallback_used': item['selection']['fallback_used'],
                'validation_precision': validation['precision'],
                'validation_recall': validation['recall'],
                'validation_f1': validation['f1'],
                'validation_fbeta': validation['fbeta'],
                'validation_roc_auc': validation['roc_auc'],
                'validation_average_precision': validation['average_precision'],
                'validation_false_alarms': validation['total_false_alarms'],
                'validation_missed_critical_windows': validation[
                    'missed_critical_windows'
                ],
                'test_precision': test['precision'],
                'test_recall': test['recall'],
                'test_f1': test['f1'],
                'test_fbeta': test['fbeta'],
                'test_roc_auc': test['roc_auc'],
                'test_average_precision': test['average_precision'],
                'test_false_alarms': test['total_false_alarms'],
                'test_missed_critical_windows': test['missed_critical_windows'],
                'test_mean_anticipation_cycles': test['mean_anticipation_cycles'],
                'test_proportion_alerted_before_eol': test[
                    'proportion_alerted_before_eol'
                ],
                'test_expected_cost': item['expected_cost_test'],
            }
            for mode in ('immediate', 'consecutive_2', 'consecutive_3',
                         'smoothed_3'):
                row[f'test_{mode}_false_alarms'] = modes[mode][
                    'total_false_alarms'
                ]
                row[f'test_{mode}_recall'] = modes[mode]['mean_recall']
            writer.writerow(row)
    return path


def save_sensitivity_plot(report: Dict[str, Any], path: str) -> str:
    """Plot precision, false alarms and anticipation against recall."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    candidates = report.get('candidates', [])
    if not candidates:
        raise ValueError("No threshold candidates to plot.")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    recall = np.asarray([item['test']['recall'] for item in candidates])
    precision = np.asarray([item['test']['precision'] for item in candidates])
    false_alarms = np.asarray(
        [item['test']['total_false_alarms'] for item in candidates]
    )
    anticipation = np.asarray(
        [item['test']['mean_anticipation_cycles'] for item in candidates]
    )
    thresholds = np.asarray([item['threshold'] for item in candidates])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    scatter = axes[0].scatter(
        recall, precision, c=thresholds, cmap='viridis', s=70,
        edgecolor='black', linewidth=0.5,
    )
    axes[0].set_xlabel('Test recall (window level)')
    axes[0].set_ylabel('Test precision (window level)')
    axes[0].set_title('Precision vs recall by threshold rule')
    fig.colorbar(scatter, ax=axes[0], label='fusion threshold')

    axes[1].scatter(recall, false_alarms, color='tab:red', s=60)
    axes[1].set_xlabel('Test recall (window level)')
    axes[1].set_ylabel('False alarms (engine level)')
    axes[1].set_title('False alarms vs recall')

    axes[2].scatter(recall, anticipation, color='tab:green', s=60)
    axes[2].set_xlabel('Test recall (window level)')
    axes[2].set_ylabel('Mean anticipation (cycles)')
    axes[2].set_title('Anticipation vs recall')

    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle('Fusion threshold sensitivity (thresholds selected on validation)')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
