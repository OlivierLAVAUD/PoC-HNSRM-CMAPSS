# src/evaluation/engine_metrics.py
"""Engine-level predictive-maintenance metrics for C-MAPSS."""

import csv
import os
from typing import Any, Dict, Optional

import numpy as np

CSV_FIELDS = [
    'unit', 'windows', 'positive_windows', 'predicted_positives',
    'true_positives', 'false_positives', 'false_negatives', 'true_negatives',
    'recall', 'precision', 'false_alarms', 'first_alert_cycle',
    'first_correct_alert_cycle', 'eol_cycle', 'lead_time_cycles',
    'anticipation_cycles', 'alerted_before_eol',
]


def _mean_defined(values):
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else 0.0


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def compute_engine_metrics(unit_ids: np.ndarray, cycles: np.ndarray,
                           rul: np.ndarray, y_true: np.ndarray,
                           y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute per-engine detection, false-alarm and anticipation metrics."""
    unit_ids = np.asarray(unit_ids).ravel()
    cycles = np.asarray(cycles).ravel()
    rul = np.asarray(rul).ravel()
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if not (len(unit_ids) == len(cycles) == len(rul) == len(y_true) == len(y_pred)):
        raise ValueError("Engine metric inputs must have the same length.")

    rows = []
    for unit in sorted(np.unique(unit_ids)):
        mask = unit_ids == unit
        unit_true = y_true[mask]
        unit_pred = y_pred[mask]
        unit_cycles = cycles[mask]
        unit_rul = rul[mask]

        tp = int(np.sum((unit_pred == 1) & (unit_true == 1)))
        fp = int(np.sum((unit_pred == 1) & (unit_true == 0)))
        fn = int(np.sum((unit_pred == 0) & (unit_true == 1)))
        tn = int(np.sum((unit_pred == 0) & (unit_true == 0)))

        correct_alert = (unit_pred == 1) & (unit_true == 1)
        any_alert = unit_pred == 1
        first_correct_cycle = (
            int(unit_cycles[np.argmax(correct_alert)]) if correct_alert.any() else None
        )
        first_alert_cycle = (
            int(unit_cycles[np.argmax(any_alert)]) if any_alert.any() else None
        )
        first_correct_rul = (
            int(unit_rul[np.argmax(correct_alert)]) if correct_alert.any() else None
        )
        eol_cycle = int(unit_cycles.max())
        lead_time = (
            eol_cycle - first_correct_cycle if first_correct_cycle is not None else None
        )
        anticipation = first_correct_rul
        alerted_before_eol = bool(
            first_correct_cycle is not None and first_correct_cycle < eol_cycle
        )

        rows.append({
            'unit': int(unit),
            'windows': int(mask.sum()),
            'positive_windows': int(unit_true.sum()),
            'predicted_positives': int(unit_pred.sum()),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'recall': _ratio(tp, tp + fn),
            'precision': _ratio(tp, tp + fp),
            'false_alarms': fp,
            'first_alert_cycle': first_alert_cycle,
            'first_correct_alert_cycle': first_correct_cycle,
            'eol_cycle': eol_cycle,
            'lead_time_cycles': lead_time,
            'anticipation_cycles': anticipation,
            'alerted_before_eol': alerted_before_eol,
        })

    relevant_rows = [row for row in rows if row['positive_windows'] > 0]
    summary = {
        'engines': len(rows),
        'engines_with_positive_windows': len(relevant_rows),
        'mean_recall': _mean_defined(row['recall'] for row in rows),
        'mean_precision': _mean_defined(row['precision'] for row in rows),
        'total_false_alarms': int(sum(row['false_alarms'] for row in rows)),
        'mean_false_alarms_per_engine': _mean_defined(
            row['false_alarms'] for row in rows
        ),
        'mean_lead_time_cycles': _mean_defined(
            row['lead_time_cycles'] for row in rows
        ),
        'mean_anticipation_cycles': _mean_defined(
            row['anticipation_cycles'] for row in rows
        ),
        'proportion_alerted_before_eol': _mean_defined(
            1.0 if row['alerted_before_eol'] else 0.0 for row in rows
        ),
        'proportion_alerted_before_eol_among_relevant': _mean_defined(
            1.0 if row['alerted_before_eol'] else 0.0 for row in relevant_rows
        ),
    }
    return {'summary': summary, 'per_engine': rows}


def export_engine_metrics_csv(metrics: Dict[str, Any], path: str) -> str:
    """Write the per-engine table to CSV."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in metrics.get('per_engine', []):
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})
    return path
