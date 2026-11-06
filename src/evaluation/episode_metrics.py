# src/evaluation/episode_metrics.py
"""Episode-level evaluation for consecutive degradation and alert runs.

Window-level metrics can hide the operational picture: a degradation episode
may be detected by a fragmented alert, while a single isolated false positive
is not the same event as a sustained false-alarm episode. This module groups
consecutive positive windows per engine into *degradation episodes* and
consecutive alert windows into *alert episodes*, then reports how many real
episodes were detected, how many alert episodes were false alarms, and the
anticipation obtained at the first correct alert.

The official test split is only used for reporting when this module is called
from the pipeline; no threshold or model decision is derived here.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

CSV_FIELDS: Sequence[str] = (
    'kind', 'unit', 'episode_index', 'start_cycle', 'end_cycle',
    'cycles', 'windows', 'detected', 'matched_alert_episodes',
    'false_alarm', 'first_correct_alert_cycle', 'alert_to_end_cycles',
    'anticipation_cycles',
)


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _ordered_groups(unit_ids: np.ndarray,
                    cycles: np.ndarray) -> List[np.ndarray]:
    """Return one cycle-sorted index group per engine."""
    units = np.asarray(unit_ids).ravel()
    cycle_values = np.asarray(cycles).ravel()
    if len(units) != len(cycle_values):
        raise ValueError("unit_ids and cycles must have the same length.")
    order = np.lexsort((cycle_values, units))
    sorted_units = units[order]
    if not len(order):
        return []
    boundaries = np.flatnonzero(np.diff(sorted_units)) + 1
    return np.split(order, boundaries)


def _episodes(indices: np.ndarray, flags: np.ndarray,
              cycles: np.ndarray) -> List[Dict[str, Any]]:
    """Split a cycle-ordered index group into consecutive positive runs."""
    episodes: List[Dict[str, Any]] = []
    current: List[int] = []
    for index in indices:
        if int(flags[index]) == 1:
            current.append(int(index))
            continue
        if current:
            episodes.append(_episode_payload(current, cycles))
            current = []
    if current:
        episodes.append(_episode_payload(current, cycles))
    return episodes


def _episode_payload(indices: List[int],
                     cycles: np.ndarray) -> Dict[str, Any]:
    cycle_values = np.asarray(cycles).ravel()[indices]
    return {
        'indices': list(indices),
        'start_cycle': int(cycle_values.min()),
        'end_cycle': int(cycle_values.max()),
        'cycles': int(cycle_values.max() - cycle_values.min() + 1),
        'windows': len(indices),
    }


def compute_episode_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            unit_ids: np.ndarray, cycles: np.ndarray,
                            rul: Optional[np.ndarray] = None
                            ) -> Dict[str, Any]:
    """Compute episode-level detection, precision and anticipation metrics.

    A real episode is a maximal run of consecutive positive windows for one
    engine. An alert episode is a maximal run of consecutive predicted alerts
    for one engine. A real episode is *detected* when at least one of its
    windows is predicted positive, so a fragmented alert still counts once at
    the real-episode level. An alert episode is a *false alarm* when it does
    not overlap any real episode.

    ``rul`` is optional; when supplied, the anticipation of a detected
    episode is the RUL at its first correct alert, matching the engine-level
    definition used elsewhere in the project.
    """
    true = np.asarray(y_true, dtype=int).ravel()
    pred = np.asarray(y_pred, dtype=int).ravel()
    units = np.asarray(unit_ids).ravel()
    cycle_values = np.asarray(cycles).ravel()
    if not (len(true) == len(pred) == len(units) == len(cycle_values)):
        raise ValueError("Episode metric inputs must have the same length.")
    rul_values = None
    if rul is not None:
        rul_values = np.asarray(rul, dtype=float).ravel()
        if len(rul_values) != len(true):
            raise ValueError("rul must have one value per window.")

    real_episodes: List[Dict[str, Any]] = []
    alert_episodes: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for group in _ordered_groups(units, cycle_values):
        unit_real = _episodes(group, true, cycle_values)
        unit_alert = _episodes(group, pred, cycle_values)
        for episode in unit_real:
            episode['unit'] = int(units[group[0]])
            episode['matched_alert_indices'] = []
        for episode in unit_alert:
            episode['unit'] = int(units[group[0]])
            episode['matched_real_indices'] = []

        for real_index, real in enumerate(unit_real):
            for alert_index, alert in enumerate(unit_alert):
                if set(real['indices']) & set(alert['indices']):
                    real['matched_alert_indices'].append(alert_index)
                    alert['matched_real_indices'].append(real_index)

        real_episodes.extend(unit_real)
        alert_episodes.extend(unit_alert)

    detected = 0
    false_alarms = 0
    matched_alerts = 0
    alert_to_end: List[Optional[float]] = []
    anticipation: List[Optional[float]] = []

    for index, real in enumerate(real_episodes):
        is_detected = bool(real['matched_alert_indices'])
        detected += int(is_detected)
        first_correct_cycle: Optional[int] = None
        anticipation_cycles: Optional[float] = None
        alert_to_end_cycles: Optional[float] = None
        if is_detected:
            correct = [
                int(cycle_values[position])
                for position in real['indices']
                if int(pred[position]) == 1
            ]
            first_correct_cycle = min(correct)
            alert_to_end_cycles = float(
                real['end_cycle'] - first_correct_cycle
            )
            if rul_values is not None:
                first_position = next(
                    position for position in real['indices']
                    if int(cycle_values[position]) == first_correct_cycle
                )
                anticipation_cycles = float(rul_values[first_position])
            alert_to_end.append(alert_to_end_cycles)
            anticipation.append(anticipation_cycles)
        detail_rows.append({
            'kind': 'real',
            'unit': real['unit'],
            'episode_index': index,
            'start_cycle': real['start_cycle'],
            'end_cycle': real['end_cycle'],
            'cycles': real['cycles'],
            'windows': real['windows'],
            'detected': int(is_detected),
            'matched_alert_episodes': len(real['matched_alert_indices']),
            'false_alarm': 0,
            'first_correct_alert_cycle': first_correct_cycle,
            'alert_to_end_cycles': alert_to_end_cycles,
            'anticipation_cycles': anticipation_cycles,
        })

    for index, alert in enumerate(alert_episodes):
        is_false_alarm = not bool(alert['matched_real_indices'])
        false_alarms += int(is_false_alarm)
        matched_alerts += int(not is_false_alarm)
        detail_rows.append({
            'kind': 'alert',
            'unit': alert['unit'],
            'episode_index': index,
            'start_cycle': alert['start_cycle'],
            'end_cycle': alert['end_cycle'],
            'cycles': alert['cycles'],
            'windows': alert['windows'],
            'detected': 0,
            'matched_alert_episodes': len(alert['matched_real_indices']),
            'false_alarm': int(is_false_alarm),
            'first_correct_alert_cycle': None,
            'alert_to_end_cycles': None,
            'anticipation_cycles': None,
        })

    real_count = len(real_episodes)
    alert_count = len(alert_episodes)
    return {
        'engines': int(len(np.unique(units))) if len(units) else 0,
        'real_episodes': real_count,
        'detected_episodes': int(detected),
        'episode_recall': _ratio(int(detected), real_count),
        'alert_episodes': alert_count,
        'matched_alert_episodes': int(matched_alerts),
        'episode_precision': _ratio(int(matched_alerts), alert_count),
        'false_alarm_episodes': int(false_alarms),
        'mean_alert_to_episode_end_cycles': _mean(alert_to_end),
        'mean_anticipation_cycles': _mean(anticipation),
        'episodes': detail_rows,
    }


def export_episode_metrics_csv(metrics: Dict[str, Any], path: str) -> str:
    """Write the per-episode table to CSV."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in metrics.get('episodes', []):
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})
    return path
