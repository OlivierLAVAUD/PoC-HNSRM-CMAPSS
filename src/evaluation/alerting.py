"""Alert confirmation, smoothing and before/after engine-level analysis."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.evaluation.engine_metrics import compute_engine_metrics


def _ordered_indices(unit_ids: np.ndarray,
                     cycles: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return a cycle-sorted index order and the per-unit index groups."""
    unit_ids = np.asarray(unit_ids).ravel()
    cycles = np.asarray(cycles).ravel()
    if len(unit_ids) != len(cycles):
        raise ValueError("unit_ids and cycles must have the same length.")
    order = np.lexsort((cycles, unit_ids))
    groups = []
    sorted_units = unit_ids[order]
    if len(order):
        boundaries = np.flatnonzero(np.diff(sorted_units)) + 1
        groups = np.split(order, boundaries)
    return order, groups


def consecutive_confirmation(y_pred: np.ndarray, unit_ids: np.ndarray,
                             cycles: np.ndarray, k: int = 2) -> np.ndarray:
    """Keep only runs of at least ``k`` consecutive positive windows per unit."""
    predictions = np.asarray(y_pred, dtype=int).ravel()
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if len(predictions) != len(np.asarray(unit_ids).ravel()):
        raise ValueError("Predictions and unit_ids must have the same length.")
    confirmed = np.zeros(len(predictions), dtype=int)
    _, groups = _ordered_indices(unit_ids, cycles)
    for indices in groups:
        run: List[int] = []
        for index in indices:
            if predictions[index] == 1:
                run.append(int(index))
                continue
            if len(run) >= k:
                confirmed[run] = 1
            run = []
        if len(run) >= k:
            confirmed[run] = 1
    return confirmed


def smoothed_predictions(scores: np.ndarray, unit_ids: np.ndarray,
                         cycles: np.ndarray, threshold: float,
                         window: int = 3) -> np.ndarray:
    """Predict on a per-unit trailing moving average of risk scores."""
    values = np.asarray(scores, dtype=float).ravel()
    threshold = float(threshold)
    window = max(1, int(window))
    if len(values) != len(np.asarray(unit_ids).ravel()):
        raise ValueError("Scores and unit_ids must have the same length.")
    predictions = np.zeros(len(values), dtype=int)
    _, groups = _ordered_indices(unit_ids, cycles)
    for indices in groups:
        ordered_values = values[indices]
        if window == 1:
            predictions[indices] = (ordered_values >= threshold).astype(int)
            continue
        cumulative = np.concatenate(([0.0], np.cumsum(ordered_values)))
        for position, index in enumerate(indices):
            if position + 1 < window:
                continue
            mean_score = (
                cumulative[position + 1] - cumulative[position + 1 - window]
            ) / window
            predictions[index] = int(mean_score >= threshold)
    return predictions


def hysteresis_predictions(scores: np.ndarray, unit_ids: np.ndarray,
                           cycles: np.ndarray, enter_threshold: float,
                           exit_threshold: float) -> np.ndarray:
    """Apply a per-engine hysteresis state machine to risk scores.

    Alerts switch on when the score reaches ``enter_threshold`` and stay on
    until the score falls strictly below ``exit_threshold``. This avoids
    alert chattering around a single threshold while preserving early alerts.
    """
    values = np.asarray(scores, dtype=float).ravel()
    enter = float(enter_threshold)
    exit_value = float(exit_threshold)
    if not 0.0 <= exit_value < enter <= 1.0:
        raise ValueError(
            "Hysteresis thresholds must satisfy 0 <= exit < enter <= 1."
        )
    if len(values) != len(np.asarray(unit_ids).ravel()):
        raise ValueError("Scores and unit_ids must have the same length.")
    predictions = np.zeros(len(values), dtype=int)
    _, groups = _ordered_indices(unit_ids, cycles)
    for indices in groups:
        in_alert = False
        for index in indices:
            if in_alert:
                if values[index] < exit_value:
                    in_alert = False
            elif values[index] >= enter:
                in_alert = True
            predictions[index] = int(in_alert)
    return predictions


def resolve_hysteresis_thresholds(threshold: float,
                                  enter_threshold: Optional[float],
                                  exit_threshold: Optional[float],
                                  margin: float = 0.05) -> Tuple[float, float]:
    """Derive missing hysteresis thresholds from the validation threshold."""
    base = float(threshold)
    margin = float(margin)
    enter = (
        min(1.0, base + margin)
        if enter_threshold is None else float(enter_threshold)
    )
    exit_value = (
        max(0.0, base - margin)
        if exit_threshold is None else float(exit_threshold)
    )
    if not 0.0 <= exit_value < enter <= 1.0:
        raise ValueError(
            "Hysteresis thresholds must satisfy 0 <= exit < enter <= 1; "
            f"got enter={enter:.4f}, exit={exit_value:.4f}."
        )
    return enter, exit_value


def _summary_without_rows(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return dict(metrics.get("summary", {}))


def compare_alerting(y_true: np.ndarray, scores: np.ndarray,
                     unit_ids: np.ndarray, cycles: np.ndarray,
                     rul: np.ndarray, threshold: float,
                     consecutive_ks: Iterable[int] = (2, 3),
                     smoothing_cycles: int = 3,
                     hysteresis: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Compare immediate, confirmed, smoothed and hysteresis alerts."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    threshold = float(threshold)
    immediate_pred = (scores >= threshold).astype(int)

    modes: Dict[str, Dict[str, Any]] = {}
    immediate_metrics = compute_engine_metrics(
        unit_ids, cycles, rul, y_true, immediate_pred
    )
    modes["immediate"] = _summary_without_rows(immediate_metrics)

    for k in sorted({int(value) for value in consecutive_ks if int(value) > 1}):
        confirmed = consecutive_confirmation(
            immediate_pred, unit_ids, cycles, k=k
        )
        key = f"consecutive_{k}"
        modes[key] = _summary_without_rows(compute_engine_metrics(
            unit_ids, cycles, rul, y_true, confirmed
        ))

    if smoothing_cycles > 1:
        smoothed = smoothed_predictions(
            scores, unit_ids, cycles, threshold, window=smoothing_cycles
        )
        key = f"smoothed_{int(smoothing_cycles)}"
        modes[key] = _summary_without_rows(compute_engine_metrics(
            unit_ids, cycles, rul, y_true, smoothed
        ))

    hysteresis_thresholds = None
    if hysteresis and hysteresis.get('enabled', False):
        enter, exit_value = resolve_hysteresis_thresholds(
            threshold,
            hysteresis.get('enter_threshold'),
            hysteresis.get('exit_threshold'),
            margin=float(hysteresis.get('margin', 0.05)),
        )
        hysteretic = hysteresis_predictions(
            scores, unit_ids, cycles, enter, exit_value
        )
        modes["hysteresis"] = _summary_without_rows(compute_engine_metrics(
            unit_ids, cycles, rul, y_true, hysteretic
        ))
        hysteresis_thresholds = {
            "enter_threshold": enter,
            "exit_threshold": exit_value,
        }

    baseline = modes["immediate"]
    for name, summary in modes.items():
        if name == "immediate":
            summary["delta_vs_immediate"] = {}
            continue
        summary["delta_vs_immediate"] = {
            "recall": float(summary["mean_recall"] - baseline["mean_recall"]),
            "precision": float(
                summary["mean_precision"] - baseline["mean_precision"]
            ),
            "total_false_alarms": int(
                summary["total_false_alarms"] - baseline["total_false_alarms"]
            ),
            "mean_lead_time_cycles": float(
                summary["mean_lead_time_cycles"]
                - baseline["mean_lead_time_cycles"]
            ),
        }

    return {
        "threshold": threshold,
        "consecutive_ks": sorted({int(value) for value in consecutive_ks}),
        "smoothing_cycles": int(smoothing_cycles),
        "hysteresis_thresholds": hysteresis_thresholds,
        "modes": modes,
    }
