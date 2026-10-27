# src/evaluation/thresholding.py
"""Validation-only decision-threshold selection and precision/recall analysis."""

import os
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_recall_curve


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def precision_recall_analysis(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """Return the precision-recall curve as serialisable arrays."""
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(y_true, dtype=int).ravel(),
        np.asarray(scores, dtype=float).ravel(),
    )
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
    }


def _candidate_metrics(y_true: np.ndarray, scores: np.ndarray,
                       thresholds: np.ndarray, beta: float) -> Dict[str, np.ndarray]:
    """Evaluate candidate thresholds without using the test set."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    thresholds = np.asarray(thresholds, dtype=float)
    metrics = {"threshold": thresholds, "precision": [], "recall": [], "f1": [], "fbeta": []}

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(
            _safe_div(2 * precision * recall, precision + recall)
        )
        beta_sq = beta * beta
        metrics["fbeta"].append(
            _safe_div((1 + beta_sq) * precision * recall,
                      beta_sq * precision + recall)
        )

    for key in ("precision", "recall", "f1", "fbeta"):
        metrics[key] = np.asarray(metrics[key], dtype=float)
    return metrics


def select_threshold(y_true: np.ndarray, scores: np.ndarray,
                     strategy: str = "fbeta", min_precision: float = 0.7,
                     beta: float = 2.0, fallback: float = 0.5) -> Dict[str, Any]:
    """Select a threshold on validation scores with a configurable strategy.

    Strategies:
    - ``f1``: maximise F1.
    - ``recall_at_precision``: maximise recall subject to precision >= min_precision.
    - ``fbeta``: maximise F-beta (beta > 1 favours recall by default).
    """
    strategy = str(strategy).lower()
    if strategy not in {"f1", "recall_at_precision", "fbeta"}:
        raise ValueError(f"Unsupported threshold strategy: {strategy!r}")
    if not 0.0 <= float(min_precision) <= 1.0:
        raise ValueError("min_precision must be in [0, 1].")
    if float(beta) <= 0:
        raise ValueError("beta must be positive.")

    y_true = np.asarray(y_true, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length.")
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return _fallback_result(fallback, strategy, min_precision, beta, y_true, scores)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return _fallback_result(fallback, strategy, min_precision, beta, y_true, scores)

    candidates = _candidate_metrics(y_true, scores, thresholds, beta)
    if strategy == "f1":
        selected = int(np.argmax(candidates["f1"]))
    elif strategy == "fbeta":
        selected = int(np.argmax(candidates["fbeta"]))
    else:
        valid = candidates["precision"] >= float(min_precision)
        if not np.any(valid):
            return _fallback_result(
                fallback, strategy, min_precision, beta, y_true, scores,
                reason="no_threshold_meets_min_precision",
            )
        valid_indices = np.flatnonzero(valid)
        selected = int(valid_indices[np.argmax(candidates["recall"][valid])])

    selected_threshold = float(candidates["threshold"][selected])
    selected_metrics = {key: float(candidates[key][selected])
                        for key in ("precision", "recall", "f1", "fbeta")}

    return {
        "threshold": selected_threshold,
        "strategy": strategy,
        "beta": float(beta),
        "min_precision": float(min_precision),
        "selected_metrics": selected_metrics,
        "candidates_evaluated": int(len(thresholds)),
        "fallback_used": False,
        "reason": None,
        "precision_recall_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


def _fallback_result(fallback: float, strategy: str, min_precision: float,
                     beta: float, y_true: np.ndarray, scores: np.ndarray,
                     reason: Optional[str] = None) -> Dict[str, Any]:
    """Return the configured fallback threshold with descriptive metrics."""
    threshold = float(fallback)
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "strategy": strategy,
        "beta": float(beta),
        "min_precision": float(min_precision),
        "selected_metrics": {
            "precision": float(np.mean(np.asarray(y_true)[pred == 1] == 1))
            if np.any(pred == 1) else 0.0,
            "recall": float(np.mean(pred[np.asarray(y_true) == 1] == 1))
            if np.any(np.asarray(y_true) == 1) else 0.0,
            "f1": float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else 0.0,
            "fbeta": float(fbeta_score(y_true, pred, beta=beta, zero_division=0))
            if len(y_true) else 0.0,
        },
        "candidates_evaluated": 0,
        "fallback_used": True,
        "reason": reason or "insufficient_class_diversity",
        "precision_recall_curve": {"precision": [], "recall": [], "thresholds": []},
    }


def save_precision_recall_plot(analysis: Dict[str, Any], path: str,
                               threshold: Optional[float] = None) -> str:
    """Save a precision-recall curve, optionally marking the chosen threshold."""
    import matplotlib.pyplot as plt

    precision = np.asarray(analysis.get("precision", []), dtype=float)
    recall = np.asarray(analysis.get("recall", []), dtype=float)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    if len(precision):
        ax.plot(recall, precision, color="tab:blue", label="Precision-recall")
    if threshold is not None:
        ax.axhline(threshold, color="tab:orange", linestyle=":",
                   label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
