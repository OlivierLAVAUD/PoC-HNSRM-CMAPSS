# src/evaluation/calibration.py
"""Probability-calibration diagnostics for the lightweight classifier."""

import os
from typing import Any, Dict

import numpy as np
from sklearn.metrics import brier_score_loss


def _as_probabilities(y_prob: np.ndarray) -> np.ndarray:
    """Return a finite 1D probability vector clipped to [0, 1]."""
    prob = np.asarray(y_prob, dtype=float).ravel()
    if not np.isfinite(prob).all():
        raise ValueError("Probabilities contain NaN or infinite values.")
    return np.clip(prob, 0.0, 1.0)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray,
                      n_bins: int = 10) -> Dict[str, Any]:
    """Return a fixed-width reliability curve and bin counts."""
    n_bins = max(1, int(n_bins))
    prob = _as_probabilities(y_prob)
    true = np.asarray(y_true, dtype=int).ravel()
    if len(prob) != len(true):
        raise ValueError("y_true and y_prob must have the same length.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob, edges[1:-1], right=False)
    centers, means, fractions, counts = [], [], [], []

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        counts.append(count)
        centers.append(float((edges[bin_id] + edges[bin_id + 1]) / 2.0))
        if count:
            means.append(float(prob[mask].mean()))
            fractions.append(float(true[mask].mean()))
        else:
            means.append(float("nan"))
            fractions.append(float("nan"))

    return {
        "n_bins": n_bins,
        "bin_edges": edges.tolist(),
        "bin_centers": centers,
        "mean_predicted": means,
        "fraction_positive": fractions,
        "counts": counts,
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """Compute the standard binned expected calibration error (ECE)."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    total = int(np.sum(curve["counts"]))
    if total == 0:
        return 0.0
    ece = 0.0
    for mean_pred, frac_pos, count in zip(
        curve["mean_predicted"], curve["fraction_positive"], curve["counts"]
    ):
        if count and not (np.isnan(mean_pred) or np.isnan(frac_pos)):
            ece += (count / total) * abs(mean_pred - frac_pos)
    return float(ece)


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                        n_bins: int = 10) -> Dict[str, Any]:
    """Compute Brier score, ECE and the reliability-curve data."""
    prob = _as_probabilities(y_prob)
    true = np.asarray(y_true, dtype=int).ravel()
    return {
        "brier_score": float(brier_score_loss(true, prob)),
        "ece": expected_calibration_error(true, prob, n_bins=n_bins),
        "reliability_curve": reliability_curve(true, prob, n_bins=n_bins),
        "mean_predicted_probability": float(prob.mean()),
        "positive_rate": float(true.mean()) if len(true) else 0.0,
    }


def save_reliability_plot(metrics: Dict[str, Any], path: str,
                          title: str = "Reliability curve") -> str:
    """Save a simple reliability diagram and the perfect-calibration line."""
    import matplotlib.pyplot as plt

    curve = metrics["reliability_curve"]
    means = np.asarray(curve["mean_predicted"], dtype=float)
    fractions = np.asarray(curve["fraction_positive"], dtype=float)
    mask = np.isfinite(means) & np.isfinite(fractions)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(means[mask], fractions[mask], marker="o", color="tab:blue",
            label="Classifier")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
