# src/fusion/symbolic_contribution.py
"""Measure what the symbolic and ProbLog inputs add to the fusion.

The production fusion is fitted once on validation and its threshold is fixed
there. This module re-scores the same validation and test matrices with one or
more symbolic columns set to zero, using the *unchanged* fitted coefficients
and threshold. The report is diagnostic: it never refits the fusion and never
changes the production decision.

Two levels are reported:

- signal level: zero the whole ``symbolic_rule_score`` or
  ``problog_failure_risk`` input;
- rule level, when per-window rule activations are supplied: zero the symbolic
  score only on the windows where one rule fires. This isolates the marginal
  contribution of that rule to window F1 and to episode recall, alongside its
  activation frequency, validation precision/recall and its conflict rate with
  the neural probability.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.evaluation.engine_metrics import compute_engine_metrics
from src.evaluation.episode_metrics import compute_episode_metrics
from src.evaluation.operational_policy import expected_cost, window_metrics


SIGNALS = ('symbolic_rule_score', 'problog_failure_risk')
SYMBOLIC_COLUMN = 'symbolic_rule_score'
F1_EPSILON = 0.005


def _summary(y_true: np.ndarray, predictions: np.ndarray,
             units: np.ndarray, cycles: np.ndarray, rul: np.ndarray,
             fn_cost: float, fa_cost: float) -> Dict[str, Any]:
    window = window_metrics(y_true, predictions)
    engine = compute_engine_metrics(
        units, cycles, rul, y_true, predictions
    )['summary']
    episode = compute_episode_metrics(y_true, predictions, units, cycles, rul)
    missed_episodes = max(
        0, int(episode['real_episodes']) - int(episode['detected_episodes'])
    )
    return {
        'window': {
            key: window[key] for key in (
                'true_positives', 'false_positives', 'false_negatives',
                'true_negatives', 'precision', 'recall', 'f1', 'f2',
            )
        },
        'engine': {
            key: engine[key] for key in (
                'mean_recall', 'mean_precision', 'total_false_alarms',
                'mean_lead_time_cycles', 'mean_anticipation_cycles',
            )
        },
        'episode': {
            key: episode[key] for key in (
                'real_episodes', 'detected_episodes', 'episode_recall',
                'episode_precision', 'false_alarm_episodes',
                'mean_anticipation_cycles',
            )
        },
        'expected_cost': {
            'window': expected_cost(
                window['false_negatives'], window['false_positives'],
                fn_cost, fa_cost,
            ),
            'episode': expected_cost(
                missed_episodes, episode['false_alarm_episodes'],
                fn_cost, fa_cost,
            ),
        },
    }


def _ablated_matrix(matrix: np.ndarray, feature_order: Sequence[str],
                    removed: Iterable[str],
                    rows: Optional[np.ndarray] = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=float).copy()
    removed_set = set(removed)
    for index, name in enumerate(feature_order):
        if name in removed_set:
            if rows is None:
                values[:, index] = 0.0
            else:
                values[np.asarray(rows, dtype=bool), index] = 0.0
    return values


def _delta(full: Dict[str, Any], ablated: Dict[str, Any],
           split: str) -> Dict[str, Any]:
    full_window = full[split]['window']
    ablated_window = ablated[split]['window']
    full_episode = full[split]['episode']
    ablated_episode = ablated[split]['episode']
    return {
        'f1': float(
            (full_window.get('f1') or 0.0)
            - (ablated_window.get('f1') or 0.0)
        ),
        'recall': float(
            (full_window.get('recall') or 0.0)
            - (ablated_window.get('recall') or 0.0)
        ),
        'precision': float(
            (full_window.get('precision') or 0.0)
            - (ablated_window.get('precision') or 0.0)
        ),
        'episode_recall': float(
            (full_episode.get('episode_recall') or 0.0)
            - (ablated_episode.get('episode_recall') or 0.0)
        ),
        'false_alarm_episodes': int(
            (full_episode.get('false_alarm_episodes') or 0)
            - (ablated_episode.get('false_alarm_episodes') or 0)
        ),
        'expected_cost_window': float(
            full[split]['expected_cost']['window']
            - ablated[split]['expected_cost']['window']
        ),
        'expected_cost_episode': float(
            full[split]['expected_cost']['episode']
            - ablated[split]['expected_cost']['episode']
        ),
    }


def _as_activation_matrix(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    matrix = np.asarray(values, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("rule_activations must be a 2D boolean matrix.")
    return matrix


def _rule_statistics(activations: np.ndarray, y_true: np.ndarray,
                     neural_scores: Optional[np.ndarray],
                     neural_threshold: float) -> Dict[str, Any]:
    active = np.asarray(activations, dtype=bool).ravel()
    labels = np.asarray(y_true, dtype=int).ravel()
    support = int(active.sum())
    if support:
        precision = float(labels[active].mean())
        recall = float(labels[active].sum() / labels.sum()) if labels.sum() else 0.0
    else:
        precision = 0.0
        recall = 0.0
    conflict_rate = None
    if neural_scores is not None and support:
        neural = np.asarray(neural_scores, dtype=float).ravel()
        conflict_rate = float(np.mean(neural[active] < float(neural_threshold)))
    return {
        'activation_count': support,
        'activation_frequency': float(active.mean()) if len(active) else 0.0,
        'precision': precision,
        'recall': recall,
        'conflict_rate_with_neural': conflict_rate,
        'windows_true_positive': int(np.sum(active & (labels == 1))),
        'windows_false_positive': int(np.sum(active & (labels == 0))),
    }


def _per_rule_report(
        fusion_model: Any, feature_order: List[str],
        val_matrix: np.ndarray, y_val: np.ndarray,
        val_units: np.ndarray, val_cycles: np.ndarray, val_rul: np.ndarray,
        test_matrix: np.ndarray, y_test: np.ndarray,
        test_units: np.ndarray, test_cycles: np.ndarray, test_rul: np.ndarray,
        threshold: float, rule_names: Sequence[str],
        val_activations: np.ndarray, test_activations: np.ndarray,
        neural_val: Optional[np.ndarray], neural_test: Optional[np.ndarray],
        neural_threshold: float, fn_cost: float, fa_cost: float,
        full_variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Ablate one rule at a time on its activated validation windows."""
    if SYMBOLIC_COLUMN not in feature_order:
        return {'applicable': False, 'reason': 'no_symbolic_fusion_input'}
    if len(rule_names) != val_activations.shape[1]:
        raise ValueError(
            "rule_names must match the number of rule_activations columns."
        )
    if val_activations.shape[0] != len(y_val):
        raise ValueError(
            "rule_activations_val must have one row per validation window."
        )
    if test_activations.shape[0] != len(y_test):
        raise ValueError(
            "rule_activations_test must have one row per test window."
        )

    contributions: Dict[str, Dict[str, Any]] = {}
    for column, name in enumerate(rule_names):
        active_val = val_activations[:, column]
        active_test = test_activations[:, column]
        if not active_val.any() and not active_test.any():
            contributions[str(name)] = {
                'activation_count_validation': 0,
                'activation_frequency_validation': 0.0,
                'precision_validation': 0.0,
                'recall_validation': 0.0,
                'conflict_rate_with_neural': None,
                'delta_full_minus_ablation': {},
                'verdict': 'never_fired',
            }
            continue
        val_scores = fusion_model.positive_proba(
            _ablated_matrix(
                val_matrix, feature_order, [SYMBOLIC_COLUMN], active_val
            )
        )
        test_scores = fusion_model.positive_proba(
            _ablated_matrix(
                test_matrix, feature_order, [SYMBOLIC_COLUMN], active_test
            )
        )
        ablated = {
            'validation': _summary(
                y_val, (val_scores >= threshold).astype(int),
                val_units, val_cycles, val_rul, fn_cost, fa_cost,
            ),
            'test': _summary(
                y_test, (test_scores >= threshold).astype(int),
                test_units, test_cycles, test_rul, fn_cost, fa_cost,
            ),
        }
        deltas = {
            split: _delta(full_variants['full'], ablated, split)
            for split in ('validation', 'test')
        }
        stats = _rule_statistics(
            active_val, y_val, neural_val, neural_threshold
        )
        validation_f1 = deltas['validation']['f1']
        verdict = (
            'helps' if validation_f1 > F1_EPSILON
            else 'hurts' if validation_f1 < -F1_EPSILON
            else 'neutral'
        )
        contributions[str(name)] = {
            'activation_count_validation': stats['activation_count'],
            'activation_frequency_validation': stats['activation_frequency'],
            'precision_validation': stats['precision'],
            'recall_validation': stats['recall'],
            'windows_true_positive': stats['windows_true_positive'],
            'windows_false_positive': stats['windows_false_positive'],
            'conflict_rate_with_neural': stats[
                'conflict_rate_with_neural'
            ],
            'activation_count_test': int(active_test.sum()),
            'delta_full_minus_ablation': deltas,
            'marginal_f1_validation': deltas['validation']['f1'],
            'marginal_episode_recall_validation': deltas['validation'][
                'episode_recall'
            ],
            'verdict': verdict,
            'reason': (
                'F1 and episode-recall deltas compare the full fusion with '
                'the same fixed coefficients and threshold after zeroing the '
                'symbolic score only on this rule\'s activated windows.'
            ),
        }

    ranked = sorted(
        (
            (name, value) for name, value in contributions.items()
            if value['verdict'] != 'never_fired'
        ),
        key=lambda item: item[1]['marginal_f1_validation'],
        reverse=True,
    )
    useful = [name for name, value in ranked
              if value['verdict'] == 'helps']
    harmful = [name for name, value in ranked
               if value['verdict'] == 'hurts']
    conflicted = [
        name for name, value in contributions.items()
        if value['verdict'] == 'never_fired'
        or (value.get('conflict_rate_with_neural') or 0.0) >= 0.5
    ]
    inactive = [
        name for name, value in contributions.items()
        if value['verdict'] == 'never_fired'
    ]
    return {
        'applicable': True,
        'diagnostic': True,
        'ablation_unit': 'rule',
        'rule_count': len(rule_names),
        'contributions': contributions,
        'summary': {
            'most_useful_rules': useful[:5],
            'rules_to_remove_or_examine': (
                harmful + [name for name in conflicted if name not in harmful]
            )[:5],
            'inactive_rules': inactive[:10],
            'answer': (
                'Rules improving validation F1: '
                + (', '.join(useful) if useful else 'none')
                + '; rules degrading it: '
                + (', '.join(harmful) if harmful else 'none')
                + '; rules to examine for neural conflict: '
                + (', '.join(conflicted) if conflicted else 'none') + '.'
            ),
        },
    }


def symbolic_contribution(fusion_model: Any,
                          val_matrix: np.ndarray, y_val: np.ndarray,
                          val_units: np.ndarray, val_cycles: np.ndarray,
                          val_rul: np.ndarray, test_matrix: np.ndarray,
                          y_test: np.ndarray, test_units: np.ndarray,
                          test_cycles: np.ndarray, test_rul: np.ndarray,
                          threshold: float,
                          config: Optional[Dict[str, Any]] = None,
                          rule_names: Optional[Sequence[str]] = None,
                          rule_activations_val: Optional[np.ndarray] = None,
                          rule_activations_test: Optional[np.ndarray] = None,
                          neural_val: Optional[np.ndarray] = None,
                          neural_test: Optional[np.ndarray] = None,
                          neural_threshold: Optional[float] = None
                          ) -> Dict[str, Any]:
    """Compare the full fusion with symbolic/ProbLog inputs zeroed out.

    Rule-level diagnostics are added when ``rule_names`` and the two activation
    matrices are supplied. All ablations reuse the production coefficients and
    threshold, so the report never selects or tunes the fusion.
    """
    config = config or {}
    feature_order = list(
        getattr(fusion_model, 'feature_order', None)
        or [name for name in SIGNALS]
    )
    present = [name for name in SIGNALS if name in feature_order]
    if not present:
        return {
            'applicable': False,
            'reason': 'no_symbolic_fusion_input',
            'signals': [],
        }
    fn_cost = float(config.get('false_negative_cost', 10.0))
    fa_cost = float(config.get('false_alarm_cost', 1.0))

    variants = {
        'full': list(feature_order),
        'without_symbolic_score': ['symbolic_rule_score'],
        'without_problog_risk': ['problog_failure_risk'],
        'without_symbolic_or_problog': list(SIGNALS),
        'neural_and_rul_only': list(SIGNALS),
    }
    variant_reports: Dict[str, Dict[str, Any]] = {}
    for variant, removed in variants.items():
        if variant != 'full' and not set(removed) & set(present):
            continue
        val_scores = fusion_model.positive_proba(
            _ablated_matrix(val_matrix, feature_order, removed)
            if variant != 'full' else np.asarray(val_matrix, dtype=float)
        )
        test_scores = fusion_model.positive_proba(
            _ablated_matrix(test_matrix, feature_order, removed)
            if variant != 'full' else np.asarray(test_matrix, dtype=float)
        )
        val_pred = (val_scores >= float(threshold)).astype(int)
        test_pred = (test_scores >= float(threshold)).astype(int)
        variant_reports[variant] = {
            'removed_inputs': list(removed) if variant != 'full' else [],
            'validation': _summary(
                y_val, val_pred, val_units, val_cycles, val_rul,
                fn_cost, fa_cost,
            ),
            'test': _summary(
                y_test, test_pred, test_units, test_cycles, test_rul,
                fn_cost, fa_cost,
            ),
        }

    full = variant_reports['full']
    contributions = {}
    for signal in present:
        variant = (
            'without_symbolic_score' if signal == 'symbolic_rule_score'
            else 'without_problog_risk'
        )
        if variant not in variant_reports:
            continue
        ablated = variant_reports[variant]
        split_deltas = {
            split: _delta(full, ablated, split)
            for split in ('validation', 'test')
        }
        validation_f1 = split_deltas['validation']['f1']
        contributions[signal] = {
            'ablation_variant': variant,
            'delta_full_minus_ablation': split_deltas,
            'verdict': (
                'helps' if validation_f1 > F1_EPSILON
                else 'hurts' if validation_f1 < -F1_EPSILON
                else 'neutral'
            ),
            'note': (
                'Positive deltas mean the full fusion is better than the '
                'same model with this input zeroed; the verdict uses the '
                'validation F1 delta.'
            ),
        }

    combined = variant_reports.get('without_symbolic_or_problog')
    combined_delta = (
        {
            split: _delta(full, combined, split)
            for split in ('validation', 'test')
        } if combined else {}
    )
    helpful = [
        name for name, value in contributions.items()
        if value['verdict'] == 'helps'
    ]
    harmful = [
        name for name, value in contributions.items()
        if value['verdict'] == 'hurts'
    ]
    rule_report: Dict[str, Any] = {
        'applicable': False,
        'reason': 'rule_activations_not_supplied',
    }
    val_activations = _as_activation_matrix(rule_activations_val)
    test_activations = _as_activation_matrix(rule_activations_test)
    if (rule_names is not None and val_activations is not None
            and test_activations is not None):
        rule_report = _per_rule_report(
            fusion_model, feature_order,
            val_matrix, y_val, val_units, val_cycles, val_rul,
            test_matrix, y_test, test_units, test_cycles, test_rul,
            float(threshold), list(rule_names), val_activations,
            test_activations, neural_val, neural_test,
            0.5 if neural_threshold is None else float(neural_threshold),
            fn_cost, fa_cost, variant_reports,
        )

    return {
        'applicable': True,
        'diagnostic': True,
        'scope': 'validation_and_test_reporting',
        'threshold': float(threshold),
        'feature_order': feature_order,
        'signals': present,
        'variants': variant_reports,
        'contributions': contributions,
        'combined_ablation': {
            'removed_inputs': list(SIGNALS),
            'delta_full_minus_ablation': combined_delta,
        },
        'rule_contributions': rule_report,
        'summary': {
            'helpful_signals': helpful,
            'harmful_signals': harmful,
            'missing_inputs': [
                name for name in SIGNALS if name not in present
            ],
            'answer': (
                'Symbolic evidence improves validation F1 for: '
                + (', '.join(helpful) if helpful else 'none')
                + '; it degrades it for: '
                + (', '.join(harmful) if harmful else 'none') + '.'
            ),
        },
    }


def save_symbolic_contribution(report: Dict[str, Any], path: str) -> str:
    """Persist the symbolic-contribution report as JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(report, file, indent=2)
    return path
