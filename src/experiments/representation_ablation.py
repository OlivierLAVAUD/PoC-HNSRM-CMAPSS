# src/experiments/representation_ablation.py
"""Controlled physical-representation ablation on C-MAPSS FD001-FD004.

Protocol per variant:
1. fit only on FD001 training engines;
2. calibrate only on FD001 training windows;
3. select the source threshold only on FD001 validation engines;
4. score FD001, FD002, FD003 and FD004 official test engines with that
   frozen threshold;
5. also report each target under a fixed alert budget selected on FD001
   validation (never on target labels).

Target labels are used only to compute reporting metrics.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.alert_budget import select_alert_budget_threshold
from src.features.physical_representation import VARIANTS
from src.pipeline.pipeline import ANSRDTPipeline

CSV_FIELDS = [
    'variant', 'dataset', 'threshold_strategy', 'threshold', 'windows',
    'engines', 'roc_auc', 'average_precision', 'f1', 'precision', 'recall',
    'brier_score', 'ece', 'real_episodes', 'detected_episodes',
    'episode_recall', 'episode_precision', 'false_alarm_episodes',
    'false_alarms_per_engine', 'false_alarms_per_100_cycles',
    'mean_anticipation_cycles', 'expected_cost_episode',
]


def _safe(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _false_alarms_per_100(engine: Dict[str, Any],
                          units: np.ndarray, cycles: np.ndarray) -> float:
    observed = int(len(np.unique(np.column_stack([units, cycles]), axis=0)))
    if not observed:
        return 0.0
    return float(100.0 * _safe(engine.get('total_false_alarms')) / observed)


def _build_row(variant: str, dataset: str, strategy: str,
               threshold: float, metrics: Dict[str, Any],
               calibration: Dict[str, Any], engine: Dict[str, Any],
               episode: Dict[str, Any], units: np.ndarray,
               cycles: np.ndarray) -> Dict[str, Any]:
    engines = int(len(np.unique(units)))
    return {
        'variant': variant,
        'dataset': dataset,
        'threshold_strategy': strategy,
        'threshold': float(threshold),
        'windows': int(len(units)),
        'engines': engines,
        'roc_auc': metrics.get('roc_auc'),
        'average_precision': metrics.get('average_precision'),
        'f1': metrics.get('f1'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'brier_score': calibration.get('brier_score'),
        'ece': calibration.get('ece'),
        'real_episodes': episode.get('real_episodes'),
        'detected_episodes': episode.get('detected_episodes'),
        'episode_recall': episode.get('episode_recall'),
        'episode_precision': episode.get('episode_precision'),
        'false_alarm_episodes': episode.get('false_alarm_episodes'),
        'false_alarms_per_engine': (
            _safe(engine.get('total_false_alarms')) / engines
            if engines else 0.0
        ),
        'false_alarms_per_100_cycles': _false_alarms_per_100(
            engine, units, cycles
        ),
        'mean_anticipation_cycles': episode.get('mean_anticipation_cycles'),
        'expected_cost_episode': (
            _safe(episode.get('real_episodes'))
            - _safe(episode.get('detected_episodes'))
            + _safe(episode.get('false_alarm_episodes'))
        ),
    }


def _target_payload(source: ANSRDTPipeline, prepared: Dict[str, Any],
                    target_config: Dict[str, Any], target_path: str,
                    logger: Any) -> Dict[str, Any]:
    target = ANSRDTPipeline(target_config, target_path, logger)
    X, y = target.data_loader.load_test_data()
    units = np.asarray(target.data_loader.unit_ids)
    cycles = np.asarray(target.data_loader.window_end_cycles)
    rul = np.asarray(target.data_loader.window_rul)
    ops = getattr(target.data_loader, 'window_op_settings', None)
    X = source._sanitize_non_finite(X, 'ablation-target')
    X_model = source.transform_external_windows(
        prepared, X, units=units, cycles=cycles, op_settings=ops
    )
    return {
        'X_model': X_model, 'y': y, 'units': units, 'cycles': cycles,
        'rul': rul,
    }


def _payload_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'metrics': payload['metrics'],
        'calibration': payload['calibration'],
        'engine': payload['engine_metrics'],
        'episode': payload['episode_metrics'],
    }


def _conclusion(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    statements = []
    for dataset in ('FD002', 'FD004'):
        target_rows = [row for row in rows if row['dataset'] == dataset]
        best_roc = max((_safe(row['roc_auc']) for row in target_rows),
                       default=0.0)
        best_ap = max((_safe(row['average_precision']) for row in target_rows),
                      default=0.0)
        if best_roc < 0.65 or best_ap < 0.10:
            statements.append(
                f"{dataset}: best ROC-AUC {best_roc:.4f} / AP {best_ap:.4f} "
                "across variants; the simple physical representation does "
                "not fix the ranking problem."
            )
        else:
            statements.append(
                f"{dataset}: a variant reaches ROC-AUC {best_roc:.4f} / AP "
                f"{best_ap:.4f}; the representation may help ranking, subject "
                "to validation confirmation."
            )
    fd003 = [
        row for row in rows
        if row['dataset'] == 'FD003' and row['threshold_strategy'] == 'source'
    ]
    if fd003:
        best = max(fd003, key=lambda row: _safe(row['roc_auc']))
        if _safe(best['roc_auc']) >= 0.80 and _safe(best['recall']) < 0.50:
            statements.append(
                "FD003 keeps a high ROC-AUC with low recall at the source "
                "threshold: this is primarily a threshold/calibration "
                "problem, not a ranking problem."
            )
    statements.append(
        "These are descriptive results from one source protocol; no causal "
        "claim is made."
    )
    return {'statements': statements}


def run_representation_ablation(
        config: Dict[str, Any], config_path: str,
        targets: Sequence[Dict[str, Any]], logger: Any,
        results_dir: Optional[str] = None,
        architecture: Optional[str] = None) -> Dict[str, Any]:
    """Run the five-variant ablation and persist CSV/JSON/plots."""
    architecture = (
        architecture
        or config.get('model', {}).get('architecture', 'random_forest')
    )
    architecture = str(architecture).lower()
    if architecture in {'lgbm', 'lightgbm_classifier'}:
        architecture = 'lightgbm'
    if architecture in {'light_classifier'}:
        architecture = 'random_forest'
    output_dir = results_dir or os.path.join(
        config.get('paths', {}).get('results_dir', 'results'),
        'representation_ablation',
    )
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    base_budget = dict(config.get('alert_budget', {}) or {})
    if not base_budget.get('max_alerts_per_engine') and not base_budget.get(
            'max_alerts_per_100_cycles'):
        base_budget = {
            'max_alerts_per_engine': 1.0,
            'selection_split': 'validation',
            'unreachable_policy': 'most_constraining',
        }

    rows: List[Dict[str, Any]] = []
    variant_reports: Dict[str, Any] = {}
    for variant in VARIANTS:
        logger.info("Representation ablation: variant=%s", variant)
        variant_config = copy.deepcopy(config)
        representation = variant_config.setdefault('representation', {})
        representation.update({
            'enabled': True,
            'variant': variant,
            'apply_to_architectures': [architecture],
            'n_conditions': representation.get(
                'n_conditions',
                (variant_config.get('dataset', {}) or {}).get(
                    'operating_conditions', 1
                ),
            ),
        })
        variant_config.setdefault('fusion', {})['enabled'] = False
        variant_config.setdefault('rul_regression', {})['enabled'] = False
        variant_config.setdefault('hitl', {})['enabled'] = False
        variant_config.setdefault('audit', {})['enabled'] = False
        variant_config.setdefault('knowledge_graph', {})['enabled'] = False
        variant_config.setdefault('alerting', {}).setdefault(
            'policy_comparison', {}
        )['enabled'] = False
        variant_config.setdefault('operational_policy', {})['enabled'] = False
        # Never overwrite the published FD001 artifacts: every variant writes
        # its model, scaler and representation under the ablation directory.
        variant_config.setdefault('paths', {})['results_dir'] = os.path.join(
            output_dir, f'run_{variant}'
        )
        source = ANSRDTPipeline(variant_config, config_path, logger)
        splits = source._load_cmapss_splits()
        prepared = source._prepare_model_data(splits, architecture)
        evaluation = source._evaluate_architecture(prepared, splits)
        threshold = float(evaluation['threshold'])
        budget = select_alert_budget_threshold(
            evaluation['val_scores'], splits['val_units'],
            splits['val_cycles'], base_budget,
        )
        budget_threshold = float(budget['threshold'])
        report: Dict[str, Any] = {
            'variant': variant,
            'representation': prepared.get('representation_report', {}),
            'source_threshold': threshold,
            'budget_threshold': budget_threshold,
            'budget_selection': budget,
            'datasets': {},
        }

        source_test = {
            'metrics': evaluation['test_metrics'],
            'calibration': evaluation['calibration_report']['test'],
            'engine': evaluation['engine_metrics']['test']['summary'],
            'episode': evaluation['episode_metrics']['test'],
        }
        rows.append(_build_row(
            variant, 'FD001', 'source', threshold,
            source_test['metrics'], source_test['calibration'],
            source_test['engine'], source_test['episode'],
            splits['test_units'], splits['test_cycles'],
        ))
        budget_eval = source.evaluate_external_model_input(
            prepared, prepared['X_test'], splits['y_test'],
            splits['test_units'], splits['test_cycles'], splits['test_rul'],
            budget_threshold,
        )
        budget_row = _payload_row(budget_eval)
        rows.append(_build_row(
            variant, 'FD001', 'alert_budget', budget_threshold,
            budget_row['metrics'], budget_row['calibration'],
            budget_row['engine'], budget_row['episode'],
            splits['test_units'], splits['test_cycles'],
        ))
        report['datasets']['FD001'] = {
            'source_threshold': rows[-2], 'alert_budget': rows[-1],
        }

        for target in targets:
            name = str(target.get('name', '')).upper()
            payload = _target_payload(
                source, prepared, copy.deepcopy(target['config']),
                target.get('config_path'), logger,
            )
            source_eval = source.evaluate_external_model_input(
                prepared, payload['X_model'], payload['y'],
                payload['units'], payload['cycles'], payload['rul'], threshold,
            )
            source_data = _payload_row(source_eval)
            rows.append(_build_row(
                variant, name, 'source', threshold,
                source_data['metrics'], source_data['calibration'],
                source_data['engine'], source_data['episode'],
                payload['units'], payload['cycles'],
            ))
            budget_eval = source.evaluate_external_model_input(
                prepared, payload['X_model'], payload['y'],
                payload['units'], payload['cycles'], payload['rul'],
                budget_threshold,
            )
            budget_data = _payload_row(budget_eval)
            rows.append(_build_row(
                variant, name, 'alert_budget', budget_threshold,
                budget_data['metrics'], budget_data['calibration'],
                budget_data['engine'], budget_data['episode'],
                payload['units'], payload['cycles'],
            ))
            report['datasets'][name] = {
                'source_threshold': rows[-2],
                'alert_budget': rows[-1],
            }
        variant_reports[variant] = report

    conclusion = _conclusion(rows)
    summary = {
        'protocol': 'representation_ablation',
        'source_dataset': 'FD001',
        'architecture': architecture,
        'selection_split': 'validation',
        'test_usage': 'reporting_only',
        'representation_selection': (
            'all variants are reported; no test-based winner is selected'
        ),
        'budget': base_budget,
        'variants': list(VARIANTS),
        'rows': rows,
        'conclusion': conclusion,
        'timestamp': datetime.now().isoformat(),
    }
    csv_path = os.path.join(output_dir, 'summary.csv')
    json_path = os.path.join(output_dir, 'summary.json')
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plots = _save_plots(rows, os.path.join(output_dir, 'plots'))
    summary['artifacts'] = {
        'summary_csv': csv_path, 'summary_json': json_path, 'plots': plots,
    }
    with open(json_path, 'w') as file:
        json.dump(summary, file, indent=2, default=str)
    return summary


def _save_plots(rows: List[Dict[str, Any]], plot_dir: str) -> List[str]:
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    paths = []
    for metric in ('roc_auc', 'average_precision'):
        fig, ax = plt.subplots(figsize=(8, 4))
        datasets = sorted({row['dataset'] for row in rows})
        width = 0.15
        for index, variant in enumerate(VARIANTS):
            values = []
            for dataset in datasets:
                matches = [
                    row for row in rows
                    if row['dataset'] == dataset
                    and row['variant'] == variant
                    and row['threshold_strategy'] == 'source'
                ]
                values.append(_safe(matches[0][metric]) if matches else 0.0)
            ax.bar(
                np.arange(len(datasets)) + index * width, values,
                width=width, label=variant,
            )
        ax.set_xticks(np.arange(len(datasets)) + 2 * width)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by representation variant (source threshold)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(plot_dir, f'{metric}.png')
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths
