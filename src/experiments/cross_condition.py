# src/experiments/cross_condition.py
"""Cross-condition generalisation on C-MAPSS FD001-FD004.

The source dataset is split, calibrated and thresholded exactly as in the
normal pipeline. The fitted model, scaler/extractor and threshold are then
applied unchanged to the official test engines of each target dataset. Target
labels are used only to compute reporting metrics: no target window ever
participates in training, calibration, threshold selection or feature fitting.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

CSV_FIELDS = [
    'target_dataset', 'architecture', 'threshold', 'windows', 'engines',
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision',
    'brier_score', 'ece',
    'engine_mean_recall', 'engine_mean_precision', 'total_false_alarms',
    'episode_recall', 'episode_precision', 'false_alarm_episodes',
    'mean_anticipation_cycles',
    'source_validation_f1', 'source_validation_recall',
    'source_validation_precision', 'source_validation_average_precision',
    'relative_degradation_f1', 'relative_degradation_recall',
    'relative_degradation_precision',
    'relative_degradation_average_precision',
    'target_source',
]


def _relative_degradation(source_value: Any, target_value: Any) -> Optional[float]:
    """Return (source - target) / |source|; positive means degradation."""
    try:
        source = float(source_value)
        target = float(target_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(source) or not np.isfinite(target) or source == 0.0:
        return None
    return float((source - target) / abs(source))


def _target_specs(targets: Sequence[Any]) -> List[Dict[str, Any]]:
    """Normalise target specifications to name/config/config_path entries."""
    specs = []
    for item in targets:
        if not isinstance(item, dict):
            raise ValueError(
                "Each cross-condition target must be a dict with 'name', "
                "'config' and 'config_path'."
            )
        config = copy.deepcopy(item.get('config') or {})
        name = str(item.get('name') or _dataset_id(config))
        specs.append({
            'name': name,
            'config': config,
            'config_path': item.get('config_path'),
        })
    if not specs:
        raise ValueError("At least one target dataset is required.")
    return specs


def _dataset_id(config: Dict[str, Any]) -> str:
    dataset = config.get('dataset', {}) or {}
    if dataset.get('id'):
        return str(dataset['id'])
    if dataset.get('name'):
        return str(dataset['name']).replace('NASA C-MAPSS ', '')
    data_file = str(config.get('paths', {}).get('data_file', ''))
    for token in ('FD001', 'FD002', 'FD003', 'FD004'):
        if token in data_file:
            return token
    return 'target'


def _metric_row(name: str, architecture: str, threshold: float,
                windows: int, engines: int, metrics: Dict[str, Any],
                engine: Dict[str, Any], episode: Dict[str, Any],
                calibration: Dict[str, Any],
                source_validation: Dict[str, Any],
                target_source: str) -> Dict[str, Any]:
    """Build one target row, including relative degradation vs validation."""
    keys = ('f1', 'recall', 'precision', 'average_precision')
    degradation = {
        f'relative_degradation_{key}': _relative_degradation(
            source_validation.get(key), metrics.get(key)
        )
        for key in keys
    }
    return {
        'target_dataset': name,
        'architecture': architecture,
        'threshold': threshold,
        'windows': windows,
        'engines': engines,
        'accuracy': metrics.get('accuracy'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'f1': metrics.get('f1'),
        'roc_auc': metrics.get('roc_auc'),
        'average_precision': metrics.get('average_precision'),
        'brier_score': calibration.get('brier_score'),
        'ece': calibration.get('ece'),
        'engine_mean_recall': engine.get('mean_recall'),
        'engine_mean_precision': engine.get('mean_precision'),
        'total_false_alarms': engine.get('total_false_alarms'),
        'episode_recall': episode.get('episode_recall'),
        'episode_precision': episode.get('episode_precision'),
        'false_alarm_episodes': episode.get('false_alarm_episodes'),
        'mean_anticipation_cycles': episode.get('mean_anticipation_cycles'),
        'source_validation_f1': source_validation.get('f1'),
        'source_validation_recall': source_validation.get('recall'),
        'source_validation_precision': source_validation.get('precision'),
        'source_validation_average_precision': source_validation.get(
            'average_precision'
        ),
        **degradation,
        'target_source': target_source,
    }


def save_cross_condition_report(report: Dict[str, Any],
                                path: str) -> str:
    """Persist the cross-condition report as JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(report, file, indent=2, default=_json_default)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def export_cross_condition_csv(report: Dict[str, Any], path: str) -> str:
    """Write one target row per cross-condition evaluation."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS,
                                extrasaction='ignore')
        writer.writeheader()
        for row in report.get('rows', []):
            writer.writerow(row)
    return path


def run_cross_condition(
        source_config: Dict[str, Any], source_config_path: str,
        targets: Sequence[Any], logger: Any,
        results_dir: Optional[str] = None,
        architecture: Optional[str] = None) -> Dict[str, Any]:
    """Train on one condition and evaluate unchanged on other conditions."""
    from src.pipeline.pipeline import ANSRDTPipeline

    source_config = copy.deepcopy(source_config)
    source_dataset = _dataset_id(source_config)
    base_dir = os.path.abspath(
        results_dir or source_config['paths']['results_dir']
    )
    cross_dir = os.path.join(base_dir, 'cross_condition')
    source_dir = os.path.join(cross_dir, source_dataset.lower())
    os.makedirs(source_dir, exist_ok=True)
    source_config['paths']['results_dir'] = source_dir
    source_config.setdefault('hitl', {})['enabled'] = False
    source_config.setdefault('audit', {})['enabled'] = False
    source_config.setdefault('monitoring', {}).setdefault('drift', {})[
        'enabled'
    ] = False
    source_config.setdefault('experiments', {}).setdefault(
        'multi_seed', {}
    )['enabled'] = False

    logger.info(
        "Cross-condition source %s: training and validation-only threshold.",
        source_dataset,
    )
    source = ANSRDTPipeline(source_config, source_config_path, logger)
    splits = source.load_splits()
    source_architecture = source._normalise_architecture(
        architecture or source_config.get('model', {}).get(
            'architecture', 'random_forest'
        )
    )
    prepared = source._prepare_model_data(splits, source_architecture)
    source_eval = source._evaluate_architecture(prepared, splits)
    threshold = float(source_eval['threshold'])
    source_calibration = source_eval.get('calibration_report') or {}
    source_metrics = {
        'validation': source_eval['validation_metrics'],
        'test': source_eval['test_metrics'],
        'calibration': {
            split: {
                'brier_score': (source_calibration.get(split) or {}).get(
                    'brier_score'
                ),
                'ece': (source_calibration.get(split) or {}).get('ece'),
            }
            for split in ('validation', 'test')
        },
        'threshold': threshold,
        'test_source': splits['test_source'],
        'dataset': source.dataset_name,
    }

    rows: List[Dict[str, Any]] = []
    target_reports: Dict[str, Dict[str, Any]] = {}
    for spec in _target_specs(targets):
        name = spec['name']
        target_config = spec['config']
        target_dir = os.path.join(cross_dir, name.lower())
        os.makedirs(target_dir, exist_ok=True)
        target_config['paths']['results_dir'] = target_dir
        target_config.setdefault('hitl', {})['enabled'] = False
        target_config.setdefault('audit', {})['enabled'] = False
        target_config.setdefault('monitoring', {}).setdefault('drift', {})[
            'enabled'
        ] = False
        target_config.setdefault('experiments', {}).setdefault(
            'multi_seed', {}
        )['enabled'] = False
        target_path = spec.get('config_path') or source_config_path
        if not target_config.get('paths', {}).get('test_file'):
            raise ValueError(
                f"Cross-condition target {name} has no official test_file; "
                "a reporting-only target test split is required."
            )
        if list(target_config['model'].get('feature_names', [])) != list(
                source.feature_names):
            raise ValueError(
                f"Cross-condition target {name} uses different feature_names "
                "from the source dataset."
            )
        target = ANSRDTPipeline(target_config, target_path, logger)
        X_target, y_target = target.data_loader.load_test_data()
        units = np.asarray(target.data_loader.unit_ids)
        cycles = np.asarray(target.data_loader.window_end_cycles)
        rul = np.asarray(target.data_loader.window_rul)
        X_target = source._sanitize_non_finite(X_target, f'cross-{name}')
        evaluation = source.evaluate_external_windows(
            prepared, X_target, y_target, units, cycles, rul, threshold,
        )
        row = _metric_row(
            name, source_architecture, threshold, len(y_target),
            int(len(np.unique(units))), evaluation['metrics'],
            evaluation['engine_metrics'], evaluation['episode_metrics'],
            evaluation.get('calibration', {}),
            source_eval['validation_metrics'],
            target_config['paths']['test_file'],
        )
        rows.append(row)
        target_reports[name] = {
            'dataset': target.dataset_name,
            'metrics': evaluation['metrics'],
            'calibration': evaluation.get('calibration', {}),
            'engine_metrics': evaluation['engine_metrics'],
            'episode_metrics': evaluation['episode_metrics'],
            'relative_degradation_vs_source_validation': {
                key: row[f'relative_degradation_{key}']
                for key in ('f1', 'recall', 'precision', 'average_precision')
            },
            'test_source': target_config['paths']['test_file'],
            'windows': int(len(y_target)),
            'engines': int(len(np.unique(units))),
        }
        logger.info(
            "Cross-condition %s -> %s: F1=%.4f recall=%.4f precision=%.4f",
            source_dataset, name, float(row['f1'] or 0.0),
            float(row['recall'] or 0.0), float(row['precision'] or 0.0),
        )

    report = {
        'protocol': {
            'source_dataset': source_dataset,
            'target_datasets': [spec['name'] for spec in _target_specs(targets)],
            'train_conditions': [source_dataset],
            'test_conditions': [spec['name'] for spec in _target_specs(targets)],
            'architecture': source_architecture,
            'selection_split': 'source_validation',
            'target_usage': 'reporting_only',
            'source_transforms_reused': True,
            'threshold': threshold,
            'degradation_reference': 'source_validation',
        },
        'source': source_metrics,
        'targets': target_reports,
        'rows': rows,
        'timestamp': datetime.now().isoformat(),
    }
    json_path = save_cross_condition_report(
        report, os.path.join(base_dir, 'cross_condition_report.json')
    )
    csv_path = export_cross_condition_csv(
        report, os.path.join(base_dir, 'cross_condition_summary.csv')
    )
    report['artifacts'] = {'report_path': json_path, 'csv_path': csv_path}
    logger.info("Cross-condition report saved to %s.", json_path)
    return report
