# src/experiments/cross_condition.py
"""Cross-condition generalisation on C-MAPSS FD001-FD004.

The source dataset is split, calibrated and thresholded exactly as in the
normal pipeline. The fitted model, scaler/extractor and threshold are then
applied to the official test engines of each target dataset. Target labels are
used only to compute reporting metrics: no target window ever participates in
training, calibration or feature fitting.

The optional adaptation block separates two label-free corrections:

* ``normalization: per_condition`` standardises target windows with statistics
  fitted per operating condition on the unlabelled target test set;
* ``threshold_strategy: target_alert_rate`` matches the alert rate observed on
  the source validation split instead of reusing its raw threshold.

Both are transductive and are reported as such.  When enabled, the report
contains the full 2x2 ablation (source/per-condition normalisation x
source/alert-rate threshold) so the effect of each correction is visible.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.experiments.condition_adaptation import (
    ConditionNormalizer,
    achieved_alert_rate,
    select_alert_rate_threshold,
)

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
    'variant', 'normalization', 'threshold_strategy',
    'source_validation_alert_rate', 'target_alert_rate',
    'baseline_f1', 'baseline_precision', 'baseline_recall',
    'baseline_false_alarm_episodes',
    'target_source',
]

ADAPTATION_NORMALIZATIONS = ('source', 'per_condition')
ADAPTATION_THRESHOLDS = ('source', 'target_alert_rate')


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


def _adaptation_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Read and validate the label-free cross-condition adaptation block."""
    section = (config.get('experiments', {}) or {}).get(
        'cross_condition', {}
    ) or {}
    adaptation = section.get('adaptation', {}) or {}
    normalization = str(
        adaptation.get('normalization', 'source')
    ).strip().lower()
    threshold_strategy = str(
        adaptation.get('threshold_strategy', 'source')
    ).strip().lower()
    if normalization not in ADAPTATION_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported cross-condition normalization: {normalization!r}. "
            f"Allowed: {list(ADAPTATION_NORMALIZATIONS)}."
        )
    if threshold_strategy not in ADAPTATION_THRESHOLDS:
        raise ValueError(
            "Unsupported cross-condition threshold strategy: "
            f"{threshold_strategy!r}. Allowed: {list(ADAPTATION_THRESHOLDS)}."
        )
    n_conditions = adaptation.get('n_conditions')
    if n_conditions is not None and (
            not isinstance(n_conditions, int) or isinstance(n_conditions, bool)
            or n_conditions < 1):
        raise ValueError(
            "experiments.cross_condition.adaptation.n_conditions must be an "
            "integer >= 1."
        )
    return {
        'normalization': normalization,
        'threshold_strategy': threshold_strategy,
        'n_conditions': n_conditions,
        'random_state': int(adaptation.get('random_state', 42)),
    }


def _primary_variant(normalization: str, threshold_strategy: str) -> str:
    """Return the report variant selected by the configuration."""
    if (normalization == 'per_condition'
            and threshold_strategy == 'target_alert_rate'):
        return 'per_condition_alert_rate'
    if normalization == 'per_condition':
        return 'per_condition'
    if threshold_strategy == 'target_alert_rate':
        return 'alert_rate'
    return 'zero_shot'


def _variant(evaluation: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Keep the reporting payload of one evaluation variant."""
    return {
        'threshold': float(threshold),
        'achieved_alert_rate': achieved_alert_rate(
            evaluation['scores'], threshold
        ),
        'metrics': evaluation['metrics'],
        'calibration': evaluation.get('calibration', {}),
        'engine_metrics': evaluation['engine_metrics'],
        'episode_metrics': evaluation['episode_metrics'],
    }


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
                target_source: str, *,
                variant: str = 'zero_shot',
                normalization: str = 'source',
                threshold_strategy: str = 'source',
                source_validation_alert_rate: Optional[float] = None,
                target_alert_rate: Optional[float] = None,
                baseline_metrics: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
    """Build one target row, including relative degradation vs validation."""
    keys = ('f1', 'recall', 'precision', 'average_precision')
    degradation = {
        f'relative_degradation_{key}': _relative_degradation(
            source_validation.get(key), metrics.get(key)
        )
        for key in keys
    }
    baseline = baseline_metrics or {}
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
        'variant': variant,
        'normalization': normalization,
        'threshold_strategy': threshold_strategy,
        'source_validation_alert_rate': source_validation_alert_rate,
        'target_alert_rate': target_alert_rate,
        'baseline_f1': baseline.get('f1'),
        'baseline_precision': baseline.get('precision'),
        'baseline_recall': baseline.get('recall'),
        'baseline_false_alarm_episodes': baseline.get(
            'false_alarm_episodes'
        ),
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
    """Train on one condition and evaluate on other conditions.

    The optional adaptation block inspects unlabelled target features only.
    Target labels remain reporting-only.
    """
    from src.pipeline.pipeline import ANSRDTPipeline

    source_config = copy.deepcopy(source_config)
    adaptation = _adaptation_settings(source_config)
    normalization = adaptation['normalization']
    threshold_strategy = adaptation['threshold_strategy']
    primary_name = _primary_variant(normalization, threshold_strategy)
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
        "Cross-condition source %s: training and validation-only threshold "
        "(normalization=%s, threshold=%s).",
        source_dataset, normalization, threshold_strategy,
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
    source_validation_scores = np.asarray(source_eval['val_scores'])
    source_alert_rate = float(np.mean(
        source_validation_scores >= threshold
    ))
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
        'validation_alert_rate': source_alert_rate,
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
        op_settings = getattr(
            target.data_loader, 'window_op_settings', None
        )
        X_target = source._sanitize_non_finite(X_target, f'cross-{name}')

        baseline_input = source.transform_external_windows(
            prepared, X_target
        )
        baseline_eval = source.evaluate_external_model_input(
            prepared, baseline_input, y_target, units, cycles, rul, threshold,
        )
        variants = {'zero_shot': _variant(baseline_eval, threshold)}

        adapted_input = None
        normalizer = None
        if normalization == 'per_condition':
            if (prepared.get('uses_temporal_features')
                    or prepared.get('scaler') is None):
                raise ValueError(
                    "cross_condition.adaptation.normalization='per_condition' "
                    "requires a source architecture with a fitted scaler "
                    "(for example random_forest). LightGBM uses raw window "
                    "features and cannot reuse per-condition z-scores safely."
                )
            if op_settings is None:
                raise ValueError(
                    f"Cross-condition target {name} has no operating settings; "
                    "per-condition normalization is unavailable."
                )
            n_conditions = adaptation['n_conditions']
            if n_conditions is None:
                n_conditions = int(
                    (target_config.get('dataset', {}) or {}).get(
                        'operating_conditions', 1
                    )
                )
            normalizer = ConditionNormalizer.fit(
                op_settings, X_target, n_conditions,
                random_state=adaptation['random_state'],
            )
            adapted_input = normalizer.transform(X_target, op_settings)
            adapted_eval = source.evaluate_external_model_input(
                prepared, adapted_input, y_target, units, cycles, rul,
                threshold,
            )
            variants['per_condition'] = _variant(adapted_eval, threshold)
            logger.info(
                "Cross-condition %s: fitted %d condition(s) on unlabelled "
                "target windows (%s).", name, normalizer.n_conditions,
                normalizer.condition_counts,
            )

        if threshold_strategy == 'target_alert_rate':
            baseline_threshold = select_alert_rate_threshold(
                baseline_eval['scores'], source_alert_rate
            )
            baseline_alert_eval = source.evaluate_external_model_input(
                prepared, baseline_input, y_target, units, cycles, rul,
                baseline_threshold,
            )
            variants['alert_rate'] = _variant(
                baseline_alert_eval, baseline_threshold
            )
            if adapted_input is not None:
                adapted_scores = source.model.positive_proba(adapted_input)
                adapted_threshold = select_alert_rate_threshold(
                    adapted_scores, source_alert_rate,
                )
                adapted_alert_eval = source.evaluate_external_model_input(
                    prepared, adapted_input, y_target, units, cycles, rul,
                    adapted_threshold,
                )
                variants['per_condition_alert_rate'] = _variant(
                    adapted_alert_eval, adapted_threshold
                )

        primary = variants[primary_name]
        baseline_row = _metric_row(
            name, source_architecture, threshold, len(y_target),
            int(len(np.unique(units))), variants['zero_shot']['metrics'],
            variants['zero_shot']['engine_metrics'],
            variants['zero_shot']['episode_metrics'],
            variants['zero_shot']['calibration'],
            source_eval['validation_metrics'],
            target_config['paths']['test_file'],
            variant='zero_shot',
            normalization='source',
            threshold_strategy='source',
            source_validation_alert_rate=source_alert_rate,
            target_alert_rate=variants['zero_shot']['achieved_alert_rate'],
        )
        row = _metric_row(
            name, source_architecture, primary['threshold'], len(y_target),
            int(len(np.unique(units))), primary['metrics'],
            primary['engine_metrics'], primary['episode_metrics'],
            primary['calibration'], source_eval['validation_metrics'],
            target_config['paths']['test_file'],
            variant=primary_name,
            normalization=normalization,
            threshold_strategy=threshold_strategy,
            source_validation_alert_rate=source_alert_rate,
            target_alert_rate=primary['achieved_alert_rate'],
            baseline_metrics=baseline_row,
        )
        rows.append(row)
        target_reports[name] = {
            'dataset': target.dataset_name,
            'metrics': primary['metrics'],
            'calibration': primary['calibration'],
            'engine_metrics': primary['engine_metrics'],
            'episode_metrics': primary['episode_metrics'],
            'variants': variants,
            'baseline': variants['zero_shot'],
            'selected_variant': primary_name,
            'adaptation': {
                'normalization': normalization,
                'threshold_strategy': threshold_strategy,
                'n_conditions': (
                    normalizer.n_conditions if normalizer is not None else None
                ),
                'condition_counts': (
                    normalizer.condition_counts
                    if normalizer is not None else None
                ),
            },
            'relative_degradation_vs_source_validation': {
                key: row[f'relative_degradation_{key}']
                for key in ('f1', 'recall', 'precision', 'average_precision')
            },
            'test_source': target_config['paths']['test_file'],
            'windows': int(len(y_target)),
            'engines': int(len(np.unique(units))),
        }
        logger.info(
            "Cross-condition %s -> %s (%s): F1=%.4f recall=%.4f "
            "precision=%.4f false_alarm_episodes=%s (zero-shot F1=%.4f, "
            "precision=%.4f).",
            source_dataset, name, primary_name,
            float(primary['metrics'].get('f1') or 0.0),
            float(primary['metrics'].get('recall') or 0.0),
            float(primary['metrics'].get('precision') or 0.0),
            primary['episode_metrics'].get('false_alarm_episodes'),
            float(variants['zero_shot']['metrics'].get('f1') or 0.0),
            float(variants['zero_shot']['metrics'].get('precision') or 0.0),
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
            'variant': primary_name,
            'normalization': normalization,
            'threshold_strategy': threshold_strategy,
            'adaptation_usage': (
                'none'
                if normalization == 'source' and threshold_strategy == 'source'
                else 'unlabeled_target_features'
            ),
            'source_transforms_reused': normalization == 'source',
            'source_threshold_reused': threshold_strategy == 'source',
            'source_validation_alert_rate': source_alert_rate,
            'variants_evaluated': list(variants),
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
