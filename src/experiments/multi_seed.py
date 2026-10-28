# src/experiments/multi_seed.py
"""Multi-seed validation of the FD001 protocol.

Each seed re-splits the *training* engines into training and validation
engines, retrains the selected architecture, recalibrates the probabilities,
re-selects the decision and fusion thresholds on validation, refits the fusion
on validation and evaluates on the unchanged official test engines.

Every seed also re-derives the symbolic thresholds from its own training
engines and writes a per-seed ``rules.pl`` copy, so the provenance gate of the
pipeline is satisfied without ever touching the packaged rules file.

Outputs:

- ``results/multi_seed_report.json``: per-seed metrics, aggregates and checks;
- ``results/multi_seed_summary.csv``: one row per seed plus summary rows;
- ``results/visualization/multi_seed_distributions.png``: five distributions.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.reasoning.thresholds import derive_thresholds, update_rules_file

CSV_FIELDS: List[str] = [
    'row', 'seed', 'train_engines', 'validation_engines', 'neural_threshold',
    'fusion_threshold', 'validation_f1', 'validation_recall',
    'validation_precision', 'validation_fbeta', 'validation_roc_auc',
    'validation_average_precision', 'test_f1', 'test_recall',
    'test_precision', 'test_fbeta', 'test_roc_auc', 'test_average_precision',
    'validation_brier', 'validation_ece', 'test_brier', 'test_ece',
    'engine_mean_recall', 'engine_mean_precision', 'total_false_alarms',
    'mean_anticipation_cycles', 'mean_lead_time_cycles',
    'proportion_alerted_before_eol', 'selected_alert_mode',
    'validation_expected_cost', 'test_expected_cost',
    'test_immediate_recall', 'test_immediate_false_alarms',
    'test_consecutive_3_recall', 'test_consecutive_3_false_alarms',
    'test_smoothed_3_recall', 'test_smoothed_3_false_alarms',
    'validation_selection_score',
]

SUMMARY_ROWS: Sequence[str] = ('mean', 'std', 'min', 'max', 'ci95_low',
                               'ci95_high')


def parse_seeds(values: Any,
                config: Optional[Dict[str, Any]] = None) -> List[int]:
    """Parse a seed list from the CLI or from ``experiments.multi_seed``."""
    if values is None:
        values = (
            (config or {}).get('experiments', {})
            .get('multi_seed', {})
            .get('seeds', [42])
        )
    if isinstance(values, str):
        raw = [item for item in values.replace(' ', '').split(',') if item]
    elif isinstance(values, Iterable):
        raw = list(values)
    else:
        raise ValueError("Seeds must be a comma-separated string or a list.")
    seeds: List[int] = []
    for item in raw:
        try:
            seed = int(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid seed: {item!r}.") from error
        if seed < 0:
            raise ValueError(f"Seeds must be >= 0, got {seed}.")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def fbeta(precision: float, recall: float, beta: float) -> float:
    """Return the F-beta score from precision and recall."""
    beta_sq = float(beta) ** 2
    denominator = beta_sq * float(precision) + float(recall)
    if not denominator:
        return 0.0
    return float(
        (1.0 + beta_sq) * float(precision) * float(recall) / denominator
    )


def aggregate(values: Iterable[Any]) -> Dict[str, Any]:
    """Return mean, standard deviation, extremes and the 95% interval."""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None,
            'ci95_low': None, 'ci95_high': None,
        }
    array = np.asarray(clean, dtype=float)
    count = int(len(array))
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    margin = 1.96 * std / np.sqrt(count) if count > 1 else 0.0
    return {
        'count': count,
        'mean': mean,
        'std': std,
        'min': float(array.min()),
        'max': float(array.max()),
        'ci95_low': float(mean - margin),
        'ci95_high': float(mean + margin),
    }


def selection_score(metrics: Dict[str, Any], metric: str,
                    beta: float) -> float:
    """Return the configured validation-only selection score."""
    name = str(metric).lower()
    if name == 'fbeta':
        return fbeta(
            float(metrics.get('precision') or 0.0),
            float(metrics.get('recall') or 0.0),
            beta,
        )
    if name == 'average_precision':
        return float(metrics.get('average_precision') or 0.0)
    if name == 'roc_auc':
        return float(metrics.get('roc_auc') or 0.0)
    return float(metrics.get(name) or 0.0)


def _metrics_block(metrics: Dict[str, Any], beta: float) -> Dict[str, Any]:
    """Return the reported metrics for one split, including F-beta."""
    precision = float(metrics.get('precision') or 0.0)
    recall = float(metrics.get('recall') or 0.0)
    return {
        'accuracy': float(metrics.get('accuracy') or 0.0),
        'precision': precision,
        'recall': recall,
        'f1': float(metrics.get('f1') or 0.0),
        'fbeta': fbeta(precision, recall, beta),
        'roc_auc': metrics.get('roc_auc'),
        'average_precision': metrics.get('average_precision'),
        'confusion_matrix': metrics.get('confusion_matrix'),
    }


def _seed_record(seed: int, results: Dict[str, Any],
                 train_units: Sequence[int], val_units: Sequence[int],
                 beta: float, policy_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the per-seed report entry from a pipeline result."""
    fusion = results.get('probability_fusion', {}) or {}
    fusion_applied = bool(fusion.get('applied'))
    neural_threshold = fusion.get(
        'neural_threshold',
        results.get('decision_threshold', {}).get('threshold'),
    )
    fusion_threshold = fusion.get('threshold') if fusion_applied else None
    validation = _metrics_block(results['metrics']['validation'], beta)
    test = _metrics_block(results['metrics']['test'], beta)

    if fusion_applied:
        validation_calibration = fusion.get('validation_calibration', {}) or {}
        test_calibration = fusion.get('test_calibration', {}) or {}
    else:
        calibration = results.get('calibration', {}).get('metrics', {}) or {}
        validation_calibration = calibration.get('validation', {}) or {}
        test_calibration = calibration.get('test', {}) or {}

    engine_summary = (
        results.get('engine_metrics', {}).get('test', {}) or {}
    ).get('summary', {}) or {}
    modes = (results.get('alerting', {}) or {}).get('modes', {}) or {}
    policy = results.get('operational_policy', {}) or {}
    selected_mode = policy.get('selected_mode')
    policy_validation = (policy.get('validation', {}) or {}).get(selected_mode, {})
    policy_test = (policy.get('test', {}) or {}).get(selected_mode, {})

    def mode_entry(name: str, split: str = 'test') -> Dict[str, Any]:
        source = (policy.get(split, {}) or {}).get(name)
        if source is not None:
            return {
                'recall': source.get('mean_recall'),
                'precision': source.get('precision'),
                'f1': source.get('f1'),
                'total_false_alarms': source.get('total_false_alarms'),
                'missed_critical_windows': source.get('missed_critical_windows'),
                'mean_anticipation_cycles': source.get('mean_anticipation_cycles'),
                'expected_cost': source.get('expected_cost'),
            }
        if split == 'test' and name in modes:
            summary = modes[name]
            return {
                'recall': summary.get('mean_recall'),
                'precision': summary.get('mean_precision'),
                'f1': fbeta(
                    float(summary.get('mean_precision') or 0.0),
                    float(summary.get('mean_recall') or 0.0),
                    2.0,
                ),
                'total_false_alarms': summary.get('total_false_alarms'),
                'missed_critical_windows': None,
                'mean_anticipation_cycles': summary.get('mean_anticipation_cycles'),
                'expected_cost': None,
            }
        return {}

    alert_modes = {
        name: mode_entry(name) for name in (
            'immediate', 'consecutive_2', 'consecutive_3',
            'smoothed_3', 'smoothed_5',
        )
    }
    return {
        'seed': int(seed),
        'train_units': [int(unit) for unit in train_units],
        'validation_units': [int(unit) for unit in val_units],
        'train_engines': int(len(train_units)),
        'validation_engines': int(len(val_units)),
        'architecture': results.get('architecture'),
        'test_source': results.get('test_source'),
        'thresholds': {
            'neural': None if neural_threshold is None else float(neural_threshold),
            'fusion': None if fusion_threshold is None else float(fusion_threshold),
            'fusion_strategy': fusion.get('threshold_strategy'),
        },
        'splits': results.get('splits', {}),
        'validation': validation,
        'test': test,
        'calibration': {
            'validation': {
                'brier_score': validation_calibration.get('brier_score'),
                'ece': validation_calibration.get('ece'),
            },
            'test': {
                'brier_score': test_calibration.get('brier_score'),
                'ece': test_calibration.get('ece'),
            },
        },
        'engine_level': {
            'mean_recall': engine_summary.get('mean_recall'),
            'mean_precision': engine_summary.get('mean_precision'),
            'total_false_alarms': engine_summary.get('total_false_alarms'),
            'mean_false_alarms_per_engine': engine_summary.get(
                'mean_false_alarms_per_engine'
            ),
            'mean_anticipation_cycles': engine_summary.get(
                'mean_anticipation_cycles'
            ),
            'mean_lead_time_cycles': engine_summary.get('mean_lead_time_cycles'),
            'proportion_alerted_before_eol': engine_summary.get(
                'proportion_alerted_before_eol'
            ),
            'proportion_alerted_before_eol_among_relevant': engine_summary.get(
                'proportion_alerted_before_eol_among_relevant'
            ),
        },
        'alert_modes': alert_modes,
        'operational_policy': {
            'selected_mode': selected_mode,
            'selection_split': policy.get('selection_split'),
            'validation_expected_cost': policy_validation.get('expected_cost'),
            'test_expected_cost': policy_test.get('expected_cost'),
            'false_negative_cost': policy_config.get(
                'false_negative_cost', 10.0
            ),
            'false_alarm_cost': policy_config.get('false_alarm_cost', 1.0),
        },
    }


def flatten_seed(record: Dict[str, Any], beta: float,
                 selection_metric: str = 'fbeta') -> Dict[str, Any]:
    """Flatten one seed record into the CSV row representation."""
    validation = record['validation']
    test = record['test']
    calibration = record['calibration']
    engine = record['engine_level']
    modes = record['alert_modes']
    policy = record['operational_policy']

    def mode_value(name: str, key: str) -> Any:
        entry = modes.get(name) or {}
        return entry.get(key)

    precision = float(validation['precision'])
    recall = float(validation['recall'])
    return {
        'row': 'seed',
        'seed': record['seed'],
        'train_engines': record['train_engines'],
        'validation_engines': record['validation_engines'],
        'neural_threshold': record['thresholds']['neural'],
        'fusion_threshold': record['thresholds']['fusion'],
        'validation_f1': validation['f1'],
        'validation_recall': recall,
        'validation_precision': precision,
        'validation_fbeta': validation['fbeta'],
        'validation_roc_auc': validation['roc_auc'],
        'validation_average_precision': validation['average_precision'],
        'test_f1': test['f1'],
        'test_recall': test['recall'],
        'test_precision': test['precision'],
        'test_fbeta': test['fbeta'],
        'test_roc_auc': test['roc_auc'],
        'test_average_precision': test['average_precision'],
        'validation_brier': calibration['validation']['brier_score'],
        'validation_ece': calibration['validation']['ece'],
        'test_brier': calibration['test']['brier_score'],
        'test_ece': calibration['test']['ece'],
        'engine_mean_recall': engine['mean_recall'],
        'engine_mean_precision': engine['mean_precision'],
        'total_false_alarms': engine['total_false_alarms'],
        'mean_anticipation_cycles': engine['mean_anticipation_cycles'],
        'mean_lead_time_cycles': engine['mean_lead_time_cycles'],
        'proportion_alerted_before_eol': engine[
            'proportion_alerted_before_eol'
        ],
        'selected_alert_mode': policy['selected_mode'],
        'validation_expected_cost': policy['validation_expected_cost'],
        'test_expected_cost': policy['test_expected_cost'],
        'test_immediate_recall': mode_value('immediate', 'recall'),
        'test_immediate_false_alarms': mode_value(
            'immediate', 'total_false_alarms'
        ),
        'test_consecutive_3_recall': mode_value('consecutive_3', 'recall'),
        'test_consecutive_3_false_alarms': mode_value(
            'consecutive_3', 'total_false_alarms'
        ),
        'test_smoothed_3_recall': mode_value('smoothed_3', 'recall'),
        'test_smoothed_3_false_alarms': mode_value(
            'smoothed_3', 'total_false_alarms'
        ),
        'validation_selection_score': selection_score(
            validation, selection_metric, beta
        ),
    }


def build_report(per_seed: Sequence[Dict[str, Any]],
                 config: Dict[str, Any],
                 leakage: Dict[str, Any],
                 protocol: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Aggregate the per-seed records into the multi-seed report."""
    multi_config = config.get('experiments', {}).get('multi_seed', {})
    beta = float(multi_config.get('beta', 2.0))
    metric = str(multi_config.get('selection_metric', 'fbeta'))
    rows = [flatten_seed(record, beta, metric) for record in per_seed]
    numeric_keys = [
        key for key in CSV_FIELDS
        if key not in {'row', 'seed', 'selected_alert_mode'}
    ]
    aggregate_fields = {
        key: aggregate(row.get(key) for row in rows)
        for key in numeric_keys
    }

    scores = [
        (row['validation_selection_score'], -row['seed'], row['seed'])
        for row in rows
    ]
    best_score, _, best_seed = max(scores) if scores else (None, None, None)

    return {
        'dataset': 'NASA C-MAPSS FD001',
        'protocol': {
            'seeds': [record['seed'] for record in per_seed],
            'selection_metric': metric,
            'beta': beta,
            'selection_split': 'validation',
            'test_usage': 'reporting_only',
            'resplit': 'training engines only',
            'official_test_unchanged': bool(
                leakage.get('test_windows_identical', True)
            ),
            **(protocol or {}),
        },
        'per_seed': list(per_seed),
        'rows': rows,
        'aggregate': aggregate_fields,
        'selection': {
            'metric': metric,
            'beta': beta,
            'selected_seed': best_seed,
            'validation_score': None if best_score is None else float(best_score),
            'note': (
                'The selected seed only illustrates which split maximised the '
                'validation metric; the official test split never drives it.'
            ),
        },
        'leakage_check': leakage,
        'timestamp': datetime.now().isoformat(),
    }


def write_summary_csv(report: Dict[str, Any], path: str) -> str:
    """Write one row per seed followed by the aggregate rows."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    rows = report.get('rows', [])
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})
        for summary_name in SUMMARY_ROWS:
            row = {'row': summary_name, 'selected_alert_mode': 'n/a'}
            for key in CSV_FIELDS:
                if key in {'row', 'seed', 'selected_alert_mode'}:
                    continue
                stats = report.get('aggregate', {}).get(key, {})
                row[key] = stats.get(summary_name)
            writer.writerow(row)
    return path


def save_distribution_plots(report: Dict[str, Any], path: str) -> str:
    """Plot the seed distributions of the five headline quantities."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = report.get('rows', [])
    if not rows:
        raise ValueError("No seed rows to plot.")
    panels = [
        ('fusion_threshold', 'Fusion threshold'),
        ('test_f1', 'Test F1 (window level)'),
        ('test_recall', 'Test recall (window level)'),
        ('total_false_alarms', 'False alarms (engine level)'),
        ('mean_anticipation_cycles', 'Mean anticipation (cycles)'),
    ]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3.8))
    seeds = [row['seed'] for row in rows]
    for axis, (key, label) in zip(axes, panels):
        values = [
            np.nan if row.get(key) is None else float(row[key])
            for row in rows
        ]
        axis.plot(seeds, values, marker='o', linestyle='-', color='tab:blue')
        axis.set_title(label, fontsize=10)
        axis.set_xlabel('Seed')
        axis.grid(alpha=0.3)
        if seeds:
            axis.set_xticks(seeds)
        mean = np.nanmean(values) if any(np.isfinite(values)) else np.nan
        if np.isfinite(mean):
            axis.axhline(mean, color='tab:orange', linestyle='--',
                         label=f'mean = {mean:.3f}')
            axis.legend(fontsize=7)
    fig.suptitle('Multi-seed stability on C-MAPSS FD001 (official test unchanged)')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def test_split_fingerprint(splits: Dict[str, Any]) -> Dict[str, Any]:
    """Return a content fingerprint of the official test split.

    Engine identifiers are only meaningful inside one file: the official test
    engines are different machines that happen to share the numbering of the
    training engines. The multi-seed guarantee is therefore checked on the
    content of the test split, not on identifier overlap.
    """
    import hashlib

    digest = hashlib.sha256()
    for key in ('X_test', 'y_test', 'test_units', 'test_cycles', 'test_rul'):
        array = np.ascontiguousarray(np.asarray(splits[key]))
        digest.update(str(array.shape).encode('utf-8'))
        digest.update(array.tobytes())
    return {
        'fingerprint': digest.hexdigest()[:32],
        'windows': int(len(np.asarray(splits['y_test']))),
        'engines': int(len(np.unique(splits['test_units']))),
    }


def _leakage_report(per_seed: Sequence[Dict[str, Any]],
                    fingerprints: Sequence[Dict[str, Any]]
                    ) -> Dict[str, Any]:
    """Check train/validation disjointness and test-split identity."""
    details = []
    disjoint = True
    for record, fingerprint in zip(per_seed, fingerprints):
        overlap = sorted(
            set(record['train_units']) & set(record['validation_units'])
        )
        if overlap:
            disjoint = False
        details.append({
            'seed': record['seed'],
            'train_validation_overlap': overlap,
            'train_engines': record['train_engines'],
            'validation_engines': record['validation_engines'],
            'test_windows': fingerprint['windows'],
            'test_engines': fingerprint['engines'],
            'test_fingerprint': fingerprint['fingerprint'],
        })
    hashes = [entry['test_fingerprint'] for entry in details]
    identical = bool(hashes) and all(value == hashes[0] for value in hashes)
    return {
        'ok': bool(disjoint and identical),
        'train_validation_disjoint': disjoint,
        'test_windows_identical': identical,
        'test_units_identical': bool(
            details and len({entry['test_engines'] for entry in details}) == 1
        ),
        'test_engines': details[0]['test_engines'] if details else 0,
        'test_windows': details[0]['test_windows'] if details else 0,
        'test_source': 'official FD001 test file (engine ids are file-local)',
        'details': details,
    }


def run_multi_seed(config: Dict[str, Any], config_path: str,
                   logger: Any, seeds: Any = None,
                   results_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run the full protocol once per seed and persist the aggregated report."""
    from src.pipeline.pipeline import ANSRDTPipeline, NumpyEncoder

    multi_config = config.get('experiments', {}).get('multi_seed', {})
    seed_list = parse_seeds(seeds, config)
    base_results = os.path.abspath(
        results_dir or config['paths']['results_dir']
    )
    multi_dir = os.path.join(base_results, 'multi_seed')
    os.makedirs(multi_dir, exist_ok=True)
    source_rules = os.path.abspath(
        config['paths']['reasoning_rules_path']
    )
    policy_config = config.get('operational_policy', {}) or {}

    per_seed: List[Dict[str, Any]] = []
    fingerprints: List[Dict[str, Any]] = []
    for seed in seed_list:
        seed_dir = os.path.join(multi_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        seed_rules = os.path.join(seed_dir, 'rules.pl')
        symbolic_enabled = bool(
            (config.get('symbolic_reasoning', {}) or {}).get('enabled', True)
        )
        if symbolic_enabled:
            shutil.copyfile(source_rules, seed_rules)

        seed_config = copy.deepcopy(config)
        seed_config['training']['random_state'] = int(seed)
        seed_config['paths']['results_dir'] = seed_dir
        if symbolic_enabled:
            seed_config['paths']['reasoning_rules_path'] = seed_rules
        seed_config.setdefault('hitl', {})['enabled'] = False
        seed_config.setdefault('audit', {})['enabled'] = False
        seed_config.setdefault('monitoring', {}).setdefault('drift', {})[
            'enabled'
        ] = False
        seed_config.setdefault('experiments', {}).setdefault(
            'multi_seed', {}
        )['enabled'] = False
        seed_config.setdefault('operational_policy', {})['enabled'] = True

        logger.info("Multi-seed run %d/%d (seed %d).",
                    seed_list.index(seed) + 1, len(seed_list), seed)
        pipeline = ANSRDTPipeline(seed_config, config_path, logger)
        splits = pipeline.load_splits()
        train_units = sorted({
            int(unit) for unit in np.asarray(splits['train_units']).tolist()
        })
        val_units = sorted({
            int(unit) for unit in np.asarray(splits['val_units']).tolist()
        })
        if symbolic_enabled:
            frame = pipeline.data_loader._read_frame(
                seed_config['paths']['data_file']
            )
            thresholds = derive_thresholds(frame, train_units)
            update_rules_file(seed_rules, thresholds)

        results = pipeline.run(compare_models=False)
        fingerprints.append(test_split_fingerprint(splits))
        per_seed.append(_seed_record(
            seed, results, train_units, val_units,
            beta=float(multi_config.get('beta', 2.0)),
            policy_config=policy_config,
        ))
        logger.info(
            "Seed %d: val F1=%.4f test F1=%.4f test recall=%.4f "
            "false alarms=%s",
            seed,
            per_seed[-1]['validation']['f1'],
            per_seed[-1]['test']['f1'],
            per_seed[-1]['test']['recall'],
            per_seed[-1]['engine_level']['total_false_alarms'],
        )

    leakage = _leakage_report(per_seed, fingerprints)
    report = build_report(
        per_seed, config, leakage,
        protocol={
            'train_fraction': config['training'].get('train_fraction'),
            'validation_split': config['training'].get('validation_split'),
            'architecture': config.get('model', {}).get('architecture'),
            'per_seed_directory': multi_dir,
        },
    )
    report_path = os.path.join(base_results, 'multi_seed_report.json')
    with open(report_path, 'w') as file:
        json.dump(report, file, indent=2, cls=NumpyEncoder)
    csv_path = write_summary_csv(
        report, os.path.join(base_results, 'multi_seed_summary.csv')
    )
    plot_path = save_distribution_plots(
        report,
        os.path.join(base_results, 'visualization',
                     'multi_seed_distributions.png'),
    )
    report['artifacts'] = {
        'report_path': report_path,
        'summary_csv': csv_path,
        'plot_path': plot_path,
        'per_seed_directory': multi_dir,
    }
    with open(report_path, 'w') as file:
        json.dump(report, file, indent=2, cls=NumpyEncoder)
    logger.info(
        "Multi-seed report saved to %s (%d seeds, leakage_ok=%s).",
        report_path, len(per_seed), leakage['ok'],
    )
    return report
