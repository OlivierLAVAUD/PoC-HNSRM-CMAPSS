# src/evaluation/robustness_report.py
"""Consolidated robustness dossier for an existing C-MAPSS run.

The report is post-hoc: it reads the artifacts already produced by a run and
never retrains a model or changes a decision. It brings seed stability,
threshold sensitivity, fusion diagnostics, symbolic contribution, calibration,
drift, HITL risk ordering and cross-condition results into one decision-ready
JSON and Markdown document.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from typing import Any, Dict, List, Optional

CHECK_PASS = 'pass'
CHECK_WARN = 'warn'
CHECK_FAIL = 'fail'
CHECK_SKIP = 'skip'


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as file:
            payload = json.load(file)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _mean_std(values: List[Any]) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {'count': 0, 'mean': None, 'std': None}
    return {
        'count': len(clean),
        'mean': float(statistics.fmean(clean)),
        'std': float(statistics.pstdev(clean)) if len(clean) > 1 else 0.0,
    }


def _check(name: str, status: str, detail: str,
           value: Any = None) -> Dict[str, Any]:
    return {
        'name': name,
        'status': status,
        'detail': detail,
        'value': value,
    }


def build_robustness_report(results_dir: str) -> Dict[str, Any]:
    """Load all available artifacts and build the robustness dossier."""
    paths = {
        'core': os.path.join(results_dir, 'cmapss_results.json'),
        'final': os.path.join(results_dir, 'final_run_results.json'),
        'multi_seed': os.path.join(results_dir, 'multi_seed_report.json'),
        'threshold_sensitivity': os.path.join(
            results_dir, 'threshold_sensitivity.json'
        ),
        'fusion_diagnostics': os.path.join(
            results_dir, 'fusion_diagnostics.json'
        ),
        'fusion_ablation': os.path.join(results_dir, 'fusion_ablation.json'),
        'nested_validation': os.path.join(
            results_dir, 'fusion_nested_validation.json'
        ),
        'symbolic_contribution': os.path.join(
            results_dir, 'symbolic_contribution.json'
        ),
        'alert_policy': os.path.join(
            results_dir, 'alert_policy_comparison.json'
        ),
        'cross_condition': os.path.join(
            results_dir, 'cross_condition_report.json'
        ),
        'drift': os.path.join(results_dir, 'drift_report.json'),
        'hitl': os.path.join(results_dir, 'hitl_queue.json'),
    }
    sources = {name: os.path.exists(path) for name, path in paths.items()}
    loaded = {
        name: _load(path) for name, path in paths.items()
        if sources[name]
    }
    core = loaded.get('core') or loaded.get('final') or {}
    checks: List[Dict[str, Any]] = []

    if core:
        test_metrics = (core.get('metrics') or {}).get('test', {})
        checks.append(_check(
            'core_run', CHECK_PASS,
            'C-MAPSS results are present.',
            {'test_f1': test_metrics.get('f1')},
        ))
        if (test_metrics.get('f1') or 0.0) <= 0.0:
            checks.append(_check(
                'test_signal', CHECK_FAIL,
                'The official test F1 is zero or missing.',
            ))
    else:
        checks.append(_check(
            'core_run', CHECK_FAIL,
            'cmapss_results.json / final_run_results.json is missing.',
        ))

    calibration = (
        (core.get('calibration') or {}).get('metrics', {}) or {}
    ).get('test', {}) or {}
    ece = calibration.get('ece')
    if ece is None:
        checks.append(_check(
            'calibration', CHECK_SKIP, 'No test ECE recorded.'
        ))
    elif float(ece) <= 0.10:
        checks.append(_check(
            'calibration', CHECK_PASS,
            f'Test ECE {float(ece):.4f} <= 0.10.', float(ece),
        ))
    else:
        checks.append(_check(
            'calibration', CHECK_WARN,
            f'Test ECE {float(ece):.4f} > 0.10; recalibrate on new labelled data.',
            float(ece),
        ))

    seed_report = loaded.get('multi_seed')
    seed_stability = {'available': seed_report is not None}
    if seed_report:
        aggregate = seed_report.get('aggregate', {}) or {}
        stability = {
            key: (
                aggregate.get(key, {}).get('mean'),
                aggregate.get(key, {}).get('std'),
            )
            for key in ('test_f1', 'test_recall', 'test_precision',
                        'fusion_threshold', 'total_false_alarms')
        }
        seed_stability.update({
            'seeds': seed_report.get('protocol', {}).get('seeds'),
            'stability': stability,
            'leakage_check': seed_report.get('leakage_check'),
        })
        std = aggregate.get('test_f1', {}).get('std')
        leakage = seed_report.get('leakage_check', {}) or {}
        if not leakage.get('ok', False):
            checks.append(_check(
                'multi_seed_leakage', CHECK_FAIL,
                'The multi-seed leakage check failed.',
            ))
        else:
            checks.append(_check(
                'multi_seed_leakage', CHECK_PASS,
                'Training/validation engines are disjoint and the official '
                'test split is unchanged.',
            ))
        if std is None:
            checks.append(_check(
                'seed_stability', CHECK_SKIP, 'No test F1 dispersion recorded.'
            ))
        elif float(std) <= 0.05:
            checks.append(_check(
                'seed_stability', CHECK_PASS,
                f'Test F1 standard deviation {float(std):.4f} <= 0.05.',
                float(std),
            ))
        else:
            checks.append(_check(
                'seed_stability', CHECK_WARN,
                f'Test F1 standard deviation {float(std):.4f} > 0.05; '
                'results are split-sensitive.', float(std),
            ))
    else:
        checks.append(_check(
            'seed_stability', CHECK_SKIP,
            'multi_seed_report.json is absent; run --multi-seed.',
        ))

    sensitivity = loaded.get('threshold_sensitivity')
    if sensitivity:
        production = sensitivity.get('production_candidate', {}) or {}
        test = production.get('test', {}) or {}
        checks.append(_check(
            'threshold_sensitivity', CHECK_PASS,
            'Validation-selected threshold was replayed across the grid.',
            {
                'threshold': production.get('threshold'),
                'test_f1': test.get('f1'),
                'test_precision': test.get('precision'),
                'test_recall': test.get('recall'),
            },
        ))
    else:
        checks.append(_check(
            'threshold_sensitivity', CHECK_SKIP,
            'threshold_sensitivity.json is absent.',
        ))

    nested = loaded.get('nested_validation')
    nested_summary = {'available': nested is not None}
    if nested:
        metrics = nested.get('metrics', {}) or {}
        nested_summary.update({
            'folds_evaluated': nested.get('folds_evaluated'),
            'f1': metrics.get('f1'),
            'threshold': metrics.get('threshold'),
        })
        checks.append(_check(
            'fusion_nested_validation', CHECK_PASS,
            f"{nested.get('folds_evaluated', 0)} engine-grouped folds evaluated.",
        ))
    else:
        checks.append(_check(
            'fusion_nested_validation', CHECK_SKIP,
            'fusion_nested_validation.json is absent.',
        ))

    diagnostics = loaded.get('fusion_diagnostics')
    ablation = loaded.get('fusion_ablation')
    fusion_summary = {
        'diagnostics_available': diagnostics is not None,
        'ablation_available': ablation is not None,
    }
    if diagnostics:
        fusion_summary.update({
            'folds_evaluated': diagnostics.get('folds_evaluated'),
            'relative_importance': diagnostics.get('relative_importance'),
            'threshold_stability': diagnostics.get('threshold_stability'),
        })
        checks.append(_check(
            'fusion_stability', CHECK_PASS,
            'Engine-grouped fusion diagnostics are available.',
        ))
    if ablation:
        fusion_summary.update({
            'ablation_scope': ablation.get('scope'),
            'ablation_summary': ablation.get('summary'),
            'ablations': ablation.get('ablations'),
        })
    else:
        checks.append(_check(
            'fusion_stability', CHECK_SKIP,
            'fusion_diagnostics.json is absent.',
        ))

    contribution = loaded.get('symbolic_contribution')
    symbolic_summary = {'available': contribution is not None}
    if contribution:
        symbolic_summary.update({
            'signals': contribution.get('signals'),
            'summary': contribution.get('summary'),
            'contributions': contribution.get('contributions'),
        })
        helpful = (contribution.get('summary') or {}).get(
            'helpful_signals', []
        )
        checks.append(_check(
            'symbolic_contribution',
            CHECK_PASS if helpful else CHECK_WARN,
            (contribution.get('summary') or {}).get('answer')
            or 'Symbolic contribution report is available.',
            helpful,
        ))
    else:
        checks.append(_check(
            'symbolic_contribution', CHECK_SKIP,
            'symbolic_contribution.json is absent (fusion may be disabled).',
        ))

    policy = loaded.get('alert_policy')
    policy_summary = {'available': policy is not None}
    if policy:
        policy_summary.update({
            'selected_mode': policy.get('selected_mode'),
            'cost_unit': policy.get('cost_unit'),
            'rationale': policy.get('rationale'),
        })
        checks.append(_check(
            'alert_policy', CHECK_PASS,
            f"Selected {policy.get('selected_mode')} on validation "
            f"({policy.get('cost_unit')} cost).",
        ))
    else:
        checks.append(_check(
            'alert_policy', CHECK_SKIP,
            'alert_policy_comparison.json is absent.',
        ))

    cross = loaded.get('cross_condition')
    cross_summary = {'available': cross is not None}
    if cross:
        cross_summary.update({
            'source': cross.get('protocol', {}).get('source_dataset'),
            'targets': cross.get('protocol', {}).get('target_datasets'),
            'rows': cross.get('rows', []),
        })
        checks.append(_check(
            'cross_condition', CHECK_PASS,
            'Cross-condition report is available; targets are reporting only.',
        ))
    else:
        checks.append(_check(
            'cross_condition', CHECK_SKIP,
            'cross_condition_report.json is absent.',
        ))

    drift = loaded.get('drift')
    drift_summary = {'available': drift is not None}
    if drift:
        level = drift.get('level')
        drift_summary.update({
            'level': level,
            'score': drift.get('score'),
            'recommendation': drift.get('recommendation'),
        })
        checks.append(_check(
            'drift', CHECK_PASS if level == 'ok' else CHECK_WARN,
            f"Drift level is {level!r}."
            + (' On the official test engines a non-ok level is expected.'
               if level != 'ok' else ''),
            level,
        ))
    else:
        checks.append(_check(
            'drift', CHECK_SKIP, 'drift_report.json is absent.'
        ))

    rul_regression = core.get('rul_regression') or {}
    uncertainty = (
        rul_regression.get('uncertainty')
        or rul_regression.get('uncertainty_summary')
        or {}
    )
    rul_uncertainty_summary = {
        'available': bool(uncertainty),
        'rul_enabled': rul_regression.get('enabled'),
        'uncertainty': uncertainty,
    }
    if uncertainty:
        checks.append(_check(
            'rul_uncertainty', CHECK_PASS,
            'The RUL uncertainty ensemble is recorded in the run summary.',
            uncertainty,
        ))
    else:
        checks.append(_check(
            'rul_uncertainty', CHECK_SKIP,
            'RUL uncertainty is disabled or absent from the run summary.',
        ))

    hitl = loaded.get('hitl')
    hitl_summary = {'available': hitl is not None}
    if hitl:
        metadata = hitl.get('metadata', {}) or {}
        cases = hitl.get('cases', []) or []
        bands: Dict[str, int] = {}
        for case in cases:
            band = case.get('risk_band', 'low')
            bands[band] = bands.get(band, 0) + 1
        hitl_summary.update({
            'total_cases': len(cases),
            'prioritized': metadata.get('prioritization_enabled'),
            'risk_bands': bands,
            'highest_risk_score': max(
                (case.get('risk_score', 0.0) for case in cases), default=None
            ),
        })
        checks.append(_check(
            'hitl_prioritization',
            CHECK_PASS if metadata.get('prioritization_enabled') else CHECK_WARN,
            'HITL cases carry risk bands.'
            if metadata.get('prioritization_enabled')
            else 'HITL risk fields exist but risk ordering is disabled.',
            bands,
        ))
    else:
        checks.append(_check(
            'hitl_prioritization', CHECK_SKIP, 'hitl_queue.json is absent.'
        ))

    statuses = [check['status'] for check in checks]
    if CHECK_FAIL in statuses:
        overall = CHECK_FAIL
    elif CHECK_WARN in statuses:
        overall = CHECK_WARN
    else:
        overall = CHECK_PASS
    counts = {
        status: statuses.count(status)
        for status in (CHECK_PASS, CHECK_WARN, CHECK_FAIL, CHECK_SKIP)
    }
    return {
        'results_dir': os.path.abspath(results_dir),
        'dataset': core.get('dataset'),
        'sources': sources,
        'overall_status': overall,
        'check_counts': counts,
        'checks': checks,
        'core': {
            'dataset': core.get('dataset'),
            'architecture': core.get('architecture'),
            'splits': core.get('splits'),
            'metrics': core.get('metrics'),
            'calibration': calibration,
            'engine_metrics': core.get('engine_metrics'),
            'episode_metrics': core.get('episode_metrics'),
        },
        'seed_stability': seed_stability,
        'threshold_sensitivity': (
            sensitivity.get('production_candidate')
            if sensitivity else None
        ),
        'fusion_stability': fusion_summary,
        'nested_validation': nested_summary,
        'symbolic_contribution': symbolic_summary,
        'alert_policy': policy_summary,
        'cross_condition': cross_summary,
        'drift': drift_summary,
        'rul_uncertainty': rul_uncertainty_summary,
        'hitl_prioritization': hitl_summary,
        'summary': (
            f"Robustness status: {overall.upper()} "
            f"({counts[CHECK_PASS]} pass, {counts[CHECK_WARN]} warn, "
            f"{counts[CHECK_FAIL]} fail, {counts[CHECK_SKIP]} skipped)."
        ),
    }


def render_robustness_markdown(report: Dict[str, Any]) -> str:
    """Render the dossier as a compact Markdown document."""
    lines = [
        "# C-MAPSS robustness report",
        "",
        report.get('summary', ''),
        "",
        f"- Dataset: `{report.get('dataset')}`",
        f"- Results directory: `{report.get('results_dir')}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in report.get('checks', []):
        detail = str(check.get('detail', '')).replace('|', '/')
        lines.append(
            f"| {check['name']} | {check['status']} | {detail} |"
        )
    core_metrics = (report.get('core', {}).get('metrics') or {})
    if core_metrics:
        lines.extend([
            "",
            "## Core metrics",
            "",
            "| Split | Precision | Recall | F1 | ROC-AUC | AP |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for split in ('validation', 'test'):
            metrics = core_metrics.get(split, {}) or {}
            lines.append(
                f"| {split} | {float(metrics.get('precision') or 0):.4f} | "
                f"{float(metrics.get('recall') or 0):.4f} | "
                f"{float(metrics.get('f1') or 0):.4f} | "
                f"{float(metrics.get('roc_auc') or 0):.4f} | "
                f"{float(metrics.get('average_precision') or 0):.4f} |"
            )
    seeds = report.get('seed_stability', {}) or {}
    stability = seeds.get('stability') or {}
    if stability:
        lines.extend([
            "",
            "## Multi-seed stability",
            "",
            "| Quantity | Mean | Std |",
            "|---|---:|---:|",
        ])
        for key, values in stability.items():
            mean, std = values
            lines.append(
                f"| {key} | {_fmt(mean)} | {_fmt(std)} |"
            )
    cross = report.get('cross_condition', {}) or {}
    if cross.get('rows'):
        lines.extend([
            "",
            "## Cross-condition targets",
            "",
            "| Target | F1 | Recall | Precision |",
            "|---|---:|---:|---:|",
        ])
        for row in cross['rows']:
            lines.append(
                f"| {row.get('target_dataset')} | "
                f"{_fmt(row.get('f1'))} | {_fmt(row.get('recall'))} | "
                f"{_fmt(row.get('precision'))} |"
            )
    lines.append('')
    return '\n'.join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return 'n/a'
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def export_robustness_csv(report: Dict[str, Any], path: str) -> str:
    """Write one row per robustness check to CSV."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fields = ['check', 'status', 'value', 'detail']
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for check in report.get('checks', []):
            writer.writerow({
                'check': check.get('name'),
                'status': check.get('status'),
                'value': json.dumps(check.get('value')),
                'detail': check.get('detail'),
            })
    return path


def render_robustness_plot(report: Dict[str, Any], path: str) -> Optional[str]:
    """Render the check-status distribution as a PNG, or skip gracefully."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return None
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    counts = report.get('check_counts', {}) or {}
    labels = ['pass', 'warn', 'fail', 'skip']
    values = [int(counts.get(label, 0)) for label in labels]
    colors = ['#2e7d32', '#ef6c00', '#c62828', '#9e9e9e']
    figure, axis = plt.subplots(figsize=(6, 3.5))
    axis.bar(labels, values, color=colors)
    axis.set_ylabel('Checks')
    axis.set_title(
        f"Robustness: {str(report.get('overall_status', 'unknown')).upper()}"
    )
    for index, value in enumerate(values):
        axis.text(index, value, str(value), ha='center', va='bottom')
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def run_robustness_report(results_dir: str,
                          json_path: Optional[str] = None,
                          markdown_path: Optional[str] = None,
                          csv_path: Optional[str] = None,
                          plot_path: Optional[str] = None
                          ) -> Dict[str, Any]:
    """Build and persist the JSON, Markdown, CSV and PNG dossier."""
    report = build_robustness_report(results_dir)
    json_target = json_path or os.path.join(
        results_dir, 'robustness_report.json'
    )
    markdown_target = markdown_path or os.path.join(
        results_dir, 'robustness_report.md'
    )
    csv_target = csv_path or os.path.join(
        results_dir, 'robustness_report.csv'
    )
    plot_target = plot_path or os.path.join(
        results_dir, 'visualization', 'robustness_report.png'
    )
    os.makedirs(results_dir, exist_ok=True)
    with open(json_target, 'w') as file:
        json.dump(report, file, indent=2)
    with open(markdown_target, 'w') as file:
        file.write(render_robustness_markdown(report))
    export_robustness_csv(report, csv_target)
    rendered_plot = render_robustness_plot(report, plot_target)
    report['artifacts'] = {
        'json_path': json_target,
        'markdown_path': markdown_target,
        'csv_path': csv_target,
        'plot_path': rendered_plot,
    }
    return report
