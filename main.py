#!/usr/bin/env python3
"""Lightweight HNSRM proof-of-concept on C-MAPSS FD001."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from src.config.config_manager import load_config
from src.logging.logging_setup import setup_logging
from src.pipeline.pipeline import ANSRDTPipeline, NumpyEncoder
from src.utils.clear_results import clear_results


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve(base: str, path: str) -> str:
    """Resolve a possibly relative path against a base directory."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def _print_summary(summary: dict) -> None:
    """Print run metrics, model comparison or representative ProbLog risks."""
    print("\n=== ANSR-DT / C-MAPSS FD001 ===")
    if 'selected_model' in summary and 'models' in summary:
        print(
            f"Comparison selected: {summary['selected_model']} "
            f"({summary.get('selection_metric')}, validation only)"
        )
        for name, entry in summary['models'].items():
            metrics = entry.get('validation_metrics', {})
            test_metrics = entry.get('test_metrics', {})
            print(
                f"  {name:<14} val_f1={metrics.get('f1', 0):.4f} "
                f"val_recall={metrics.get('recall', 0):.4f} "
                f"test_f1={test_metrics.get('f1', 0):.4f} "
                f"test_recall={test_metrics.get('recall', 0):.4f} "
                f"threshold={entry.get('decision_threshold', 0):.4f}"
            )
        return

    splits = summary.get('splits', {})
    print(
        "Windows: train={train_windows} val={validation_windows} test={test_windows}".format(
            **splits
        )
    )
    print(f"Architecture: {summary.get('architecture', 'unknown')}")
    for split_name in ('validation', 'test'):
        metrics = summary.get('metrics', {}).get(split_name, {})
        print(
            f"{split_name.capitalize():<10} "
            f"acc={metrics.get('accuracy', 0):.4f} "
            f"precision={metrics.get('precision', 0):.4f} "
            f"recall={metrics.get('recall', 0):.4f} "
            f"f1={metrics.get('f1', 0):.4f} "
            f"roc_auc={metrics.get('roc_auc') or 0:.4f}"
        )
        print(f"{'':<10} confusion_matrix={metrics.get('confusion_matrix')}")

    fusion = summary.get('probability_fusion', {})
    if fusion.get('applied'):
        print(
            f"Fusion: threshold={fusion.get('threshold', 0):.4f} "
            f"coefficients={fusion.get('coefficients')}"
        )
    alerting = summary.get('alerting', {}).get('modes', {})
    if alerting:
        immediate = alerting.get('immediate', {})
        print(
            f"Alerts: immediate false_alarms={immediate.get('total_false_alarms', 0)} "
            f"recall={immediate.get('mean_recall', 0):.4f}"
        )
        for name, mode in alerting.items():
            if name != 'immediate':
                print(
                    f"  {name:<14} false_alarms={mode.get('total_false_alarms', 0)} "
                    f"recall={mode.get('mean_recall', 0):.4f}"
                )

    representative = summary.get('representative_windows', {})
    if representative:
        print("\nProbLog representative windows:")
        for label, info in representative.items():
            print(
                f"  {label:<8} decision={info['neural_probability']:.4f} "
                f"problog_failure_risk={info['problog_failure_risk']:.4f} "
                f"insights={info['insights']}"
            )


def _print_multi_seed_summary(report: dict) -> None:
    """Print the headline of a multi-seed report."""
    aggregate = report.get('aggregate', {})
    print("\n=== ANSR-DT multi-seed ===")
    print(f"Seeds: {report['protocol']['seeds']}")
    for key in ('fusion_threshold', 'test_f1', 'test_recall',
                'total_false_alarms', 'mean_anticipation_cycles',
                'proportion_alerted_before_eol'):
        stats = aggregate.get(key, {})
        if stats.get('mean') is None:
            continue
        print(
            f"  {key:<34} mean={stats['mean']:.4f} "
            f"std={stats['std']:.4f} "
            f"[{stats['min']:.4f}, {stats['max']:.4f}]"
        )
    leakage = report.get('leakage_check', {})
    print(
        f"  leakage check: ok={leakage.get('ok')} "
        f"test_engines={leakage.get('test_engines')}"
    )
    artifacts = report.get('artifacts', {})
    print(f"  report: {artifacts.get('report_path')}")
    print(f"  csv:    {artifacts.get('summary_csv')}")
    print(f"  plot:   {artifacts.get('plot_path')}")


def main(clear: bool = False, compare_models: bool = False,
         architecture: str = None, multi_seed: bool = False,
         seeds: str = None, threshold_sensitivity: bool = False,
         operational_policy: bool = False) -> int:
    """Run the C-MAPSS pipeline and persist a run summary."""
    log_path = os.path.join(PROJECT_ROOT, 'logs', 'ansr_dt_main.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logging(log_file=log_path, log_level=logging.INFO)
    logger.info("--- Starting ANSR-DT run (%s) ---", datetime.now())

    config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    config = load_config(config_path)
    if architecture:
        config.setdefault('model', {})['architecture'] = architecture
        logger.info("Architecture overridden from CLI: %s", architecture)

    results_dir = _resolve(PROJECT_ROOT, config['paths'].get('results_dir', 'results'))
    config['paths']['results_dir'] = results_dir
    config['paths']['reasoning_rules_path'] = _resolve(
        PROJECT_ROOT,
        config['paths'].get('reasoning_rules_path', 'src/reasoning/rules.pl'),
    )
    os.makedirs(results_dir, exist_ok=True)
    if clear:
        logger.warning("Clearing results directory: %s", results_dir)
        clear_results(results_dir)

    if seeds or multi_seed:
        logger.info("Multi-seed mode enabled.")
        from src.experiments.multi_seed import parse_seeds, run_multi_seed

        seed_list = parse_seeds(seeds, config)
        config.setdefault('experiments', {}).setdefault(
            'multi_seed', {}
        )['enabled'] = True
        report = run_multi_seed(
            config, config_path, logger, seeds=seed_list,
            results_dir=results_dir,
        )
        _print_multi_seed_summary(report)
        logger.info("--- ANSR-DT multi-seed run finished ---")
        return 0

    if threshold_sensitivity:
        config.setdefault('threshold_sensitivity', {})['enabled'] = True
    if operational_policy:
        config.setdefault('operational_policy', {})['enabled'] = True

    pipeline = ANSRDTPipeline(config, config_path, logger)
    results = pipeline.run(compare_models=True if compare_models else None)

    summary = dict(results)
    summary['run_timestamp'] = datetime.now().isoformat()
    summary['rule_statistics'] = pipeline.get_rule_statistics()
    summary_path = os.path.join(config['paths']['results_dir'], 'final_run_results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    _print_summary(summary)
    logger.info("Run summary saved to %s", summary_path)
    logger.info("--- ANSR-DT run finished ---")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the lightweight ANSR-DT PoC.')
    parser.add_argument('--clear-results', action='store_true',
                        help='Clear the results directory before running.')
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare RandomForest and LightGBM on validation.')
    parser.add_argument('--architecture', choices=['random_forest', 'lightgbm', 'cnn_lstm'],
                        default=None, help='Override model.architecture.')
    parser.add_argument('--multi-seed', action='store_true',
                        help='Re-run the protocol over several split seeds.')
    parser.add_argument('--seeds', default=None,
                        help='Comma-separated seeds for --multi-seed (e.g. 42,43,44).')
    parser.add_argument('--threshold-sensitivity', action='store_true',
                        help='Sweep validation-selected decision thresholds.')
    parser.add_argument('--operational-policy', action='store_true',
                        help='Select the alert mode from maintenance costs.')
    parser.add_argument(
        "--benchmark",
        choices=["fd001", "skab"],
        default="fd001",
        help="Benchmark dataset to use (fd001 or skab)."
    )
    args = parser.parse_args()
    sys.exit(main(
        clear=args.clear_results,
        compare_models=args.compare_models,
        architecture=args.architecture,
        multi_seed=args.multi_seed,
        seeds=args.seeds,
        threshold_sensitivity=args.threshold_sensitivity,
        operational_policy=args.operational_policy,
    ))
