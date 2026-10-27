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


def main(clear: bool = False, compare_models: bool = False,
         architecture: str = None) -> int:
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
    os.makedirs(results_dir, exist_ok=True)
    if clear:
        logger.warning("Clearing results directory: %s", results_dir)
        clear_results(results_dir)

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
    parser.add_argument('--architecture', choices=['random_forest', 'lightgbm'],
                        default=None, help='Override model.architecture.')
    args = parser.parse_args()
    sys.exit(main(
        clear=args.clear_results,
        compare_models=args.compare_models,
        architecture=args.architecture,
    ))
