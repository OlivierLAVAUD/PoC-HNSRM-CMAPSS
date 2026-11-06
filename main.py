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


CMAPSS_DATASETS = ('fd001', 'fd002', 'fd003', 'fd004')


def _config_path_for_dataset(project_root: str, dataset: str) -> str:
    """Return the configuration file for a C-MAPSS sub-dataset."""
    name = str(dataset or 'fd001').lower()
    if name not in CMAPSS_DATASETS:
        raise ValueError(f"Unsupported C-MAPSS dataset: {dataset!r}")
    filename = 'config.yaml' if name == 'fd001' else f'config_{name}.yaml'
    return os.path.join(project_root, 'configs', filename)


def _parse_datasets(values: str) -> list:
    """Parse and validate a comma-separated C-MAPSS dataset list."""
    names = [
        item.strip().lower() for item in str(values or '').split(',')
        if item.strip()
    ]
    if not names:
        raise ValueError("At least one C-MAPSS dataset is required.")
    invalid = [name for name in names if name not in CMAPSS_DATASETS]
    if invalid:
        raise ValueError(
            f"Unsupported C-MAPSS dataset(s): {invalid}. "
            f"Allowed: {list(CMAPSS_DATASETS)}."
        )
    return names


def _print_summary(summary: dict) -> None:
    """Print run metrics, model comparison or representative ProbLog risks."""
    print(f"\n=== ANSR-DT / {summary.get('dataset', 'C-MAPSS')} ===")
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


def _print_robustness_summary(report: dict) -> None:
    """Print the headline of a robustness dossier."""
    print("\n=== C-MAPSS robustness report ===")
    print(report.get('summary', ''))
    for check in report.get('checks', []):
        print(
            f"  {check['name']:<28} {check['status']:<5} "
            f"{check.get('detail', '')}"
        )
    artifacts = report.get('artifacts', {})
    print(f"  report:   {artifacts.get('json_path')}")
    print(f"  markdown: {artifacts.get('markdown_path')}")
    print(f"  csv:      {artifacts.get('csv_path')}")
    print(f"  plot:     {artifacts.get('plot_path')}")


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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by the FD001 and SKAB entry points."""
    parser = argparse.ArgumentParser(
        description='Run the lightweight ANSR-DT PoC.'
    )
    parser.add_argument('--clear-results', action='store_true',
                        help='Clear the results directory before running.')
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare RandomForest and LightGBM on validation.')
    parser.add_argument(
        '--architecture',
        choices=['random_forest', 'lightgbm', 'cnn_lstm'],
        default=None, help='Override model.architecture.',
    )
    parser.add_argument('--multi-seed', action='store_true',
                        help='Re-run the protocol over several split seeds.')
    parser.add_argument('--seeds', default=None,
                        help='Comma-separated seeds for --multi-seed (e.g. 42,43,44).')
    parser.add_argument('--threshold-sensitivity', action='store_true',
                        help='Sweep validation-selected decision thresholds.')
    parser.add_argument('--operational-policy', action='store_true',
                        help='Select the alert mode from maintenance costs.')
    parser.add_argument('--robustness-report', action='store_true',
                        help='Build a robustness dossier from existing artifacts.')
    parser.add_argument('--representation-ablation', action='store_true',
                        help='Run the controlled physical-representation ablation.')
    parser.add_argument('--few-shot-target', default=None,
                        choices=['fd002', 'fd003', 'fd004'],
                        help='Run few-shot target adaptation (not zero-shot).')
    parser.add_argument(
        '--cmapss-dataset',
        choices=list(CMAPSS_DATASETS),
        default='fd001',
        help='C-MAPSS sub-dataset configuration (fd001-fd004).',
    )
    parser.add_argument(
        '--cross-condition', action='store_true',
        help='Train on one C-MAPSS condition and test unchanged on others.',
    )
    parser.add_argument(
        '--cross-train-dataset', choices=list(CMAPSS_DATASETS), default=None,
        help='Source dataset for --cross-condition (defaults to --cmapss-dataset).',
    )
    parser.add_argument(
        '--cross-test-datasets', default=None,
        help='Comma-separated target datasets for --cross-condition. '
             'Defaults to experiments.cross_condition.test_conditions.',
    )
    parser.add_argument(
        '--cross-normalization',
        choices=['source', 'per_condition'], default=None,
        help='Label-free target normalisation for --cross-condition.',
    )
    parser.add_argument(
        '--cross-threshold',
        choices=['source', 'target_alert_rate'], default=None,
        help='Threshold strategy for --cross-condition: reuse the source '
             'threshold or match the source validation alert rate.',
    )
    parser.add_argument(
        '--benchmark',
        choices=['fd001', 'skab'],
        default='fd001',
        help='Benchmark dataset to use (fd001 or skab).',
    )
    from src.skab.run import add_skab_arguments
    return add_skab_arguments(parser)


def main(clear: bool = False, compare_models: bool = False,
         architecture: str = None, multi_seed: bool = False,
         seeds: str = None, threshold_sensitivity: bool = False,
         operational_policy: bool = False, benchmark: str = 'fd001',
         cmapss_dataset: str = 'fd001', cross_condition: bool = False,
         robustness_report: bool = False, representation_ablation: bool = False,
         few_shot_target: str = None,
         cross_train_dataset: str = None, cross_test_datasets: str = None,
         cross_normalization: str = None, cross_threshold: str = None,
         modes=None, data_path: str = None,
         model_type: str = 'random_forest', epochs: int = 50,
         symbolic: bool = True, min_support: int = 3,
         max_rules: int = 50, min_confidence: float = 0.7) -> int:
    """Run the configured benchmark and persist its summary."""
    if benchmark == 'skab':
        from src.skab.run import run_skab_from_args
        skab_args = argparse.Namespace(
            benchmark='skab', modes=modes or ['random_forest', 'symbolic'],
            data_path=data_path, model_type=model_type, epochs=epochs,
            symbolic=symbolic, min_support=min_support,
            max_rules=max_rules, min_confidence=min_confidence,
        )
        return run_skab_from_args(skab_args, project_root=PROJECT_ROOT)

    log_path = os.path.join(PROJECT_ROOT, 'logs', 'ansr_dt_main.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logging(log_file=log_path, log_level=logging.INFO)
    logger.info("--- Starting ANSR-DT run (%s) ---", datetime.now())

    source_dataset = (cross_train_dataset or cmapss_dataset or 'fd001').lower()
    config_path = _config_path_for_dataset(PROJECT_ROOT, source_dataset)
    config = load_config(config_path)
    cross_settings = (
        config.get('experiments', {}) or {}
    ).get('cross_condition', {}) or {}
    if cross_settings.get('enabled', False):
        cross_condition = True
    if cross_condition and not cross_train_dataset:
        train_conditions = [
            str(name).strip().lower()
            for name in cross_settings.get('train_conditions') or []
            if str(name).strip()
        ]
        if train_conditions and train_conditions[0] != source_dataset:
            source_dataset = train_conditions[0]
            config_path = _config_path_for_dataset(
                PROJECT_ROOT, source_dataset
            )
            config = load_config(config_path)
            cross_settings = (
                config.get('experiments', {}) or {}
            ).get('cross_condition', {}) or {}
    if cross_normalization or cross_threshold:
        adaptation = cross_settings.setdefault('adaptation', {})
        if cross_normalization:
            adaptation['normalization'] = cross_normalization
        if cross_threshold:
            adaptation['threshold_strategy'] = cross_threshold
        config.setdefault('experiments', {})['cross_condition'] = (
            cross_settings
        )
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

    if robustness_report:
        from src.evaluation.robustness_report import run_robustness_report

        report = run_robustness_report(results_dir)
        _print_robustness_summary(report)
        logger.info("--- ANSR-DT robustness report finished ---")
        return 0

    if representation_ablation:
        from src.experiments.representation_ablation import (
            run_representation_ablation,
        )

        ablation_config_path = _config_path_for_dataset(PROJECT_ROOT, 'fd001')
        ablation_config = load_config(ablation_config_path)
        targets = []
        for name in ('fd002', 'fd003', 'fd004'):
            target_path = _config_path_for_dataset(PROJECT_ROOT, name)
            targets.append({
                'name': name.upper(),
                'config': load_config(target_path),
                'config_path': target_path,
            })
        report = run_representation_ablation(
            ablation_config, ablation_config_path, targets, logger,
            results_dir=os.path.join(
                results_dir, 'representation_ablation'
            ),
            architecture=architecture,
        )
        print("\n=== Representation ablation ===")
        for statement in report['conclusion']['statements']:
            print(f"  {statement}")
        print(f"  summary: {report['artifacts']['summary_json']}")
        print(f"  csv:     {report['artifacts']['summary_csv']}")
        logger.info("--- Representation ablation finished ---")
        return 0

    if few_shot_target:
        from src.experiments.few_shot import run_few_shot_target

        source_path = _config_path_for_dataset(PROJECT_ROOT, 'fd001')
        source_config = load_config(source_path)
        target_name = few_shot_target.lower()
        target_path = _config_path_for_dataset(PROJECT_ROOT, target_name)
        target_config = load_config(target_path)
        target_config['few_shot_target'] = {
            **(target_config.get('few_shot_target', {}) or {}),
            **((config.get('few_shot_target', {}) or {})),
            'enabled': True,
            'target_dataset': target_name,
        }
        report = run_few_shot_target(
            source_config, target_config, target_path, logger,
            results_dir=os.path.join(
                results_dir, 'few_shot_target', target_name
            ),
            architecture=architecture,
        )
        print("\n=== Few-shot target adaptation ===")
        print(
            f"source=FD001 target={target_name.upper()} "
            f"protocol={report['protocol']}"
        )
        print(
            f"threshold={report['threshold']:.4f} "
            f"test_f1={float(report['metrics'].get('f1') or 0):.4f} "
            f"test_recall={float(report['metrics'].get('recall') or 0):.4f}"
        )
        print(f"  warning: {report['warning']}")
        print(f"  report: {report['artifacts']['report_path']}")
        logger.info("--- Few-shot target adaptation finished ---")
        return 0

    if cross_condition:
        from src.experiments.cross_condition import run_cross_condition

        configured_targets = [
            str(name).strip().lower()
            for name in cross_settings.get('test_conditions') or []
            if str(name).strip()
        ]
        target_names = _parse_datasets(
            cross_test_datasets
            or (','.join(configured_targets) if configured_targets
                else 'fd002,fd004')
        )
        if source_dataset in target_names:
            raise ValueError(
                "The cross-condition source dataset cannot also be a target."
            )
        targets = []
        for name in target_names:
            target_path = _config_path_for_dataset(PROJECT_ROOT, name)
            targets.append({
                'name': name.upper(),
                'config': load_config(target_path),
                'config_path': target_path,
            })
        report = run_cross_condition(
            config, config_path, targets, logger,
            results_dir=results_dir, architecture=architecture,
        )
        print("\n=== C-MAPSS cross-condition ===")
        print(
            f"source={report['protocol']['source_dataset']} "
            f"targets={report['protocol']['target_datasets']} "
            f"threshold={report['protocol']['threshold']:.4f}"
        )
        for row in report['rows']:
            print(
                f"  {row['target_dataset']:<6} f1={float(row['f1'] or 0):.4f} "
                f"recall={float(row['recall'] or 0):.4f} "
                f"precision={float(row['precision'] or 0):.4f}"
            )
        print(f"  report: {report['artifacts']['report_path']}")
        logger.info("--- ANSR-DT cross-condition run finished ---")
        return 0

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
    args = build_parser().parse_args()
    sys.exit(main(
        clear=args.clear_results,
        compare_models=args.compare_models,
        architecture=args.architecture,
        multi_seed=args.multi_seed,
        seeds=args.seeds,
        threshold_sensitivity=args.threshold_sensitivity,
        operational_policy=args.operational_policy,
        benchmark=args.benchmark,
        cmapss_dataset=args.cmapss_dataset,
        cross_condition=args.cross_condition,
        cross_train_dataset=args.cross_train_dataset,
        cross_test_datasets=args.cross_test_datasets,
        cross_normalization=args.cross_normalization,
        cross_threshold=args.cross_threshold,
        robustness_report=args.robustness_report,
        representation_ablation=args.representation_ablation,
        few_shot_target=args.few_shot_target,
        modes=args.modes,
        data_path=args.data_path,
        model_type=args.model_type,
        epochs=args.epochs,
        symbolic=args.symbolic,
        min_support=args.min_support,
        max_rules=args.max_rules,
        min_confidence=args.min_confidence,
    ))
