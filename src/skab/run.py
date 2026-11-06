"""SKAB benchmark runner.

The FD001 entry point (``main.py``) delegates ``--benchmark skab`` here so the
SKAB argument handling exists in one place only. The parser helper is reused
by ``main.py`` to expose the same options on both entry points.
"""

import argparse
import csv
import importlib.util
import logging
import os
from datetime import datetime

from src.skab.config import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    SKAB_DATA_PATH,
    SKAB_FEATURES,
)
from src.skab.pipeline import run_skab_benchmark

logger = logging.getLogger(__name__)

_CSV_FIELDS = [
    'run_id', 'benchmark', 'model_type', 'rf_accuracy',
    'symbolic_evaluated', 'symbolic_accepted', 'symbolic_rejected',
    'symbolic_rules_count', 'feature_count', 'timestamp',
]


def _tensorflow_available() -> bool:
    return importlib.util.find_spec('tensorflow') is not None


def add_skab_arguments(parser: argparse.ArgumentParser
                       ) -> argparse.ArgumentParser:
    """Add the SKAB-specific options to an existing parser."""
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['random_forest', 'symbolic'],
        default=['random_forest', 'symbolic'],
        help='Modes to run (default: random_forest symbolic)',
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to the SKAB data directory',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['random_forest', 'keras'],
        default='random_forest',
        help='Model type for the RF baseline',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs for Keras training',
    )
    parser.add_argument(
        '--symbolic',
        action='store_true',
        default=True,
        help='Run symbolic reasoning mode',
    )
    parser.add_argument(
        '--no-symbolic',
        action='store_false',
        dest='symbolic',
        help='Disable symbolic reasoning mode',
    )
    parser.add_argument(
        '--min-support',
        type=int,
        default=3,
        help='Minimum support for symbolic rules',
    )
    parser.add_argument(
        '--max-rules',
        type=int,
        default=50,
        help='Maximum number of symbolic rules',
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.7,
        help='Minimum confidence for symbolic rules',
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone SKAB parser."""
    parser = argparse.ArgumentParser(
        description='SKAB benchmark: Random Forest baseline and symbolic reasoning'
    )
    parser.add_argument(
        '--benchmark',
        choices=['fd001', 'skab'],
        default='fd001',
        help='Benchmark dataset to use (default: fd001)',
    )
    return add_skab_arguments(parser)


def run_skab_from_args(args: argparse.Namespace,
                       project_root: str = None) -> int:
    """Execute the SKAB benchmark from a parsed argument namespace."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    root = os.path.abspath(project_root or os.getcwd())
    modes = list(getattr(args, 'modes', None) or ['random_forest', 'symbolic'])
    symbolic = 'symbolic' in modes and bool(getattr(args, 'symbolic', True))
    model_type = str(getattr(args, 'model_type', 'random_forest'))
    if model_type == 'keras' and not _tensorflow_available():
        logger.warning(
            'TensorFlow not available, switching to Random Forest mode'
        )
        model_type = 'random_forest'

    run_id = f"skab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.path.join(root, 'results', 'skab_separate', run_id)
    os.makedirs(results_dir, exist_ok=True)

    logger.info('Running SKAB benchmark (modes=%s)', modes)
    results = run_skab_benchmark(
        data_path=getattr(args, 'data_path', None) or os.path.join(
            root, SKAB_DATA_PATH
        ),
        model_type=model_type,
        symbolic=symbolic,
        epochs=getattr(args, 'epochs', 50),
        n_estimators=DEFAULT_N_ESTIMATORS,
        max_depth=DEFAULT_MAX_DEPTH,
        symbolic_min_support=getattr(args, 'min_support', 3),
        symbolic_max_rules=getattr(args, 'max_rules', 50),
        symbolic_min_confidence=getattr(args, 'min_confidence', 0.7),
    )

    summary_data = {
        'run_id': run_id,
        'benchmark': 'skab',
        'model_type': results.get('model_type', model_type),
        'rf_accuracy': results.get('rf_accuracy', 0.0),
        'symbolic_evaluated': results.get('symbolic_evaluated', 0),
        'symbolic_accepted': results.get('symbolic_accepted', 0),
        'symbolic_rejected': results.get('symbolic_rejected', 0),
        'symbolic_rules_count': len(results.get('symbolic_rules', [])),
        'feature_count': results.get('feature_count', len(SKAB_FEATURES)),
        'timestamp': datetime.now().isoformat(),
    }
    summary_path = os.path.join(results_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(summary_data)

    logger.info('SKAB benchmark completed')
    logger.info('Results saved to: %s', summary_path)
    logger.info('RF Accuracy: %.4f', summary_data['rf_accuracy'])
    logger.info(
        'Symbolic: %d evaluated, %d accepted, %d rejected',
        summary_data['symbolic_evaluated'],
        summary_data['symbolic_accepted'],
        summary_data['symbolic_rejected'],
    )
    return 0


def main(argv=None) -> int:
    """Standalone entry point for ``python -m src.skab.run``."""
    args = build_parser().parse_args(argv)
    if args.benchmark == 'skab':
        return run_skab_from_args(args)
    logger.info(
        '--benchmark fd001 selected; use `python main.py` for the FD001 pipeline.'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
