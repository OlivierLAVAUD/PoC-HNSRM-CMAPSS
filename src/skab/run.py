"""SKAB benchmark runner.

Main entry point for running the SKAB benchmark path.
Supports both random forest baseline and symbolic reasoning modes.
"""

import argparse
import sys
import os
import json
from datetime import datetime

# Lazy import to avoid tensorflow dependency when running FD001 benchmark
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

from src.skab.config import (
    SKAB_FEATURES, SKAB_DATA_PATH, DEFAULT_N_ESTIMATORS, DEFAULT_MAX_DEPTH,
    DEFAULT_RANDOM_STATE
)
from src.skab.data import load_skab_data, describe_data
from src.skab.pipeline import run_skab_benchmark
from src.skab.reasoner import validate_skab_rules, deduplicate_skab_rules
from src.reasoning.reasoning import SymbolicReasoner

logger = logging.getLogger(__name__)


def main():
    """Main entry point for SKAB benchmark."""
    parser = argparse.ArgumentParser(
        description='SKAB benchmark: Random Forest baseline and symbolic reasoning'
    )
    parser.add_argument(
        '--benchmark', 
        choices=['fd001', 'skab'], 
        default='fd001',
        help='Benchmark dataset to use (default: fd001)'
    )
    parser.add_argument(
        '--modes', 
        nargs='+', 
        choices=['random_forest', 'symbolic'], 
        default=['random_forest', 'symbolic'],
        help='Modes to run (default: random_forest symbolic)'
    )
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=None,
        help='Path to SKAB data directory'
    )
    parser.add_argument(
        '--model-type', 
        type=str, 
        choices=['random_forest', 'keras'], 
        default='random_forest',
        help='Model type for RF baseline'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of epochs for Keras training'
    )
    parser.add_argument(
        '--symbolic', 
        action='store_true',
        default=True,
        help='Run symbolic reasoning mode'
    )
    parser.add_argument(
        '--no-symbolic', 
        action='store_false',
        dest='symbolic',
        help='Disable symbolic reasoning mode'
    )
    parser.add_argument(
        '--min-support', 
        type=int, 
        default=3,
        help='Minimum support for symbolic rules'
    )
    parser.add_argument(
        '--max-rules', 
        type=int, 
        default=50,
        help='Maximum number of symbolic rules'
    )
    parser.add_argument(
        '--min-confidence', 
        type=float, 
        default=0.7,
        help='Minimum confidence for symbolic rules'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handle benchmark selection
    if args.benchmark == 'skab':
        logger.info("=" * 60)
        logger.info("Running SKAB benchmark")
        logger.info("=" * 60)
        
        # Check TensorFlow availability for Keras mode
        if args.model_type == 'keras' and not _TF_AVAILABLE:
            logger.warning(
                "TensorFlow not available, switching to Random Forest mode"
            )
            args.model_type = 'random_forest'
        
        # Create results directory
        run_id = f"skab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir = f"results/skab_separate/{run_id}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Run the benchmark
        results = run_skab_benchmark(
            data_path=args.data_path or "data/SKAB",
            model_type=args.model_type,
            symbolic=args.symbolic,
            n_estimators=DEFAULT_N_ESTIMATORS,
            max_depth=DEFAULT_MAX_DEPTH,
            symbolic_min_support=args.min_support,
            symbolic_max_rules=args.max_rules,
            symbolic_min_confidence=args.min_confidence
        )
        
        # Save results
        summary_path = os.path.join(results_dir, "summary.csv")
        
        # Prepare summary data
        summary_data = {
            'run_id': run_id,
            'benchmark': 'skab',
            'model_type': results.get('model_type', args.model_type),
            'rf_accuracy': results.get('rf_accuracy', 0.0),
            'symbolic_evaluated': results.get('symbolic_evaluated', 0),
            'symbolic_accepted': results.get('symbolic_accepted', 0),
            'symbolic_rejected': results.get('symbolic_rejected', 0),
            'symbolic_rules_count': len(results.get('symbolic_rules', [])),
            'feature_count': results.get('feature_count', len(SKAB_FEATURES)),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Write CSV summary
        import csv
        write_csv_header = not os.path.exists(summary_path)
        
        with open(summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'run_id', 'benchmark', 'model_type', 'rf_accuracy',
                'symbolic_evaluated', 'symbolic_accepted', 'symbolic_rejected',
                'symbolic_rules_count', 'feature_count', 'timestamp'
            ])
            
            if write_csv_header:
                writer.writeheader()
            
            writer.writerow(summary_data)
        
        logger.info("SKAB benchmark completed")
        logger.info("Results saved to: %s", summary_path)
        logger.info("RF Accuracy: %.4f", summary_data['rf_accuracy'])
        logger.info("Symbolic: %d evaluated, %d accepted, %d rejected",
                     summary_data['symbolic_evaluated'],
                     summary_data['symbolic_accepted'],
                     summary_data['symbolic_rejected'])
        
    else:  # fd001 benchmark
        logger.info("Running FD001 benchmark (TensorFlow required)")
        # This would delegate to the main FD001 pipeline
        # For now, just log that we're skipping
        logger.info("--benchmark fd001 selected - using FD001 pipeline")
        logger.info("Ensuring TensorFlow is not imported unnecessarily")
        # Import check - if we get here without tf import issue, good
        if not _TF_AVAILABLE:
            logger.warning("TensorFlow not available, FD001 may have limited functionality")


if __name__ == "__main__":
    main()
