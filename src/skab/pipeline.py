"""SKAB pipeline module.

Orchestrates the SKAB benchmark: data loading, model training,
and symbolic rule extraction.
"""

import logging
import numpy as np
from src.skab.config import (
    SKAB_FEATURES, DEFAULT_N_ESTIMATORS, DEFAULT_MAX_DEPTH,
    DEFAULT_RANDOM_STATE, SYMBOLIC_MIN_SUPPORT, SYMBOLIC_MAX_RULES,
    SYMBOLIC_MIN_CONFIDENCE
)
from src.skab.data import load_skab_data, describe_data
from src.skab.model import create_skab_model, SKABClassifier, RandomForestModel
from src.reasoning.reasoning import SymbolicReasoner

logger = logging.getLogger(__name__)


def run_skab_benchmark(
    data_path: str = None,
    model_type: str = "random_forest",
    symbolic: bool = True,
    n_estimators: int = None,
    max_depth: int = None,
    symbolic_min_support: int = None,
    symbolic_max_rules: int = None,
    symbolic_min_confidence: float = None
) -> dict:
    """Run the full SKAB benchmark.
    
    Args:
        data_path: Path to SKAB data directory
        model_type: "random_forest" or "keras"
        symbolic: Whether to run symbolic rule extraction
        n_estimators: Number of trees for RF (uses default if None)
        max_depth: Max depth for RF (uses default if None)
        symbolic_min_support: Min support for symbolic rules
        symbolic_max_rules: Max rules for symbolic extraction
        symbolic_min_confidence: Min confidence for symbolic rules
    
    Returns:
        Dictionary with benchmark results
    """
    # Load data
    logger.info("Loading SKAB data from: %s", data_path or config.SKAB_DATA_PATH)
    X, y, feature_names = load_skab_data(data_path)
    
    logger.info("SKAB dataset: %d samples, %d features", X.shape[0], X.shape[1])
    
    # Describe data
    desc = describe_data(X, y, feature_names)
    logger.info("Dataset description: %s", desc)
    
    # Initialize models
    n_est = n_estimators or DEFAULT_N_ESTIMATORS
    depth = max_depth or DEFAULT_MAX_DEPTH
    
    rf_model = create_skab_model("random_forest", n_estimators=n_est, max_depth=depth)
    rf_model.fit(X, y)
    
    rf_predictions = rf_model.predict(X)
    rf_accuracy = np.mean(rf_predictions == y)
    logger.info("Random Forest accuracy: %.4f", rf_accuracy)
    
    # Initialize symbolic reasoner if requested
    results = {
        'model_type': model_type,
        'rf_accuracy': rf_accuracy,
        'symbolic_rules': [],
        'symbolic_evaluated': 0,
        'symbolic_accepted': 0,
        'symbolic_rejected': 0,
    }
    
    if symbolic:
        sym_min_support = symbolic_min_support or SYMBOLIC_MIN_SUPPORT
        sym_max_rules = symbolic_max_rules or SYMBOLIC_MAX_RULES
        sym_min_conf = symbolic_min_confidence or SYMBOLIC_MIN_CONFIDENCE
        
        # Extract rules from the RF model using symbolic reasoning
        # This is a simplified version - real implementation would
        # extract if-then rules from decision trees
        try:
            reasoner = SymbolicReasoner(
                rules_path="src/reasoning/rules.pl",
                input_shape=(1, X.shape[1],),
            )
            
            # Create synthetic rule data based on feature thresholds
            # In a real implementation, this would use actual tree extraction
            potential_rules = []
            
            for i, feat_name in enumerate(feature_names[:8]):  # Use first 8 features
                # Create a simple rule based on feature threshold
                rule_body = f"feature_threshold({feat_name}, _, high)"
                rule_string = f"neural_rule_{i+1} :- {rule_body}."
                confidence = 0.8  # placeholder
                potential_rules.append((rule_string, confidence))
            
            # Update rules (this will trigger dedup and validation)
            if potential_rules:
                reasoner.update_rules(
                    potential_rules,
                    min_confidence=sym_min_conf,
                    max_learned_rules=sym_max_rules,
                    pruning_strategy='confidence',
                )
                
                stats = reasoner.get_rule_statistics()
                results['symbolic_evaluated'] = stats.get('total_rules_evaluated', 0)
                results['symbolic_accepted'] = len(reasoner.learned_rules)
                results['symbolic_rejected'] = max(0, len(potential_rules) - len(reasoner.learned_rules))
                results['symbolic_rules'] = list(reasoner.learned_rules.keys())
                
                logger.info("Symbolic reasoning: %d rules evaluated, %d accepted, %d rejected",
                           results['symbolic_evaluated'], results['symbolic_accepted'], results['symbolic_rejected'])
                
        except Exception as e:
            logger.error("Symbolic reasoning failed: %s", e, exc_info=True)
            results['symbolic_error'] = str(e)
    
    return results
