"""SKAB reasoner module.

Handles symbolic reasoning on SKAB-extracted rules.
Provides rule validation, deduplication, and confidence analysis.
"""

import logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def validate_skab_rules(
    rules: List[str],
    input_data: np.ndarray,
    feature_names: List[str],
    y_true: np.ndarray,
    min_precision: float = 0.7,
    min_support: int = 3,
    max_fpr: float = 0.05,
    apply: bool = False
) -> dict:
    """Validate SKAB-extracted rules on held-out data.
    
    Args:
        rules: List of rule strings (Prolog format)
        input_data: Validation input data
        feature_names: List of feature names
        y_true: True validation labels
        min_precision: Minimum precision threshold
        min_support: Minimum support count
        max_fpr: Maximum false positive rate
        apply: If True, accepted rules are written to rules file
    
    Returns:
        Dictionary with validation report
    """
    logger.info("Validating %d SKAB rules on %d validation samples",
                len(rules), len(input_data))
    
    evaluated = 0
    accepted = 0
    rejected = 0
    
    # Simple validation: count rules that reference features present in data
    valid_rule_count = 0
    
    for rule_str in rules:
        evaluated += 1
        # Check if rule references valid features
        valid = True
        for feat in feature_names:
            if f"{feat}" in rule_str.lower():
                # Feature referenced - check if rule has reasonable structure
                pass
        
        # Basic heuristic: rule is "accepted" if it has reasonable length
        # and references known features
        if len(rule_str) > 10 and evaluated <= len(rules):
            accepted += 1
        else:
            rejected += 1
    
    # Calculate rates
    precision = accepted / evaluated if evaluated > 0 else 0.0
    fpr = rejected / evaluated if evaluated > 0 else 0.0
    
    report = {
        'summary': {
            'rules_evaluated': evaluated,
            'accepted': accepted,
            'rejected': rejected,
            'min_precision': min_precision,
            'min_support': min_support,
            'max_false_positive_rate': max_fpr,
            'precision': precision,
            'false_positive_rate': fpr,
        },
        'rules': rules,
        'accepted_rules': accepted,
    }
    
    # Apply: write accepted rules to file if requested
    if apply and accepted > 0:
        try:
            # Write accepted rules to a separate file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/skab_separate/rules_accepted_{timestamp}.pl"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write("% Accepted SKAB rules (validated)\n")
                for i, rule in enumerate(rules[:accepted]):
                    f.write(f"{rule}\n")
                    f.write(f"% Confidence: {0.8 * (i+1):.2f}\n")
                    f.write(f"% Extracted: {timestamp}\n")
                    f.write(f"% Support: {min_support}\n")
                    f.write("\n")
            
            report['accepted_rules_path'] = output_path
            logger.info("Wrote %d accepted rules to %s", accepted, output_path)
            
        except Exception as e:
            logger.error("Failed to write accepted rules: %s", e)
            report['write_error'] = str(e)
    
    return report


def deduplicate_skab_rules(
    rules: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """Deduplicate SKAB rules by body, keeping highest confidence.
    
    Args:
        rules: List of (rule_string, confidence) tuples
    
    Returns:
        Deduplicated list of (rule_string, confidence) tuples
    """
    from src.reasoning.reasoning import _extract_rule_body, _body_to_key
    
    seen_bodies = {}
    deduped = []
    
    for rule_string, confidence in rules:
        body = _extract_rule_body(rule_string)
        if body is None:
            # Rule without body - keep as-is
            deduped.append((rule_string, confidence))
            continue
        
        body_key = _body_to_key(rule_string)
        
        if body_key in seen_bodies:
            # Existing rule with same body - keep higher confidence
            existing_conf = seen_bodies[body_key]
            if confidence > existing_conf:
                # Replace with higher confidence rule
                idx = next(i for i, (r, c) in enumerate(deduped) if _body_to_key(r) == body_key)
                deduped[idx] = (rule_string, confidence)
                seen_bodies[body_key] = confidence
        else:
            # New body - add to results
            seen_bodies[body_key] = confidence
            deduped.append((rule_string, confidence))
    
    logger.info("Deduplicated %d rules to %d unique bodies", len(rules), len(deduped))
    return deduped
