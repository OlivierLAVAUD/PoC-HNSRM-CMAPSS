"""SKAB utility functions.

Miscellaneous utilities for the SKAB benchmark path.
"""

import logging
import numpy as np
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    logger.debug("Random seed set to %d", seed)


def count_parameters(model) -> int:
    """Count trainable parameters in a Keras model."""
    try:
        import tensorflow as tf
        return int(model.count_params())
    except Exception:
        # For sklearn models, count differently
        return 0


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def check_tf_available() -> bool:
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False


def validate_file_path(filepath: str, must_exist: bool = True) -> bool:
    """Validate that a file path is valid (optionally must exist)."""
    if not filepath:
        return False
    
    if must_exist and not os.path.exists(filepath):
        logger.error("File not found: %s", filepath)
        return False
    
    return True


def extract_numeric_from_rule(rule_string: str) -> Optional[float]:
    """Extract numeric threshold from a Prolog rule string.
    
    Args:
        rule_string: Prolog rule like "neural_rule_1 :- feature_threshold(s4, _, 1426.015)."
    
    Returns:
        Numeric threshold value, or None if not found
    """
    import re
    match = re.search(r'([0-9]*\.?[0-9]+)', rule_string)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None
