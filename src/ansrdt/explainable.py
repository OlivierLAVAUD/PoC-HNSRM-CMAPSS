# src/ansrdt/explainable.py

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .core import ANSRDTCore


class ExplainableANSRDT(ANSRDTCore):
    """ANSRDT core with human-readable decision explanations."""

    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None,
                 classifier: Optional[Any] = None):
        """Initialize the explainable core."""
        super().__init__(config_path, logger, classifier)
        self.decision_history: List[Dict[str, Any]] = []

    def explain_decision(self, decision: Dict[str, Any]) -> str:
        """Build a short explanation from neural confidence and symbolic insights."""
        try:
            confidence = decision.get('failure_probability', decision.get('confidence', 0.0))
            parts = [f"Failure probability: {confidence:.2%}"]
            insights = decision.get('insights') or []
            if insights:
                parts.append("Insights: " + "; ".join(insights))
            else:
                parts.append("No symbolic rule activated.")
            return " | ".join(parts)
        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return "Error generating explanation"

    def track_decision(self, decision: Dict[str, Any]) -> None:
        """Record a decision and its explanation, bounded to 1000 entries."""
        self.decision_history.append({
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'explanation': self.explain_decision(decision),
        })
        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

    def save_decision_history(self, output_path: str) -> None:
        """Persist decision history to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({'decisions': self.decision_history}, f, indent=2, default=str)
