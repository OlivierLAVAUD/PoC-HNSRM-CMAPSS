"""Human-in-the-loop queue for uncertain or conflicting FD001 cases."""

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


RULE_ALARM_INSIGHTS = {
    'HPC outlet temperature elevated',
    'HPC outlet temperature critical',
    'HPC static pressure high',
    'HPC outlet pressure low',
    'Fuel flow low',
    'Coolant bleed low',
    'HPC degradation pattern',
    'HPC critical pattern',
    'HPC thermal overload',
    'HPC fuel starvation',
    'HPC temperature rising trend',
    'HPC outlet pressure falling trend',
    'HPC static pressure persistent high',
    'HPC thermal-pressure degradation',
    'HPC temperature variability high',
    'HPC moving-average temperature high',
}

REASON_PRIORITY = {
    'rule_probability_conflict_high': 0,
    'rule_probability_conflict_low': 1,
    'problog_probability_conflict_low': 2,
    'uncertain_probability': 3,
}


@dataclass
class HITLCase:
    """One test window that needs a human decision."""

    case_id: str
    window_index: int
    neural_probability: float
    problog_failure_risk: float
    reason: str
    insights: List[str]
    sensor_values: Dict[str, float]
    unit: Optional[int] = None
    cycle: Optional[int] = None
    status: str = 'pending'
    human_label: Optional[str] = None
    reviewer: str = ''
    comment: str = ''
    rule_added: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    reviewed_at: Optional[str] = None


class HITLQueue:
    """In-memory queue with CSV/JSON export and human feedback updates."""

    def __init__(self, uncertain_low: float = 0.4, uncertain_high: float = 0.6,
                 output_dir: str = 'results', max_cases: int = 200,
                 decision_threshold: Optional[float] = None):
        """Configure uncertainty bounds and persistence paths."""
        self.uncertain_low = float(uncertain_low)
        self.uncertain_high = float(uncertain_high)
        self.output_dir = output_dir
        self.max_cases = int(max_cases)
        self.decision_threshold = (
            None if decision_threshold is None else float(decision_threshold)
        )
        self.cases: List[HITLCase] = []
        self._next_id = 1
        self.csv_path = os.path.join(output_dir, 'hitl_queue.csv')
        self.json_path = os.path.join(output_dir, 'hitl_queue.json')
        os.makedirs(output_dir, exist_ok=True)

    @classmethod
    def classify_reason(cls, neural_probability: float, insights: List[str],
                        problog_failure_risk: float,
                        uncertain_low: float = 0.4,
                        uncertain_high: float = 0.6) -> Optional[str]:
        """Return the queue reason, or None when the case is not uncertain."""
        rule_alarm = any(
            insight in RULE_ALARM_INSIGHTS
            or insight.startswith('Learned Rule Activated: hitl_rule_')
            for insight in insights
        )
        problog_alarm = problog_failure_risk >= 0.5

        if uncertain_low <= neural_probability <= uncertain_high:
            return 'uncertain_probability'
        if neural_probability > uncertain_high and not rule_alarm:
            return 'rule_probability_conflict_high'
        if neural_probability < uncertain_low and rule_alarm:
            return 'rule_probability_conflict_low'
        if neural_probability < uncertain_low and problog_alarm:
            return 'problog_probability_conflict_low'
        return None

    def _new_case_id(self) -> str:
        """Generate a stable sequential case identifier."""
        case_id = f"case_{self._next_id:04d}"
        self._next_id += 1
        return case_id

    def _is_duplicate(self, window_index: int, unit: Optional[int],
                      cycle: Optional[int]) -> bool:
        """Avoid enqueuing the same window twice."""
        for case in self.cases:
            if (case.window_index == window_index and case.unit == unit
                    and case.cycle == cycle):
                return True
        return False

    def add_case(self, window_index: int, neural_probability: float,
                 problog_failure_risk: float, reason: str, insights: List[str],
                 sensor_values: Dict[str, float], unit: Optional[int] = None,
                 cycle: Optional[int] = None) -> Optional[HITLCase]:
        """Append one case when capacity and deduplication allow it."""
        if len(self.cases) >= self.max_cases:
            return None
        if self._is_duplicate(window_index, unit, cycle):
            return None
        case = HITLCase(
            case_id=self._new_case_id(),
            window_index=int(window_index),
            neural_probability=float(neural_probability),
            problog_failure_risk=float(problog_failure_risk),
            reason=reason,
            insights=list(insights),
            sensor_values={k: float(v) for k, v in sensor_values.items()},
            unit=None if unit is None else int(unit),
            cycle=None if cycle is None else int(cycle),
        )
        self.cases.append(case)
        return case

    def add_from_prediction(self, window_index: int, neural_probability: float,
                            problog_failure_risk: float, insights: List[str],
                            sensor_values: Dict[str, float],
                            unit: Optional[int] = None,
                            cycle: Optional[int] = None) -> Optional[HITLCase]:
        """Classify and append a prediction when it needs review."""
        reason = self.classify_reason(
            neural_probability, insights, problog_failure_risk,
            self.uncertain_low, self.uncertain_high,
        )
        if reason is None:
            return None
        return self.add_case(
            window_index, neural_probability, problog_failure_risk, reason,
            insights, sensor_values, unit, cycle,
        )

    def pending(self) -> List[HITLCase]:
        """Return cases that still need a human label."""
        return [case for case in self.cases if case.status == 'pending']

    def get(self, case_id: str) -> Optional[HITLCase]:
        """Look up a case by identifier."""
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None

    def update_feedback(self, case_id: str, label: str, reviewer: str = '',
                        comment: str = '', rule_added: Optional[str] = None) -> HITLCase:
        """Mark a case reviewed and store the human label."""
        case = self.get(case_id)
        if case is None:
            raise KeyError(f"Unknown case_id: {case_id}")
        case.status = 'reviewed'
        case.human_label = label
        case.reviewer = reviewer
        case.comment = comment
        case.rule_added = rule_added
        case.reviewed_at = datetime.now().isoformat()
        return case

    def export_csv(self, path: Optional[str] = None) -> str:
        """Write all cases to a flat CSV file."""
        target = path or self.csv_path
        directory = os.path.dirname(target)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fields = [
            'case_id', 'window_index', 'unit', 'cycle', 'neural_probability',
            'problog_failure_risk', 'reason', 'status', 'human_label',
            'reviewer', 'comment', 'rule_added', 'insights', 'sensor_values',
            'created_at', 'reviewed_at',
        ]
        with open(target, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for case in self.cases:
                row = asdict(case)
                row['insights'] = '|'.join(case.insights)
                row['sensor_values'] = json.dumps(case.sensor_values)
                writer.writerow({key: row.get(key) for key in fields})
        return target

    def save_json(self, path: Optional[str] = None) -> str:
        """Persist the queue for an interactive review session."""
        target = path or self.json_path
        directory = os.path.dirname(target)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            'metadata': {
                'uncertain_low': self.uncertain_low,
                'uncertain_high': self.uncertain_high,
                'decision_threshold': self.decision_threshold,
                'max_cases': self.max_cases,
                'exported_at': datetime.now().isoformat(),
            },
            'cases': [asdict(case) for case in self.cases],
        }
        with open(target, 'w') as file:
            json.dump(payload, file, indent=2)
        return target

    @classmethod
    def load_json(cls, path: str) -> "HITLQueue":
        """Load a queue saved by :meth:`save_json`."""
        with open(path, 'r') as file:
            payload = json.load(file)
        metadata = payload.get('metadata', {})
        queue = cls(
            uncertain_low=metadata.get('uncertain_low', 0.4),
            uncertain_high=metadata.get('uncertain_high', 0.6),
            output_dir=os.path.dirname(os.path.abspath(path)),
            max_cases=metadata.get('max_cases', 200),
            decision_threshold=metadata.get('decision_threshold'),
        )
        for case_data in payload.get('cases', []):
            queue.cases.append(HITLCase(**case_data))
        if queue.cases:
            queue._next_id = max(
                int(case.case_id.split('_')[-1]) for case in queue.cases
            ) + 1
        return queue

    def summary(self) -> Dict[str, Any]:
        """Return counts by status and by queue reason."""
        reasons: Dict[str, int] = {}
        for case in self.cases:
            reasons[case.reason] = reasons.get(case.reason, 0) + 1
        return {
            'total_cases': len(self.cases),
            'pending': len(self.pending()),
            'reviewed': len(self.cases) - len(self.pending()),
            'reasons': reasons,
            'csv_path': self.csv_path,
            'json_path': self.json_path,
        }
