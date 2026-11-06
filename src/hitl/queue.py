"""Human-in-the-loop queue for uncertain or conflicting FD001 cases.

Three review-ordering strategies are available:

- ``uncertainty``: the existing neural/ProbLog/rule-conflict risk ordering;
- ``expected_cost``: rank cases by the maintenance cost exposure of leaving
  them unreviewed, using the failure probability, ProbLog risk, predicted RUL,
  optional RUL uncertainty and the drift level;
- ``hybrid``: blend the uncertainty risk score with the expected-cost score.

Prioritization only changes review order. It never changes the model decision,
the alert, or the learned-rule validation.
"""

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


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

PRIORITIZATION_STRATEGIES = ('uncertainty', 'expected_cost', 'hybrid')

DRIFT_FACTORS = {'ok': 1.0, 'warning': 1.1, 'critical': 1.25}

PRIORITY_ORDER = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}


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
    risk_score: float = 0.0
    risk_band: str = 'low'
    priority_reason: str = ''
    predicted_rul: Optional[float] = None
    rul_uncertainty: Optional[float] = None
    drift_level: Optional[str] = None
    expected_cost: float = 0.0
    inspection_cost: float = 0.0
    priority_score: float = 0.0
    hitl_priority: str = 'low'


class HITLQueue:
    """In-memory queue with CSV/JSON export and human feedback updates."""

    def __init__(self, uncertain_low: float = 0.4, uncertain_high: float = 0.6,
                 output_dir: str = 'results', max_cases: int = 200,
                 decision_threshold: Optional[float] = None,
                 prioritization_enabled: bool = False,
                 risk_thresholds: Optional[Dict[str, float]] = None,
                 strategy: str = 'uncertainty',
                 false_negative_cost: float = 10.0,
                 false_alarm_cost: float = 1.0,
                 rul_threshold: float = 30.0):
        """Configure uncertainty bounds, prioritization and persistence."""
        self.uncertain_low = float(uncertain_low)
        self.uncertain_high = float(uncertain_high)
        self.output_dir = output_dir
        self.max_cases = int(max_cases)
        self.decision_threshold = (
            None if decision_threshold is None else float(decision_threshold)
        )
        self.prioritization_enabled = bool(prioritization_enabled)
        name = str(strategy or 'uncertainty').lower()
        if name not in PRIORITIZATION_STRATEGIES:
            raise ValueError(
                f"hitl.prioritization.strategy must be one of "
                f"{PRIORITIZATION_STRATEGIES}, got {strategy!r}."
            )
        self.strategy = name
        self.false_negative_cost = float(false_negative_cost)
        self.false_alarm_cost = float(false_alarm_cost)
        if self.false_negative_cost < 0 or self.false_alarm_cost < 0:
            raise ValueError(
                "hitl.prioritization costs must be non-negative."
            )
        if self.false_negative_cost == 0 and self.false_alarm_cost == 0:
            raise ValueError(
                "hitl.prioritization costs cannot both be zero."
            )
        if not np.isfinite(float(rul_threshold)) or float(rul_threshold) <= 0:
            raise ValueError("hitl.prioritization.rul_threshold must be > 0.")
        self.rul_threshold = float(rul_threshold)
        thresholds = risk_thresholds or {}
        self.risk_thresholds = {
            'medium': float(thresholds.get('medium', 0.4)),
            'high': float(thresholds.get('high', 0.6)),
            'critical': float(thresholds.get('critical', 0.8)),
        }
        self.cases: List[HITLCase] = []
        self._next_id = 1
        self.csv_path = os.path.join(output_dir, 'hitl_queue.csv')
        self.json_path = os.path.join(output_dir, 'hitl_queue.json')
        os.makedirs(output_dir, exist_ok=True)

    @classmethod
    def compute_risk_score(cls, neural_probability: float,
                           problog_failure_risk: float,
                           reason: str = '',
                           decision_threshold: Optional[float] = None,
                           risk_weights: Optional[Dict[str, float]] = None
                           ) -> Dict[str, Any]:
        """Return a transparent HITL risk score and its dominant driver.

        The score is the maximum of the neural probability, the ProbLog risk
        and the disagreement exposed by the queue reason. A low-probability
        rule conflict is therefore ranked by the rule's confidence proxy
        (``1 - neural_probability``) instead of being buried at the bottom.
        """
        weights = risk_weights or {}
        neural_weight = float(weights.get('neural', 1.0))
        problog_weight = float(weights.get('problog', 1.0))
        conflict_weight = float(weights.get('conflict', 1.0))
        neural = float(np.clip(float(neural_probability), 0.0, 1.0))
        problog = float(np.clip(float(problog_failure_risk), 0.0, 1.0))
        reason_name = str(reason or '')
        components = {
            'neural_probability': neural_weight * neural,
            'problog_failure_risk': problog_weight * problog,
            'rule_conflict': 0.0,
        }
        if reason_name in {
                'rule_probability_conflict_low',
                'problog_probability_conflict_low'}:
            components['rule_conflict'] = conflict_weight * (1.0 - neural)
        elif reason_name == 'rule_probability_conflict_high':
            components['rule_conflict'] = conflict_weight * neural
        return {
            'score': float(np.clip(max(components.values()), 0.0, 1.0)),
            'components': components,
            'driver': max(components, key=components.get),
        }

    @staticmethod
    def risk_band_for_score(score: float,
                            thresholds: Optional[Dict[str, float]] = None
                            ) -> str:
        """Map a risk score to a readable operational band."""
        values = thresholds or {}
        if float(score) >= float(values.get('critical', 0.8)):
            return 'critical'
        if float(score) >= float(values.get('high', 0.6)):
            return 'high'
        if float(score) >= float(values.get('medium', 0.4)):
            return 'medium'
        return 'low'

    @classmethod
    def evaluate_priority(
            cls, neural_probability: float, problog_failure_risk: float,
            reason: str = '', predicted_rul: Optional[float] = None,
            rul_uncertainty: Optional[float] = None,
            drift_level: Optional[str] = None,
            strategy: str = 'expected_cost',
            risk_weights: Optional[Dict[str, float]] = None,
            risk_thresholds: Optional[Dict[str, float]] = None,
            false_negative_cost: float = 10.0,
            false_alarm_cost: float = 1.0,
            rul_threshold: float = 30.0) -> Dict[str, Any]:
        """Score one review case with the configured prioritization strategy.

        ``expected_cost`` estimates the loss exposure of leaving the window
        unreviewed: failure probability x RUL imminence x drift factor x
        false-negative cost. RUL uncertainty and the inspection cost only
        affect the reported trade-off, not the model decision.
        """
        name = str(strategy or 'uncertainty').lower()
        if name not in PRIORITIZATION_STRATEGIES:
            raise ValueError(
                f"Unsupported HITL prioritization strategy: {strategy!r}."
            )
        risk = cls.compute_risk_score(
            neural_probability, problog_failure_risk, reason,
            None, risk_weights,
        )
        neural = float(np.clip(float(neural_probability), 0.0, 1.0))
        problog = float(np.clip(float(problog_failure_risk), 0.0, 1.0))
        conflict = float(risk['components'].get('rule_conflict', 0.0))
        failure_probability = float(np.clip(
            max(neural, problog, conflict), 0.0, 1.0
        ))
        imminence = 0.5
        if predicted_rul is not None and np.isfinite(float(predicted_rul)):
            imminence = float(np.clip(
                1.0 - float(predicted_rul) / max(float(rul_threshold), 1.0),
                0.0, 1.0,
            ))
        uncertainty_factor = 0.0
        if rul_uncertainty is not None and np.isfinite(float(rul_uncertainty)):
            uncertainty_factor = float(np.clip(
                float(rul_uncertainty) / max(float(rul_threshold), 1.0),
                0.0, 1.0,
            ))
        drift_factor = float(DRIFT_FACTORS.get(
            str(drift_level or '').lower(), 1.0
        ))

        missed_cost = (
            failure_probability * (1.0 + imminence)
            * drift_factor * float(false_negative_cost)
        )
        inspection_cost = float(false_alarm_cost) * (
            1.0 + 0.5 * uncertainty_factor
        )
        expected_cost = max(0.0, missed_cost - inspection_cost)
        scale = max(
            float(false_negative_cost) * 2.25,
            float(false_alarm_cost),
            1e-9,
        )
        cost_score = float(np.clip(expected_cost / scale, 0.0, 1.0))

        if name == 'uncertainty':
            priority_score = float(risk['score'])
            driver = str(risk['driver'])
        elif name == 'expected_cost':
            priority_score = cost_score
            driver = max(
                [
                    ('failure_probability', failure_probability),
                    ('rul_imminence', imminence),
                    ('rul_uncertainty', uncertainty_factor),
                    ('drift', drift_factor - 1.0),
                ],
                key=lambda item: item[1],
            )[0]
        else:
            priority_score = float(np.clip(
                0.6 * risk['score'] + 0.4 * cost_score, 0.0, 1.0
            ))
            driver = (
                f"{risk['driver']}+cost"
                if cost_score >= risk['score'] else risk['driver']
            )

        priority = cls.risk_band_for_score(priority_score, risk_thresholds)
        reason_text = (
            f"strategy={name}; score={priority_score:.3f}; "
            f"risk={risk['score']:.3f}; driver={driver}; "
            f"expected_cost={expected_cost:.2f}; "
            f"p_failure={failure_probability:.3f}; "
            f"inspection_cost={inspection_cost:.2f}; "
            f"rul={predicted_rul if predicted_rul is None else round(float(predicted_rul), 2)}; "
            f"drift={drift_level or 'unknown'}; queue_reason={reason}"
        )
        return {
            'strategy': name,
            'priority': priority,
            'priority_score': priority_score,
            'priority_reason': reason_text,
            'risk_score': float(risk['score']),
            'risk_band': cls.risk_band_for_score(
                float(risk['score']), risk_thresholds
            ),
            'failure_probability': failure_probability,
            'expected_cost': float(expected_cost),
            'inspection_cost': float(inspection_cost),
            'missed_cost': float(missed_cost),
            'rul_imminence': imminence,
            'rul_uncertainty_factor': uncertainty_factor,
            'drift_factor': drift_factor,
            'driver': driver,
        }

    def _band(self, score: float) -> str:
        return self.risk_band_for_score(score, self.risk_thresholds)

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
                 cycle: Optional[int] = None,
                 risk_weights: Optional[Dict[str, float]] = None,
                 predicted_rul: Optional[float] = None,
                 rul_uncertainty: Optional[float] = None,
                 drift_level: Optional[str] = None
                 ) -> Optional[HITLCase]:
        """Append one case when capacity and deduplication allow it."""
        if len(self.cases) >= self.max_cases:
            return None
        if self._is_duplicate(window_index, unit, cycle):
            return None
        priority = self.evaluate_priority(
            neural_probability, problog_failure_risk, reason,
            predicted_rul, rul_uncertainty, drift_level,
            self.strategy, risk_weights, self.risk_thresholds,
            self.false_negative_cost, self.false_alarm_cost,
            self.rul_threshold,
        )
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
            risk_score=float(priority['risk_score']),
            risk_band=str(priority['risk_band']),
            priority_reason=str(priority['priority_reason']),
            predicted_rul=(
                None if predicted_rul is None else float(predicted_rul)
            ),
            rul_uncertainty=(
                None if rul_uncertainty is None else float(rul_uncertainty)
            ),
            drift_level=drift_level,
            expected_cost=float(priority['expected_cost']),
            inspection_cost=float(priority['inspection_cost']),
            priority_score=float(priority['priority_score']),
            hitl_priority=str(priority['priority']),
        )
        self.cases.append(case)
        return case

    def add_from_prediction(self, window_index: int, neural_probability: float,
                            problog_failure_risk: float, insights: List[str],
                            sensor_values: Dict[str, float],
                            unit: Optional[int] = None,
                            cycle: Optional[int] = None,
                            risk_weights: Optional[Dict[str, float]] = None,
                            predicted_rul: Optional[float] = None,
                            rul_uncertainty: Optional[float] = None,
                            drift_level: Optional[str] = None
                            ) -> Optional[HITLCase]:
        """Classify and append a prediction when it needs review."""
        reason = self.classify_reason(
            neural_probability, insights, problog_failure_risk,
            self.uncertain_low, self.uncertain_high,
        )
        if reason is None:
            return None
        return self.add_case(
            window_index, neural_probability, problog_failure_risk, reason,
            insights, sensor_values, unit, cycle, risk_weights,
            predicted_rul, rul_uncertainty, drift_level,
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

    def _ordered_cases(self) -> List[HITLCase]:
        """Return cases in review order.

        Disabling prioritization keeps the legacy reason-first order. Within
        prioritization, ``uncertainty`` preserves the historical risk score
        ordering, while ``expected_cost`` and ``hybrid`` order by the
        strategy-specific priority score.
        """
        if not self.prioritization_enabled:
            return sorted(
                self.cases,
                key=lambda c: (
                    REASON_PRIORITY.get(c.reason, 99),
                    -c.neural_probability,
                ),
            )
        if self.strategy == 'uncertainty':
            return sorted(
                self.cases,
                key=lambda c: (
                    -float(c.risk_score),
                    REASON_PRIORITY.get(c.reason, 99),
                    -float(c.neural_probability),
                ),
            )
        return sorted(
            self.cases,
            key=lambda c: (
                -float(c.priority_score),
                PRIORITY_ORDER.get(c.hitl_priority, 99),
                -float(c.risk_score),
                REASON_PRIORITY.get(c.reason, 99),
            ),
        )

    def export_csv(self, path: Optional[str] = None) -> str:
        """Write all cases to a flat CSV file in review order."""
        target = path or self.csv_path
        directory = os.path.dirname(target)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fields = [
            'case_id', 'window_index', 'unit', 'cycle', 'neural_probability',
            'problog_failure_risk', 'risk_score', 'risk_band',
            'hitl_priority', 'priority_score', 'priority_reason',
            'expected_cost', 'inspection_cost', 'predicted_rul',
            'rul_uncertainty', 'drift_level',
            'reason', 'status', 'human_label',
            'reviewer', 'comment', 'rule_added', 'insights', 'sensor_values',
            'created_at', 'reviewed_at',
        ]
        with open(target, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for case in self._ordered_cases():
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
                'prioritization_enabled': self.prioritization_enabled,
                'prioritization_strategy': self.strategy,
                'false_negative_cost': self.false_negative_cost,
                'false_alarm_cost': self.false_alarm_cost,
                'rul_threshold': self.rul_threshold,
                'risk_thresholds': self.risk_thresholds,
                'exported_at': datetime.now().isoformat(),
            },
            'cases': [asdict(case) for case in self._ordered_cases()],
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
            prioritization_enabled=metadata.get(
                'prioritization_enabled', False
            ),
            risk_thresholds=metadata.get('risk_thresholds'),
            strategy=metadata.get('prioritization_strategy', 'uncertainty'),
            false_negative_cost=metadata.get('false_negative_cost', 10.0),
            false_alarm_cost=metadata.get('false_alarm_cost', 1.0),
            rul_threshold=metadata.get('rul_threshold', 30.0),
        )
        for case_data in payload.get('cases', []):
            queue.cases.append(HITLCase(**case_data))
        if queue.cases:
            queue._next_id = max(
                int(case.case_id.split('_')[-1]) for case in queue.cases
            ) + 1
        return queue

    def summary(self) -> Dict[str, Any]:
        """Return counts by status, reason, priority and risk band."""
        reasons: Dict[str, int] = {}
        bands: Dict[str, int] = {}
        priorities: Dict[str, int] = {}
        for case in self.cases:
            reasons[case.reason] = reasons.get(case.reason, 0) + 1
            bands[case.risk_band] = bands.get(case.risk_band, 0) + 1
            priorities[case.hitl_priority] = (
                priorities.get(case.hitl_priority, 0) + 1
            )
        ordered = self._ordered_cases()
        return {
            'total_cases': len(self.cases),
            'pending': len(self.pending()),
            'reviewed': len(self.cases) - len(self.pending()),
            'reasons': reasons,
            'risk_bands': bands,
            'priorities': priorities,
            'strategy': self.strategy,
            'prioritized': self.prioritization_enabled,
            'highest_risk_score': (
                float(ordered[0].risk_score) if ordered else None
            ),
            'highest_priority_score': (
                float(ordered[0].priority_score) if ordered else None
            ),
            'highest_priority_case': (
                ordered[0].case_id if ordered else None
            ),
            'highest_risk_case': (
                ordered[0].case_id if ordered else None
            ),
            'csv_path': self.csv_path,
            'json_path': self.json_path,
        }
