"""Human feedback recording and learned-rule updates for HITL cases."""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.hitl.queue import HITLQueue


FAILURE_LABELS = {'failure', 'fault', 'anomaly', '1', 'true', 'yes'}
NORMAL_LABELS = {'normal', 'no_failure', 'healthy', '0', 'false', 'no'}


class FeedbackRecorder:
    """Turn human labels into queue updates and learned Prolog rules."""

    def __init__(self, queue: HITLQueue, reasoner: Any,
                 log_path: Optional[str] = None):
        """Store the queue, reasoner and JSONL log destination."""
        self.queue = queue
        self.reasoner = reasoner
        self.log_path = log_path or os.path.join(queue.output_dir, 'hitl_log.jsonl')
        directory = os.path.dirname(self.log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def normalise_label(label: str) -> str:
        """Map a free-form human answer to 'failure' or 'normal'."""
        normalised = str(label).strip().lower()
        if normalised in FAILURE_LABELS:
            return 'failure'
        if normalised in NORMAL_LABELS:
            return 'normal'
        raise ValueError(f"Unsupported feedback label: {label!r}")

    def _next_rule_name(self) -> str:
        """Find an unused hitl_rule_N head name."""
        existing = set()
        for rule_string in self.reasoner.learned_rules:
            head = rule_string.split(':-', 1)[0].strip().rstrip('.')
            existing.add(head)
        index = 1
        while f"hitl_rule_{index}" in existing:
            index += 1
        return f"hitl_rule_{index}"

    def record(self, case_id: str, label: str, reviewer: str = 'human',
               comment: str = '') -> Dict[str, Any]:
        """Record one label, update the queue and learn a rule for failures."""
        case = self.queue.get(case_id)
        if case is None:
            raise KeyError(f"Unknown case_id: {case_id}")

        normalised_label = self.normalise_label(label)
        conditions: List[str] = []
        rule_added = None

        if normalised_label == 'failure':
            conditions = self.reasoner.conditions_for_sensor_values(case.sensor_values)
            if conditions:
                rule_name = self._next_rule_name()
                rule_added = f"{rule_name} :- {', '.join(conditions)}."
                self.reasoner.update_rules([(rule_added, 0.95)], min_confidence=0.7)

        self.queue.update_feedback(
            case_id, normalised_label, reviewer=reviewer,
            comment=comment, rule_added=rule_added,
        )
        self.queue.export_csv()

        entry = {
            'timestamp': datetime.now().isoformat(),
            'case_id': case.case_id,
            'unit': case.unit,
            'cycle': case.cycle,
            'neural_probability': case.neural_probability,
            'problog_failure_risk': case.problog_failure_risk,
            'reason': case.reason,
            'human_label': normalised_label,
            'reviewer': reviewer,
            'comment': comment,
            'conditions': conditions,
            'rule_added': rule_added,
        }
        with open(self.log_path, 'a') as file:
            file.write(json.dumps(entry) + '\n')
        return entry

    def review_interactive(self, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
        """Prompt a human through pending cases and record their answers."""
        entries = []
        pending = self.queue.pending()
        if max_cases is not None:
            pending = pending[:int(max_cases)]

        for case in pending:
            print('\n' + '=' * 72)
            print(f"Case {case.case_id} | unit={case.unit} cycle={case.cycle}")
            print(f"Reason: {case.reason}")
            print(f"Neural probability: {case.neural_probability:.3f}")
            print(f"ProbLog failure risk: {case.problog_failure_risk:.3f}")
            print(f"Insights: {', '.join(case.insights) if case.insights else 'none'}")
            print(f"Sensors: {json.dumps(case.sensor_values, indent=2)}")

            answer = input("Label [failure/normal/skip/quit]: ").strip().lower()
            if answer in {'q', 'quit', 'exit'}:
                break
            if answer in {'s', 'skip', ''}:
                continue

            comment = input("Comment (optional): ").strip()
            reviewer = input("Reviewer [human]: ").strip() or 'human'
            try:
                entry = self.record(case.case_id, answer, reviewer, comment)
            except ValueError as error:
                print(f"Skipped: {error}")
                continue

            entries.append(entry)
            if entry['rule_added']:
                print(f"Rule added: {entry['rule_added']}")
            else:
                print("Feedback recorded without a new rule.")

        return entries


def _build_reasoner(config_path: str):
    """Build a symbolic reasoner from the project configuration."""
    from src.config.config_manager import load_config
    from src.reasoning.reasoning import SymbolicReasoner

    absolute_config = os.path.abspath(config_path)
    config = load_config(absolute_config)
    project_root = os.path.dirname(os.path.dirname(absolute_config))
    rules_path = config['paths'].get('reasoning_rules_path', 'src/reasoning/rules.pl')
    if not os.path.isabs(rules_path):
        rules_path = os.path.join(project_root, rules_path)
    input_shape = tuple(config['model']['input_shape'])
    return SymbolicReasoner(rules_path, input_shape, model=None)


def main() -> None:
    """Run the interactive HITL review from the command line."""
    parser = argparse.ArgumentParser(description='Review uncertain FD001 HITL cases.')
    parser.add_argument('--queue', default='results/hitl_queue.json',
                        help='Path to the HITL queue JSON file.')
    parser.add_argument('--config', default='configs/config.yaml',
                        help='Path to the project configuration file.')
    parser.add_argument('--max-cases', type=int, default=None,
                        help='Maximum number of cases to review.')
    args = parser.parse_args()

    queue = HITLQueue.load_json(args.queue)
    reasoner = _build_reasoner(args.config)
    recorder = FeedbackRecorder(queue, reasoner)
    entries = recorder.review_interactive(args.max_cases)
    print(f"\nReviewed {len(entries)} case(s). Log: {recorder.log_path}")


if __name__ == '__main__':
    main()
