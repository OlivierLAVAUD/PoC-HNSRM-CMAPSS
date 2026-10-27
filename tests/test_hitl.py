"""Tests for the human-in-the-loop queue and feedback loop."""

import csv
import json
import logging
import os
import shutil
import tempfile
import unittest

from src.hitl.feedback import FeedbackRecorder
from src.hitl.queue import HITLQueue
from src.reasoning.prob_query import EXAMPLES
from src.reasoning.reasoning import SymbolicReasoner


class TestHITLQueueAndFeedback(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='ansrdt_hitl_')
        self.rules_path = os.path.join(self.temp_dir, 'rules.pl')
        shutil.copy('src/reasoning/rules.pl', self.rules_path)
        self.queue = HITLQueue(output_dir=self.temp_dir, max_cases=10)

        self.queue.add_from_prediction(
            0, 0.50, 0.0, [], EXAMPLES['critical'], unit=1, cycle=100
        )
        self.queue.add_from_prediction(
            1, 0.45, 0.0, [], EXAMPLES['normal'], unit=2, cycle=80
        )
        self.queue.add_from_prediction(
            2, 0.55, 0.0, [], EXAMPLES['normal'], unit=3, cycle=90
        )
        self.queue.add_from_prediction(
            3, 0.90, 0.0, [], EXAMPLES['normal'], unit=4, cycle=70
        )
        self.queue.add_from_prediction(
            4, 0.10, 0.0, ['HPC critical pattern'], EXAMPLES['critical'],
            unit=5, cycle=120,
        )

        self.assertIsNone(self.queue.add_from_prediction(
            5, 0.90, 0.0, ['HPC degradation pattern'], EXAMPLES['degraded']
        ))
        self.assertIsNone(self.queue.add_from_prediction(
            6, 0.10, 0.0, [], EXAMPLES['normal']
        ))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_queue_contains_uncertain_and_conflicting_cases(self):
        self.assertEqual(len(self.queue.cases), 5)
        self.assertEqual(len(self.queue.pending()), 5)
        summary = self.queue.summary()
        self.assertEqual(summary['total_cases'], 5)
        self.assertEqual(summary['pending'], 5)
        self.assertEqual(summary['reasons']['uncertain_probability'], 3)
        self.assertEqual(summary['reasons']['rule_probability_conflict_high'], 1)
        self.assertEqual(summary['reasons']['rule_probability_conflict_low'], 1)

    def test_queue_exports_csv_and_json(self):
        csv_path = self.queue.export_csv()
        json_path = self.queue.save_json()
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))

        with open(csv_path, newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]['case_id'], 'case_0001')
        self.assertEqual(rows[0]['reason'], 'uncertain_probability')

        loaded = HITLQueue.load_json(json_path)
        self.assertEqual(len(loaded.cases), 5)
        self.assertEqual(loaded.cases[-1].case_id, 'case_0005')
        self.assertEqual(loaded.cases[0].sensor_values['s4'], EXAMPLES['critical']['s4'])

    def test_feedback_updates_queue_log_and_learned_rules(self):
        logger = logging.getLogger('TestHITL')
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)
        reasoner = SymbolicReasoner(
            self.rules_path, input_shape=(30, 14), model=None, logger=logger
        )
        recorder = FeedbackRecorder(self.queue, reasoner)

        failure_entry = recorder.record(
            'case_0001', 'failure', reviewer='tester', comment='confirmed critical'
        )
        normal_entry = recorder.record(
            'case_0002', 'normal', reviewer='tester', comment='healthy'
        )

        self.assertEqual(failure_entry['human_label'], 'failure')
        self.assertIsNotNone(failure_entry['rule_added'])
        self.assertIn('hitl_rule_1', failure_entry['rule_added'])
        self.assertIsNone(normal_entry['rule_added'])

        self.assertEqual(len(self.queue.pending()), 3)
        self.assertEqual(self.queue.get('case_0001').status, 'reviewed')
        self.assertEqual(self.queue.get('case_0002').human_label, 'normal')

        learned_path = os.path.join(self.temp_dir, 'rules_learned.pl')
        self.assertTrue(os.path.exists(learned_path))
        with open(learned_path) as file:
            learned_content = file.read()
        self.assertIn('hitl_rule_1 :-', learned_content)

        log_path = os.path.join(self.temp_dir, 'hitl_log.jsonl')
        with open(log_path) as file:
            log_entries = [json.loads(line) for line in file if line.strip()]
        self.assertEqual(len(log_entries), 2)
        self.assertEqual(log_entries[0]['case_id'], 'case_0001')
        self.assertEqual(log_entries[0]['human_label'], 'failure')
        self.assertEqual(log_entries[1]['human_label'], 'normal')

        insights = reasoner.reason(EXAMPLES['critical'])
        self.assertTrue(any(
            insight.startswith('Learned Rule Activated: hitl_rule_1')
            for insight in insights
        ))


if __name__ == '__main__':
    unittest.main()
