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
from src.reasoning.prob_query import EXAMPLES, RULES_PATH, evaluate_state
from src.reasoning.reasoning import PYSWIP_AVAILABLE, SymbolicReasoner


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
        # Cases are exported in priority order (highest priority first), then
        # by descending neural probability:
        #   case_0004 (rule_probability_conflict_high, 0.90),
        #   case_0005 (rule_probability_conflict_low, 0.10),
        #   case_0003 (uncertain_probability, 0.55),
        #   case_0001 (uncertain_probability, 0.50),
        #   case_0002 (uncertain_probability, 0.45).
        self.assertEqual(rows[0]['case_id'], 'case_0004')
        self.assertEqual(rows[0]['reason'], 'rule_probability_conflict_high')
        self.assertEqual(rows[-1]['case_id'], 'case_0002')
        self.assertEqual(rows[-1]['reason'], 'uncertain_probability')

        loaded = HITLQueue.load_json(json_path)
        self.assertEqual(len(loaded.cases), 5)
        self.assertEqual(loaded.cases[0].case_id, 'case_0004')
        self.assertEqual(loaded.cases[0].reason, 'rule_probability_conflict_high')
        self.assertEqual(loaded.cases[-1].case_id, 'case_0002')
        self.assertEqual(loaded.cases[-1].reason, 'uncertain_probability')
        # The critical-example case is the low-priority conflict (case_0005).
        self.assertEqual(
            loaded.cases[1].sensor_values['s4'], EXAMPLES['critical']['s4']
        )

    @unittest.skipUnless(PYSWIP_AVAILABLE, "SWI-Prolog/PySWIP not available")
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

        active_learned = reasoner.get_active_learned_rule_names()
        self.assertIn('hitl_rule_1', active_learned)
        with open(RULES_PATH, 'r') as file:
            problog_program = file.read()
        base_risk = evaluate_state(problog_program, EXAMPLES['critical'])
        learned_risk = evaluate_state(
            problog_program, EXAMPLES['critical'],
            learned_rules=active_learned, learned_rule_weight=0.15,
        )
        self.assertGreater(learned_risk['failure_risk'],
                           base_risk['failure_risk'])


if __name__ == '__main__':
    unittest.main()


class TestHITLRiskPrioritization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='ansrdt_hitl_risk_')
        self.queue = HITLQueue(
            output_dir=self.temp_dir, max_cases=10,
            prioritization_enabled=True,
            risk_thresholds={'medium': 0.4, 'high': 0.6, 'critical': 0.8},
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_score_is_risk_or_conflict_not_just_probability(self):
        high = HITLQueue.compute_risk_score(
            0.95, 0.0, 'rule_probability_conflict_high'
        )
        low_conflict = HITLQueue.compute_risk_score(
            0.10, 0.0, 'rule_probability_conflict_low'
        )
        problog = HITLQueue.compute_risk_score(
            0.10, 0.90, 'problog_probability_conflict_low'
        )
        self.assertEqual(high['driver'], 'neural_probability')
        self.assertEqual(low_conflict['driver'], 'rule_conflict')
        self.assertEqual(problog['driver'], 'problog_failure_risk')
        self.assertAlmostEqual(low_conflict['score'], 0.9)
        self.assertGreaterEqual(low_conflict['score'], 0.4)

    def test_risk_bands_and_weights(self):
        self.assertEqual(HITLQueue.risk_band_for_score(0.1), 'low')
        self.assertEqual(HITLQueue.risk_band_for_score(0.4), 'medium')
        self.assertEqual(HITLQueue.risk_band_for_score(0.6), 'high')
        self.assertEqual(HITLQueue.risk_band_for_score(0.8), 'critical')
        weighted = HITLQueue.compute_risk_score(
            0.4, 0.2, '', risk_weights={'neural': 2.0}
        )
        self.assertAlmostEqual(weighted['score'], 0.8)

    def test_high_risk_case_is_ordered_first(self):
        self.queue.add_from_prediction(
            0, 0.50, 0.0, [], EXAMPLES['normal'], unit=1, cycle=10
        )
        self.queue.add_from_prediction(
            1, 0.90, 0.0, [], EXAMPLES['normal'], unit=2, cycle=20
        )
        ordered = self.queue._ordered_cases()
        self.assertEqual(ordered[0].neural_probability, 0.90)
        self.assertEqual(ordered[0].risk_band, 'critical')
        self.assertIn('risk=0.900', ordered[0].priority_reason)

    def test_risk_fields_survive_json_and_csv(self):
        self.queue.add_from_prediction(
            0, 0.50, 0.0, [], EXAMPLES['normal'], unit=1, cycle=10
        )
        csv_path = self.queue.export_csv()
        json_path = self.queue.save_json()
        with open(csv_path, newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertIn('risk_score', rows[0])
        self.assertIn('risk_band', rows[0])
        self.assertIn('priority_reason', rows[0])
        loaded = HITLQueue.load_json(json_path)
        self.assertTrue(loaded.prioritization_enabled)
        self.assertIsNotNone(loaded.cases[0].risk_score)
        self.assertEqual(loaded.cases[0].risk_band, 'medium')

    def test_legacy_order_remains_the_default(self):
        queue = HITLQueue(output_dir=self.temp_dir)
        queue.add_from_prediction(
            0, 0.45, 0.0, [], EXAMPLES['normal'], unit=1, cycle=10
        )
        queue.add_from_prediction(
            1, 0.55, 0.0, [], EXAMPLES['normal'], unit=2, cycle=20
        )
        ordered = queue._ordered_cases()
        self.assertEqual(ordered[0].neural_probability, 0.55)


class TestHITLCostPrioritization(unittest.TestCase):
    """Review ordering must follow cost exposure without touching decisions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='ansrdt_hitl_cost_')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _queue(self, strategy='expected_cost', enabled=True):
        return HITLQueue(
            output_dir=self.temp_dir, max_cases=20,
            prioritization_enabled=enabled, strategy=strategy,
            false_negative_cost=10.0, false_alarm_cost=1.0,
            rul_threshold=30.0,
        )

    def test_imminent_high_probability_case_is_ordered_first(self):
        queue = self._queue()
        queue.add_case(
            0, 0.90, 0.0, 'uncertain_probability', [],
            EXAMPLES['normal'], unit=1, cycle=10,
            predicted_rul=5.0, drift_level='critical',
        )
        queue.add_case(
            1, 0.20, 0.0, 'uncertain_probability', [],
            EXAMPLES['normal'], unit=2, cycle=20,
            predicted_rul=29.0, drift_level='ok',
        )
        ordered = queue._ordered_cases()
        self.assertEqual(ordered[0].window_index, 0)
        self.assertGreater(ordered[0].expected_cost, ordered[1].expected_cost)
        self.assertEqual(ordered[0].hitl_priority, 'critical')
        self.assertIn('expected_cost', ordered[0].priority_reason)

    def test_conflicting_low_probability_rule_is_still_prioritized(self):
        queue = self._queue()
        queue.add_case(
            0, 0.10, 0.0, 'rule_probability_conflict_low', [],
            EXAMPLES['critical'], unit=1, cycle=10,
            predicted_rul=10.0,
        )
        queue.add_from_prediction(
            1, 0.45, 0.0, [], EXAMPLES['normal'], unit=2, cycle=20,
            predicted_rul=28.0,
        )
        ordered = queue._ordered_cases()
        self.assertEqual(ordered[0].window_index, 0)
        self.assertGreater(ordered[0].priority_score, 0.0)
        self.assertIn('driver=', ordered[0].priority_reason)

    def test_uncertainty_strategy_keeps_the_historical_risk_order(self):
        queue = self._queue(strategy='uncertainty')
        queue.add_from_prediction(
            0, 0.45, 0.0, [], EXAMPLES['normal'], unit=1, cycle=10,
            predicted_rul=2.0,
        )
        queue.add_from_prediction(
            1, 0.95, 0.0, [], EXAMPLES['normal'], unit=2, cycle=20,
            predicted_rul=29.0,
        )
        ordered = queue._ordered_cases()
        self.assertEqual(ordered[0].neural_probability, 0.95)

    def test_disabling_prioritization_restores_reason_first_order(self):
        queue = self._queue(enabled=False)
        queue.add_case(
            0, 0.90, 0.0, 'uncertain_probability', [],
            EXAMPLES['normal'], unit=1, cycle=10
        )
        queue.add_case(
            1, 0.10, 0.0, 'rule_probability_conflict_low', [],
            EXAMPLES['critical'], unit=2, cycle=20,
        )
        ordered = queue._ordered_cases()
        self.assertEqual(ordered[0].reason, 'rule_probability_conflict_low')

    def test_priority_fields_are_exported_and_reloaded(self):
        queue = self._queue()
        queue.add_from_prediction(
            0, 0.80, 0.4, [], EXAMPLES['degraded'], unit=1, cycle=10,
            predicted_rul=8.0, rul_uncertainty=3.0, drift_level='warning',
        )
        csv_path = queue.export_csv()
        json_path = queue.save_json()
        with open(csv_path, newline='') as file:
            rows = list(csv.DictReader(file))
        for column in ('hitl_priority', 'priority_score',
                       'priority_reason', 'expected_cost', 'predicted_rul',
                       'rul_uncertainty', 'drift_level'):
            self.assertIn(column, rows[0])
        loaded = HITLQueue.load_json(json_path)
        self.assertEqual(loaded.strategy, 'expected_cost')
        self.assertIn(
            loaded.cases[0].hitl_priority,
            {'low', 'medium', 'high', 'critical'},
        )
        self.assertIsNotNone(loaded.cases[0].priority_score)
        self.assertEqual(loaded.cases[0].drift_level, 'warning')

    def test_invalid_strategy_and_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            HITLQueue(output_dir=self.temp_dir, strategy='cheapest')
        with self.assertRaises(ValueError):
            HITLQueue(
                output_dir=self.temp_dir, strategy='expected_cost',
                false_negative_cost=0.0, false_alarm_cost=0.0,
            )
        with self.assertRaises(ValueError):
            HITLQueue(
                output_dir=self.temp_dir, strategy='expected_cost',
                rul_threshold=0.0,
            )
        with self.assertRaises(ValueError):
            HITLQueue.evaluate_priority(0.5, 0.0, strategy='cheapest')
