# tests/test_rule_validation.py

import logging
import os
import shutil
import tempfile
import unittest

import numpy as np

from src.reasoning.prob_query import EXAMPLES
from src.reasoning.reasoning import SymbolicReasoner

FEATURES = list(EXAMPLES['normal'].keys())


def _validation_windows():
    normal = np.array([EXAMPLES['normal'][name] for name in FEATURES], dtype=float)
    critical = normal.copy()
    critical[FEATURES.index('s4')] = 1450.0
    positive = np.tile(critical, (30, 1))
    negative = np.tile(normal, (30, 1))
    X = np.stack([positive] * 6 + [negative] * 10)
    y = np.array([1] * 6 + [0] * 10)
    return X, y


class TestRuleValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='ansrdt_ruleval_')
        rules_path = os.path.join(self.temp_dir, 'rules.pl')
        shutil.copy('src/reasoning/rules.pl', rules_path)
        logger = logging.getLogger('TestRuleValidation')
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)
        self.reasoner = SymbolicReasoner(
            rules_path, input_shape=(30, len(FEATURES)), model=None, logger=logger
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _add_rules(self):
        self.reasoner.update_rules([
            ('hitl_rule_1 :- hpc_temp_critical.', 0.95),
            ('hitl_rule_bad :- current_sensor_value(s2, _).', 0.95),
        ])

    def test_validation_accepts_precise_rule_and_rejects_broad_rule(self):
        self._add_rules()
        X, y = _validation_windows()
        report = self.reasoner.validate_learned_rules(
            X, FEATURES, y, min_precision=0.9, min_support=5,
            max_false_positive_rate=0.05, apply=False,
        )
        rows = {row['rule'].split(' ')[0]: row for row in report['rules']}

        self.assertTrue(rows['hitl_rule_1']['accepted'])
        self.assertEqual(rows['hitl_rule_1']['support'], 6)
        self.assertEqual(rows['hitl_rule_1']['precision'], 1.0)
        self.assertEqual(rows['hitl_rule_1']['false_positive_rate'], 0.0)

        bad = rows['hitl_rule_bad']
        self.assertFalse(bad['accepted'])
        self.assertEqual(bad['support'], 16)
        self.assertAlmostEqual(bad['precision'], 6 / 16)
        self.assertEqual(bad['false_positive_rate'], 1.0)
        self.assertIn('low_precision', bad['rejection_reasons'])
        self.assertIn('high_false_positive_rate', bad['rejection_reasons'])

        self.assertEqual(len(self.reasoner.learned_rules), 2)
        self.assertEqual(report['summary']['accepted'], 1)
        self.assertEqual(report['summary']['rejected'], 1)

    def test_apply_removes_rejected_rule_and_updates_sidecar(self):
        self._add_rules()
        X, y = _validation_windows()
        report = self.reasoner.validate_learned_rules(
            X, FEATURES, y, apply=True,
        )
        self.assertEqual(report['summary']['rejected'], 1)
        self.assertIn(
            'hitl_rule_1 :- hpc_temp_critical.',
            self.reasoner.learned_rules,
        )
        self.assertNotIn(
            'hitl_rule_bad :- current_sensor_value(s2, _).',
            self.reasoner.learned_rules,
        )
        with open(self.reasoner.learned_rules_path) as file:
            content = file.read()
        self.assertIn('hitl_rule_1', content)
        self.assertNotIn('hitl_rule_bad', content)

    def test_no_learned_rules_returns_empty_report(self):
        X, y = _validation_windows()
        report = self.reasoner.validate_learned_rules(X, FEATURES, y)
        self.assertEqual(report['summary']['rules_evaluated'], 0)
        self.assertEqual(report['rules'], [])


if __name__ == '__main__':
    unittest.main()
