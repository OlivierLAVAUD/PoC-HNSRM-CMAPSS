# tests/test_reasoning.py

import logging
import unittest

from src.reasoning.prob_query import EXAMPLES
from src.reasoning.reasoning import SymbolicReasoner


class TestSymbolicReasoner(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger('TestReasoning')
        logger.addHandler(logging.NullHandler())
        self.reasoner = SymbolicReasoner(
            'src/reasoning/rules.pl', input_shape=(30, 14), model=None, logger=logger
        )

    def test_normal_state_has_no_hpc_alarm(self):
        insights = self.reasoner.reason(EXAMPLES['normal'])
        self.assertNotIn('HPC degradation pattern', insights)
        self.assertNotIn('HPC critical pattern', insights)
        self.assertEqual(insights, [])

    def test_degraded_state_triggers_early_warning(self):
        insights = self.reasoner.reason(EXAMPLES['degraded'])
        for expected in [
            'HPC outlet temperature elevated',
            'HPC static pressure high',
            'HPC outlet pressure low',
            'Fuel flow low',
            'HPC degradation pattern',
        ]:
            self.assertIn(expected, insights)

    def test_critical_state_triggers_severe_pattern(self):
        insights = self.reasoner.reason(EXAMPLES['critical'])
        for expected in [
            'HPC outlet temperature critical',
            'HPC static pressure high',
            'HPC outlet pressure low',
            'Coolant bleed low',
            'HPC critical pattern',
            'HPC thermal overload',
            'HPC fuel starvation',
        ]:
            self.assertIn(expected, insights)

    def test_history_records_activations(self):
        self.reasoner.reason(EXAMPLES['normal'])
        self.reasoner.reason(EXAMPLES['critical'])
        history = self.reasoner.get_rule_activations()
        self.assertEqual(len(history), 2)
        self.assertGreater(len(history[-1]['activated_rules_detailed']), 0)
        self.assertEqual(history[-1]['sensor_values']['s4'], EXAMPLES['critical']['s4'])


if __name__ == '__main__':
    unittest.main()
