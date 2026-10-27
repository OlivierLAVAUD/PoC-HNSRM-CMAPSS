# tests/test_symbolic_reasoning.py

import logging
import unittest

from src.reasoning.prob_query import EXAMPLES
from src.reasoning.reasoning import SymbolicReasoner


class TestSymbolicReasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger('TestSymbolicReasoning')
        cls.logger.setLevel(logging.DEBUG)
        if not cls.logger.handlers:
            cls.logger.addHandler(logging.NullHandler())
        cls.reasoner = SymbolicReasoner(
            'src/reasoning/rules.pl', input_shape=(30, 14), model=None, logger=cls.logger
        )

    def test_temperature_elevated(self):
        insights = self.reasoner.reason(EXAMPLES['degraded'])
        self.assertIn('HPC outlet temperature elevated', insights)
        self.assertNotIn('HPC outlet temperature critical', insights)

    def test_static_pressure_high(self):
        insights = self.reasoner.reason(EXAMPLES['degraded'])
        self.assertIn('HPC static pressure high', insights)

    def test_outlet_pressure_low(self):
        insights = self.reasoner.reason(EXAMPLES['critical'])
        self.assertIn('HPC outlet pressure low', insights)

    def test_coolant_bleed_low(self):
        insights = self.reasoner.reason(EXAMPLES['critical'])
        self.assertIn('Coolant bleed low', insights)

    def test_graded_risk_between_examples(self):
        normal = self.reasoner.reason(EXAMPLES['normal'])
        degraded = self.reasoner.reason(EXAMPLES['degraded'])
        critical = self.reasoner.reason(EXAMPLES['critical'])
        self.assertLess(len(normal), len(degraded))
        self.assertLess(len(degraded), len(critical))


if __name__ == '__main__':
    unittest.main()
