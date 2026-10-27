# tests/test_prob_query.py

import unittest

from src.reasoning.prob_query import EXAMPLES, RULES_PATH, evaluate_state


class TestProbLogHPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(RULES_PATH, 'r') as file:
            cls.program = file.read()

    def test_failure_risk_increases_with_degradation(self):
        normal = evaluate_state(self.program, EXAMPLES['normal'])
        degraded = evaluate_state(self.program, EXAMPLES['degraded'])
        critical = evaluate_state(self.program, EXAMPLES['critical'])

        self.assertEqual(normal['failure_risk'], 0.0)
        self.assertGreater(degraded['failure_risk'], normal['failure_risk'])
        self.assertGreater(critical['failure_risk'], degraded['failure_risk'])
        self.assertAlmostEqual(degraded['failure_risk'], 0.4318, places=3)
        self.assertAlmostEqual(critical['failure_risk'], 0.6452, places=3)

    def test_evidence_matches_thresholds(self):
        degraded = evaluate_state(self.program, EXAMPLES['degraded'])
        critical = evaluate_state(self.program, EXAMPLES['critical'])

        self.assertIn('hpc_temp_elevated', degraded['active_conditions'])
        self.assertNotIn('hpc_temp_critical', degraded['active_conditions'])
        self.assertIn('hpc_temp_critical', critical['active_conditions'])
        self.assertIn('coolant_bleed_low', critical['active_conditions'])


if __name__ == '__main__':
    unittest.main()
