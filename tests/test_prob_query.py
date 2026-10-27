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

    def test_learned_rule_increases_risk_and_is_capped(self):
        base = evaluate_state(self.program, EXAMPLES['degraded'])
        one = evaluate_state(
            self.program, EXAMPLES['degraded'],
            learned_rules=['hitl_rule_1'], learned_rule_weight=0.15,
            max_active_learned_rules=3,
        )
        four = evaluate_state(
            self.program, EXAMPLES['degraded'],
            learned_rules=['hitl_rule_1', 'hitl_rule_2', 'hitl_rule_3', 'hitl_rule_4'],
            learned_rule_weight=0.15, max_active_learned_rules=3,
        )

        self.assertGreater(one['failure_risk'], base['failure_risk'])
        self.assertLessEqual(four['failure_risk'],
                             1 - (1 - base['failure_risk']) * (1 - 0.15) ** 3 + 1e-9)
        self.assertEqual(len(four['learned_rules_active']), 3)
        self.assertIn('hitl_rule_1', four['learned_rules_active'])

    def test_zero_learned_weight_has_no_effect(self):
        base = evaluate_state(self.program, EXAMPLES['critical'])
        weighted = evaluate_state(
            self.program, EXAMPLES['critical'],
            learned_rules=['hitl_rule_1'], learned_rule_weight=0.0,
        )
        self.assertAlmostEqual(base['failure_risk'], weighted['failure_risk'])


if __name__ == '__main__':
    unittest.main()
