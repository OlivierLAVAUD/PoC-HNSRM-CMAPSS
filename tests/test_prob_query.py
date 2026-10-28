# tests/test_prob_query.py

import os
import tempfile
import unittest

from src.reasoning.prob_query import (
    EXAMPLES, RULES_PATH, THRESHOLDS, evaluate_state,
    load_hpc_thresholds,
)


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
        self.assertAlmostEqual(degraded['failure_risk'], 0.5218, places=3)
        self.assertAlmostEqual(critical['failure_risk'], 0.6939, places=3)

    def test_pressure_flow_pathway_without_elevated_temperature(self):
        """Pressure/flow evidence alone must not report a zero risk."""
        state = dict(EXAMPLES['normal'])
        state.update({'s4': 1400.0, 's11': 47.8, 's7': 552.0,
                      's12': 520.5})
        result = evaluate_state(self.program, state)

        self.assertNotIn('hpc_temp_elevated', result['active_conditions'])
        self.assertEqual(
            set(result['active_conditions']),
            {'hpc_static_pressure_high', 'hpc_outlet_pressure_low',
             'fuel_flow_low'},
        )
        self.assertAlmostEqual(result['failure_risk'], 0.2321, places=3)
        self.assertGreater(result['failure_risk'], 0.0)
        self.assertLess(
            result['failure_risk'],
            evaluate_state(self.program, EXAMPLES['degraded'])['failure_risk'],
        )

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


class TestThresholdSource(unittest.TestCase):
    def test_thresholds_come_from_the_prolog_rule_file(self):
        self.assertEqual(
            set(THRESHOLDS),
            {
                's4_elevated', 's4_critical', 's11_high',
                's7_low', 's12_low', 's21_low',
            },
        )
        with tempfile.NamedTemporaryFile(
            'w', suffix='.pl', delete=False
        ) as file:
            file.write("hpc_rule_threshold(s4, elevated, 1512.5).\n")
            path = file.name
        try:
            self.assertEqual(
                load_hpc_thresholds(path), {'s4_elevated': 1512.5}
            )
        finally:
            os.remove(path)


if __name__ == '__main__':
    unittest.main()
