# tests/test_rule_thresholds.py
"""Lock rules.pl thresholds to the FD001 20% training split."""

import os
import re
import unittest

import numpy as np

from src.data.cmapss_loader import CMAPSSLoader
from src.reasoning.prob_query import RULES_PL_PATH

FD001_PARQUET = os.path.join('data', 'raw', 'CMAPSS', 'FD001_processed.parquet')


def _parse_binary_facts(rules, name):
    pattern = rf"{name}\(\s*([a-z0-9_]+)\s*,\s*([-0-9.]+)\s*\)\."
    return {
        sensor: float(value)
        for sensor, value in re.findall(pattern, rules)
    }


def _parse_named_facts(rules, name):
    pattern = (
        rf"{name}\(\s*([a-z0-9_]+)\s*,\s*([a-z0-9_]+)\s*,\s*"
        rf"([-0-9.]+)\s*\)\."
    )
    return {
        (sensor, kind): float(value)
        for sensor, kind, value in re.findall(pattern, rules)
    }


@unittest.skipUnless(os.path.exists(FD001_PARQUET), "FD001 parquet not downloaded")
class TestFD001RuleThresholds(unittest.TestCase):
    """The hard-coded thresholds must come from the 20 training engines only."""

    @classmethod
    def setUpClass(cls):
        cls.loader = CMAPSSLoader(FD001_PARQUET, window_size=30, rul_threshold=30)
        X, y = cls.loader.load_data()
        cls.loader.split_by_unit(
            X, y, validation_split=0.2, test_split=0.6, random_state=42
        )
        train_mask = cls.loader.last_split_masks['train']
        cls.train_units = sorted(
            set(np.asarray(cls.loader.unit_ids)[train_mask].tolist())
        )

        frame = cls.loader._read_frame(FD001_PARQUET)
        cls.train_frame = frame[frame['unit_number'].isin(cls.train_units)]
        with open(RULES_PL_PATH) as rules_file:
            cls.rules = rules_file.read()

    def test_split_selects_twenty_training_engines(self):
        self.assertEqual(len(self.train_units), 20)

    def test_high_thresholds_match_training_percentiles(self):
        facts = _parse_binary_facts(self.rules, 'sensor_high_threshold')
        self.assertTrue(facts)
        for sensor, value in facts.items():
            expected = round(
                float(np.percentile(self.train_frame[sensor], 95)), 3
            )
            self.assertAlmostEqual(
                value, expected, places=9,
                msg=f"{sensor} high threshold is not the training p95",
            )

    def test_low_thresholds_match_training_percentiles(self):
        facts = _parse_binary_facts(self.rules, 'sensor_low_threshold')
        self.assertTrue(facts)
        for sensor, value in facts.items():
            expected = round(
                float(np.percentile(self.train_frame[sensor], 5)), 3
            )
            self.assertAlmostEqual(
                value, expected, places=9,
                msg=f"{sensor} low threshold is not the training p5",
            )

    def test_named_hpc_thresholds_match_training_percentiles(self):
        facts = _parse_named_facts(self.rules, 'hpc_rule_threshold')
        self.assertTrue(facts)
        percentiles = {
            'elevated': 75, 'high': 75, 'critical': 95, 'low': 25,
        }
        for (sensor, kind), value in facts.items():
            expected = round(
                float(np.percentile(
                    self.train_frame[sensor], percentiles[kind]
                )), 3
            )
            self.assertAlmostEqual(
                value, expected, places=9,
                msg=f"{sensor} {kind} threshold is not the training percentile",
            )

    def test_gradient_thresholds_match_training_percentiles(self):
        facts = _parse_binary_facts(self.rules, 'gradient_threshold')
        self.assertTrue(facts)
        deltas = (
            self.train_frame
            .groupby('unit_number')[list(facts)]
            .diff()
            .abs()
            .dropna()
        )
        for sensor, value in facts.items():
            expected = round(float(np.percentile(deltas[sensor], 95)), 3)
            self.assertAlmostEqual(
                value, expected, places=9,
                msg=f"{sensor} gradient threshold is not the training p95",
            )

    def test_thresholds_are_not_derived_from_all_engines(self):
        """Guard against silently reverting to the full 100-engine file."""
        all_engines = self.loader._read_frame(FD001_PARQUET)
        train_p95 = float(np.percentile(self.train_frame['s4'], 95))
        all_p95 = float(np.percentile(all_engines['s4'], 95))
        self.assertGreater(abs(train_p95 - all_p95), 1e-3)


if __name__ == '__main__':
    unittest.main()
