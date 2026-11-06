# tests/test_representation_ablation.py

import unittest

from main import build_parser
from src.experiments.representation_ablation import _conclusion


class TestRepresentationAblation(unittest.TestCase):
    def test_conclusion_flags_random_ranking(self):
        rows = [
            {'dataset': 'FD002', 'threshold_strategy': 'source',
             'variant': 'raw', 'roc_auc': 0.58, 'average_precision': 0.06},
            {'dataset': 'FD004', 'threshold_strategy': 'source',
             'variant': 'raw', 'roc_auc': 0.59, 'average_precision': 0.05},
        ]
        statements = _conclusion(rows)['statements']
        self.assertTrue(any('does not fix the ranking' in item for item in statements))

    def test_conclusion_flags_fd003_threshold_problem(self):
        rows = [
            {'dataset': 'FD003', 'threshold_strategy': 'source',
             'variant': 'raw', 'roc_auc': 0.95, 'average_precision': 0.44,
             'recall': 0.35},
        ]
        statements = _conclusion(rows)['statements']
        self.assertTrue(any('threshold/calibration' in item for item in statements))

    def test_cli_exposes_new_modes(self):
        parser = build_parser()
        args = parser.parse_args(['--representation-ablation'])
        self.assertTrue(args.representation_ablation)
        args = parser.parse_args(['--few-shot-target', 'fd003'])
        self.assertEqual(args.few_shot_target, 'fd003')


if __name__ == '__main__':
    unittest.main()
