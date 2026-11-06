# tests/test_skab_cli.py

import csv
import os
import tempfile
import unittest
from unittest.mock import patch

from main import build_parser
from src.skab.run import run_skab_from_args


class TestSKABCLI(unittest.TestCase):
    def test_main_parser_exposes_skab_options(self):
        args = build_parser().parse_args([
            '--benchmark', 'skab', '--modes', 'random_forest',
            '--min-support', '2', '--max-rules', '7',
        ])
        self.assertEqual(args.benchmark, 'skab')
        self.assertEqual(args.modes, ['random_forest'])
        self.assertEqual(args.min_support, 2)
        self.assertEqual(args.max_rules, 7)

    @patch('src.skab.run.run_skab_benchmark')
    def test_runner_writes_summary_csv(self, benchmark):
        benchmark.return_value = {
            'model_type': 'random_forest',
            'rf_accuracy': 0.875,
            'symbolic_evaluated': 5,
            'symbolic_accepted': 3,
            'symbolic_rejected': 2,
            'symbolic_rules': [{}, {}, {}],
            'feature_count': 9,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            code = run_skab_from_args(
                build_parser().parse_args([
                    '--benchmark', 'skab', '--modes', 'random_forest',
                ]),
                project_root=tmpdir,
            )
            summaries = []
            for root, _, files in os.walk(tmpdir):
                if 'summary.csv' in files:
                    summaries.append(os.path.join(root, 'summary.csv'))
            self.assertEqual(len(summaries), 1)
            with open(summaries[0], newline='') as file:
                row = next(csv.DictReader(file))
        self.assertEqual(code, 0)
        self.assertEqual(row['benchmark'], 'skab')
        self.assertEqual(row['rf_accuracy'], '0.875')
        self.assertEqual(row['symbolic_rules_count'], '3')
        benchmark.assert_called_once()


if __name__ == '__main__':
    unittest.main()
