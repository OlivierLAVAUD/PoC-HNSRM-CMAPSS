# tests/test_robustness_report.py

import json
import os
import tempfile
import unittest

from src.evaluation.robustness_report import (
    build_robustness_report,
    render_robustness_markdown,
    run_robustness_report,
)


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        json.dump(payload, file)


def _core():
    return {
        'dataset': 'NASA C-MAPSS FD001',
        'architecture': 'random_forest',
        'splits': {'training': 10, 'validation': 3, 'test': 6},
        'metrics': {
            'validation': {
                'precision': 0.9, 'recall': 0.8, 'f1': 0.85,
                'roc_auc': 0.95, 'average_precision': 0.9,
            },
            'test': {
                'precision': 0.8, 'recall': 0.7, 'f1': 0.75,
                'roc_auc': 0.9, 'average_precision': 0.8,
            },
        },
        'calibration': {'metrics': {'test': {'ece': 0.02}}},
    }


class TestRobustnessReport(unittest.TestCase):
    def test_missing_results_are_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = build_robustness_report(tmpdir)
        self.assertEqual(report['overall_status'], 'fail')
        self.assertTrue(any(
            check['name'] == 'core_run' and check['status'] == 'fail'
            for check in report['checks']
        ))

    def test_available_artifacts_are_consolidated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write(os.path.join(tmpdir, 'cmapss_results.json'), _core())
            _write(os.path.join(tmpdir, 'threshold_sensitivity.json'), {
                'production_candidate': {
                    'threshold': 0.22,
                    'test': {
                        'f1': 0.76, 'precision': 0.77, 'recall': 0.75,
                    },
                },
            })
            _write(os.path.join(tmpdir, 'multi_seed_report.json'), {
                'protocol': {'seeds': [42, 43, 44]},
                'aggregate': {
                    'test_f1': {'mean': 0.76, 'std': 0.01},
                    'test_recall': {'mean': 0.75, 'std': 0.02},
                    'test_precision': {'mean': 0.77, 'std': 0.01},
                    'fusion_threshold': {'mean': 0.22, 'std': 0.005},
                    'total_false_alarms': {'mean': 30.0, 'std': 2.0},
                },
                'leakage_check': {'ok': True, 'test_engines': 100},
            })
            _write(os.path.join(tmpdir, 'symbolic_contribution.json'), {
                'signals': ['symbolic_rule_score'],
                'summary': {
                    'helpful_signals': ['symbolic_rule_score'],
                    'harmful_signals': [],
                    'answer': 'Symbolic evidence improves validation F1.',
                },
                'contributions': {
                    'symbolic_rule_score': {'verdict': 'helps'},
                },
            })
            _write(os.path.join(tmpdir, 'alert_policy_comparison.json'), {
                'selected_mode': 'consecutive_2',
                'cost_unit': 'episode',
                'rationale': {'selected_mode': 'consecutive_2'},
            })
            _write(os.path.join(tmpdir, 'hitl_queue.json'), {
                'metadata': {'prioritization_enabled': True},
                'cases': [
                    {'risk_band': 'critical', 'risk_score': 0.9},
                    {'risk_band': 'medium', 'risk_score': 0.5},
                ],
            })
            report = run_robustness_report(tmpdir)

            self.assertEqual(report['overall_status'], 'pass')
            checks = {check['name']: check for check in report['checks']}
            self.assertEqual(checks['seed_stability']['status'], 'pass')
            self.assertEqual(checks['symbolic_contribution']['status'], 'pass')
            self.assertEqual(checks['hitl_prioritization']['status'], 'pass')
            self.assertTrue(os.path.exists(
                os.path.join(tmpdir, 'robustness_report.json')
            ))
            self.assertTrue(os.path.exists(
                os.path.join(tmpdir, 'robustness_report.md')
            ))
            self.assertTrue(os.path.exists(
                os.path.join(tmpdir, 'robustness_report.csv')
            ))
            self.assertIn('csv_path', report['artifacts'])
            self.assertIn('plot_path', report['artifacts'])
            if report['artifacts']['plot_path']:
                self.assertTrue(os.path.exists(
                    report['artifacts']['plot_path']
                ))
            markdown = render_robustness_markdown(report)
            self.assertIn('# C-MAPSS robustness report', markdown)
            self.assertIn('consecutive_2', json.dumps(report))

    def test_warnings_and_skips_do_not_hide_the_core_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write(os.path.join(tmpdir, 'cmapss_results.json'), _core())
            _write(os.path.join(tmpdir, 'multi_seed_report.json'), {
                'protocol': {'seeds': [42, 43]},
                'aggregate': {'test_f1': {'mean': 0.7, 'std': 0.10}},
                'leakage_check': {'ok': True},
            })
            report = build_robustness_report(tmpdir)
            self.assertEqual(report['overall_status'], 'warn')
            self.assertTrue(any(
                check['name'] == 'seed_stability'
                and check['status'] == 'warn'
                for check in report['checks']
            ))
            self.assertTrue(any(
                check['status'] == 'skip' for check in report['checks']
            ))


if __name__ == '__main__':
    unittest.main()
