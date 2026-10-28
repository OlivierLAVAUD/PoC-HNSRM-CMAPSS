# tests/test_readme_metrics.py
"""Lock the published README fusion and alerting numbers to the FD001 run.

Regenerating ``results/final_run_results.json`` requires refreshing the
"LightGBM with fusion" and "Alert confirmation" tables in ``README.md``;
this test fails when the documentation and the recorded run disagree.
"""

import json
import os
import unittest

README_PATH = 'README.md'
RESULTS_PATH = os.path.join('results', 'final_run_results.json')
TOLERANCE = 5e-4


def _table_rows(text, heading):
    """Return the body rows of the first markdown table under a heading."""
    lines = text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError as error:
        raise AssertionError(f"Missing README heading: {heading!r}") from error

    rows = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if stripped.startswith('|'):
            rows.append([
                cell.strip().strip('*`') for cell in stripped.strip('|').split('|')
            ])
        elif rows:
            break
    if len(rows) < 2:
        raise AssertionError(f"No table found under heading {heading!r}")
    body = rows[1:]
    return [
        row for row in body
        if not all(set(cell) <= {'-', ':'} and cell for cell in row)
    ]


class TestReadmeFusionMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(README_PATH) as readme_file:
            cls.readme = readme_file.read()
        with open(RESULTS_PATH) as results_file:
            cls.results = json.load(results_file)

    def test_fusion_table_matches_recorded_run(self):
        rows = _table_rows(self.readme, '### LightGBM with fusion')
        expected_splits = {'Validation': 'validation', 'Official test': 'test'}
        columns = (
            'accuracy', 'precision', 'recall', 'f1',
            'roc_auc', 'average_precision',
        )
        seen = set()
        for row in rows:
            split_label = row[0]
            self.assertIn(split_label, expected_splits)
            seen.add(split_label)
            metrics = self.results['metrics'][expected_splits[split_label]]
            for column, published in zip(columns, row[1:]):
                with self.subTest(split=split_label, metric=column):
                    self.assertAlmostEqual(
                        float(published), float(metrics[column]),
                        delta=TOLERANCE,
                        msg=f"{split_label} {column} in README is stale",
                    )
        self.assertEqual(seen, set(expected_splits))

    def test_alert_confirmation_table_matches_recorded_run(self):
        rows = _table_rows(self.readme, '### Alert confirmation')
        expected_modes = {
            'Immediate': 'immediate',
            'Consecutive 2': 'consecutive_2',
            'Consecutive 3': 'consecutive_3',
            'Smoothed 3': 'smoothed_3',
        }
        seen = set()
        for row in rows:
            mode_label = row[0]
            self.assertIn(mode_label, expected_modes)
            seen.add(mode_label)
            mode = self.results['alerting']['modes'][expected_modes[mode_label]]
            false_alarms, mean_recall, mean_precision = row[1:]
            with self.subTest(mode=mode_label):
                self.assertEqual(
                    int(false_alarms), int(mode['total_false_alarms']),
                    f"{mode_label} false alarms in README are stale",
                )
                self.assertAlmostEqual(
                    float(mean_recall), float(mode['mean_recall']),
                    delta=TOLERANCE,
                    msg=f"{mode_label} mean recall in README is stale",
                )
                self.assertAlmostEqual(
                    float(mean_precision), float(mode['mean_precision']),
                    delta=TOLERANCE,
                    msg=f"{mode_label} mean precision in README is stale",
                )
        self.assertEqual(seen, set(expected_modes))


if __name__ == '__main__':
    unittest.main()
