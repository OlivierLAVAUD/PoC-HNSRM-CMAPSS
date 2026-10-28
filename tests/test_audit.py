# tests/test_audit.py

import json
import os
import tempfile
import unittest

from src.utils.audit import (
    AuditTrail,
    build_artifact_versions,
    main,
    make_decision_id,
    read_audit,
    sha256_file,
    verify_audit,
)


def write_file(path, content):
    """Write a small text file and return its path."""
    with open(path, 'w') as file:
        file.write(content)
    return path


def build_run(tmpdir):
    """Create a small but coherent audit run inside ``tmpdir``."""
    model_path = write_file(os.path.join(tmpdir, 'model.joblib'), 'model-v1')
    rules_path = write_file(os.path.join(tmpdir, 'rules.pl'), 'rule(a).\n')
    with open(os.path.join(tmpdir, 'fusion_threshold.json'), 'w') as file:
        json.dump({'threshold': 0.22}, file)
    with open(os.path.join(tmpdir, 'decision_threshold.json'), 'w') as file:
        json.dump({'threshold': 0.07}, file)
    artifacts = build_artifact_versions(
        architecture='lightgbm', model_path=model_path, rules_path=rules_path,
    )
    trail = AuditTrail(
        path=os.path.join(tmpdir, 'decision_audit.jsonl'),
        run_id='20260101T000000',
        artifacts=artifacts,
        thresholds={'source': 'probability_fusion', 'fusion': 0.22,
                    'neural': 0.07},
    )
    trail.add(
        split='test', window_index=0, unit=1, cycle=31,
        neural_probability=0.4, fused_probability=0.5, threshold=0.22,
        decision=1, alert_mode='consecutive_2', neural_threshold=0.07,
        symbolic_rule_score=0.8, problog_failure_risk=0.52,
        predicted_rul=18.0, insights=['critical temperature'],
        active_rules=['high_s4'], data_quality={'non_finite_values': 0},
    )
    trail.add(
        split='test', window_index=1, unit=1, cycle=32,
        neural_probability=0.1, fused_probability=0.1, threshold=0.22,
        decision=0, alert_mode='consecutive_2', neural_threshold=0.07,
        data_quality={'non_finite_values': 0},
    )
    trail.flush()
    return trail, model_path, rules_path, artifacts


class TestIdentifiersAndHashes(unittest.TestCase):
    def test_decision_ids_are_unique_and_deterministic(self):
        first = make_decision_id('run', 'test', 3, 7, 40)
        same = make_decision_id('run', 'test', 3, 7, 40)
        other = make_decision_id('run', 'test', 4, 7, 40)
        other_run = make_decision_id('run2', 'test', 3, 7, 40)
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)
        self.assertNotEqual(first, other_run)
        self.assertEqual(len(first), 24)

    def test_hash_is_content_addressed(self):
        self.assertIsNone(sha256_file('/nonexistent/file'))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_file(os.path.join(tmpdir, 'artifact.bin'), 'a')
            first = sha256_file(path)
            write_file(path, 'b')
            second = sha256_file(path)
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_versions_are_derived_from_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = write_file(os.path.join(tmpdir, 'model.joblib'), 'm')
            rules_path = write_file(os.path.join(tmpdir, 'rules.pl'), 'r')
            metadata_path = write_file(
                os.path.join(tmpdir, 'features.json'), '{}'
            )
            artifacts = build_artifact_versions(
                'lightgbm', model_path=model_path, rules_path=rules_path,
                feature_metadata_path=metadata_path,
            )
            model_hash = sha256_file(model_path)
            rules_hash = sha256_file(rules_path)
        self.assertTrue(
            artifacts['versions']['model_version'].startswith('lightgbm-')
        )
        self.assertTrue(
            artifacts['versions']['features_version'].startswith('temporal-')
        )
        self.assertTrue(artifacts['versions']['rules_version'].startswith('pl-'))
        self.assertEqual(artifacts['hashes']['model_sha256'], model_hash)
        self.assertEqual(artifacts['hashes']['rules_pl_sha256'], rules_hash)


class TestAuditTrail(unittest.TestCase):
    def test_journal_contains_manifest_and_decisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trail, _, _, artifacts = build_run(tmpdir)
            records = read_audit(
                os.path.join(tmpdir, 'decision_audit.jsonl')
            )
        self.assertEqual(trail.count, 2)
        self.assertEqual(len(records), 3)
        manifest = records[0]
        self.assertEqual(manifest['record_type'], 'run_manifest')
        self.assertEqual(manifest['run_id'], '20260101T000000')
        self.assertEqual(manifest['decision_records'], 2)
        self.assertEqual(manifest['versions'], artifacts['versions'])
        decision = records[1]
        self.assertEqual(decision['record_type'], 'decision')
        self.assertEqual(decision['decision_id'],
                         make_decision_id('20260101T000000', 'test', 0, 1, 31))
        self.assertEqual(decision['model_version'],
                         artifacts['versions']['model_version'])
        self.assertEqual(decision['fusion_threshold'], 0.22)
        self.assertEqual(decision['decision'], 1)
        self.assertEqual(decision['insights'], ['critical temperature'])

    def test_record_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trail, _, _, _ = build_run(tmpdir)
            capped = AuditTrail(
                path=os.path.join(tmpdir, 'capped.jsonl'), run_id='run',
                artifacts=trail.artifacts, enabled=True,
            )
            capped.max_records = 1
            for index in range(3):
                capped.add(
                    split='test', window_index=index, unit=1, cycle=index,
                    neural_probability=0.5, fused_probability=0.5,
                    threshold=0.5, decision=1, alert_mode='immediate',
                )
        self.assertEqual(capped.count, 1)
        self.assertEqual(capped.dropped, 2)

    def test_disabled_trail_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'audit.jsonl')
            trail = AuditTrail(path=path, run_id='run', artifacts={},
                               enabled=False)
            trail.add(split='test', window_index=0, unit=1, cycle=1,
                      neural_probability=0.1, fused_probability=None,
                      threshold=0.5, decision=0, alert_mode='immediate')
            self.assertIsNone(trail.flush())
            self.assertFalse(os.path.exists(path))


class TestVerification(unittest.TestCase):
    def test_coherent_run_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_run(tmpdir)
            report = verify_audit(tmpdir)
        self.assertTrue(report['ok'], report['issues'])
        self.assertEqual(report['decisions'], 2)
        self.assertEqual(
            report['artifacts']['model']['status'], 'ok'
        )

    def test_missing_journal_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = verify_audit(tmpdir)
        self.assertFalse(report['ok'])
        self.assertEqual(report['issues'][0]['type'], 'missing_journal')

    def test_modified_artifact_is_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, model_path, _, _ = build_run(tmpdir)
            write_file(model_path, 'tampered')
            report = verify_audit(tmpdir)
        self.assertFalse(report['ok'])
        types = {issue['type'] for issue in report['issues']}
        self.assertIn('artifact_hash_mismatch', types)

    def test_threshold_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_run(tmpdir)
            with open(os.path.join(tmpdir, 'fusion_threshold.json'), 'w') as file:
                json.dump({'threshold': 0.9}, file)
            report = verify_audit(tmpdir)
        self.assertFalse(report['ok'])
        types = {issue['type'] for issue in report['issues']}
        self.assertIn('threshold_mismatch', types)

    def test_duplicate_and_inconsistent_records_are_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_run(tmpdir)
            journal = os.path.join(tmpdir, 'decision_audit.jsonl')
            with open(journal) as file:
                lines = file.read().splitlines()
            duplicate = json.loads(lines[1])
            inconsistent = json.loads(lines[2])
            inconsistent['decision_id'] = 'other-id'
            inconsistent['decision'] = 1
            with open(journal, 'w') as file:
                file.write('\n'.join(lines + [
                    json.dumps(duplicate), json.dumps(inconsistent),
                ]) + '\n')
            report = verify_audit(tmpdir)
        types = {issue['type'] for issue in report['issues']}
        self.assertFalse(report['ok'])
        self.assertIn('duplicate_decision_id', types)
        self.assertIn('inconsistent_decision', types)

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            build_run(tmpdir)
            self.assertEqual(main(['--results', tmpdir]), 0)
            write_file(os.path.join(tmpdir, 'model.joblib'), 'tampered')
            self.assertEqual(main(['--results', tmpdir]), 1)


if __name__ == '__main__':
    unittest.main()
