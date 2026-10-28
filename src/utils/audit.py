# src/utils/audit.py
"""Decision audit trail: identifiers, artifact hashes and JSONL verification.

Every decision written by the pipeline or the inference module carries a
unique ``decision_id``, the versions and SHA-256 digests of the artifacts that
produced it, both decision thresholds, the four fusion signals and the final
binary decision. The journal is an append-free JSONL file rewritten per run:
one ``run_manifest`` record followed by one record per decided window.

Verify a run with::

    python -m src.utils.audit --results results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

DECISION_REQUIRED_FIELDS: Sequence[str] = (
    'decision_id', 'record_type', 'created_at', 'split', 'window_index',
    'model_version', 'features_version', 'rules_version',
    'neural_threshold', 'alert_mode', 'neural_probability',
    'fused_probability', 'threshold', 'decision', 'data_quality',
)

ARTIFACT_FIELDS: Sequence[str] = (
    'model_sha256', 'scaler_sha256', 'feature_metadata_sha256',
    'rules_pl_sha256', 'rules_learned_pl_sha256', 'problog_rules_sha256',
)

AUDIT_FILENAME = 'decision_audit.jsonl'


def sha256_file(path: Optional[str]) -> Optional[str]:
    """Return the SHA-256 digest of a file, or ``None`` when it is absent."""
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(value: Optional[str], length: int = 12) -> str:
    """Return a short readable digest or ``'none'``."""
    return value[:length] if value else 'none'


def make_decision_id(run_id: str, split: str, window_index: int,
                     unit: Optional[int], cycle: Optional[int]) -> str:
    """Build a deterministic, unique identifier for one decided window."""
    payload = '|'.join([
        str(run_id), str(split), str(int(window_index)),
        '' if unit is None else str(int(unit)),
        '' if cycle is None else str(int(cycle)),
    ])
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]


def build_artifact_versions(
        architecture: str,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        feature_metadata_path: Optional[str] = None,
        rules_path: Optional[str] = None,
        learned_rules_path: Optional[str] = None,
        problog_rules_path: Optional[str] = None) -> Dict[str, Any]:
    """Hash the artifacts that determine a decision and derive versions."""
    hashes = {
        'model_sha256': sha256_file(model_path),
        'scaler_sha256': sha256_file(scaler_path),
        'feature_metadata_sha256': sha256_file(feature_metadata_path),
        'rules_pl_sha256': sha256_file(rules_path),
        'rules_learned_pl_sha256': sha256_file(learned_rules_path),
        'problog_rules_sha256': sha256_file(problog_rules_path),
    }
    versions = {
        'model_version': f"{architecture}-{short_hash(hashes['model_sha256'])}",
        'features_version': (
            f"temporal-{short_hash(hashes['feature_metadata_sha256'])}"
            if hashes['feature_metadata_sha256'] else 'flattened-window'
        ),
        'rules_version': f"pl-{short_hash(hashes['rules_pl_sha256'], 8)}",
    }
    return {
        'versions': versions,
        'hashes': hashes,
        'paths': {
            'model': model_path,
            'scaler': scaler_path,
            'feature_metadata': feature_metadata_path,
            'rules': rules_path,
            'learned_rules': learned_rules_path,
            'problog_rules': problog_rules_path,
        },
    }


class AuditTrail:
    """Collect decision records and write them as a JSONL journal."""

    def __init__(self, path: str, run_id: str,
                 artifacts: Dict[str, Any],
                 thresholds: Optional[Dict[str, Any]] = None,
                 enabled: bool = True,
                 logger: Any = None):
        self.path = path
        self.run_id = run_id
        self.artifacts = artifacts
        self.thresholds = thresholds or {}
        self.enabled = bool(enabled)
        self.logger = logger
        self.records: List[Dict[str, Any]] = []
        self.dropped = 0
        self.max_records: Optional[int] = None

    def manifest(self, **extra: Any) -> Dict[str, Any]:
        """Return the run-level manifest record."""
        return {
            'record_type': 'run_manifest',
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'versions': self.artifacts.get('versions', {}),
            'hashes': self.artifacts.get('hashes', {}),
            'artifact_paths': self.artifacts.get('paths', {}),
            'thresholds': self.thresholds,
            **extra,
        }

    def add(self, split: str, window_index: int,
            unit: Optional[int], cycle: Optional[int],
            neural_probability: Optional[float],
            fused_probability: Optional[float],
            threshold: float,
            decision: int,
            alert_mode: str,
            neural_threshold: Optional[float] = None,
            symbolic_rule_score: Optional[float] = None,
            problog_failure_risk: Optional[float] = None,
            predicted_rul: Optional[float] = None,
            insights: Optional[Iterable[str]] = None,
            active_rules: Optional[Iterable[str]] = None,
            data_quality: Optional[Dict[str, Any]] = None,
            alert_confirmed: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """Append one decision record, honouring ``max_records``."""
        if not self.enabled:
            return None
        if self.max_records is not None and len(self.records) >= self.max_records:
            self.dropped += 1
            return None
        hashes = self.artifacts.get('hashes', {})
        versions = self.artifacts.get('versions', {})
        record = {
            'record_type': 'decision',
            'decision_id': make_decision_id(
                self.run_id, split, window_index, unit, cycle
            ),
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'split': split,
            'window_index': int(window_index),
            'unit': None if unit is None else int(unit),
            'cycle': None if cycle is None else int(cycle),
            'model_version': versions.get('model_version'),
            'features_version': versions.get('features_version'),
            'rules_version': versions.get('rules_version'),
            **{field: hashes.get(field) for field in ARTIFACT_FIELDS},
            'neural_threshold': (
                None if neural_threshold is None else float(neural_threshold)
            ),
            'fusion_threshold': (
                None if self.thresholds.get('fusion') is None
                else float(self.thresholds['fusion'])
            ),
            'threshold': float(threshold),
            'alert_mode': str(alert_mode),
            'alert_confirmed': (
                bool(alert_confirmed) if alert_confirmed is not None
                else bool(decision)
            ),
            'neural_probability': (
                None if neural_probability is None
                else float(neural_probability)
            ),
            'symbolic_rule_score': (
                None if symbolic_rule_score is None
                else float(symbolic_rule_score)
            ),
            'problog_failure_risk': (
                None if problog_failure_risk is None
                else float(problog_failure_risk)
            ),
            'predicted_rul': (
                None if predicted_rul is None else float(predicted_rul)
            ),
            'fused_probability': (
                None if fused_probability is None else float(fused_probability)
            ),
            'decision': int(decision),
            'active_rules': list(active_rules or []),
            'insights': list(insights or []),
            'data_quality': dict(data_quality or {}),
        }
        self.records.append(record)
        return record

    def flush(self) -> Optional[str]:
        """Write the manifest and all decision records to the JSONL journal."""
        if not self.enabled:
            return None
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, 'w') as file:
            file.write(json.dumps(self.manifest(
                decision_records=len(self.records),
                dropped_records=self.dropped,
            )) + '\n')
            for record in self.records:
                file.write(json.dumps(record) + '\n')
        if self.logger is not None:
            self.logger.info(
                "Decision audit trail: %d records written to %s.",
                len(self.records), self.path,
            )
        return self.path

    @property
    def count(self) -> int:
        """Return the number of buffered decision records."""
        return len(self.records)


def read_audit(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL audit journal and return its records."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audit journal not found at {path}")
    records = []
    with open(path) as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON: {error}"
                ) from error
    return records


def verify_audit(results_dir: str,
                 max_issues: int = 20) -> Dict[str, Any]:
    """Verify the coherence of an audit journal and its artifacts."""
    path = os.path.join(results_dir, AUDIT_FILENAME)
    issues: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return {
            'ok': False,
            'journal': path,
            'records': 0,
            'decisions': 0,
            'issues': [{'type': 'missing_journal', 'detail': path}],
        }
    try:
        records = read_audit(path)
    except (OSError, ValueError) as error:
        return {
            'ok': False,
            'journal': path,
            'records': 0,
            'decisions': 0,
            'issues': [{'type': 'unreadable_journal', 'detail': str(error)}],
        }

    manifests = [item for item in records
                 if item.get('record_type') == 'run_manifest']
    decisions = [item for item in records
                 if item.get('record_type') == 'decision']
    if not manifests:
        issues.append({
            'type': 'missing_manifest',
            'detail': 'no run_manifest record at the start of the journal',
        })
    manifest = manifests[0] if manifests else {}

    seen = set()
    for record in decisions:
        missing = [field for field in DECISION_REQUIRED_FIELDS
                   if field not in record]
        if missing:
            issues.append({
                'type': 'missing_field',
                'detail': f"{record.get('decision_id')}: {missing}",
            })
        identifier = record.get('decision_id')
        if identifier in seen:
            issues.append({
                'type': 'duplicate_decision_id',
                'detail': str(identifier),
            })
        seen.add(identifier)
        try:
            probability = record.get('fused_probability')
            if probability is None:
                probability = record.get('neural_probability')
            threshold = record.get('threshold')
            if probability is not None and threshold is not None:
                expected = int(float(probability) >= float(threshold))
                if int(record.get('decision', -1)) != expected:
                    issues.append({
                        'type': 'inconsistent_decision',
                        'detail': (
                            f"{identifier}: decision={record.get('decision')} "
                            f"but probability {float(probability):.6f} vs "
                            f"threshold {float(threshold):.6f}"
                        ),
                    })
        except (TypeError, ValueError) as error:
            issues.append({
                'type': 'invalid_decision_payload',
                'detail': f"{identifier}: {error}",
            })
        if record.get('alert_mode') not in {
                'immediate', 'consecutive_2', 'consecutive_3',
                'smoothed_3', 'smoothed_5'}:
            issues.append({
                'type': 'unknown_alert_mode',
                'detail': f"{identifier}: {record.get('alert_mode')}",
            })

    checked_artifacts = {}
    artifact_paths = manifest.get('artifact_paths', {}) or {}
    manifest_hashes = manifest.get('hashes', {}) or {}
    field_by_path = {
        'model': 'model_sha256',
        'scaler': 'scaler_sha256',
        'feature_metadata': 'feature_metadata_sha256',
        'rules': 'rules_pl_sha256',
        'learned_rules': 'rules_learned_pl_sha256',
        'problog_rules': 'problog_rules_sha256',
    }
    for name, field in field_by_path.items():
        artifact_path = artifact_paths.get(name)
        expected = manifest_hashes.get(field)
        if not artifact_path or not expected:
            continue
        current = sha256_file(artifact_path)
        status = 'ok' if current == expected else 'mismatch'
        checked_artifacts[name] = {'path': artifact_path, 'status': status}
        if status == 'mismatch':
            issues.append({
                'type': 'artifact_hash_mismatch',
                'detail': f"{name}: {artifact_path}",
            })

    thresholds = manifest.get('thresholds', {}) or {}
    threshold_files = {
        'fusion': os.path.join(results_dir, 'fusion_threshold.json'),
        'neural': os.path.join(results_dir, 'decision_threshold.json'),
    }
    for name, threshold_path in threshold_files.items():
        value = thresholds.get(name)
        if value is None or not os.path.exists(threshold_path):
            continue
        try:
            with open(threshold_path) as file:
                expected = float(json.load(file)['threshold'])
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if abs(float(value) - expected) > 1e-9:
            issues.append({
                'type': 'threshold_mismatch',
                'detail': f"{name}: journal {value} vs {threshold_path} {expected}",
            })

    return {
        'ok': not issues,
        'journal': path,
        'records': len(records),
        'decisions': len(decisions),
        'run_id': manifest.get('run_id'),
        'versions': manifest.get('versions', {}),
        'artifacts': checked_artifacts,
        'issues': issues[:int(max_issues)],
        'issue_count': len(issues),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: verify an audit journal and print the report."""
    parser = argparse.ArgumentParser(
        description='Verify the decision audit trail of a run.'
    )
    parser.add_argument('--results', default='results',
                        help='Results directory containing the audit journal.')
    parser.add_argument('--max-issues', type=int, default=20)
    args = parser.parse_args(argv)

    report = verify_audit(args.results, max_issues=args.max_issues)
    print(json.dumps(report, indent=2))
    if report['ok']:
        print(
            f"Audit OK: {report['decisions']} decisions, "
            f"{len(report['artifacts'])} artifacts verified."
        )
        return 0
    print(
        f"Audit FAILED: {report['issue_count']} issue(s).",
        file=sys.stderr,
    )
    return 1


if __name__ == '__main__':
    sys.exit(main())
