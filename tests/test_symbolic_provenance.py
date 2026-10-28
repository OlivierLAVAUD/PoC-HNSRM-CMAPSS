# tests/test_symbolic_provenance.py
"""Check that symbolic thresholds are verified against the training split."""

import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.cmapss_loader import CMAPSSLoader, USEFUL_SENSORS
from src.pipeline.pipeline import ANSRDTPipeline
from src.reasoning.prob_query import RULES_PL_PATH
from src.reasoning.thresholds import (
    derive_thresholds,
    format_facts,
    verify_train_provenance,
)

FD001_PARQUET = os.path.join('data', 'raw', 'CMAPSS', 'FD001_processed.parquet')


def _frame() -> pd.DataFrame:
    """Build a deterministic two-engine frame covering every used sensor."""
    rows = []
    for unit in (1, 2):
        for cycle in range(40):
            row = {'unit_number': unit}
            for index, sensor in enumerate(USEFUL_SENSORS):
                row[sensor] = float(unit * 100 + cycle + index)
            rows.append(row)
    return pd.DataFrame(rows)


class _StubLoader:
    """Minimal C-MAPSS loader exposing the frame read used by the pipeline."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def _read_frame(self, path: str) -> pd.DataFrame:
        return self._frame


def _pipeline(frame: pd.DataFrame, rules_path: str,
              data_file: str) -> ANSRDTPipeline:
    """Build a pipeline instance without loading data or models."""
    pipeline = object.__new__(ANSRDTPipeline)
    pipeline.logger = logging.getLogger('test.symbolic_provenance')
    pipeline.symbolic_enabled = True
    pipeline.is_cmapss = True
    pipeline.data_loader = _StubLoader(frame)
    pipeline.config = {
        'paths': {
            'data_file': data_file,
            'reasoning_rules_path': rules_path,
        },
    }
    return pipeline


class TestVerifyTrainProvenance(unittest.TestCase):
    def setUp(self):
        self.frame = _frame()
        self.rules_text = format_facts(
            derive_thresholds(self.frame, [1, 2]), train_count=2
        )

    def test_matching_rules_report_no_mismatch(self):
        report = verify_train_provenance(self.rules_text, self.frame, [1, 2])
        self.assertEqual(report['mismatches'], [])
        self.assertGreater(report['checked_facts'], 0)

    def test_stale_threshold_is_reported(self):
        stale = self.rules_text.replace('hpc_rule_threshold(s4, critical, ',
                                        'hpc_rule_threshold(s4, critical, 1')
        report = verify_train_provenance(stale, self.frame, [1, 2])
        self.assertEqual(len(report['mismatches']), 1)
        mismatch = report['mismatches'][0]
        self.assertEqual(mismatch['fact'], 'hpc_rule_threshold')
        self.assertEqual(mismatch['key'], 's4 critical')

    def test_thresholds_from_all_engines_are_rejected(self):
        """Thresholds computed on every engine must fail the training check."""
        all_engine_rules = format_facts(
            derive_thresholds(self.frame, [1, 2]), train_count=2
        )
        # Derive the same facts from a different unit set: the values change.
        other = format_facts(derive_thresholds(self.frame, [2]), train_count=1)
        self.assertNotEqual(all_engine_rules, other)
        report = verify_train_provenance(other, self.frame, [1, 2])
        self.assertTrue(report['mismatches'])


class TestPipelineProvenanceCheck(unittest.TestCase):
    def setUp(self):
        self.frame = _frame()
        handle, self.data_file = tempfile.mkstemp(suffix='.parquet')
        os.close(handle)
        handle, self.rules_file = tempfile.mkstemp(suffix='.pl')
        os.close(handle)
        with open(self.rules_file, 'w') as rules:
            rules.write(format_facts(
                derive_thresholds(self.frame, [1, 2]), train_count=2
            ))
        self.splits = {'train_units': np.array([1, 1, 2, 2])}

    def tearDown(self):
        for path in (self.data_file, self.rules_file):
            if os.path.exists(path):
                os.remove(path)

    def test_pipeline_accepts_training_derived_thresholds(self):
        pipeline = _pipeline(self.frame, self.rules_file, self.data_file)
        report = pipeline._verify_symbolic_train_provenance(self.splits)

        self.assertTrue(report['checked'])
        self.assertEqual(report['split'], 'train')
        self.assertEqual(report['train_engines'], 2)
        self.assertEqual(report['mismatches'], [])

    def test_pipeline_rejects_thresholds_from_another_split(self):
        pipeline = _pipeline(self.frame, self.rules_file, self.data_file)
        pipeline.data_loader = _StubLoader(_frame().assign(
            s4=lambda frame: frame['s4'] + 500.0
        ))
        with self.assertRaises(RuntimeError) as context:
            pipeline._verify_symbolic_train_provenance(self.splits)
        self.assertIn('do not match the training split', str(context.exception))
        self.assertIn('src.reasoning.thresholds --write', str(context.exception))

    def test_check_is_skipped_without_symbolic_layer(self):
        pipeline = _pipeline(self.frame, self.rules_file, self.data_file)
        pipeline.symbolic_enabled = False
        report = pipeline._verify_symbolic_train_provenance(self.splits)
        self.assertFalse(report['checked'])

    def test_check_is_skipped_for_non_cmapss_runs(self):
        pipeline = _pipeline(self.frame, self.rules_file, self.data_file)
        pipeline.is_cmapss = False
        report = pipeline._verify_symbolic_train_provenance(self.splits)
        self.assertFalse(report['checked'])


@unittest.skipUnless(os.path.exists(FD001_PARQUET), "FD001 parquet not downloaded")
class TestFD001PipelineProvenance(unittest.TestCase):
    def test_committed_rules_match_the_twenty_training_engines(self):
        loader = CMAPSSLoader(FD001_PARQUET, window_size=30, rul_threshold=30)
        X, y = loader.load_data()
        loader.split_by_unit(
            X, y, validation_split=0.2, test_split=0.6, random_state=42
        )
        train_mask = loader.last_split_masks['train']
        train_units = np.asarray(loader.unit_ids)[train_mask]

        pipeline = object.__new__(ANSRDTPipeline)
        pipeline.logger = logging.getLogger('test.symbolic_provenance')
        pipeline.symbolic_enabled = True
        pipeline.is_cmapss = True
        pipeline.data_loader = loader
        pipeline.config = {
            'paths': {
                'data_file': FD001_PARQUET,
                'reasoning_rules_path': RULES_PL_PATH,
            },
        }

        report = pipeline._verify_symbolic_train_provenance(
            {'train_units': train_units}
        )

        self.assertTrue(report['checked'])
        self.assertEqual(report['train_engines'], 20)
        self.assertEqual(report['mismatches'], [])


if __name__ == '__main__':
    unittest.main()
