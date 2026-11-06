# tests/test_pipeline_representation.py

import logging
import os
import tempfile
import unittest

from src.pipeline.pipeline import ANSRDTPipeline
from tests.test_pipeline_analyses import write_project


class TestPipelineRepresentation(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('TestPipelineRepresentation')
        self.logger.addHandler(logging.NullHandler())
        self.logger.setLevel(logging.WARNING)

    def test_disabled_representation_preserves_legacy_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, path = write_project(tmpdir)
            pipeline = ANSRDTPipeline(config, path, self.logger)
            splits = pipeline._load_cmapss_splits()
            prepared = pipeline._prepare_model_data(splits, 'random_forest')
            self.assertFalse(prepared['uses_representation'])
            self.assertIsNone(prepared['representation'])
            self.assertEqual(prepared['X_train'].shape[0], len(splits['X_train']))

    def test_enabled_representation_is_saved_and_applied_before_scaler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, path = write_project(tmpdir)
            config['representation'] = {
                'enabled': True,
                'variant': 'deltas',
                'early_life_cycles': 5,
                'apply_to_architectures': ['random_forest'],
            }
            pipeline = ANSRDTPipeline(config, path, self.logger)
            splits = pipeline._load_cmapss_splits()
            prepared = pipeline._prepare_model_data(splits, 'random_forest')
            self.assertTrue(prepared['uses_representation'])
            self.assertIsNotNone(prepared['representation'])
            self.assertTrue(os.path.exists(
                prepared['paths']['representation']
            ))
            self.assertEqual(
                prepared['X_train'].shape,
                (len(splits['X_train']),
                 len(prepared['representation'].feature_names_out)),
            )
            self.assertEqual(
                prepared['representation'].training_hash,
                prepared['representation_report']['training_hash'],
            )


if __name__ == '__main__':
    unittest.main()
