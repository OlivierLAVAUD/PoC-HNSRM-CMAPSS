# src/data/loader_factory.py
"""Select the data loader that matches the configured input file.

C-MAPSS parquet/text inputs use :class:`~src.data.cmapss_loader.CMAPSSLoader`.
Anything else falls back to the synthetic ``.npz`` loader, which lives in the
``legacy`` package because it is not part of the FD001 pipeline.
"""

from __future__ import annotations

from src.data.cmapss_loader import CMAPSSLoader


def create_data_loader(data_file: str, window_size: int, feature_names=None,
                       test_file: str = None, rul_file: str = None,
                       logger=None, rul_threshold: int = 30):
    """Return a CMAPSS loader for parquet/text data, else the synthetic loader."""
    if data_file.lower().endswith(('.parquet', '.txt')):
        return CMAPSSLoader(
            data_file,
            window_size=window_size,
            feature_names=feature_names,
            test_file=test_file or None,
            rul_file=rul_file or None,
            rul_threshold=rul_threshold,
            logger=logger,
        )
    from legacy.synthetic_data_loader import DataLoader
    return DataLoader(data_file, window_size=window_size)
