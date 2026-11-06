# src/data/cmapss_datasets.py
"""Generic C-MAPSS subset definitions and dataset-specific validation.

FD001 remains the default. FD002-FD004 share the same 21 raw telemetry columns
but differ in operating conditions and fault modes, so their configurations
inherit ``configs/config.yaml`` and override identity, paths and window shape.

This module validates those effective configurations *before* any data file is
read: the sensor count, the window shape and the file naming convention are
checked explicitly, which turns a mis-typed sensor list or a missing download
into an actionable message instead of a cryptic loader error.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

CMAPSS_DATASETS: tuple = ('fd001', 'fd002', 'fd003', 'fd004')

DATASET_CONDITIONS: Dict[str, int] = {
    'fd001': 1,
    'fd002': 6,
    'fd003': 2,
    'fd004': 6,
}

RAW_COLUMNS: tuple = tuple(
    ['unit_number', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    + [f's{i}' for i in range(1, 22)]
)

DATA_CONVENTIONS: Dict[str, str] = {
    'data_file': 'data/raw/CMAPSS/{ID}_processed.parquet',
    'test_file': 'data/raw/CMAPSS/test_{ID}.txt',
    'rul_file': 'data/raw/CMAPSS/RUL_{ID}.txt',
}


def dataset_id(config: Dict[str, Any]) -> str:
    """Return the normalised dataset identifier (``fd001``-``fd004``)."""
    dataset = config.get('dataset', {}) or {}
    value = str(dataset.get('id') or '').lower()
    if value in CMAPSS_DATASETS:
        return value
    for token in CMAPSS_DATASETS:
        if token in str(config.get('paths', {}).get('data_file', '')).lower():
            return token
    return value


def missing_data_message(config: Dict[str, Any], missing: Sequence[str]) -> str:
    """Return a readable setup message for absent C-MAPSS files."""
    identifier = dataset_id(config).upper() or 'FD001'
    lines = [
        f"C-MAPSS {identifier} data files are missing:",
    ]
    for key in missing:
        configured = str(config.get('paths', {}).get(key, ''))
        lines.append(f"  - {key}: {configured}")
    lines.append("Expected layout:")
    for key, template in DATA_CONVENTIONS.items():
        lines.append(f"  - {template.format(ID=identifier)}  ({key})")
    lines.append(
        "Download the NASA C-MAPSS archive, build the processed parquet with "
        "the documented columns, or point paths.data_file / test_file / "
        "rul_file at existing files. Use --cmapss-dataset fd001 (the default) "
        "to run the published FD001 protocol."
    )
    return "\n".join(lines)


def validate_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate dataset identity, sensors, window shape and path conventions.

    Returns a small report describing the effective dataset configuration.
    Raises ``ValueError`` for an unsupported dataset id, a duplicated or
    unknown sensor name, an inconsistent window/input shape, or paths that do
    not belong to the configured dataset.
    """
    identifier = dataset_id(config)
    if identifier not in CMAPSS_DATASETS:
        raise ValueError(
            f"Unsupported C-MAPSS dataset id: {identifier!r}. "
            f"Allowed: {list(CMAPSS_DATASETS)}."
        )

    model = config.get('model', {}) or {}
    feature_names: List[str] = list(model.get('feature_names') or [])
    if not feature_names:
        raise ValueError("model.feature_names must list the C-MAPSS sensors.")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("model.feature_names contains duplicate sensors.")
    unknown = [name for name in feature_names if name not in RAW_COLUMNS]
    if unknown:
        raise ValueError(
            f"Unknown C-MAPSS sensor(s) in model.feature_names: {unknown}. "
            "Expected names such as 's2', 's3', 's11', 's21'."
        )

    window_size = model.get('window_size')
    input_shape = model.get('input_shape')
    expected_shape = [int(window_size), len(feature_names)]
    if list(input_shape or []) != expected_shape:
        raise ValueError(
            f"model.input_shape must be {expected_shape} for {identifier.upper()} "
            f"({len(feature_names)} sensors, window {window_size}), "
            f"got {input_shape!r}."
        )

    paths = config.get('paths', {}) or {}
    wrong_paths = []
    for key in ('data_file', 'test_file', 'rul_file'):
        value = str(paths.get(key) or '')
        if value and identifier.upper() not in value.upper():
            wrong_paths.append(f"{key}={value!r}")
    if wrong_paths:
        raise ValueError(
            f"Paths for {identifier.upper()} must reference the same dataset: "
            f"{wrong_paths}."
        )
    results_dir = str(paths.get('results_dir') or '')
    if results_dir and identifier != 'fd001' and identifier not in results_dir.lower():
        raise ValueError(
            f"paths.results_dir for {identifier.upper()} must contain "
            f"{identifier!r}, got {results_dir!r}; datasets must not share "
            "a results directory."
        )

    missing = [
        key for key in ('data_file', 'test_file', 'rul_file')
        if not paths.get(key) or not os.path.exists(str(paths[key]))
    ]
    operating_conditions = (config.get('dataset', {}) or {}).get(
        'operating_conditions'
    )
    expected_conditions = DATASET_CONDITIONS[identifier]
    warnings = []
    if (operating_conditions is not None
            and int(operating_conditions) != expected_conditions):
        warnings.append(
            f"dataset.operating_conditions is {operating_conditions}, but "
            f"{identifier.upper()} has {expected_conditions} condition(s)."
        )
    return {
        'dataset': identifier,
        'sensor_count': len(feature_names),
        'window_size': int(window_size),
        'input_shape': expected_shape,
        'operating_conditions': expected_conditions,
        'results_dir': results_dir,
        'missing_files': missing,
        'warnings': warnings,
    }
