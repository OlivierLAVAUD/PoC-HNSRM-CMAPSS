# src/reasoning/thresholds.py
"""Derive the symbolic FD001 thresholds from the pipeline training split."""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.data.cmapss_loader import CMAPSSLoader, USEFUL_SENSORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_FILE = os.path.join(
    PROJECT_ROOT, 'data', 'raw', 'CMAPSS', 'FD001_processed.parquet'
)
DEFAULT_RULES_PATH = os.path.join(PROJECT_ROOT, 'src', 'reasoning', 'rules.pl')

HIGH_SENSORS: Tuple[str, ...] = (
    's2', 's3', 's4', 's8', 's9', 's11', 's13', 's14', 's15', 's17',
)
LOW_SENSORS: Tuple[str, ...] = ('s7', 's12', 's20', 's21')
NAMED_THRESHOLDS: Tuple[Tuple[str, str, int], ...] = (
    ('s4', 'elevated', 75),
    ('s4', 'critical', 95),
    ('s11', 'high', 75),
    ('s7', 'low', 25),
    ('s12', 'low', 25),
    ('s21', 'low', 25),
)


FACT_NAMES: Tuple[str, ...] = (
    'sensor_high_threshold',
    'sensor_low_threshold',
    'hpc_rule_threshold',
    'gradient_threshold',
)

_FACT_LINE = re.compile(
    r"^(sensor_high_threshold|sensor_low_threshold|gradient_threshold)"
    r"\(\s*([a-z][a-z0-9_]*)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)\."
)
_NAMED_FACT_LINE = re.compile(
    r"^hpc_rule_threshold\(\s*([a-z][a-z0-9_]*)\s*,\s*([a-z][a-z0-9_]*)\s*,"
    r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)\."
)


def parse_threshold_facts(rules_text: str) -> Dict[str, Dict]:
    """Read the threshold facts currently declared in a rules.pl file."""
    facts: Dict[str, Dict] = {name: {} for name in FACT_NAMES}
    for line in rules_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        match = _FACT_LINE.match(stripped)
        if match:
            facts[match.group(1)][match.group(2)] = float(match.group(3))
            continue
        match = _NAMED_FACT_LINE.match(stripped)
        if match:
            facts['hpc_rule_threshold'][
                (match.group(1), match.group(2))
            ] = float(match.group(3))
    return facts


def verify_train_provenance(rules_text: str, frame: pd.DataFrame,
                            train_units: Iterable[int]) -> Dict:
    """Compare rules.pl facts with thresholds derived from the training split."""
    derived = derive_thresholds(frame, train_units)
    facts = parse_threshold_facts(rules_text)
    checked = 0
    mismatches = []
    for name in FACT_NAMES:
        for key, expected in derived[name].items():
            checked += 1
            found = facts[name].get(key)
            if found is None or abs(found - expected) > 1e-9:
                mismatches.append({
                    'fact': name,
                    'key': ' '.join(key) if isinstance(key, tuple) else key,
                    'expected': float(expected),
                    'found': None if found is None else float(found),
                })
        for key, found in facts[name].items():
            if key in derived[name]:
                continue
            mismatches.append({
                'fact': name,
                'key': ' '.join(key) if isinstance(key, tuple) else key,
                'expected': None,
                'found': float(found),
            })
    return {'checked_facts': checked, 'mismatches': mismatches}


def derive_thresholds(frame: pd.DataFrame,
                      train_units: Iterable[int]) -> Dict[str, Dict]:
    """Return threshold facts computed on the given training engines only."""
    units = set(int(unit) for unit in train_units)
    training = frame[frame['unit_number'].isin(units)]
    if training.empty:
        raise ValueError("No rows found for the supplied training units.")

    high = {
        sensor: round(float(np.percentile(training[sensor], 95)), 3)
        for sensor in HIGH_SENSORS
    }
    low = {
        sensor: round(float(np.percentile(training[sensor], 5)), 3)
        for sensor in LOW_SENSORS
    }
    named = {
        (sensor, kind): round(
            float(np.percentile(training[sensor], percentile)), 3
        )
        for sensor, kind, percentile in NAMED_THRESHOLDS
    }
    deltas = training.groupby('unit_number')[list(USEFUL_SENSORS)].diff().abs()
    gradient = {
        sensor: round(float(np.percentile(deltas[sensor].dropna(), 95)), 3)
        for sensor in USEFUL_SENSORS
    }
    return {
        'sensor_high_threshold': high,
        'sensor_low_threshold': low,
        'hpc_rule_threshold': named,
        'gradient_threshold': gradient,
    }


def format_facts(thresholds: Dict[str, Dict],
                 train_count: int | None = None) -> str:
    """Format threshold values as a copy-pasteable Prolog fact block."""
    engine_label = (
        f"the {train_count} training engines" if train_count else "the training engines"
    )
    lines = [
        f'% 95th percentile of {engine_label}.',
    ]
    for sensor, value in thresholds['sensor_high_threshold'].items():
        lines.append(f"sensor_high_threshold({sensor}, {value:.3f}).")
    lines.extend([
        '',
        f'% 5th percentile of {engine_label}.',
    ])
    for sensor, value in thresholds['sensor_low_threshold'].items():
        lines.append(f"sensor_low_threshold({sensor}, {value:.3f}).")
    lines.extend([
        '',
        '% Named HPC rule thresholds: 75th / 95th / 25th percentile of',
        f'% {engine_label}.',
    ])
    for (sensor, kind), value in thresholds['hpc_rule_threshold'].items():
        lines.append(f"hpc_rule_threshold({sensor}, {kind}, {value:.3f}).")
    lines.extend([
        '',
        '% 95th percentile of absolute cycle-to-cycle change on',
        f'% {engine_label} (rule extraction only).',
    ])
    for sensor, value in thresholds['gradient_threshold'].items():
        lines.append(f"gradient_threshold({sensor}, {value:.3f}).")
    return '\n'.join(lines) + '\n'


def _rewrite_facts(rules_text: str, name: str,
                   values: Dict, named: bool = False) -> str:
    """Replace existing fact lines in place, failing if any is missing."""
    for key, value in values.items():
        if named:
            sensor, kind = key
            pattern = (
                rf"^{name}\(\s*{sensor}\s*,\s*{kind}\s*,\s*"
                rf"[-0-9.]+\s*\)\."
            )
            replacement = f"{name}({sensor}, {kind}, {value:.3f})."
        else:
            sensor = key
            pattern = rf"^{name}\(\s*{sensor}\s*,\s*[-0-9.]+\s*\)\."
            replacement = f"{name}({sensor}, {value:.3f})."
        rules_text, count = re.subn(
            pattern, replacement, rules_text, count=1, flags=re.MULTILINE
        )
        if count != 1:
            raise ValueError(f"Could not find a unique {name} fact for {key}.")
    return rules_text


def update_rules_file(path: str, thresholds: Dict[str, Dict]) -> None:
    """Rewrite the threshold facts in an existing rules.pl file."""
    with open(path) as rules_file:
        rules_text = rules_file.read()
    rules_text = _rewrite_facts(
        rules_text, 'sensor_high_threshold',
        thresholds['sensor_high_threshold'],
    )
    rules_text = _rewrite_facts(
        rules_text, 'sensor_low_threshold',
        thresholds['sensor_low_threshold'],
    )
    rules_text = _rewrite_facts(
        rules_text, 'hpc_rule_threshold',
        thresholds['hpc_rule_threshold'], named=True,
    )
    rules_text = _rewrite_facts(
        rules_text, 'gradient_threshold',
        thresholds['gradient_threshold'],
    )
    temp_path = path + '.tmp'
    with open(temp_path, 'w') as rules_file:
        rules_file.write(rules_text)
    os.replace(temp_path, path)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute rules.pl symbolic thresholds from the FD001 training "
            "split used by the pipeline."
        )
    )
    parser.add_argument('--data', default=DEFAULT_DATA_FILE,
                        help='FD001 processed parquet or raw train file.')
    parser.add_argument('--rules', default=DEFAULT_RULES_PATH,
                        help='rules.pl to read/update.')
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--rul-threshold', type=int, default=30)
    parser.add_argument('--validation-split', type=float, default=0.2)
    parser.add_argument('--test-split', type=float, default=0.6)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--write', action='store_true',
                        help='Update the threshold facts in --rules in place.')
    args = parser.parse_args(argv)

    loader = CMAPSSLoader(
        args.data,
        window_size=args.window_size,
        rul_threshold=args.rul_threshold,
    )
    X, y = loader.load_data()
    loader.split_by_unit(
        X, y,
        validation_split=args.validation_split,
        test_split=args.test_split,
        random_state=args.random_state,
    )
    train_mask = loader.last_split_masks['train']
    train_units = sorted(set(np.asarray(loader.unit_ids)[train_mask].tolist()))
    frame = loader._read_frame(args.data)
    thresholds = derive_thresholds(frame, train_units)

    print(format_facts(thresholds, len(train_units)))
    print(f"Training engines ({len(train_units)}): {train_units}")

    if args.write:
        update_rules_file(args.rules, thresholds)
        print(f"Updated threshold facts in {args.rules}")
    else:
        print("Pass --write to update the threshold facts in rules.pl.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
