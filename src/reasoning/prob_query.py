# src/reasoning/prob_query.py
"""Evaluate the ProbLog HPC failure model on representative FD001 states."""

import json
import logging
import os
import re
import sys

from problog import get_evaluatable
from problog.program import PrologString

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RULES_PATH = os.path.join(SCRIPT_DIR, 'prob_rules.pl')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Named HPC thresholds are read from rules.pl, which is the single source of
# truth (elevated / high = 75th percentile, critical = 95th, low = 25th).
RULES_PL_PATH = os.path.join(SCRIPT_DIR, 'rules.pl')
_THRESHOLD_PATTERN = re.compile(
    r"hpc_rule_threshold\(\s*([a-z][a-z0-9_]*)\s*,\s*"
    r"([a-z][a-z0-9_]*)\s*,\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\)\s*\."
)


def load_hpc_thresholds(path: str = RULES_PL_PATH) -> dict:
    """Read the named HPC thresholds declared in a Prolog rule file."""
    thresholds = {}
    with open(path, 'r') as file:
        for line in file:
            if line.lstrip().startswith('%'):
                continue
            for sensor, kind, value in _THRESHOLD_PATTERN.findall(line):
                thresholds[f'{sensor}_{kind}'] = float(value)
    if not thresholds:
        raise ValueError(f"No hpc_rule_threshold facts found in {path}")
    return thresholds


THRESHOLDS = load_hpc_thresholds()

EXAMPLES = {
    'normal': {
        's2': 641.82, 's3': 1589.70, 's4': 1400.60, 's7': 554.36,
        's8': 2388.06, 's9': 9046.19, 's11': 47.47, 's12': 521.66,
        's13': 2388.02, 's14': 8138.62, 's15': 8.419, 's17': 392.0,
        's20': 39.06, 's21': 23.419,
    },
    'degraded': {
        's2': 642.67, 's3': 1598.60, 's4': 1420.83, 's7': 552.62,
        's8': 2388.18, 's9': 9043.90, 's11': 47.72, 's12': 520.39,
        's13': 2388.19, 's14': 8119.00, 's15': 8.457, 's17': 395.0,
        's20': 38.77, 's21': 23.237,
    },
    'critical': {
        's2': 643.54, 's3': 1601.41, 's4': 1427.20, 's7': 551.25,
        's8': 2388.32, 's9': 9033.22, 's11': 48.25, 's12': 520.08,
        's13': 2388.32, 's14': 8110.93, 's15': 8.511, 's17': 396.0,
        's20': 38.48, 's21': 22.965,
    },
}


def extract_conditions(sensor_values, thresholds=None):
    """Map raw sensors to deterministic HPC conditions used by ProbLog.

    ``thresholds`` defaults to the named facts of the packaged ``rules.pl``.
    A caller that reasons with a different rule file (for example a per-seed
    copy produced by a multi-seed run) must pass its own thresholds so the
    evidence stays consistent with the symbolic layer.
    """
    values = THRESHOLDS if thresholds is None else thresholds
    return {
        'hpc_temp_elevated': sensor_values['s4'] > values['s4_elevated'],
        'hpc_temp_critical': sensor_values['s4'] > values['s4_critical'],
        'hpc_static_pressure_high': sensor_values['s11'] > values['s11_high'],
        'hpc_outlet_pressure_low': sensor_values['s7'] < values['s7_low'],
        'fuel_flow_low': sensor_values['s12'] < values['s12_low'],
        'coolant_bleed_low': sensor_values['s21'] < values['s21_low'],
    }


def _probability(result, name):
    """Read a query probability from a ProbLog result mapping."""
    for key, value in result.items():
        if str(key) == name:
            return float(value)
    return 0.0


def _normalise_learned_rules(learned_rules):
    """Accept rule names or metadata dictionaries and return unique names."""
    names = []
    for item in learned_rules or []:
        if isinstance(item, dict):
            name = item.get('rule') or item.get('name')
        else:
            name = item
        if not name:
            continue
        name = str(name)
        if name not in names:
            names.append(name)
    return names


def evaluate_state(base_program, sensor_values, learned_rules=None,
                   learned_rule_weight=0.15, max_active_learned_rules=3,
                   hpc_thresholds=None):
    """Append evidence for one state and evaluate failure-risk queries.

    Active learned HITL rules are injected as independent probabilistic facts
    with a capped weight. If ``w`` is the weight and ``n`` active rules are
    retained, their combined contribution is the noisy-or
    ``1 - (1 - w)**n``. A single rule therefore cannot dominate the final risk,
    and ``max_active_learned_rules`` bounds the total contribution.
    """
    conditions = extract_conditions(sensor_values, hpc_thresholds)
    evidence = [name for name, active in conditions.items() if active]
    program = base_program + '\n' + ''.join(f'{name}.\n' for name in evidence)

    weight = float(learned_rule_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("learned_rule_weight must be in [0, 1].")
    max_active = max(0, int(max_active_learned_rules))
    active_learned = _normalise_learned_rules(learned_rules)[:max_active]

    for index, rule_name in enumerate(active_learned):
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', rule_name)
        evidence_name = f"learned_evidence_{index}_{safe_name}"
        program += f"\n{weight:.6f}::{evidence_name}.\n"
        program += f"failure_risk :- {evidence_name}.\n"

    result = get_evaluatable().create_from(PrologString(program)).evaluate()
    return {
        'failure_risk': _probability(result, 'failure_risk'),
        'hpc_degradation': _probability(result, 'hpc_degradation'),
        'active_conditions': evidence,
        'learned_rules_active': active_learned,
        'learned_rule_weight': weight,
        'learned_rules_cap': max_active,
    }


def run_prob_log_queries():
    """Evaluate normal, degraded and critical examples and persist the table."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(LOG_DIR, 'prob_log_queries.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    try:
        with open(RULES_PATH, 'r') as file:
            base_program = file.read()
    except FileNotFoundError:
        print(f"Error: ProbLog rules file not found at {RULES_PATH}")
        sys.exit(1)

    output = {}
    print(f"{'Example':<10} {'P(failure_risk)':>16} {'P(hpc_degradation)':>19}  Evidence")
    print('-' * 96)
    for label, sensor_values in EXAMPLES.items():
        result = evaluate_state(base_program, sensor_values)
        output[label] = result
        evidence = ', '.join(result['active_conditions']) or '-'
        print(f"{label:<10} {result['failure_risk']:>16.4f} "
              f"{result['hpc_degradation']:>19.4f}  {evidence}")
        logging.info("%s: %s", label, result)

    output_path = os.path.join(RESULTS_DIR, 'prob_log_examples.json')
    with open(output_path, 'w') as file:
        json.dump(output, file, indent=2)
    print(f"\nSaved ProbLog example results to {output_path}")


if __name__ == "__main__":
    run_prob_log_queries()
