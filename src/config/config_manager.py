# src/config/config_manager.py

import copy
import os
from typing import Any, Dict

import yaml
import logging


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` and return a new mapping.

    Nested dictionaries are merged key by key, so a dataset configuration can
    override only its paths without duplicating the whole base file. Lists and
    scalars from the overlay replace the base value.
    """
    merged = copy.deepcopy(base or {})
    for key, value in (overlay or {}).items():
        if (isinstance(value, dict) and isinstance(merged.get(key), dict)):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: str) -> Dict[str, Any]:
    """Read one YAML mapping, raising a clear error when it is malformed."""
    with open(path, 'r') as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping.")
    return payload


def _resolve_extends(config_path: str,
                     payload: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an optional ``extends`` chain relative to the config file."""
    current_path = os.path.abspath(config_path)
    current = dict(payload)
    seen = set()
    while current.get('extends'):
        if current_path in seen:
            raise ValueError(
                f"Circular 'extends' chain detected in {config_path}."
            )
        seen.add(current_path)
        parent = str(current.pop('extends'))
        parent_path = (
            parent if os.path.isabs(parent)
            else os.path.normpath(os.path.join(
                os.path.dirname(current_path), parent
            ))
        )
        if not os.path.exists(parent_path):
            raise FileNotFoundError(
                f"Base configuration referenced by 'extends' not found: "
                f"{parent_path}"
            )
        parent_payload = _resolve_extends(parent_path, _read_yaml(parent_path))
        current = deep_merge(parent_payload, current)
        current_path = parent_path
    return current


def load_config(config_path: str) -> dict:
    """
    Loads the YAML configuration file with validation.

    A configuration may declare ``extends: config.yaml`` to inherit a base
    file and override only the dataset-specific keys. The base is resolved
    relative to the extending file before validation.

    Raises:
        KeyError: If required keys are missing.
        FileNotFoundError: If an ``extends`` target does not exist.
        ValueError: If the YAML payload or an ``extends`` chain is invalid.
    """
    try:
        config = _resolve_extends(config_path, _read_yaml(config_path))

        # Validate essential keys
        required_sections = ['model', 'training', 'paths']
        for section in required_sections:
            if section not in config:
                raise KeyError(f"Missing required configuration section: '{section}'")

        # Validate model-specific keys
        model_required_keys = ['window_size', 'feature_names', 'input_shape', 'architecture']
        for key in model_required_keys:
            if key not in config['model']:
                raise KeyError(f"Missing required model configuration key: '{key}'")

        logging.getLogger(__name__).info(f"Configuration loaded and validated from {config_path}.")
        return config
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load configuration: {e}")
        raise
