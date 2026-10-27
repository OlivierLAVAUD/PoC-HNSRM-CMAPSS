# src/utils/model_utils.py

import logging
import os
from typing import Any

import joblib


def save_scaler(scaler: Any, path: str, logger: logging.Logger) -> None:
    """Persist a fitted scaler with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")


def load_scaler(path: str, logger: logging.Logger) -> Any:
    """Load a fitted scaler from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found at {path}")
    scaler = joblib.load(path)
    logger.info(f"Scaler loaded from {path}")
    return scaler


def save_model(model: Any, path: str, logger: logging.Logger) -> None:
    """Persist a generic Python model with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model_with_initialization(path: str, logger: logging.Logger, input_shape=None) -> Any:
    """Load a joblib-serialized model (input_shape kept for API compatibility)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
