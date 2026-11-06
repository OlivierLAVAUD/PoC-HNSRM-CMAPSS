"""Auxiliary Remaining-Useful-Life regressor used by the LightGBM path."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np


class RULLightGBMRegressor:
    """Small wrapper around LightGBM regression with persistence and metrics."""

    def __init__(self, random_state: int = 42,
                 logger: Optional[logging.Logger] = None, **kwargs):
        self.random_state = int(random_state)
        self.logger = logger or logging.getLogger(__name__)
        self.kwargs = dict(kwargs)
        self.model = self._build_estimator()
        self.feature_names: Optional[list] = None

    def _build_estimator(self):
        """Build the LightGBM regressor, failing with a clear message."""
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:  # pragma: no cover - exercised by install docs
            raise ImportError(
                "LightGBM is required for RUL regression. Install it with "
                "`pip install 'lightgbm>=4.5,<5'`."
            ) from error

        params = {
            "objective": "regression",
            "n_estimators": int(self.kwargs.get("n_estimators", 300)),
            "learning_rate": float(self.kwargs.get("learning_rate", 0.05)),
            "num_leaves": int(self.kwargs.get("num_leaves", 31)),
            "max_depth": int(self.kwargs.get("max_depth", -1)),
            "min_child_samples": int(self.kwargs.get("min_child_samples", 20)),
            "subsample": float(self.kwargs.get("subsample", 0.9)),
            "colsample_bytree": float(self.kwargs.get("colsample_bytree", 0.9)),
            "reg_lambda": float(self.kwargs.get("reg_lambda", 0.0)),
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if params["subsample"] < 1.0:
            params["subsample_freq"] = 1
        return LGBMRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RULLightGBMRegressor":
        """Fit the regressor on features and continuous RUL targets."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError("Feature and RUL arrays must have the same length.")
        self.feature_names = (
            list(self.kwargs["feature_names"])
            if "feature_names" in self.kwargs else None
        )
        self.model.fit(X, y)
        self.logger.info("RUL regressor fitted on %d samples.", len(X))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous RUL values."""
        return np.asarray(self.model.predict(np.asarray(X, dtype=np.float32)),
                          dtype=float).ravel()

    @staticmethod
    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Return RMSE, MAE and rank concordance for RUL predictions."""
        true = np.asarray(y_true, dtype=float).ravel()
        pred = np.asarray(y_pred, dtype=float).ravel()
        if len(true) != len(pred):
            raise ValueError("RUL targets and predictions must have the same length.")
        if not len(true):
            return {"rmse": 0.0, "mae": 0.0, "spearman": 0.0}
        error = pred - true
        rmse = float(np.sqrt(np.mean(error ** 2)))
        mae = float(np.mean(np.abs(error)))
        if len(true) < 2 or np.allclose(true, true[0]) or np.allclose(pred, pred[0]):
            spearman = 0.0
        else:
            true_rank = np.argsort(np.argsort(true))
            pred_rank = np.argsort(np.argsort(pred))
            spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])
        return {"rmse": rmse, "mae": mae, "spearman": spearman}

    def save(self, path: str) -> str:
        """Persist the fitted regressor with joblib."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str,
             logger: Optional[logging.Logger] = None) -> "RULLightGBMRegressor":
        """Load a persisted regressor."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"RUL regressor not found at {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Object at {path} is not a RULLightGBMRegressor")
        if logger is not None:
            instance.logger = logger
        return instance



class RULEnsembleRegressor:
    """Small LightGBM ensemble exposing a median RUL and its dispersion."""

    def __init__(self, n_models: int = 5, random_state: int = 42,
                 logger: Optional[logging.Logger] = None, **kwargs):
        self.n_models = max(1, int(n_models))
        self.random_state = int(random_state)
        self.logger = logger or logging.getLogger(__name__)
        self.kwargs = dict(kwargs)
        self.members = [
            RULLightGBMRegressor(
                random_state=self.random_state + index,
                logger=self.logger,
                **self.kwargs,
            )
            for index in range(self.n_models)
        ]
        self.feature_names: Optional[list] = None

    @staticmethod
    def _sanitize(X: np.ndarray) -> np.ndarray:
        """Keep missing values as NaN and reject infinities."""
        values = np.asarray(X, dtype=np.float32)
        return np.where(np.isfinite(values), values, np.nan).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RULEnsembleRegressor":
        """Fit every member on the same training data with its own seed."""
        values = self._sanitize(X)
        for member in self.members:
            member.fit(values, y)
        self.logger.info(
            "RUL ensemble fitted with %d members.", len(self.members)
        )
        return self

    def predict_distribution(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return median, standard deviation and a [q10, q90] interval."""
        values = self._sanitize(X)
        predictions = np.vstack([
            member.predict(values) for member in self.members
        ])
        median = np.median(predictions, axis=0)
        std = np.std(predictions, axis=0, ddof=0)
        lower = np.quantile(predictions, 0.10, axis=0)
        upper = np.quantile(predictions, 0.90, axis=0)
        return {
            'median': np.asarray(median, dtype=float).ravel(),
            'std': np.asarray(std, dtype=float).ravel(),
            'q10': np.asarray(lower, dtype=float).ravel(),
            'q90': np.asarray(upper, dtype=float).ravel(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the median ensemble RUL prediction."""
        return self.predict_distribution(X)['median']

    @staticmethod
    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Return the same metrics as the single RUL regressor."""
        return RULLightGBMRegressor.metrics(y_true, y_pred)

    def save(self, path: str) -> str:
        """Persist the fitted ensemble with joblib."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str,
             logger: Optional[logging.Logger] = None
             ) -> "RULEnsembleRegressor":
        """Load a persisted ensemble."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"RUL ensemble not found at {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Object at {path} is not a RULEnsembleRegressor")
        if logger is not None:
            instance.logger = logger
        return instance


def out_of_fold_predictions(
        regressor_factory,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5) -> np.ndarray:
    """Generate unit-grouped out-of-fold RUL predictions.

    Grouping by engine prevents adjacent windows from the same trajectory from
    appearing in both the regressor's fitting and prediction folds.
    """
    from sklearn.model_selection import GroupKFold

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    n_groups = len(np.unique(groups))
    splits = min(max(2, int(n_splits)), n_groups)
    if splits < 2:
        full = regressor_factory().fit(X, y)
        return full.predict(X)
    predictions = np.zeros(len(y), dtype=float)
    for train_index, valid_index in GroupKFold(n_splits=splits).split(
            X, y, groups):
        fold_model = regressor_factory().fit(X[train_index], y[train_index])
        predictions[valid_index] = fold_model.predict(X[valid_index])
    return predictions
